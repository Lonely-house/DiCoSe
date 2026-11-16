"""Two-stage training and .npy inference; run with python -m ...train --help."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .model import HybridSeparator, ModelConfig


def load_array(path, audio=False):
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{path}: expected a finite numeric .npy array")
    with np.errstate(over="ignore", invalid="ignore"):
        value = value.astype(np.float32, copy=True)
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: values cannot be represented as finite float32")
    value = torch.from_numpy(value)
    if audio and value.ndim == 1:
        value = value[None]
    return value


class QueryDataset(Dataset):
    """Row paths are relative to the manifest; shared aligned random crops."""

    def __init__(self, manifest, config, sample_rate, crop_frames):
        self.path = Path(manifest).resolve()
        with self.path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError("manifest must contain at least one row")
        if crop_frames < 1 or sample_rate < 1:
            raise ValueError("crop_frames and sample_rate must be positive")
        self.config, self.sample_rate, self.crop_frames = config, sample_rate, crop_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get("sample_rate") != self.sample_rate:
            raise ValueError("all manifest sample rates must match the configured sample rate")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("each row must describe its query in a nonempty prompt string")
        present = row.get("target_present", True)
        if not isinstance(present, bool):
            raise ValueError("target_present must be a JSON boolean")
        mixture = load_array(self.path.parent / row["mix"], audio=True)
        if mixture.ndim != 2 or mixture.shape[0] != self.config.channels or mixture.shape[1] < 1:
            raise ValueError("audio must be (channels, frames>0), or (frames,) for mono")
        if row.get("target") is None:
            if present:
                raise ValueError("present queries require a target file")
            target = torch.zeros_like(mixture)
        else:
            target = load_array(self.path.parent / row["target"], audio=True)
        if target.shape != mixture.shape:
            raise ValueError("mix and target must have identical, aligned shapes")
        if not present and target.abs().max() > 1e-6:
            raise ValueError("absent-query target must be zero across the full file")
        query = load_array(self.path.parent / row["prompt_embedding"])
        if query.shape != (self.config.query_dim,) or query.norm() <= 1e-12:
            raise ValueError("prompt_embedding must be a nonzero vector with configured query_dim")
        frames = mixture.shape[-1]
        if frames >= self.crop_frames:
            start = int(torch.randint(frames - self.crop_frames + 1, ()))
            mixture = mixture[:, start:start + self.crop_frames]
            target = target[:, start:start + self.crop_frames]
        else:
            mixture = F.pad(mixture, (0, self.crop_frames - frames))
            target = F.pad(target, (0, self.crop_frames - frames))
        return mixture, target, query, present


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = HybridSeparator(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.crop_frames < 1:
        raise ValueError("epochs, batch_size and crop_frames must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(args.absence_weight) or args.absence_weight < 0:
        raise ValueError("absence_weight must be finite and nonnegative")
    torch.manual_seed(args.seed)
    if args.stage == "refiner":
        if not args.det_checkpoint:
            raise ValueError("refiner training requires --det-checkpoint")
        model, checkpoint = load_checkpoint(args.det_checkpoint, args.device)
        if checkpoint["embedding_encoder"] != args.embedding_encoder or checkpoint["sample_rate"] != args.sample_rate:
            raise ValueError("checkpoint and training data must use the same embedding encoder and sample rate")
        if checkpoint["stage"] != "det":
            raise ValueError("--det-checkpoint must be from deterministic-stage training")
        if args.sigma_data is not None:
            model.config = replace(model.config, sigma_data=args.sigma_data)
        model.freeze_separator()
        parameters = model.refiner.parameters()
    else:
        if args.det_checkpoint:
            raise ValueError("--det-checkpoint only applies to the refiner stage")
        config = ModelConfig(args.channels, args.query_dim, args.hidden, args.layers,
                             0.1 if args.sigma_data is None else args.sigma_data)
        model = HybridSeparator(config).to(args.device)
        parameters = model.det.parameters()
    dataset = QueryDataset(args.manifest, model.config, args.sample_rate, args.crop_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total, count = 0.0, 0
        for mixture, target, query, present in loader:
            mixture, target, query, present = [t.to(args.device) for t in (mixture, target, query, present)]
            optimizer.zero_grad(set_to_none=True)
            if args.stage == "det":
                prediction = model.deterministic(mixture, query)
                loss = F.mse_loss(prediction, target)
                if (~present).any():
                    loss = loss + args.absence_weight * prediction[~present].square().mean()
            else:
                loss = model.diffusion_loss(mixture, target, query, present,
                                            absence_weight=args.absence_weight)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total += loss.item() * mixture.shape[0]
            count += mixture.shape[0]
        print(json.dumps({"epoch": epoch + 1, "stage": args.stage, "mean_loss": total / count}), flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = model.checkpoint(args.stage, args.sample_rate, args.embedding_encoder)
    checkpoint["training_seed"] = args.seed
    checkpoint["manifest"] = str(Path(args.manifest).resolve())
    checkpoint["training_settings"] = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "crop_frames": args.crop_frames, "learning_rate": args.lr,
        "absence_weight": args.absence_weight,
        "log_sigma_mean": -2.0, "log_sigma_std": 1.0,
    }
    torch.save(checkpoint, destination)


@torch.no_grad()
def infer(args):
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    if checkpoint["embedding_encoder"] != args.embedding_encoder or checkpoint["sample_rate"] != args.sample_rate:
        raise ValueError("inference must use the checkpoint's embedding encoder and sample rate")
    if not args.no_refine and checkpoint["stage"] != "refiner":
        raise ValueError("refined inference requires a trained refiner checkpoint, or use --no-refine")
    mixture = load_array(args.mix, audio=True).unsqueeze(0).to(args.device)
    vectors = [load_array(path) for path in args.queries]
    if any(vector.shape != (model.config.query_dim,) for vector in vectors):
        raise ValueError("each query file must be a single vector matching checkpoint query_dim")
    queries = torch.stack(vectors).unsqueeze(1).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    targets, remainders = model.separate_queries(
        mixture, queries, steps=args.steps, sigma_min=args.sigma_min,
        sigma_max=args.sigma_max, generator=generator, refine=not args.no_refine)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for i in range(len(vectors)):
        np.save(destination / f"query_{i:03d}.npy", targets[i, 0].cpu().numpy())
        np.save(destination / f"query_{i:03d}_remainder.npy", remainders[i, 0].cpu().numpy())
    (destination / "inference.json").write_text(json.dumps({
        "queries": args.queries, "sample_rate": args.sample_rate,
        "seed": args.seed, "refine": not args.no_refine,
        "refiner_evaluations_per_query": 0 if args.no_refine else args.steps,
        "sigma_min": args.sigma_min, "sigma_max": args.sigma_max,
        "note": "Queries are independent and may overlap; remainder is mixture-target bookkeeping."
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--stage", choices=("det", "refiner"), required=True)
    training.add_argument("--manifest", required=True)
    training.add_argument("--det-checkpoint")
    training.add_argument("--channels", type=int, default=2)
    training.add_argument("--query-dim", type=int, default=512)
    training.add_argument("--hidden", type=int, default=64)
    training.add_argument("--layers", type=int, default=6)
    training.add_argument("--sigma-data", type=float, default=None,
                          help="EDM residual scale; defaults to 0.1 for det, or the loaded value for refiner")
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--crop-frames", type=int, default=32768)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--absence-weight", type=float, default=1.0)
    inference = subparsers.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--mix", required=True)
    inference.add_argument("--queries", nargs="+", required=True)
    inference.add_argument("--no-refine", action="store_true")
    inference.add_argument("--steps", type=int, default=8)
    inference.add_argument("--sigma-min", type=float, default=0.002)
    inference.add_argument("--sigma-max", type=float, default=0.5)
    for command in (training, inference):
        command.add_argument("--embedding-encoder", required=True,
                             help="Fixed encoder/checkpoint identity; must match cached vector provenance")
        command.add_argument("--sample-rate", type=int, default=22050)
        command.add_argument("--device", default="cpu")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    (train if args.command == "train" else infer)(args)


if __name__ == "__main__":
    main()
