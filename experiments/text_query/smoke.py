"""Synthetic interface/optimization check only; random vectors are not text."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .model import HybridSeparator, ModelConfig
from .train import QueryDataset, load_checkpoint


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = HybridSeparator(ModelConfig(channels=1, query_dim=8, hidden=8, layers=2))
    mixture = torch.randn(3, 1, 127) * 0.1
    target = mixture * 0.6
    target[2] = 0
    query = torch.randn(3, 8)  # Tensor wiring fixture, NOT a semantic text encoder.
    present = torch.tensor([True, True, False])
    optimizer = torch.optim.Adam(model.det.parameters(), lr=1e-3)
    loss = (model.deterministic(mixture, query) - target).square().mean()
    optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.det.parameters())
    optimizer.step()
    model.zero_grad(set_to_none=True)
    model.freeze_separator()
    frozen = {name: value.clone() for name, value in model.det.state_dict().items()}
    optimizer = torch.optim.Adam(model.refiner.parameters(), lr=1e-3)
    metrics = model.diffusion_loss(mixture, target, query, present)
    optimizer.zero_grad()
    metrics["loss"].backward()
    assert all(p.grad is None for p in model.det.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.refiner.parameters())
    optimizer.step()
    assert all(torch.equal(value, frozen[name]) for name, value in model.det.state_dict().items())
    outputs, rest = model.separate_queries(mixture, query[None].repeat(5, 1, 1), steps=2,
                                           generator=torch.Generator().manual_seed(17))
    assert outputs.shape == (5, 3, 1, 127)
    assert torch.isfinite(outputs).all()
    assert torch.allclose(outputs + rest, mixture[None].expand_as(outputs), atol=1e-6)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        np.save(directory / "mix.npy", mixture[0].numpy())
        np.save(directory / "vector.npy", query[0].numpy())
        manifest = {"mix": "mix.npy", "target": None, "target_present": False,
                    "prompt": "synthetic absent fixture", "prompt_embedding": "vector.npy",
                    "sample_rate": 22050}
        (directory / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        dataset = QueryDataset(directory / "manifest.jsonl", model.config, 22050, 256)
        x, y, q, p = dataset[0]
        assert x.shape == y.shape == (1, 256) and y.count_nonzero() == 0 and p is False
        checkpoint = directory / "model.pt"
        torch.save(model.checkpoint("refiner", 22050, "synthetic-fixture-not-CLAP"), checkpoint)
        loaded, _ = load_checkpoint(checkpoint, "cpu")
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))
    print(json.dumps({"status": "passed", "checks": ["deterministic backward", "EDM backward",
          "frozen seed", "five independent queries", "finite Euler sampling",
          "target/remainder arithmetic", "negative manifest padding", "checkpoint roundtrip"],
          "quality_claim": "none; no real text embeddings, training data or pretrained weights used"}))


if __name__ == "__main__":
    main()
