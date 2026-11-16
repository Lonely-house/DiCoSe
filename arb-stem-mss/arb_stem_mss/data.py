"""Metadata-driven multitrack queries and aligned waveform triplets.

The mixture is re-synthesized from *all* listed tracks. A query selects complete
label groups, never an arbitrary one of several indistinguishably named tracks.
Original mastered mixture files are intentionally not substituted for this sum.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset


def stable_number(*parts: Any) -> int:
    value = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def validate_record(row: dict, *, base_dir: Path | None = None) -> dict:
    """Validate one query and resolve its paths without reading GT into the model."""
    if not isinstance(row, dict):
        raise ValueError("manifest rows must be objects")
    for key in ("song_id", "dataset", "split", "prompt"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"manifest row requires nonempty {key}")
    if row["split"] not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split {row['split']!r}")
    if not isinstance(row.get("stems"), list) or not row["stems"]:
        raise ValueError(f"{row['song_id']}: stems must be a nonempty list")
    stems, ids, paths = [], set(), set()
    for stem in row["stems"]:
        if not isinstance(stem, dict):
            raise ValueError("each stem must be an object")
        for key in ("id", "label", "path"):
            if not isinstance(stem.get(key), str) or not stem[key].strip():
                raise ValueError(f"{row['song_id']}: each stem requires {key}")
        path = Path(stem["path"]).expanduser()
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{row['song_id']}: missing track {path}")
        if stem["id"] in ids or str(path) in paths:
            raise ValueError(f"{row['song_id']}: duplicate stem id or audio path")
        ids.add(stem["id"])
        paths.add(str(path))
        stems.append({**stem, "path": str(path)})
    targets = row.get("target_ids")
    if not isinstance(targets, list) or any(not isinstance(item, str) for item in targets):
        raise ValueError("target_ids must be a list of stem id strings")
    if len(set(targets)) != len(targets) or not set(targets) <= ids:
        raise ValueError(f"{row['song_id']}: duplicate or unknown target_ids")
    if not isinstance(row.get("target_present"), bool):
        raise ValueError("target_present must be an explicit JSON boolean")
    if bool(targets) != row["target_present"]:
        raise ValueError("target_present must agree with whether target_ids is empty")
    if "query_labels" in row:
        labels = row["query_labels"]
        if (not isinstance(labels, list) or not labels
                or any(not isinstance(label, str) or not label.strip() for label in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError("query_labels must be a nonempty list of distinct nonempty label strings")
        label_set = set(labels)
        available_labels = {stem["label"] for stem in stems}
        if row["target_present"]:
            if not label_set <= available_labels:
                raise ValueError("present query_labels must all exist in the song's stem labels")
            expected_ids = {stem["id"] for stem in stems if stem["label"] in label_set}
            if set(targets) != expected_ids:
                raise ValueError("target_ids must select all tracks matching query_labels, including same-label tracks")
        elif label_set & available_labels:
            raise ValueError("absent query_labels must not occur in the song's stem labels")
    return {**row, "stems": stems}


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(validate_record(json.loads(line), base_dir=path.parent))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    by_song: dict[str, tuple] = {}
    for row in rows:
        signature = (row["split"], tuple((s["id"], s["label"], s["path"]) for s in row["stems"]))
        if row["song_id"] in by_song and by_song[row["song_id"]] != signature:
            raise ValueError(f"inconsistent song definition or split: {row['song_id']}")
        by_song[row["song_id"]] = signature
    return rows


def audio_layout(row: dict) -> tuple[int, int]:
    """Return native sample rate and longest length; tails are zero-padded."""
    rates, longest = set(), 0
    for stem in row["stems"]:
        info = sf.info(stem["path"])
        if info.frames < 1 or info.channels < 1 or info.samplerate < 1:
            raise ValueError(f"invalid/empty audio: {stem['path']}")
        rates.add(info.samplerate)
        longest = max(longest, info.frames)
    if len(rates) != 1:
        raise ValueError(f"{row['song_id']}: within-song native sample rates differ: {sorted(rates)}; align/resample offline")
    return rates.pop(), longest


def load_query_audio(row: dict, sample_rate: int = 22050,
                     segment_samples: int | None = None,
                     start_frame: int = 0) -> dict[str, torch.Tensor | int]:
    """Read one aligned query triplet, including full-song evaluation.

    ``start_frame`` uses the song's *native* sample rate; all tracks share it.
    Unequal track durations are interpreted only as trailing silence (right pad).
    This does not repair unknown offsets or drifting/misaligned recordings.
    """
    if not isinstance(sample_rate, int) or sample_rate < 1:
        raise ValueError("sample_rate must be a positive integer")
    if segment_samples is not None and (not isinstance(segment_samples, int) or segment_samples < 1):
        raise ValueError("segment_samples must be a positive integer or None")
    native_rate, longest = audio_layout(row)
    if not isinstance(start_frame, int) or start_frame < 0 or start_frame >= longest:
        raise ValueError("start_frame must lie within the native song")
    available = longest - start_frame
    native_count = available if segment_samples is None else min(
        available, math.ceil(segment_samples * native_rate / sample_rate))
    valid_samples = math.ceil(native_count * sample_rate / native_rate)
    desired = valid_samples if segment_samples is None else segment_samples
    valid_samples = min(valid_samples, desired)
    gcd = math.gcd(native_rate, sample_rate)
    mixture = np.zeros(desired, dtype=np.float32)
    source = np.zeros(desired, dtype=np.float32)
    target_ids = set(row["target_ids"])
    for stem in row["stems"]:
        info = sf.info(stem["path"])
        native = np.zeros(native_count, dtype=np.float32)
        if start_frame < info.frames:
            with sf.SoundFile(stem["path"]) as handle:
                handle.seek(start_frame)
                audio = handle.read(min(native_count, info.frames-start_frame), dtype="float32", always_2d=True)
            if not np.isfinite(audio).all():
                raise ValueError(f"non-finite audio values: {stem['path']}")
            native[:len(audio)] = audio.mean(axis=1)
        converted = native if native_rate == sample_rate else resample_poly(
            native, sample_rate // gcd, native_rate // gcd).astype(np.float32)
        waveform = np.zeros(desired, dtype=np.float32)
        count = min(len(converted), desired)
        waveform[:count] = converted[:count]
        mixture += waveform
        if stem["id"] in target_ids:
            source += waveform
    if not np.isfinite(mixture).all() or not np.isfinite(source).all():
        raise ValueError(f"summed waveform overflow: {row['song_id']}")
    mix_tensor = torch.from_numpy(mixture[None])
    source_tensor = torch.from_numpy(source[None])
    return {"mixture": mix_tensor, "source": source_tensor,
            "submix": mix_tensor-source_tensor, "num_samples": valid_samples}


class QueryDataset(Dataset):
    """Offline queries with deterministic seed/epoch/index training crops."""

    def __init__(self, manifest, embedding_cache, sample_rate=22050,
                 segment_samples=262144, training=False, seed=0):
        from .text import load_embedding_cache
        self.rows = read_manifest(manifest)
        cache = load_embedding_cache(embedding_cache, allow_mock=True)
        self.encoder_id = cache["encoder_id"]
        self.query_dim = cache["query_dim"]
        self.is_mock = cache["is_mock"]
        self.vectors = cache["vectors"]
        missing = sorted({row["prompt"] for row in self.rows} - set(self.vectors))
        if missing:
            raise ValueError(f"embedding cache misses {len(missing)} prompts; first: {missing[0]!r}")
        if not isinstance(sample_rate, int) or sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if segment_samples is not None and (not isinstance(segment_samples, int) or segment_samples < 1):
            raise ValueError("segment_samples must be positive or None")
        self.sample_rate, self.segment_samples = sample_rate, segment_samples
        self.training, self.seed, self.epoch = bool(training), int(seed), 0
        self._layouts = {}
        for row in self.rows:
            if row["song_id"] not in self._layouts:
                self._layouts[row["song_id"]] = audio_layout(row)

    def set_epoch(self, epoch):
        if int(epoch) < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        native_rate, frames = self._layouts[row["song_id"]]
        needed = frames if self.segment_samples is None else math.ceil(
            self.segment_samples * native_rate / self.sample_rate)
        last_start = max(0, frames-needed)
        if self.training:
            start = stable_number(self.seed, self.epoch, index) % (last_start+1)
        else:
            start = last_start // 2
        audio = load_query_audio(row, self.sample_rate, self.segment_samples, start)
        return {**audio, "embedding": self.vectors[row["prompt"]].clone(),
                "prompt": row["prompt"], "present": row["target_present"],
                "song_id": row["song_id"]}
