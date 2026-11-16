"""Synthetic fixtures validate engineering behavior, not separation quality."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)


def write_audio(path, values, sample_rate=8000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(values, dtype=np.float32), sample_rate, subtype="FLOAT")
    return str(path)


@pytest.fixture
def query_fixture(tmp_path):
    t = np.arange(997, dtype=np.float32) / 8000
    stems = []
    for index, frequency in enumerate((170, 290, 410)):
        signal = (index + 1) * 0.02 * np.sin(2 * np.pi * frequency * t)
        path = write_audio(tmp_path / f"stem_{index}.wav", signal)
        stems.append({"id": str(index), "label": "guitar" if index < 2 else "piano", "path": path})
    prompt = "The sound of guitar"
    row = {"song_id": "fixture/song", "dataset": "fixture", "split": "train",
           "stems": stems, "target_ids": ["0", "1"], "prompt": prompt,
           "target_present": True}
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cache = tmp_path / "text.pt"
    torch.save({"encoder_id": "synthetic-test-fixture", "query_dim": 8,
                "vectors": {prompt: torch.arange(1, 9, dtype=torch.float32).div(torch.arange(1, 9, dtype=torch.float32).norm())},
                "is_mock": True}, cache)
    return {"directory": tmp_path, "row": row, "manifest": manifest, "cache": cache,
            "write_audio": write_audio}


@pytest.fixture
def write_manifest():
    def write(path, rows):
        Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return Path(path)
    return write
