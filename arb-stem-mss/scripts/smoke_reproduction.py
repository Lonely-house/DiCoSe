"""Synthetic CPU workflow check using the actual vendored DiCoSe U-Net.

This produces no evidence about language understanding or music separation
quality. All generated tracks and explicitly marked mock embeddings are fixtures.
Existing nonempty output directories are preserved rather than overwritten.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="artifacts/smoke")
    args = parser.parse_args()
    work = Path(args.work_dir).expanduser()
    if not work.is_absolute():
        work = ROOT / work
    work = work.resolve()
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be absent or empty; choose a fresh directory")
    work.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    events = []

    def run(script, arguments, *, rejected=False, rejection_text=None):
        command = [sys.executable, str(ROOT / script), *map(str, arguments)]
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with (work / "commands.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"command": command, "returncode": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr}) + "\n")
        if rejected:
            if result.returncode == 0 or (rejection_text and rejection_text not in result.stderr):
                raise RuntimeError(f"Expected protected workflow to reject {script}: {result.stderr}")
        elif result.returncode:
            raise RuntimeError(f"Workflow failed: {script}\n{result.stdout}\n{result.stderr}")
        events.append({"script": script, "expected_rejection": rejected, "status": "passed"})

    raw = work / "raw/MUSDB18HQ"
    assignments = {}
    native_rate, frames = 8000, 769
    for official, songs in (("train", ["train_a", "train_b", "validation_a"]),
                            ("test", ["test_a"])):
        for song_index, song in enumerate(songs):
            directory = raw / official / song
            directory.mkdir(parents=True)
            for index, label in enumerate(("vocals", "drums", "bass", "other")):
                t = np.arange(frames, dtype=np.float32) / native_rate
                left = 0.02 * np.sin(2 * np.pi * (120 + index * 130 + song_index * 7) * t)
                right = left * 0.8
                sf.write(directory / (label + ".wav"), np.stack([left, right], axis=-1),
                         native_rate, subtype="FLOAT")
            assignments["fixture:" + song] = "validation" if song.startswith("validation") else official
    split_file = work / "splits.json"
    split_file.write_text(json.dumps(assignments), encoding="utf-8")
    manifests = work / "manifests"
    data_config = {
        "seed": 42, "output_dir": str(manifests),
        "datasets": {"fixture": {"type": "musdb18hq", "enabled": True,
                                   "root": str(raw), "split_file": str(split_file)}},
        "queries": {"group_queries_per_song": 1, "max_group_size": 2,
                    "negative_queries_per_song": 0},
    }
    data_path = work / "data.yaml"
    data_path.write_text(yaml.safe_dump(data_config), encoding="utf-8")
    run("prepare_data.py", ["--config", data_path, "--workspace-root", work])
    manifest_files = [manifests / (split + ".jsonl") for split in ("train", "validation", "test")]
    cache = work / "text_embeddings.pt"
    run("cache_text_embeddings.py", ["--manifests", *manifest_files, "--output", cache,
                                     "--mock", "--query-dim", "16", "--seed", "42"])
    config = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text(encoding="utf-8"))
    config["data"] = {"train_manifest": str(manifest_files[0]),
                      "validation_manifest": str(manifest_files[1]),
                      "embedding_cache": str(cache)}
    config["train"]["epochs"] = 1
    config_path = work / "train_one_epoch.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    resumed_run, continuous_run, partial_run = work / "resumed_run", work / "continuous_run", work / "partial_run"
    base = ["--config", config_path, "--device", "cpu", "--output-dir", resumed_run]
    run("train_determination.py", base, rejected=True, rejection_text="mock")
    run("train_determination.py", [*base, "--allow-mock-embeddings"])
    config["train"]["epochs"] = 2
    extended_path = work / "train_two_epochs.yaml"
    extended_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run("train_determination.py", ["--config", extended_path, "--device", "cpu",
                                  "--output-dir", resumed_run, "--resume", resumed_run / "last.pt",
                                  "--allow-mock-embeddings"])
    run("train_determination.py", ["--config", extended_path, "--device", "cpu",
                                  "--output-dir", continuous_run, "--allow-mock-embeddings"])
    resumed = torch.load(resumed_run / "last.pt", map_location="cpu", weights_only=True)
    continuous = torch.load(continuous_run / "last.pt", map_location="cpu", weights_only=True)
    assert resumed["epochs_completed"] == continuous["epochs_completed"] == 2
    assert isinstance(resumed["runtime"], dict) and (resumed_run / "runtime.json").is_file()
    assert resumed["global_step"] == continuous["global_step"]
    assert all(torch.equal(resumed["model"][name], value) for name, value in continuous["model"].items())
    events.append({"check": "epoch-boundary resume exactly matches uninterrupted CPU parameters", "status": "passed"})
    original_manifest = manifest_files[0].read_bytes()
    try:
        # A harmless whitespace edit still changes the registered data identity.
        manifest_files[0].write_bytes(original_manifest + b"\n")
        run("train_determination.py", ["--config", extended_path, "--device", "cpu",
                                      "--output-dir", resumed_run, "--resume", resumed_run / "last.pt",
                                      "--allow-mock-embeddings"], rejected=True, rejection_text="hashes differ")
    finally:
        manifest_files[0].write_bytes(original_manifest)
    altered_cache = torch.load(cache, map_location="cpu", weights_only=True)
    altered_cache["encoder_id"] = "different-encoder-fixture"
    altered_path = work / "wrong_encoder.pt"
    torch.save(altered_cache, altered_path)
    run("evaluate.py", ["--checkpoint", resumed_run / "last.pt", "--manifest", manifest_files[2],
                        "--embedding-cache", altered_path, "--output", work / "must_not_exist.json",
                        "--device", "cpu", "--allow-mock-embeddings"],
        rejected=True, rejection_text="encoder identity")
    run("train_determination.py", ["--config", extended_path, "--device", "cpu",
                                  "--output-dir", partial_run, "--max-steps", "1",
                                  "--allow-mock-embeddings"])
    run("train_determination.py", ["--config", extended_path, "--device", "cpu",
                                  "--output-dir", partial_run, "--resume", partial_run / "last.pt",
                                  "--allow-mock-embeddings"], rejected=True, rejection_text="epoch-boundary")
    metrics = work / "evaluation.json"
    run("evaluate.py", ["--checkpoint", resumed_run / "last.pt", "--manifest", manifest_files[2],
                        "--embedding-cache", cache, "--output", metrics, "--device", "cpu",
                        "--allow-mock-embeddings", "--spectral-metrics"])
    evaluation = json.loads(metrics.read_text(encoding="utf-8"))
    assert evaluation["embedding_is_mock"] and evaluation["aggregate"]["num_songs"] == 1
    assert math.isfinite(evaluation["aggregate"]["macro_by_song"]["chunked_mrstft_loss"])
    assert isinstance(evaluation["runtime"], dict)
    source = raw / "test/test_a/vocals.wav"
    output = work / "separated"
    run("separate.py", ["--checkpoint", resumed_run / "last.pt", "--input", source,
                        "--text", "The sound of vocals", "--text", "The sound of drums",
                        "--embedding-cache", cache, "--output-dir", output, "--device", "cpu",
                        "--allow-mock-embeddings"])
    info = json.loads((output / "inference.json").read_text(encoding="utf-8"))
    assert info["embedding_is_mock"] and len(info["queries"]) == 2
    assert (output / "runtime.json").is_file()
    input_audio, rate = sf.read(source, dtype="float32", always_2d=True)
    for query in info["queries"]:
        target, target_rate = sf.read(output / query["target"], dtype="float32")
        rest, rest_rate = sf.read(output / query["remainder"], dtype="float32")
        assert rate == target_rate == rest_rate == native_rate
        assert target.shape == rest.shape == (frames,)
        assert np.isfinite(target).all() and np.isfinite(rest).all()
        assert np.allclose(target + rest, input_audio.mean(axis=1), atol=2e-6)
    result = {"status": "passed", "scope": "synthetic CPU engineering workflow",
              "quality_claim": "none; mock vectors and generated tones do not establish text semantics or music quality",
              "true_dicose_unet": True, "encoder_is_mock": True, "events": events,
              "software": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                           "numpy": np.__version__}}
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "result": str(work / "result.json"),
                      "quality_claim": result["quality_claim"]}))


if __name__ == "__main__":
    main()
