"""Exercise public commands, checkpoint guards and epoch-boundary restoration."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_synthetic_public_cli_workflow(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "workflow"
    result = subprocess.run(
        [sys.executable, str(root / "scripts/smoke_reproduction.py"), "--work-dir", str(output)],
        cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"), timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "passed" and report["encoder_is_mock"] is True
    assert report["true_dicose_unet"] is True
    assert report["quality_claim"].startswith("none;")
    assert sum(event.get("expected_rejection", False) for event in report["events"]) == 4
