from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from arb_stem_mss.engine import run_training

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("wrong_file", ["train", "validation"])
def test_training_rejects_wrong_manifest_split(query_fixture, write_manifest, wrong_file):
    work = query_fixture["directory"]
    train_row = deepcopy(query_fixture["row"])
    val_row = deepcopy(query_fixture["row"])
    val_row["song_id"] = "fixture/independent_validation"
    val_row["split"] = "validation"
    # Use independent source files so a split-role rejection cannot be confused
    # with the separate train/validation audio-overlap protection.
    for stem in val_row["stems"]:
        stem["path"] = query_fixture["write_audio"](
            work / ("validation_" + stem["id"] + ".wav"), np.zeros(997))
    if wrong_file == "train":
        train_row["split"] = "validation"
    else:
        val_row["split"] = "train"
    train_file = write_manifest(work / "guard_train.jsonl", [train_row])
    val_file = write_manifest(work / "guard_validation.jsonl", [val_row])
    config = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    config["model"]["query_dim"] = 8
    config["data"] = {"train_manifest": str(train_file),
                      "validation_manifest": str(val_file),
                      "embedding_cache": str(query_fixture["cache"])}
    config_file = work / "guard_config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(config=str(config_file), resume=None, device="cpu",
                           output_dir=str(work / "forbidden_run"), max_steps=None,
                           allow_mock_embeddings=True, init_upstream=None)
    with pytest.raises(ValueError, match="split"):
        run_training(args)
