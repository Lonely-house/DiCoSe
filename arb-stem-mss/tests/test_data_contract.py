"""Audio/label invariants which prevent contradictory source supervision."""

import copy
import json

import numpy as np
import pytest
import torch

from arb_stem_mss.data import QueryDataset


def dataset(fixture, **kwargs):
    return QueryDataset(fixture["manifest"], fixture["cache"], **kwargs)


def test_resampling_short_padding_preserves_triplet(query_fixture):
    sample = dataset(query_fixture, sample_rate=16000, segment_samples=2300)[0]
    assert sample["mixture"].shape == sample["source"].shape == (1, 2300)
    assert torch.allclose(sample["mixture"], sample["source"] + sample["submix"], atol=1e-6)
    assert 1990 <= sample["num_samples"] <= 1996
    assert sample["mixture"][..., sample["num_samples"]:].count_nonzero() == 0
    assert sample["source"][..., sample["num_samples"]:].count_nonzero() == 0
    assert sample["embedding"].shape == (8,)
    assert sample["present"] is True or bool(sample["present"]) is True


def test_validation_crop_is_deterministic_and_train_crop_aligned(query_fixture):
    validation = dataset(query_fixture, sample_rate=8000, segment_samples=211, training=False)
    a, b = validation[0], validation[0]
    assert torch.equal(a["mixture"], b["mixture"])
    training = dataset(query_fixture, sample_rate=8000, segment_samples=211, training=True, seed=4)
    for _ in range(3):
        item = training[0]
        assert torch.allclose(item["mixture"], item["source"] + item["submix"], atol=1e-6)


def test_same_label_tracks_belong_to_one_target(query_fixture):
    # The target includes both guitar tracks, leaving the actual piano stem.
    sample = dataset(query_fixture, sample_rate=8000, segment_samples=997)[0]
    import soundfile as sf
    piano, _ = sf.read(query_fixture["row"]["stems"][2]["path"], dtype="float32")
    assert torch.allclose(sample["submix"][0], torch.from_numpy(piano), atol=2e-6)


def test_absent_query_has_zero_target(query_fixture, write_manifest):
    row = copy.deepcopy(query_fixture["row"])
    row.update(target_ids=[], target_present=False)
    write_manifest(query_fixture["manifest"], [row])
    sample = dataset(query_fixture, sample_rate=8000, segment_samples=1200)[0]
    assert sample["source"].count_nonzero() == 0
    assert torch.equal(sample["submix"], sample["mixture"])
    assert not sample["present"]


@pytest.mark.parametrize("defect", ["unknown_target", "duplicate_id", "missing_audio", "nonfinite_audio"])
def test_invalid_supervision_fails_loudly(query_fixture, write_manifest, defect):
    row = copy.deepcopy(query_fixture["row"])
    if defect == "unknown_target":
        row["target_ids"] = ["unlisted"]
    elif defect == "duplicate_id":
        row["stems"][1]["id"] = row["stems"][0]["id"]
    elif defect == "missing_audio":
        row["stems"][0]["path"] += ".missing"
    else:
        row["stems"][0]["path"] = query_fixture["write_audio"](
            query_fixture["directory"] / "nan.wav", np.full(997, np.nan))
    write_manifest(query_fixture["manifest"], [row])
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        dataset(query_fixture, sample_rate=8000, segment_samples=128)[0]


def test_missing_or_nonfinite_cached_query_is_rejected(query_fixture):
    for vectors in ({}, {query_fixture["row"]["prompt"]: torch.full((8,), float("nan"))}):
        torch.save({"encoder_id": "fixture", "query_dim": 8,
                    "vectors": vectors, "is_mock": True}, query_fixture["cache"])
        with pytest.raises((ValueError, KeyError)):
            dataset(query_fixture, segment_samples=128)[0]


@pytest.mark.parametrize("defect", ["partial_label_group", "actually_present_negative"])
def test_declared_query_labels_cannot_contradict_stems(query_fixture, write_manifest, defect):
    row = copy.deepcopy(query_fixture["row"])
    row["query_labels"] = ["guitar"]
    if defect == "partial_label_group":
        row["target_ids"] = ["0"]  # The same label also describes stem 1.
    else:
        row.update(target_ids=[], target_present=False)
    write_manifest(query_fixture["manifest"], [row])
    with pytest.raises(ValueError):
        dataset(query_fixture, sample_rate=8000, segment_samples=128)
