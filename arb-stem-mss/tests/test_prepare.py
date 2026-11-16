"""Small raw-format fixtures exercise scanners and split/label contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from prepare_data import assert_disjoint, build_queries, prepare, scan_dataset
from conftest import write_audio


def musdb_fixture(root):
    for split, songs in (("train", ["train_a", "train_b"]), ("test", ["test_a"])):
        for song in songs:
            for index, label in enumerate(("vocals", "drums", "bass", "other")):
                values = np.sin(np.arange(97, dtype=np.float32) * (index + 1) * 0.2) * 0.01
                write_audio(root / split / song / (label + ".wav"), values)
            # A mastered mixture is deliberately inconsistent and must be ignored.
            write_audio(root / split / song / "mixture.wav", np.ones(97) * 0.8)


def test_musdb_official_test_and_stable_query_splits(tmp_path):
    root = tmp_path / "MUSDB18HQ"
    musdb_fixture(root)
    entry = {"type": "musdb18hq", "root": str(root), "validation_fraction": 0.5}
    a = scan_dataset("musdb", entry, root, 42, {}, None)
    b = scan_dataset("musdb", entry, root, 42, {}, None)
    assert [song["split"] for song in a] == [song["split"] for song in b]
    assert next(song for song in a if song["song_id"] == "musdb:test_a")["split"] == "test"
    records = build_queries(a, {"group_queries_per_song": 2, "max_group_size": 2}, 42)
    for song in a:
        rows = [row for row in records if row["song_id"] == song["song_id"]]
        assert {row["split"] for row in rows} == {song["split"]}
        assert len(song["stems"]) == 4
        assert all(Path(stem["path"]).name != "mixture.wav" for stem in song["stems"])
    incorrect = {song["song_id"]: song["split"] for song in a}
    incorrect["musdb:test_a"] = "train"
    with pytest.raises(ValueError, match="official"):
        scan_dataset("musdb", entry, root, 42, {}, incorrect)


def test_moises_same_label_tracks_aggregate_and_artist_split(tmp_path):
    root = tmp_path / "moises"
    for name in ("song_a", "song_b"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "data.json").write_text(json.dumps({"artist": "shared artist"}))
        for label, track in (("guitar", "one"), ("guitar", "two"), ("piano", "three")):
            write_audio(directory / label / (track + ".wav"), np.zeros(64))
    entry = {"type": "moisesdb", "split_group": "artist", "artist_field": "artist"}
    songs = scan_dataset("moises", entry, root, 13, {}, None)
    assert len({song["split"] for song in songs}) == 1
    records = build_queries(songs, {"max_group_size": 2}, 13)
    guitars = [row for row in records if row["prompt"] == "The sound of guitar"]
    assert len(guitars) == 2
    assert all(len(row["target_ids"]) == 2 for row in guitars)
    assert_disjoint(songs)
    leaked = copy.deepcopy(songs)
    leaked[1]["split"] = "test" if leaked[0]["split"] != "test" else "train"
    with pytest.raises(ValueError, match="leaked"):
        assert_disjoint(leaked)


def test_slakh_metadata_rendering_and_official_splits(tmp_path):
    root = tmp_path / "slakh"
    for split in ("train", "validation", "test"):
        directory = root / split / ("Track_" + split)
        directory.mkdir(parents=True)
        stems = {"S00": {"audio_rendered": True, "inst_class": "Piano"},
                 "S01": {"audio_rendered": True, "inst_class": "Guitar"},
                 "S02": {"audio_rendered": False, "inst_class": "Strings"}}
        (directory / "metadata.yaml").write_text(yaml.safe_dump({"stems": stems}))
        write_audio(directory / "stems/S00.wav", np.zeros(64))
        write_audio(directory / "stems/S01.wav", np.ones(64) * 0.01)
    songs = scan_dataset("slakh", {"type": "slakh2100"}, root, 42, {}, None)
    assert {song["split"] for song in songs} == {"train", "validation", "test"}
    assert all({stem["label"] for stem in song["stems"]} == {"piano", "guitar"} for song in songs)
    assert all(len(song["stems"]) == 2 for song in songs)


def test_unknown_other_prevents_unsupported_absence_claims(tmp_path):
    root = tmp_path / "musdb"
    musdb_fixture(root)
    songs = scan_dataset("musdb", {"type": "musdb18hq"}, root, 42, {}, None)
    records = build_queries(songs, {"negative_queries_per_song": 5, "max_group_size": 2}, 42)
    assert all(row["target_present"] for row in records)
