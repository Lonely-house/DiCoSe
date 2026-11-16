"""Scan supported raw datasets and write auditable song-disjoint query manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random

import soundfile as sf
import yaml

from arb_stem_mss.data import stable_number, validate_record


SPLITS = ("train", "validation", "test")
AUDIO_SUFFIXES = {".wav", ".flac"}
UNKNOWN_LABELS = {"other", "unknown", "unlabeled", "unlabelled", "misc", "miscellaneous"}


def aggregate_label(label):
    return label in UNKNOWN_LABELS or any(label.startswith(prefix+" ") for prefix in UNKNOWN_LABELS)


def resolve_path(value, base):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (base/path).resolve() if not path.is_absolute() else path.resolve()


def normalize_label(value, label_map):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source label must be a nonempty string")
    label = " ".join(value.replace("_", " ").strip().lower().split())
    normalized_map = {" ".join(str(k).replace("_", " ").strip().lower().split()): v
                      for k, v in label_map.items()}
    label = normalized_map.get(label, label)
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label_map values must be nonempty strings")
    return label.strip()


def metadata_string(metadata, field):
    value = metadata
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"artist split requires metadata field {field!r}; use split_group=song or supply the correct field")
        value = value[part]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metadata field {field!r} must be a nonempty string for artist splitting")
    return value.strip().lower()


def read_split_map(path):
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("split_file must be a JSON object")
    if set(data) <= set(SPLITS) and all(isinstance(v, list) for v in data.values()):
        result = {}
        for split, ids in data.items():
            for song in ids:
                if song in result:
                    raise ValueError(f"song occurs twice in split_file: {song}")
                result[song] = split
        data = result
    if any(not isinstance(key, str) or value not in SPLITS for key, value in data.items()):
        raise ValueError("split_file must map canonical song_id strings to train/validation/test")
    return data


def fraction_split(group, seed, fractions):
    if set(fractions) != set(SPLITS) or any(float(v) < 0 for v in fractions.values()):
        raise ValueError("fractions must contain nonnegative train/validation/test values")
    if abs(sum(float(v) for v in fractions.values())-1) > 1e-8:
        raise ValueError("split fractions must sum to one")
    value = stable_number(seed, group) / 2**64
    cumulative = 0.0
    for split in SPLITS:
        cumulative += float(fractions[split])
        if value < cumulative:
            return split
    return "test"


def assign_split(song_id, group, official, *, kind, entry, seed, split_map):
    if split_map is not None:
        if song_id not in split_map:
            raise ValueError(f"explicit split_file has no assignment for {song_id}")
        chosen = split_map[song_id]
        if kind == "slakh2100" and chosen != official:
            raise ValueError(f"cannot move official Slakh split: {song_id}")
        if kind == "musdb18hq" and ((official == "test") != (chosen == "test")):
            raise ValueError(f"cannot move MUSDB official train/test boundary: {song_id}")
        return chosen
    if kind == "slakh2100" or official == "test":
        return official
    if kind == "musdb18hq":
        val = float(entry.get("validation_fraction", 0.1))
        return fraction_split(group, seed, {"train": 1-val, "validation": val, "test": 0})
    return fraction_split(group, seed, entry.get("fractions", {"train": 0.8, "validation": 0.1, "test": 0.1}))


def inspect_stems(song):
    rates, frames = set(), []
    for stem in song["stems"]:
        info = sf.info(stem["path"])
        if info.frames < 1 or info.channels < 1:
            raise ValueError(f"empty audio: {stem['path']}")
        rates.add(info.samplerate)
        frames.append(info.frames)
    if len(rates) != 1:
        raise ValueError(f"{song['song_id']}: native sample rates differ; align/resample offline")
    song["native_sample_rate"] = next(iter(rates))
    song["duration_seconds"] = max(frames)/song["native_sample_rate"]
    song["unequal_track_lengths"] = len(set(frames)) > 1
    return song


def scan_dataset(name, entry, root, seed, label_map, split_map):
    kind = entry["type"]
    songs = []
    if kind not in {"musdb18hq", "moisesdb", "slakh2100"}:
        raise ValueError(f"unsupported dataset type: {kind}")
    if not root.is_dir():
        raise FileNotFoundError(f"enabled dataset root is absent: {root}; disable it or correct configs/data.yaml")
    if kind == "musdb18hq":
        for official in ("train", "test"):
            split_dir = root/official
            if not split_dir.is_dir():
                raise FileNotFoundError(f"MUSDB18HQ requires train/ and test/ directories: {split_dir}")
            for directory in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                stems = []
                for label in ("vocals", "drums", "bass", "other"):
                    path = directory/(label+".wav")
                    if not path.is_file():
                        raise FileNotFoundError(f"MUSDB18HQ missing official stem: {path}; compressed .stem.mp4 is not HQ WAV format")
                    stems.append({"id": label, "label": normalize_label(label, label_map), "path": str(path.resolve())})
                song_id = name+":"+directory.name
                split = assign_split(song_id, song_id, official, kind=kind, entry=entry, seed=seed, split_map=split_map)
                songs.append({"song_id": song_id, "dataset": name, "split": split, "split_group": song_id, "stems": stems})
    elif kind == "moisesdb":
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            metadata_path = directory/"data.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"MoisesDB raw song requires {metadata_path}; root should be moisesdb_v0.1, not its parent")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"Moises metadata must be an object: {metadata_path}")
            stems = []
            for stemdir in sorted(path for path in directory.iterdir() if path.is_dir()):
                label = normalize_label(stemdir.name, label_map)
                for path in sorted(stemdir.iterdir()):
                    if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                        stems.append({"id": stemdir.name+"/"+path.stem, "label": label, "path": str(path.resolve())})
            if not stems:
                raise ValueError(f"Moises song has no stem-folder WAV/FLAC files: {directory}")
            song_id = name+":"+directory.name
            grouping = entry.get("split_group", "song")
            if grouping not in {"song", "artist"}:
                raise ValueError("Moises split_group must be song or artist")
            group = song_id if grouping == "song" else name+":artist:"+metadata_string(metadata, entry.get("artist_field", "artist"))
            split = assign_split(song_id, group, None, kind=kind, entry=entry, seed=seed, split_map=split_map)
            songs.append({"song_id": song_id, "dataset": name, "split": split, "split_group": group,
                          "label_source": "official stem directory name", "stems": stems})
    else:
        for official in SPLITS:
            split_dir = root/official
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Slakh2100 requires train/validation/test directories: {split_dir}")
            for directory in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                metadata_path = directory/"metadata.yaml"
                if not metadata_path.is_file():
                    raise FileNotFoundError(f"missing Slakh metadata: {metadata_path}")
                metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or not isinstance(metadata.get("stems"), dict):
                    raise ValueError(f"Slakh metadata requires stems mapping: {metadata_path}")
                stems = []
                for stem_id, item in sorted(metadata["stems"].items()):
                    if item.get("audio_rendered") is not True:
                        continue
                    label = normalize_label(item.get("inst_class"), label_map)
                    candidates = [directory/"stems"/(stem_id+suffix) for suffix in (".wav", ".flac")]
                    existing = [path for path in candidates if path.is_file()]
                    if len(existing) != 1:
                        raise ValueError(f"expected exactly one rendered WAV/FLAC for {directory.name}/{stem_id}: {existing}")
                    stems.append({"id": stem_id, "label": label, "path": str(existing[0].resolve())})
                if not stems:
                    raise ValueError(f"Slakh song has no rendered stems: {directory}")
                song_id = name+":"+directory.name
                split = assign_split(song_id, song_id, official, kind=kind, entry=entry, seed=seed, split_map=split_map)
                songs.append({"song_id": song_id, "dataset": name, "split": split, "split_group": song_id,
                              "label_source": "Slakh inst_class", "stems": stems})
    if not songs:
        raise ValueError(f"no songs found at enabled dataset root: {root}")
    return [inspect_stems(song) for song in songs]


def build_queries(songs, settings, seed):
    groups_per_song = int(settings.get("group_queries_per_song", 0))
    max_group_size = int(settings.get("max_group_size", 3))
    negatives = int(settings.get("negative_queries_per_song", 0))
    if groups_per_song < 0 or negatives < 0 or max_group_size < 2:
        raise ValueError("query counts must be nonnegative and max_group_size >= 2")
    # A held-out/test-only class must not be introduced as train negative labels.
    vocab = defaultdict(set)
    for song in songs:
        for stem in song["stems"]:
            if not aggregate_label(stem["label"]):
                vocab[song["split"]].add(stem["label"])
    records = []
    for song in songs:
        label_ids = defaultdict(list)
        for stem in song["stems"]:
            label_ids[stem["label"]].append(stem["id"])
        labels = sorted(label_ids)
        chosen = [(label,) for label in labels]
        possible_groups = [combo for size in range(2, min(max_group_size, len(labels))+1)
                           for combo in itertools.combinations(labels, size)]
        rng = random.Random(stable_number(seed, song["song_id"], "query_sampling"))
        if possible_groups and groups_per_song:
            chosen.extend(rng.sample(possible_groups, min(groups_per_song, len(possible_groups))))
        for selected in chosen:
            label_text = " and ".join(selected)
            records.append({**song, "target_ids": [stem_id for label in selected for stem_id in label_ids[label]],
                            "query_labels": list(selected), "prompt": "The sound of "+label_text,
                            "target_present": True})
        # Unknown/aggregate tracks make absence claims unreliable. Disable these.
        if negatives and not any(aggregate_label(label) for label in labels):
            missing = sorted(vocab[song["split"]]-set(labels))
            for label in rng.sample(missing, min(negatives, len(missing))):
                records.append({**song, "target_ids": [], "query_labels": [label],
                                "prompt": "The sound of "+label, "target_present": False,
                                "absence_basis": "dataset coarse label metadata; not an acoustic oracle"})
    return records


def assert_disjoint(songs):
    seen_songs, seen_paths, seen_groups = {}, {}, {}
    for song in songs:
        if song["song_id"] in seen_songs:
            raise ValueError(f"duplicate canonical song id: {song['song_id']}")
        seen_songs[song["song_id"]] = song["split"]
        if song["split_group"] in seen_groups and seen_groups[song["split_group"]] != song["split"]:
            raise ValueError(f"split group leaked across splits: {song['split_group']}")
        seen_groups[song["split_group"]] = song["split"]
        for stem in song["stems"]:
            path = str(Path(stem["path"]).resolve())
            if path in seen_paths and seen_paths[path] != song["split"]:
                raise ValueError(f"same canonical audio path leaked across splits: {path}")
            seen_paths[path] = song["split"]


def prepare(config_path, workspace_root=None):
    project_root = Path(__file__).resolve().parent
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict):
        raise ValueError("data config requires a datasets mapping")
    workspace = Path(workspace_root).expanduser().resolve() if workspace_root else project_root.parent
    seed = int(config.get("seed", 42))
    settings = config.get("queries", {})
    label_map = settings.get("label_map", {"strings (continued)": "strings"})
    songs, dataset_roots = [], {}
    for name, entry in config["datasets"].items():
        if not entry.get("enabled", True):
            continue
        root = resolve_path(entry["root"], workspace)
        split_file = resolve_path(entry["split_file"], workspace) if entry.get("split_file") else None
        songs.extend(scan_dataset(name, entry, root, seed, label_map, read_split_map(split_file)))
        dataset_roots[name] = str(root)
    if not songs:
        raise ValueError("no datasets enabled")
    assert_disjoint(songs)
    records = build_queries(songs, settings, seed)
    for record in records:
        validate_record(record)
    empty_splits = [split for split in SPLITS if not any(row["split"] == split for row in records)]
    if config.get("require_nonempty_splits", False) and empty_splits:
        raise ValueError(f"empty query splits: {empty_splits}; change the dataset/split file, not the test boundary. Stable hash fractions do not guarantee nonempty tiny fixtures")
    output = resolve_path(config.get("output_dir", "artifacts/manifests"), project_root)
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = [record for record in records if record["split"] == split]
        (output/(split+".jsonl")).write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    assignments = {song["song_id"]: song["split"] for song in songs}
    (output/"splits.json").write_text(json.dumps(assignments, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    prompts = sorted({record["prompt"] for record in records})
    (output/"prompts.json").write_text(json.dumps(prompts, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    audit = {"seed": seed, "roots": dataset_roots, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
             "songs_per_split": {split: sum(song["split"] == split for song in songs) for split in SPLITS},
             "queries_per_split": {split: sum(record["split"] == split for record in records) for split in SPLITS},
             "empty_splits": empty_splits,
             "songs_per_dataset": dict(Counter(song["dataset"] for song in songs)),
             "labels_per_split": {split: sorted({stem["label"] for song in songs if song["split"] == split for stem in song["stems"]}) for split in SPLITS},
             "negative_queries": sum(not record["target_present"] for record in records),
             "unequal_duration_songs": [song["song_id"] for song in songs if song["unequal_track_lengths"]],
             "alignment_policy": "same native sample rate per song; aligned origin; right-pad short tracks; no inferred time offsets",
             "mixture_policy": "sum of all listed raw tracks; mastered mixture.wav is not used",
             "overlap_checks": ["canonical song_id", "canonical audio path", "configured split_group"],
             "limits": ["no cross-dataset audio-content/near-duplicate fingerprint check", "coarse stem labels do not prove acoustic absence", "Slakh uses inst_class, not 128-program instrument identity"]}
    (output/"audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **audit}, ensure_ascii=False, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--workspace-root", help="Parent mss-combine directory; relative dataset roots resolve here")
    args = parser.parse_args()
    prepare(args.config, args.workspace_root)


if __name__ == "__main__":
    main()
