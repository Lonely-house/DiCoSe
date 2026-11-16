"""Configuration paths are relative to the project, never the caller's cwd."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value, base=PROJECT_ROOT):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    with resolve_path(path).open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError('configuration must be a YAML mapping')
    config = deepcopy(config)
    for key in ('model', 'audio', 'data', 'train', 'loss'):
        if not isinstance(config.get(key), dict):
            raise ValueError(f'configuration requires a {key} mapping')
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
