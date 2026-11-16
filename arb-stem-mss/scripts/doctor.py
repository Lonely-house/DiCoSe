"""Check the runtime, vendored provenance, and configured dataset roots."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', default='configs/data.yaml')
    parser.add_argument('--workspace-root', default=str(ROOT.parent))
    parser.add_argument('--check-datasets', action='store_true',
                        help='Fail if an enabled dataset root is absent; does not scan full audio.')
    args = parser.parse_args()
    result = {'python': platform.python_version(), 'project_root': str(ROOT),
              'workspace_root': str(Path(args.workspace_root).resolve()),
              'packages': {}, 'vendor_integrity': {}, 'dataset_roots': {}}
    failed = False
    for name in ('torch', 'numpy', 'scipy', 'soundfile', 'PyYAML', 'einops', 'einops-exts',
                 'typing-extensions'):
        try:
            result['packages'][name] = version(name)
        except PackageNotFoundError:
            result['packages'][name] = None
            failed = True
    import torch
    result['cuda_available'] = torch.cuda.is_available()
    result['cuda_device_count'] = torch.cuda.device_count()
    upstream = json.loads((ROOT / 'UPSTREAM.json').read_text(encoding='utf-8'))
    for relative, expected in upstream['dicose']['files'].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        matches = actual == expected['sha256']
        result['vendor_integrity'][relative] = matches
        failed |= not matches
    config_path = Path(args.data_config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    for name, entry in config['datasets'].items():
        root = Path(os.path.expandvars(os.path.expanduser(entry['root'])))
        if not root.is_absolute():
            root = Path(args.workspace_root) / root
        exists = root.is_dir()
        enabled = entry.get('enabled', True)
        result['dataset_roots'][name] = {'path': str(root.resolve()),
                                         'enabled': enabled, 'exists': exists}
        failed |= args.check_datasets and enabled and not exists
    result['status'] = 'failed' if failed else 'passed'
    result['scope'] = 'Runtime and metadata preflight only; not a music quality evaluation.'
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
