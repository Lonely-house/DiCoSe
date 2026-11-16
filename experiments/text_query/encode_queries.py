"""Export genuine CLAP text embeddings. Requires optional laion-clap + weights.

This command is separate from the torch/numpy-only numerical prototype. Its
outputs are compatible with train.py's prompt_embedding and --queries inputs.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clap-checkpoint', required=True)
    parser.add_argument('--prompt', action='append', required=True,
                        help='Repeat for each distinct text query; preserve order.')
    parser.add_argument('--audio-model', default='HTSAT-base')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if any(not prompt.strip() for prompt in args.prompt):
        parser.error('prompts must be nonempty')
    checkpoint = Path(args.clap_checkpoint).resolve(strict=True)
    if not checkpoint.is_file():
        parser.error('--clap-checkpoint must point to a checkpoint file')
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        parser.error('--output must be absent or an empty directory')
    try:
        import laion_clap
    except ImportError as error:
        raise SystemExit('Install laion-clap in a compatible CLAP environment; '
                         'see LAION-AI/CLAP and this prototype README.') from error

    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    package_version = version('laion-clap')
    encoder_id = (f'laion-clap:{package_version}:{args.audio_model}:roberta:'
                  f'no-fusion:sha256={digest.hexdigest()}:l2')
    encoder = laion_clap.CLAP_Module(enable_fusion=False, amodel=args.audio_model,
                                    tmodel='roberta', device=args.device)
    encoder.load_ckpt(str(checkpoint), verbose=False)
    encoder.eval()
    with torch.no_grad():
        vectors = np.asarray(encoder.get_text_embedding(args.prompt, use_tensor=False),
                             dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(args.prompt) or not np.isfinite(vectors).all():
        raise ValueError('CLAP returned invalid text embeddings')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError('CLAP returned a zero text embedding')
    vectors = vectors / norms
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (prompt, vector) in enumerate(zip(args.prompt, vectors)):
        filename = f'query_{index:03d}.npy'
        np.save(destination / filename, vector, allow_pickle=False)
        rows.append({'prompt': prompt, 'prompt_embedding': filename})
    metadata = {'embedding_encoder': encoder_id, 'query_dim': vectors.shape[1],
                'normalization': 'l2', 'text_model': 'roberta',
                'tokenizer_max_length': 77, 'queries': rows,
                'note': 'Use this exact encoder identity for training and inference. '
                        'Original CLAP tokenization truncates long prompts.'}
    (destination / 'queries.json').write_text(json.dumps(metadata, ensure_ascii=False,
                                                       indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == '__main__':
    main()
