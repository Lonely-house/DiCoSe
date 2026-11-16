"""Extract one or more independently queried targets from a WAV file."""

import argparse
import json

import numpy as np
import soundfile as sf
import torch

from arb_stem_mss.config import resolve_path, write_json
from arb_stem_mss.engine import (load_model_checkpoint, read_embedding_cache,
                                 runtime_information, seed_everything, select_device, verify_cache_checkpoint)
from arb_stem_mss.inference import overlap_add, resample_mono, restore_length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--text', action='append', required=True,
                        help='Repeat --text for multiple independent queries')
    encoder = parser.add_mutually_exclusive_group(required=True)
    encoder.add_argument('--embedding-cache')
    encoder.add_argument('--clap-checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--allow-mock-embeddings', action='store_true')
    args = parser.parse_args()
    if any(not prompt.strip() for prompt in args.text):
        parser.error('query text must not be empty')
    device = select_device(args.device)
    seed_everything(0)
    if device.type == 'cpu':
        torch.set_num_threads(1)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device, args.allow_mock_embeddings)
    if args.embedding_cache:
        cache = read_embedding_cache(resolve_path(args.embedding_cache), args.allow_mock_embeddings)
    else:
        from arb_stem_mss.text import encode_texts
        cache = encode_texts(args.text, resolve_path(args.clap_checkpoint), device=str(device))
    verify_cache_checkpoint(cache, checkpoint)
    missing = [prompt for prompt in args.text if prompt not in cache['vectors']]
    if missing:
        raise ValueError(f'query text is not in the embedding cache: {missing}; regenerate the cache with these exact prompts')
    input_path = resolve_path(args.input)
    raw, original_rate = sf.read(input_path, dtype='float32', always_2d=True)
    if raw.shape[0] < 1 or not np.isfinite(raw).all():
        raise ValueError('input audio must contain finite samples')
    mono = raw.mean(axis=1)
    audio = checkpoint['config']['audio']
    model_rate = int(audio['sample_rate'])
    mixture = torch.from_numpy(resample_mono(raw, original_rate, model_rate))[None]
    destination = resolve_path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    query_metadata = []
    for index, prompt in enumerate(args.text):
        estimate = overlap_add(model, mixture, cache['vectors'][prompt],
                               int(audio['segment_samples']), device, args.overlap)
        target = restore_length(estimate[0].numpy(), model_rate, original_rate, len(mono))
        remainder = mono - target
        if not np.isfinite(target).all() or not np.isfinite(remainder).all():
            raise FloatingPointError('non-finite output after resampling')
        target_file, remainder_file = f'query_{index:03d}_target.wav', f'query_{index:03d}_remainder.wav'
        # FLOAT WAV keeps amplitude (including peaks >1) without PCM clipping.
        sf.write(destination / target_file, target, original_rate, subtype='FLOAT')
        sf.write(destination / remainder_file, remainder, original_rate, subtype='FLOAT')
        query_metadata.append({'text': prompt, 'target': target_file, 'remainder': remainder_file})
    metadata = {'input': str(input_path), 'checkpoint': str(resolve_path(args.checkpoint)),
                'input_sample_rate': original_rate, 'model_sample_rate': model_rate,
                'output_sample_rate': original_rate, 'output_frames': len(mono),
                'input_channels': raw.shape[1], 'output_channels': 1,
                'encoder_id': cache['encoder_id'], 'embedding_is_mock': cache['is_mock'],
                'segment_samples': audio['segment_samples'], 'overlap': args.overlap,
                'queries': query_metadata,
                'semantics': 'Independent queries may overlap; remainder equals original mono mixture minus target. '
                             'Additivity is bookkeeping, not a separation-quality metric.'}
    runtime = runtime_information(device)
    metadata['runtime'] = runtime
    write_json(destination / 'inference.json', metadata)
    write_json(destination / 'runtime.json', runtime)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
