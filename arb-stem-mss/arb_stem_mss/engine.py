"""Deterministic DET training with explicit provenance and epoch-boundary resume."""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT, file_sha256, load_config, resolve_path, write_json
from .data import QueryDataset
from .inference import overlap_add
from .losses import SeparationLoss
from .metrics import aggregate_metrics, chunked_mrstft_loss, compute_metrics
from .model import TextConditionedDET


def select_device(value):
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested, but this environment has no CUDA device')
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('this reproduction supports CPU and CUDA devices')
    return device


def runtime_information(device):
    """Plain JSON/weights-only-safe provenance; no shell or credential reads."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=3, check=False)
        commit = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    device = torch.device(device)
    return {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version.split()[0], 'torch': str(torch.__version__),
            'numpy': str(np.__version__), 'cuda_runtime': torch.version.cuda,
            'cuda_available': bool(torch.cuda.is_available()), 'device': str(device),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'source_git_commit': commit}


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': {'algorithm': state[0], 'keys': state[1].tolist(),
                      'position': state[2], 'has_gauss': state[3], 'cached_gaussian': state[4]}}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    value = state['numpy']
    np.random.set_state((value['algorithm'], np.asarray(value['keys'], dtype=np.uint32),
                         value['position'], value['has_gauss'], value['cached_gaussian']))


def read_embedding_cache(path, allow_mock=False):
    cache = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('encoder_id', 'query_dim', 'vectors', 'is_mock'):
        if key not in cache:
            raise ValueError(f'embedding cache missing {key}')
    if not isinstance(cache['encoder_id'], str) or not cache['encoder_id']:
        raise ValueError('embedding cache must identify its encoder')
    if not isinstance(cache['is_mock'], bool):
        raise ValueError('cache is_mock must be boolean')
    if cache['is_mock'] and not allow_mock:
        raise ValueError('mock vectors are test fixtures: explicitly pass --allow-mock-embeddings')
    return cache


def atomic_save(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_model_checkpoint(path, device='cpu', allow_mock=False):
    checkpoint = torch.load(resolve_path(path), map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('stage') != 'det':
        raise ValueError('expected an arb-stem-mss DET checkpoint version 1')
    if checkpoint.get('embedding_is_mock', False) and not allow_mock:
        raise ValueError('checkpoint used mock vectors; pass --allow-mock-embeddings for fixture checks')
    model = TextConditionedDET(checkpoint['config']['model'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device), checkpoint


def verify_cache_checkpoint(cache, checkpoint):
    if cache['encoder_id'] != checkpoint['encoder_id']:
        raise ValueError('text encoder identity does not match training checkpoint')
    if cache['is_mock'] != checkpoint['embedding_is_mock']:
        raise ValueError('mock and real embedding caches cannot be exchanged')
    if cache['query_dim'] != checkpoint['config']['model']['query_dim']:
        raise ValueError('embedding dimension does not match model checkpoint')


def _loader(dataset, batch_size, num_workers, training, seed, device):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      num_workers=num_workers, generator=generator,
                      worker_init_fn=seed_worker, pin_memory=device.type == 'cuda',
                      drop_last=False)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    totals, count, rows = {}, 0, []
    for batch in loader:
        mixture = batch['mixture'].to(device)
        target = batch['source'].to(device)
        prediction = model(mixture, batch['embedding'].to(device))
        losses = criterion(prediction, target, batch['num_samples'].to(device),
                           batch['present'].to(device))
        size = mixture.shape[0]
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
        for index in range(size):
            metrics = compute_metrics(prediction[index].cpu(), target[index].cpu(),
                                      mixture[index].cpu(), bool(batch['present'][index]),
                                      int(batch['num_samples'][index]))
            rows.append({'song_id': batch['song_id'][index],
                         'prompt': batch['prompt'][index], **metrics})
    if count == 0:
        raise ValueError('validation dataset must not be empty')
    means = {key: value / count for key, value in totals.items()}
    if not all(math.isfinite(value) for value in means.values()):
        raise FloatingPointError('non-finite validation loss')
    return means, aggregate_metrics(rows)


def _resume_signature(config):
    keys = ('seed', 'batch_size', 'num_workers', 'learning_rate', 'betas',
            'weight_decay', 'amp', 'grad_clip')
    return {'model': config['model'], 'audio': config['audio'], 'loss': config['loss'],
            'train': {key: config['train'].get(key) for key in keys}}


def run_training(args):
    config = load_config(args.config)
    settings = config['train']
    if args.device is not None:
        settings['device'] = args.device
    if args.output_dir is not None:
        settings['output_dir'] = args.output_dir
    device = select_device(settings.get('device', 'auto'))
    if device.type == 'cpu':
        torch.set_num_threads(int(settings.get('cpu_threads', 1)))
    seed = int(settings.get('seed', 42))
    seed_everything(seed)
    epochs, batch_size = int(settings['epochs']), int(settings['batch_size'])
    workers = int(settings.get('num_workers', 0))
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError('epochs/batch_size must be positive; num_workers nonnegative')
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('--max-steps must be positive')
    lr = float(settings.get('learning_rate', 1e-4))
    grad_clip = float(settings.get('grad_clip', 1.0))
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError('learning_rate and grad_clip must be positive finite values')
    output = resolve_path(settings['output_dir'])
    if not args.resume and (output / 'last.pt').exists():
        raise FileExistsError('output already has last.pt; use --resume or a new --output-dir')
    train_manifest = resolve_path(config['data']['train_manifest'])
    validation_manifest = resolve_path(config['data']['validation_manifest'])
    embedding_path = resolve_path(config['data']['embedding_cache'])
    cache = read_embedding_cache(embedding_path, args.allow_mock_embeddings)
    if cache['query_dim'] != config['model']['query_dim']:
        raise ValueError('model query_dim does not match embedding cache')
    identity = {'train_manifest_sha256': file_sha256(train_manifest),
                'validation_manifest_sha256': file_sha256(validation_manifest),
                'embedding_cache_sha256': file_sha256(embedding_path)}
    audio = config['audio']
    train_data = QueryDataset(train_manifest, embedding_path, int(audio['sample_rate']),
                              int(audio['segment_samples']), training=True, seed=seed)
    val_data = QueryDataset(validation_manifest, embedding_path, int(audio['sample_rate']),
                            int(audio['segment_samples']), training=False, seed=seed)
    if any(row.get('split') != 'train' for row in train_data.rows):
        raise ValueError('train_manifest must contain only split=train records')
    if any(row.get('split') != 'validation' for row in val_data.rows):
        raise ValueError('validation_manifest must contain only split=validation records')
    train_songs = {row['song_id'] for row in train_data.rows}
    validation_songs = {row['song_id'] for row in val_data.rows}
    if train_songs & validation_songs:
        raise ValueError('training and validation manifests share song IDs')
    train_audio = {str(Path(stem['path']).resolve()) for row in train_data.rows for stem in row['stems']}
    validation_audio = {str(Path(stem['path']).resolve()) for row in val_data.rows for stem in row['stems']}
    if train_audio & validation_audio:
        raise ValueError('training and validation manifests share source audio files')
    model = TextConditionedDET(config['model']).to(device)
    initialization_report = None
    if args.init_upstream:
        if args.resume:
            raise ValueError('--init-upstream and --resume are mutually exclusive')
        initialization_report = model.load_upstream_weights(resolve_path(args.init_upstream))
        initialization_report['checkpoint_sha256'] = file_sha256(resolve_path(args.init_upstream))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=tuple(settings.get('betas', (0.9, 0.99))),
                                 weight_decay=float(settings.get('weight_decay', 0.0)))
    use_amp = bool(settings.get('amp', False))
    if use_amp and device.type != 'cuda':
        raise ValueError('AMP is supported on CUDA only; set train.amp=false for CPU')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = SeparationLoss(config['loss']).to(device)
    start_epoch, global_step, best = 0, 0, float('inf')
    if args.resume:
        checkpoint = torch.load(resolve_path(args.resume), map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1 or checkpoint.get('stage') != 'det':
            raise ValueError('expected an arb-stem-mss DET checkpoint')
        if checkpoint.get('partial_epoch'):
            raise ValueError('this checkpoint stopped inside an epoch; exact resume requires an epoch-boundary checkpoint')
        if checkpoint.get('data_identity') != identity:
            raise ValueError('resume data/embedding hashes differ from the original run')
        if _resume_signature(checkpoint['config']) != _resume_signature(config):
            raise ValueError('resume architecture/audio/loss/optimizer/reproducibility settings differ')
        verify_cache_checkpoint(cache, checkpoint)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = int(checkpoint['epochs_completed'])
        global_step = int(checkpoint['global_step'])
        best = float(checkpoint['best_validation_loss'])
        initialization_report = checkpoint.get('initialization_report')
        restore_rng(checkpoint['rng_state'])
    if start_epoch >= epochs:
        raise ValueError('train.epochs must exceed the completed epoch count when resuming')
    if args.max_steps is not None and global_step >= args.max_steps:
        raise ValueError('--max-steps is a total global-step limit and has already been reached')
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime_information(device)
    write_json(output / 'runtime.json', runtime)
    write_json(output / 'config.json', config)
    if initialization_report is not None:
        write_json(output / 'upstream_initialization.json', initialization_report)
    validation_loader = _loader(val_data, batch_size, workers, False, seed, device)
    for epoch in range(start_epoch, epochs):
        if hasattr(train_data, 'set_epoch'):
            train_data.set_epoch(epoch)
        train_loader = _loader(train_data, batch_size, workers, True, seed + epoch, device)
        model.train()
        totals, count, partial = {}, 0, False
        for batch_index, batch in enumerate(train_loader):
            mixture, target = batch['mixture'].to(device), batch['source'].to(device)
            query = batch['embedding'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(mixture, query)
                losses = criterion(prediction, target, batch['num_samples'].to(device),
                                   batch['present'].to(device))
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('non-finite training loss')
            scaler.scale(losses['loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            size = mixture.shape[0]
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
            count += size
            if args.max_steps is not None and global_step >= args.max_steps:
                partial = batch_index + 1 < len(train_loader)
                break
        validation_loss, validation_metrics = validate(model, validation_loader, criterion, device)
        improved = validation_loss['loss'] < best
        if improved:
            best = validation_loss['loss']
        checkpoint = {
            'format_version': 1, 'stage': 'det', 'config': deepcopy(config),
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'epochs_completed': epoch if partial else epoch + 1,
            'partial_epoch': partial, 'epoch_index': epoch, 'global_step': global_step,
            'batches_completed_in_epoch': batch_index + 1,
            'best_validation_loss': best, 'encoder_id': cache['encoder_id'],
            'embedding_is_mock': cache['is_mock'], 'data_identity': identity,
            'rng_state': rng_state(), 'upstream': {'dicose_commit': 'a1dc0a41ad2b1829674a60ea74e74edfd1509083'},
            'software': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                         'numpy': str(np.__version__)},
            'runtime': runtime,
            'initialization_report': initialization_report,
            'resume_semantics': 'exact at completed epoch boundaries on the same software/device stack',
        }
        atomic_save(output / 'last.pt', checkpoint)
        if improved:
            atomic_save(output / 'best.pt', checkpoint)
        record = {'epoch': epoch + 1, 'partial_epoch': partial, 'global_step': global_step,
                  'train': {key: value / count for key, value in totals.items()},
                  'validation': validation_loss, 'validation_metrics': validation_metrics,
                  'embedding_is_mock': cache['is_mock']}
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        if args.max_steps is not None and global_step >= args.max_steps:
            break
    return output


def main_train():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/det_text.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--init-upstream', help='Initialize compatible DiCoSe U-Net weights; not a resume')
    parser.add_argument('--device')
    parser.add_argument('--output-dir')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--allow-mock-embeddings', action='store_true')
    run_training(parser.parse_args())


def main_evaluate():
    parser = argparse.ArgumentParser(description='Full-song, overlap-add DET evaluation')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--embedding-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--allow-mock-embeddings', action='store_true')
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--spectral-metrics', action='store_true',
                        help='CPU sample-weighted non-overlapping chunk MR-STFT; not whole-song STFT')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    device = select_device(args.device)
    seed_everything(0)
    if device.type == 'cpu':
        torch.set_num_threads(1)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device, args.allow_mock_embeddings)
    cache_path = resolve_path(args.embedding_cache)
    cache = read_embedding_cache(cache_path, args.allow_mock_embeddings)
    verify_cache_checkpoint(cache, checkpoint)
    audio = checkpoint['config']['audio']
    dataset = QueryDataset(resolve_path(args.manifest), cache_path, int(audio['sample_rate']),
                           segment_samples=None, training=False)
    spectral_criterion = None
    if args.spectral_metrics:
        spectral_config = deepcopy(checkpoint['config']['loss'])
        spectral_config.update({'wave_weight': 0.0, 'mrstft_weight': 1.0, 'absence_weight': 0.0})
        spectral_criterion = SeparationLoss(spectral_config).cpu()
    rows = []
    for index in range(len(dataset) if args.limit is None else min(len(dataset), args.limit)):
        item = dataset[index]
        predicted = overlap_add(model, item['mixture'], item['embedding'],
                                int(audio['segment_samples']), device, args.overlap)
        metrics = compute_metrics(predicted, item['source'], item['mixture'],
                                  bool(item['present']), int(item['num_samples']))
        if spectral_criterion is not None:
            metrics['chunked_mrstft_loss'] = chunked_mrstft_loss(
                predicted, item['source'], spectral_criterion,
                int(audio['segment_samples']), int(item['num_samples']))
        rows.append({'song_id': item['song_id'], 'prompt': item['prompt'], **metrics})
    if not rows:
        raise ValueError('evaluation manifest is empty')
    runtime = runtime_information(device)
    result = {'scope': 'full-song overlap-add', 'sample_rate': audio['sample_rate'],
              'runtime': runtime,
              'checkpoint': str(resolve_path(args.checkpoint)),
              'manifest_sha256': file_sha256(resolve_path(args.manifest)),
              'encoder_id': cache['encoder_id'], 'embedding_is_mock': cache['is_mock'],
              'sdr_definition': 'scale-dependent 10log10(||target||² / ||prediction-target||²); not BSS-Eval SDR',
              'silent_targets': 'silent/absent targets are excluded from SI-SDR and SDR',
              'overlap': args.overlap, 'segment_samples': audio['segment_samples'],
              'spectral_metrics': {
                  'enabled': args.spectral_metrics,
                  'definition': 'MR-STFT on non-overlapping segment_samples chunks; valid-length-weighted mean, '
                                'short tails zero-padded to FFT windows; not a whole-song STFT',
                  'fft_sizes': spectral_criterion.fft_sizes if spectral_criterion else [],
                  'hop_sizes': spectral_criterion.hop_sizes if spectral_criterion else [],
                  'win_lengths': spectral_criterion.win_lengths if spectral_criterion else [],
              },
              'aggregate': aggregate_metrics(rows), 'queries': rows}
    write_json(resolve_path(args.output), result)
    write_json(resolve_path(args.output).parent / 'runtime.json', runtime)
    print(json.dumps(result['aggregate'], ensure_ascii=False, allow_nan=False), flush=True)
