"""Bounded GPU-memory overlap-add inference. Audio remains on the CPU."""

import math

import numpy as np
import torch
from scipy.signal import resample_poly


def resample_mono(audio, original_rate, target_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size < 1 or not np.isfinite(audio).all():
        raise ValueError('audio must contain finite samples and have shape (T,) or (T,C)')
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError('sample rates must be positive')
    if original_rate == target_rate:
        return audio.copy()
    divisor = math.gcd(int(original_rate), int(target_rate))
    return resample_poly(audio, target_rate // divisor, original_rate // divisor).astype(np.float32)


@torch.inference_mode()
def overlap_add(model, mixture, embedding, segment_samples, device='cpu', overlap=0.5):
    """Return a CPU tensor (1,T); normalize every boundary by its window sum.

    The Hann window is floored above zero so the first/last file sample remains
    represented even when no adjacent chunk exists. No network call sees GT.
    """
    if mixture.ndim != 2 or mixture.shape[0] != 1 or mixture.shape[1] < 1:
        raise ValueError('mixture must have shape (1,T>0)')
    if not isinstance(segment_samples, int) or segment_samples < 1:
        raise ValueError('segment_samples must be a positive integer')
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('overlap must lie in [0,1)')
    if not torch.isfinite(mixture).all() or not torch.isfinite(embedding).all():
        raise ValueError('mixture and embedding must be finite')
    model.eval()
    device = torch.device(device)
    mixture = mixture.detach().to(device='cpu', dtype=torch.float32)
    query = embedding.detach().to(device=device, dtype=torch.float32).reshape(1, -1)
    length = mixture.shape[-1]
    stride = max(1, int(round(segment_samples * (1 - overlap))))
    starts = list(range(0, max(1, length - segment_samples + 1), stride))
    last = max(0, length - segment_samples)
    if starts[-1] != last:
        starts.append(last)
    result = torch.zeros_like(mixture)
    weights = torch.zeros(length, dtype=torch.float32)
    window = torch.hann_window(segment_samples, periodic=False).clamp_min(1e-3)
    for start in starts:
        valid = min(segment_samples, length - start)
        block = mixture[:, start:start + valid]
        if valid < segment_samples:
            block = torch.nn.functional.pad(block, (0, segment_samples - valid))
        predicted = model(block[None].to(device), query)[0, :, :valid].float().cpu()
        if predicted.shape != (1, valid) or not torch.isfinite(predicted).all():
            raise FloatingPointError('model produced invalid inference samples')
        result[:, start:start + valid] += predicted * window[:valid]
        weights[start:start + valid] += window[:valid]
    if (weights <= 0).any():
        raise RuntimeError('overlap-add left uncovered samples')
    return result / weights[None]


def restore_length(audio, source_rate, target_rate, frames):
    value = resample_mono(np.asarray(audio), source_rate, target_rate)
    if len(value) < frames:
        value = np.pad(value, (0, frames - len(value)))
    return value[:frames].astype(np.float32)
