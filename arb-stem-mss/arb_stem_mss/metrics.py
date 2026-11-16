"""Reference waveform metrics, with silence and absent-query cases separated.

``sdr_db`` is scale-dependent signal-to-error ratio, not museval/BSS Eval SDR.
Song-level macro aggregation prevents songs with many queries from dominating.
"""

from collections import defaultdict
import math

import torch


def _audio(value, name):
    value = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or min(value.shape) < 1 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite audio with shape (C,T) or (T,)")
    return value


def compute_metrics(prediction, target, mixture=None, present=True, num_samples=None,
                    silence_rms=1e-5):
    """Compute metrics for one query. Zero/silent targets have null SDR values.

    SI-SDR uses the flattened, mean-centered multichannel signal. Completely DC
    references have undefined SI-SDR and are recorded as null, too. Energy ratios
    use an epsilon floor; silent mixtures have no relative-energy denominator.
    """
    prediction = _audio(prediction, "prediction")
    target = _audio(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if not isinstance(present, bool):
        if isinstance(present, torch.Tensor) and present.numel() == 1 and present.dtype == torch.bool:
            present = bool(present.item())
        else:
            raise ValueError("present must be boolean")
    if not math.isfinite(silence_rms) or silence_rms < 0:
        raise ValueError("silence_rms must be finite and nonnegative")
    frames = prediction.shape[-1]
    if num_samples is not None:
        if isinstance(num_samples, torch.Tensor):
            if num_samples.numel() != 1 or num_samples.dtype == torch.bool or num_samples.is_floating_point():
                raise ValueError("num_samples must be one integer")
            num_samples = num_samples.item()
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or not 1 <= num_samples <= frames:
            raise ValueError("num_samples must lie in [1,T]")
        frames = num_samples
    prediction, target = prediction[..., :frames], target[..., :frames]
    if not present and target.abs().max() > 1e-6:
        raise ValueError("absent-query references must be zero")
    p, t = prediction.flatten(), target.flatten()
    error = p - t
    output_power, target_power = p.square().mean(), t.square().mean()
    target_silent = target_power.sqrt().item() <= silence_rms
    metrics = {
        "si_sdr_db": None, "sdr_db": None,
        "mse": error.square().mean().item(), "l1": error.abs().mean().item(),
        "output_rms": output_power.sqrt().item(), "target_rms": target_power.sqrt().item(),
        "relative_output_energy_db": None,
        "target_silent": target_silent, "target_present": present,
    }
    epsilon = 1e-12
    if present and not target_silent:
        metrics["sdr_db"] = (10 * torch.log10((t.square().sum() + epsilon) /
                                                   (error.square().sum() + epsilon))).item()
        centered_target, centered_prediction = t - t.mean(), p - p.mean()
        target_energy = centered_target.square().sum()
        if target_energy.item() > epsilon:
            # A completely silent estimate gives an undefined 0/0 SI-SDR.
            # Use an explicit failure floor instead of allowing epsilon/epsilon
            # to report 0 dB, and keep the example in macro aggregation.
            if centered_prediction.square().sum().item() <= epsilon:
                metrics["si_sdr_db"] = -120.0
            else:
                projected = (torch.dot(centered_prediction, centered_target) /
                             target_energy) * centered_target
                residual = centered_prediction - projected
                metrics["si_sdr_db"] = (10 * torch.log10((projected.square().sum() + epsilon) /
                                                            (residual.square().sum() + epsilon))).item()
    if mixture is not None:
        mixture = _audio(mixture, "mixture")
        if mixture.shape[0] != prediction.shape[0] or mixture.shape[-1] < frames:
            raise ValueError("mixture shape must match prediction before valid-length cropping")
        mixture_power = mixture[..., :frames].square().mean()
        if mixture_power.sqrt().item() > silence_rms:
            metrics["relative_output_energy_db"] = (10 * torch.log10(
                (output_power + epsilon) / (mixture_power + epsilon))).item()
    return metrics


def _means(rows, keys):
    output = {}
    for key in keys:
        values = [float(row[key]) for row in rows
                  if row.get(key) is not None and math.isfinite(float(row[key]))]
        output[key] = sum(values) / len(values) if values else None
    return output


@torch.no_grad()
def chunked_mrstft_loss(prediction, target, criterion, segment_samples, num_samples=None):
    """CPU, non-overlapping MR-STFT with valid-sample-weighted aggregation.

    This deliberately differs from computing an STFT across a complete song:
    each independent chunk has its own analysis boundary, and a short final
    chunk is padded only to the configured FFT windows. No waveform/remainder
    loss is included in this metric.
    """
    # Keep fp32 CPU buffers and convert only the current chunk; do not allocate
    # a float64 copy or a full-song spectrogram for this spectral metric.
    prediction = torch.as_tensor(prediction).detach().to(device='cpu')
    target = torch.as_tensor(target).detach().to(device='cpu')
    if prediction.ndim == 1:
        prediction = prediction[None]
    if target.ndim == 1:
        target = target[None]
    if prediction.ndim != 2 or min(prediction.shape) < 1:
        raise ValueError('prediction must have shape (C,T) or (T,)')
    if prediction.shape != target.shape:
        raise ValueError('prediction and target shapes must match')
    if isinstance(segment_samples, bool) or not isinstance(segment_samples, int) or segment_samples < 1:
        raise ValueError('segment_samples must be a positive integer')
    valid = prediction.shape[-1] if num_samples is None else num_samples
    if isinstance(valid, bool) or not isinstance(valid, int) or not 1 <= valid <= prediction.shape[-1]:
        raise ValueError('num_samples must lie in [1,T]')
    if not criterion.fft_sizes:
        raise ValueError('spectral metrics require at least one FFT resolution')
    total = 0.0
    for start in range(0, valid, segment_samples):
        end = min(valid, start + segment_samples)
        p, t = prediction[:, start:end].float(), target[:, start:end].float()
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():
            raise ValueError('spectral metric audio must be finite')
        score = criterion._spectral(p, t)
        if not torch.isfinite(score):
            raise FloatingPointError('non-finite chunked MR-STFT metric')
        total += float(score) * (end - start)
    return total / valid


def aggregate_metrics(rows):
    """Average within each song, then macro-average songs (not query records)."""
    groups = defaultdict(list)
    for row in rows:
        if not isinstance(row.get("song_id"), str) or not row["song_id"]:
            raise ValueError("every metric record needs a nonempty song_id")
        groups[row["song_id"]].append(row)
    fields = ["si_sdr_db", "sdr_db", "mse", "l1", "output_rms", "target_rms",
              "relative_output_energy_db"]
    energy_fields = ["output_rms", "relative_output_energy_db"]
    if any(row.get('chunked_mrstft_loss') is not None for row in rows):
        fields.append('chunked_mrstft_loss')
        energy_fields.append('chunked_mrstft_loss')
    by_song = {}
    for song_id, song_rows in sorted(groups.items()):
        active = [row for row in song_rows if row.get("target_present", True)
                  and not row.get("target_silent", False)]
        absent = [row for row in song_rows if not row.get("target_present", True)]
        silent_present = [row for row in song_rows if row.get("target_present", True)
                          and row.get("target_silent", False)]
        by_song[song_id] = {
            "num_queries": len(song_rows), "num_active_queries": len(active),
            "num_absent_queries": len(absent), "num_silent_present_queries": len(silent_present),
            "active": _means(active, fields),
            "absent": _means(absent, energy_fields),
            "silent_present": _means(silent_present, energy_fields),
        }
    song_values = list(by_song.values())
    return {
        "num_queries": len(rows), "num_songs": len(by_song),
        "num_active_queries": sum(song["num_active_queries"] for song in song_values),
        "num_absent_queries": sum(song["num_absent_queries"] for song in song_values),
        "num_silent_present_queries": sum(song["num_silent_present_queries"] for song in song_values),
        "macro_by_song": _means([song["active"] for song in song_values], fields),
        "absent_macro_by_song": _means([song["absent"] for song in song_values], energy_fields),
        "silent_present_macro_by_song": _means([song["silent_present"] for song in song_values], energy_fields),
        "by_song": by_song,
        "sdr_definition": "scale-dependent signal-to-error ratio; not BSS Eval/museval SDR",
        "si_sdr_definition": "mean-centered flattened channel/time projection; silent/DC references excluded; silent estimates on active targets use -120 dB floor",
    }
