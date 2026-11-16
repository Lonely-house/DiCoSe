"""Waveform and multi-resolution spectral objectives for single-target DET.

The target complement is defined by subtraction, so adding its L1/MSE would
duplicate the target objective rather than supply additional supervision.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SeparationLoss(nn.Module):
    """Masked waveform loss plus spectral convergence and log magnitude loss.

    ``num_samples`` contains unpadded lengths. Spectral losses are evaluated on
    each valid segment separately; right padding is used only when a segment is
    shorter than an FFT window. ``present=False`` means a verified absent query,
    whereas a present but locally silent target is not an absent-query example.
    """

    def __init__(self, config):
        super().__init__()
        self.wave_type = config.get("wave_type", "l1")
        if self.wave_type not in {"l1", "mse"}:
            raise ValueError("loss.wave_type must be l1 or mse")
        self.wave_weight = self._weight(config, "wave_weight", 1.0)
        self.mrstft_weight = self._weight(config, "mrstft_weight", 1.0)
        self.absence_weight = self._weight(config, "absence_weight", 0.0)
        self.fft_sizes = list(config.get("fft_sizes", [1024, 2048, 512]))
        self.hop_sizes = list(config.get("hop_sizes", [256, 512, 128]))
        self.win_lengths = list(config.get("win_lengths", self.fft_sizes))
        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths)):
            raise ValueError("FFT, hop and window lists must have equal lengths")
        if self.mrstft_weight and not self.fft_sizes:
            raise ValueError("MR-STFT requires at least one resolution")
        for index, (fft, hop, window) in enumerate(zip(
                self.fft_sizes, self.hop_sizes, self.win_lengths)):
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in (fft, hop, window)) or window > fft:
                raise ValueError("FFT/hop/window must be positive integers and window <= FFT")
            self.register_buffer(f"window_{index}", torch.hann_window(window), persistent=False)
        self.eps = 1e-7

    @staticmethod
    def _weight(config, name, default):
        value = float(config.get(name, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{name} must be finite and nonnegative")
        return value

    def _spectral(self, prediction, target):
        terms = []
        for index, (fft, hop, window_length) in enumerate(zip(
                self.fft_sizes, self.hop_sizes, self.win_lengths)):
            if prediction.shape[-1] < fft:
                padding = fft - prediction.shape[-1]
                p = F.pad(prediction, (0, padding))
                t = F.pad(target, (0, padding))
            else:
                p, t = prediction, target
            window = getattr(self, f"window_{index}").to(p.device, p.dtype)
            p_mag = torch.stft(p, n_fft=fft, hop_length=hop,
                               win_length=window_length, window=window,
                               center=False, return_complex=True).abs()
            t_mag = torch.stft(t, n_fft=fft, hop_length=hop,
                               win_length=window_length, window=window,
                               center=False, return_complex=True).abs()
            reference_norm = torch.linalg.vector_norm(t_mag)
            if reference_norm.detach() <= self.eps:
                # Relative spectral convergence is undefined for a silent
                # target; absolute magnitude error avoids an arbitrary 1/eps
                # amplification that could swamp all active-query examples.
                convergence = F.l1_loss(p_mag, t_mag)
            else:
                convergence = torch.linalg.vector_norm(p_mag - t_mag) / reference_norm
            log_magnitude = F.l1_loss(p_mag.clamp_min(self.eps).log(),
                                      t_mag.clamp_min(self.eps).log())
            terms.append(convergence + log_magnitude)
        return torch.stack(terms).mean()

    def forward(self, prediction, target, num_samples=None, present=None):
        if prediction.ndim != 3 or prediction.shape != target.shape:
            raise ValueError("prediction and target must have matching (B,C,T) shapes")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError("prediction and target must be floating point")
        if prediction.device != target.device:
            raise ValueError("prediction and target must be on the same device")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("prediction and target must be finite")
        batch, channels, frames = prediction.shape
        if min(batch, channels, frames) < 1:
            raise ValueError("audio tensors cannot have empty dimensions")
        if num_samples is None:
            lengths = torch.full((batch,), frames, device=prediction.device, dtype=torch.long)
        else:
            lengths = torch.as_tensor(num_samples, device=prediction.device)
            if lengths.shape != (batch,) or lengths.dtype == torch.bool or lengths.is_floating_point():
                raise ValueError("num_samples must be an integer vector with shape (B,)")
            if (lengths < 1).any() or (lengths > frames).any():
                raise ValueError("num_samples must lie in [1, T]")
        if present is None:
            present = torch.ones(batch, device=prediction.device, dtype=torch.bool)
        else:
            present = torch.as_tensor(present, device=prediction.device)
            if present.shape != (batch,) or present.dtype != torch.bool:
                raise ValueError("present must be a boolean vector with shape (B,)")
        # STFT and waveform accumulation use fp32 even under mixed precision.
        prediction, target = prediction.float(), target.float()
        waves, spectra, absent = [], [], []
        for index, length in enumerate(lengths.tolist()):
            p, t = prediction[index, :, :length], target[index, :, :length]
            waves.append(F.l1_loss(p, t) if self.wave_type == "l1" else F.mse_loss(p, t))
            if self.mrstft_weight:
                spectra.append(self._spectral(p, t))
            if not present[index]:
                if t.abs().max() > 1e-6:
                    raise ValueError("absent-query target must be zero over valid samples")
                absent.append(p.square().mean())
        zero = prediction.sum() * 0
        wave = torch.stack(waves).mean()
        mrstft = torch.stack(spectra).mean() if spectra else zero
        absence = torch.stack(absent).mean() if absent else zero
        loss = self.wave_weight * wave + self.mrstft_weight * mrstft + self.absence_weight * absence
        return {"loss": loss, "wave": wave, "mrstft": mrstft, "absence": absence}
