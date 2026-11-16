"""Small waveform prototype, not a reproduction of DiCoSe or MGE-LDM.

The EDM variable is the correction to a frozen deterministic target estimate.
The remainder is bookkeeping, x - target, rather than a second diffusion track.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    channels: int = 2
    query_dim: int = 512
    hidden: int = 64
    layers: int = 6
    sigma_data: float = 0.1

    def __post_init__(self):
        for name in ("channels", "query_dim", "hidden", "layers"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.sigma_data) or self.sigma_data <= 0:
            raise ValueError("sigma_data must be positive and finite")


class FiLMBlock(nn.Module):
    def __init__(self, hidden, condition_dim, dilation):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.affine = nn.Linear(condition_dim, 2 * hidden)

    def forward(self, x, condition):
        scale, shift = self.affine(condition).chunk(2, dim=-1)
        h = self.conv(F.silu(x))
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return (x + F.silu(h)) / math.sqrt(2)


class ConditionedConv(nn.Module):
    def __init__(self, input_channels, output_channels, config, noise_condition=False):
        super().__init__()
        self.noise_condition = noise_condition
        condition_dim = config.query_dim + int(noise_condition)
        self.input = nn.Conv1d(input_channels, config.hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            FiLMBlock(config.hidden, condition_dim, 2 ** (i % 6))
            for i in range(config.layers)
        ])
        self.output = nn.Conv1d(config.hidden, output_channels, 3, padding=1)

    def forward(self, x, query, log_sigma=None):
        condition = query
        if self.noise_condition:
            condition = torch.cat((condition, log_sigma[:, None] / 4), dim=-1)
        h = self.input(x)
        for block in self.blocks:
            h = block(h, condition)
        return self.output(F.silu(h))


class HybridSeparator(nn.Module):
    """Shared query model; query vectors must come from a fixed real encoder."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.channels
        self.det = ConditionedConv(c, c, self.config)
        # Noisy correction, mixture, deterministic target, deterministic remainder.
        self.refiner = ConditionedConv(4 * c, c, self.config, noise_condition=True)
        self._det_frozen = False

    def train(self, mode=True):
        super().train(mode)
        if self._det_frozen:
            self.det.eval()
        return self

    def freeze_separator(self):
        self._det_frozen = True
        self.det.requires_grad_(False)
        self.det.eval()

    def _validate(self, mixture, query):
        if mixture.ndim != 3 or mixture.shape[1] != self.config.channels or mixture.shape[2] < 1:
            raise ValueError("mixture must have shape (batch, configured channels, time>0)")
        if mixture.shape[0] < 1 or query.shape != (mixture.shape[0], self.config.query_dim):
            raise ValueError("query must have shape (batch, configured query_dim)")
        if not mixture.is_floating_point() or not query.is_floating_point():
            raise ValueError("mixture and query must be floating point")
        if mixture.device != query.device or mixture.dtype != query.dtype:
            raise ValueError("mixture and query must have the same device and dtype")
        if not torch.isfinite(mixture).all() or not torch.isfinite(query).all():
            raise ValueError("mixture and query must be finite")
        norm = query.norm(dim=-1)
        if not torch.isfinite(norm).all() or (norm <= 1e-12).any():
            raise ValueError("query embeddings must have finite, nonzero norms")
        return F.normalize(query, dim=-1)

    def deterministic(self, mixture, query):
        query = self._validate(mixture, query)
        return self.det(mixture, query)

    def denoise_correction(self, noisy, sigma, mixture, seed, query):
        """EDM preconditioning; sigma has one positive value per batch item."""
        if sigma.shape != (mixture.shape[0],) or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("sigma must have shape (batch,) and be finite and positive")
        if noisy.shape != mixture.shape or seed.shape != mixture.shape:
            raise ValueError("correction, seed and mixture shapes must match")
        sd = self.config.sigma_data
        s = sigma[:, None, None]
        denom = s.square() + sd ** 2
        c_skip = sd ** 2 / denom
        c_out = s * sd / denom.sqrt()
        c_in = denom.rsqrt()
        context = torch.cat((c_in * noisy, mixture, seed, mixture - seed), dim=1)
        return c_skip * noisy + c_out * self.refiner(context, query, sigma.log())

    def diffusion_loss(self, mixture, target, query, target_present=None,
                       absence_weight=1.0, log_sigma_mean=-2.0, log_sigma_std=1.0):
        query = self._validate(mixture, query)
        if (target.shape != mixture.shape or target.device != mixture.device
                or target.dtype != mixture.dtype or not torch.isfinite(target).all()):
            raise ValueError("target must be finite and match mixture shape, device and dtype")
        if absence_weight < 0 or not math.isfinite(absence_weight):
            raise ValueError("absence_weight must be finite and nonnegative")
        if log_sigma_std <= 0 or not math.isfinite(log_sigma_std) or not math.isfinite(log_sigma_mean):
            raise ValueError("invalid lognormal sigma distribution")
        if target_present is None:
            target_present = torch.ones(mixture.shape[0], dtype=torch.bool, device=mixture.device)
        if (target_present.shape != (mixture.shape[0],) or target_present.dtype != torch.bool
                or target_present.device != mixture.device):
            raise ValueError("target_present must be a boolean vector (batch,) on the mixture device")
        absent = ~target_present
        if absent.any() and target[absent].abs().max() > 1e-6:
            raise ValueError("absent-query targets must be zero")
        # Always detached, including when called outside the training CLI.
        self.det.eval()
        with torch.no_grad():
            seed = self.det(mixture, query)
        clean = target - seed
        sigma = (torch.randn(mixture.shape[0], device=mixture.device, dtype=mixture.dtype)
                 * log_sigma_std + log_sigma_mean).exp()
        noisy = clean + sigma[:, None, None] * torch.randn_like(clean)
        prediction = self.denoise_correction(noisy, sigma, mixture, seed, query)
        sd = self.config.sigma_data
        weight = (sigma.square() + sd ** 2) / (sigma * sd).square()
        edm = ((prediction - clean).square().mean(dim=(1, 2)) * weight).mean()
        absence = (seed[absent] + prediction[absent]).square().mean() if absent.any() else prediction.sum() * 0
        return {"loss": edm + absence_weight * absence, "edm": edm, "absence": absence}

    @torch.no_grad()
    def separate(self, mixture, query, steps=8, sigma_min=0.002, sigma_max=0.5,
                 rho=7.0, generator=None, refine=True):
        """Euler EDM probability-flow sampler; steps counts refiner evaluations.

        Starts with Gaussian correction noise at sigma_max, then adds the frozen
        deterministic seed. A finite maximum sigma is an approximation to the
        high-noise prior. No CFG: this prototype has no unconditional training.
        """
        query = self._validate(mixture, query)
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not (0 < sigma_min <= sigma_max and math.isfinite(sigma_max)):
            raise ValueError("require finite 0 < sigma_min <= sigma_max")
        if not math.isfinite(rho) or rho <= 0:
            raise ValueError("rho must be positive and finite")
        self.eval()
        seed = self.det(mixture, query)
        if not refine:
            return seed, mixture - seed
        ramp = torch.linspace(0, 1, steps, device=mixture.device, dtype=mixture.dtype)
        sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        state = torch.randn(seed.shape, device=seed.device, dtype=seed.dtype, generator=generator) * sigmas[0]
        for i in range(steps):
            sigma = sigmas[i].expand(mixture.shape[0])
            clean = self.denoise_correction(state, sigma, mixture, seed, query)
            derivative = (state - clean) / sigmas[i]
            state = state + (sigmas[i + 1] - sigmas[i]) * derivative
        target = seed + state
        if not torch.isfinite(target).all():
            raise FloatingPointError("non-finite sampler output")
        return target, mixture - target

    @torch.no_grad()
    def separate_queries(self, mixture, queries, **kwargs):
        """queries: (K, B, D); returns targets/rests (K, B, C, T).

        Each query is independent. Overlapping descriptions can select the same
        sound; summing K targets is therefore not a valid consistency constraint.
        """
        if queries.ndim != 3 or queries.shape[0] < 1:
            raise ValueError("queries must have shape (K>0, batch, query_dim)")
        results = [self.separate(mixture, query, **kwargs) for query in queries]
        return torch.stack([r[0] for r in results]), torch.stack([r[1] for r in results])

    def checkpoint(self, stage, sample_rate, embedding_encoder):
        return {"config": asdict(self.config), "model": self.state_dict(),
                "stage": stage, "sample_rate": sample_rate,
                "embedding_encoder": embedding_encoder}
