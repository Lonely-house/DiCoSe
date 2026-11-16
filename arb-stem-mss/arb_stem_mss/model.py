"""Text-conditioned single-target DET using the actual vendored DiCoSe U-Net."""

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.dicose import UNet1d


class TextConditionedDET(nn.Module):
    """Map only (mixture, text embedding) to one target waveform.

    The U-Net's global ``features`` path maps the adapted text embedding to
    scale/shift conditioning throughout the original DiCoSe residual blocks.
    No true source/submixture enters this forward call. The complement is
    computed by the caller as mixture minus prediction.
    """

    def __init__(self, model_config: dict[str, Any]):
        super().__init__()
        self.model_config = deepcopy(model_config)
        self.query_dim = int(model_config.get("query_dim", 512))
        self.condition_dim = int(model_config.get("condition_dim", self.query_dim))
        if self.query_dim < 1 or self.condition_dim < 1:
            raise ValueError("query_dim and condition_dim must be positive")
        arguments = deepcopy(model_config["unet"])
        if arguments.get("in_channels") != 1 or arguments.get("out_channels", 1) != 1:
            raise ValueError("Phase-1 DET requires one mixture channel and one target channel")
        if arguments.get("use_context_time", False):
            raise ValueError("DET has no diffusion timestep: use_context_time must be false")
        if arguments.get("context_channels") not in (None, []):
            raise ValueError("Phase-1 DET accepts mixture and text only, without extra GT context")
        if arguments.get("context_embedding_features") is not None:
            raise ValueError("This baseline uses pooled text features, not token cross-attention")
        configured_context = arguments.get("context_features")
        if configured_context not in (None, self.condition_dim):
            raise ValueError("unet.context_features must match model.condition_dim")
        arguments["context_features"] = self.condition_dim
        arguments["use_context_time"] = False
        arguments["out_channels"] = 1
        if model_config.get("query_adapter", True):
            self.query_adapter = nn.Sequential(
                nn.Linear(self.query_dim, self.condition_dim),
                nn.GELU(),
                nn.Linear(self.condition_dim, self.condition_dim),
            )
        else:
            if self.query_dim != self.condition_dim:
                raise ValueError("Without a query adapter, query_dim must equal condition_dim")
            self.query_adapter = nn.Identity()
        factors = arguments["factors"]
        patch_factor = arguments["patch_factor"]
        patch_blocks = arguments["patch_blocks"]
        if (not isinstance(patch_factor, int) or patch_factor < 1
                or not isinstance(patch_blocks, int) or patch_blocks < 1
                or any(not isinstance(f, int) or f < 1 for f in factors)):
            raise ValueError("Patch factors, patch blocks, and downsample factors must be positive integers")
        self.required_multiple = patch_factor ** patch_blocks * math.prod(factors)
        self.unet = UNet1d(**arguments)

    def _prepare(self, mixture: torch.Tensor, embedding: torch.Tensor):
        if mixture.ndim != 3 or mixture.shape[0] < 1 or mixture.shape[1] != 1 or mixture.shape[2] < 1:
            raise ValueError("mixture must have shape [B>0,1,T>0]")
        if embedding.shape != (mixture.shape[0], self.query_dim):
            raise ValueError(f"text embedding must have shape [B,{self.query_dim}]")
        if not mixture.is_floating_point() or not embedding.is_floating_point():
            raise ValueError("mixture and text embedding must be floating point")
        if mixture.device != embedding.device or mixture.dtype != embedding.dtype:
            raise ValueError("mixture and embedding must share dtype and device")
        if not torch.isfinite(mixture).all() or not torch.isfinite(embedding).all():
            raise ValueError("mixture and text embedding must contain finite values")
        if (embedding.float().norm(dim=-1) <= 1e-12).any():
            raise ValueError("text embeddings must be nonzero")
        length = mixture.shape[-1]
        padding = (-length) % self.required_multiple
        if padding:
            mixture = F.pad(mixture, (0, padding))
        condition = self.query_adapter(F.normalize(embedding, dim=-1))
        return mixture, condition, length

    def forward(self, mixture: torch.Tensor, query_embedding: torch.Tensor) -> torch.Tensor:
        mixture, condition, length = self._prepare(mixture, query_embedding)
        prediction = self.unet(mixture, features=condition)
        return prediction[..., :length]

    def extract_features(self, mixture: torch.Tensor, query_embedding: torch.Tensor) -> dict:
        """Feature contract for future DIF, without implementing DIF training.

        Feature maps retain padded resolutions. The estimate is cropped to the
        original waveform length. Upstream's last feature is the prediction;
        it is deliberately separated here to avoid accidental list mutation.
        """
        padded, condition, length = self._prepare(mixture, query_embedding)
        feature_list = self.unet.get_feature(padded, features=condition)
        return {
            "estimate": feature_list[-1][..., :length],
            "features": tuple(feature_list[:-1]),
            "query_condition": condition,
            "original_length": length,
            "padded_length": padded.shape[-1],
        }

    def load_upstream_weights(self, checkpoint_path: str | Path) -> dict:
        """Load shape-compatible native DiCoSe DET U-Net weights explicitly.

        Four-class feature projections do not match the new semantic condition
        width. They and the new text adapter stay initialized from this model.
        A report lists every skipped key; this is not a strict resumed run.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError("Expected a native DiCoSe state_dict checkpoint")
        own = self.unet.state_dict()
        selected, skipped = {}, {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                skipped[key] = "not a parameter tensor"
                continue
            candidate = key
            for prefix in ("model.unet.", "unet."):
                if key.startswith(prefix):
                    candidate = key[len(prefix):]
                    break
            if candidate not in own:
                skipped[key] = "not a supported U-Net key"
            elif own[candidate].shape != value.shape:
                skipped[key] = f"shape {list(value.shape)} != {list(own[candidate].shape)}"
            else:
                selected[candidate] = value
        if not selected:
            raise ValueError("No shape-compatible DiCoSe U-Net weights found")
        incompatibility = self.unet.load_state_dict(selected, strict=False)
        return {"loaded_keys": sorted(selected), "skipped_keys": skipped,
                "missing_keys": list(incompatibility.missing_keys),
                "query_adapter_initialized": True}
