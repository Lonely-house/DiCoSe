"""Vendored DiCoSe U-Net only; see LICENSE and project UPSTREAM.json.

Unlike the upstream package initializer, this does not import diffusion,
Lightning, W&B, torchaudio, or optional separator backbones.
"""

from .modules import UNet1d

__all__ = ["UNet1d"]
