# Third party sources

The DET backbone is the actual DiCoSe `UNet1d`, rather than a newly invented
small convolutional approximation. The following files are copied byte for byte
from `Lonely-house/DiCoSe`, commit
`a1dc0a41ad2b1829674a60ea74e74edfd1509083`:

- `audio_diffusion_pytorch_/modules.py` → `arb_stem_mss/vendor/dicose/modules.py`
- `audio_diffusion_pytorch_/utils.py` → `arb_stem_mss/vendor/dicose/utils.py`
- `LICENSE` → `arb_stem_mss/vendor/dicose/LICENSE`

DiCoSe is distributed under the MIT License, copyright (c) 2024 Russell Izadi.
Its license text is retained in the vendored directory. DiCoSe acknowledges
Audio Diffusion PyTorch among its own upstream sources. File hashes and source
URLs are recorded in `UPSTREAM.json`.

The vendored package initializer is intentionally minimal. It exports `UNet1d`
without importing the original Lightning, W&B, diffusion, or optional model
wrappers. Neither copied `modules.py` nor copied `utils.py` is modified.

The new `TextConditionedDET` wrapper supplies a semantic condition through
DiCoSe's existing global `features`/scale-shift pathway, optionally after a
learned MLP. It handles padding/cropping and separates the predicted waveform
from the intermediate feature tuple for future refiner integration. These are
project adaptations, not features claimed to have existed in the original
four-class training pipeline.

MGE-LDM (`Lonely-house/MGE-LDM`, commit
`e180d9a16e31a78afc6b40ca40cde60eae96382a`) supplies the mixture/source/submixture
data formulation used in the design. No MGE-LDM code, codec, checkpoint, or
three-track diffusion model is copied into this DET implementation.

Real text encoding uses the optional `laion-clap` package and a separately
obtained checkpoint. The installed package version, checkpoint SHA256,
architecture, normalization, and text-encoder identity are recorded in the
embedding cache. Its data/model license and dependencies remain those of the
respective upstream providers. Model weights and audio datasets are not included
in this repository.
