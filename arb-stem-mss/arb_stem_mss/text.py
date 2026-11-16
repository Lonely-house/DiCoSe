"""Frozen CLAP text embeddings, cache provenance, and explicit test fixtures."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def load_embedding_cache(path, allow_mock=False, expected_dim=None, expected_encoder_id=None):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    required = {"encoder_id", "query_dim", "vectors", "is_mock"}
    if not isinstance(cache, dict) or not required.issubset(cache):
        raise ValueError("Text cache requires encoder_id/query_dim/vectors/is_mock")
    if not isinstance(cache["encoder_id"], str) or not cache["encoder_id"].strip():
        raise ValueError("Text cache must name its exact encoder identity")
    if not isinstance(cache["is_mock"], bool):
        raise ValueError("Text cache is_mock must be boolean")
    if cache["is_mock"] and not allow_mock:
        raise ValueError("Mock vectors have no language semantics; explicitly allow them only for smoke tests")
    dimension = cache["query_dim"]
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Text cache query_dim must be a positive integer")
    if expected_dim is not None and dimension != expected_dim:
        raise ValueError("Text cache dimension differs from the model")
    if expected_encoder_id is not None and cache["encoder_id"] != expected_encoder_id:
        raise ValueError("Text cache encoder identity differs from the checkpoint")
    if not isinstance(cache["vectors"], dict) or not cache["vectors"]:
        raise ValueError("Text cache must contain at least one prompt vector")
    for prompt, vector in cache["vectors"].items():
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Cache prompts must be nonempty strings")
        if (not isinstance(vector, torch.Tensor) or vector.shape != (dimension,)
                or not vector.is_floating_point() or not torch.isfinite(vector).all()
                or vector.float().norm() <= 1e-12):
            raise ValueError(f"Invalid cached text vector for {prompt!r}")
        cache["vectors"][prompt] = F.normalize(vector.float(), dim=0)
    return cache


def collect_prompts(manifest_paths):
    prompts = set()
    manifest_hashes = {}
    for manifest_path in manifest_paths:
        path = Path(manifest_path).resolve(strict=True)
        data = path.read_bytes()
        manifest_hashes[str(path)] = hashlib.sha256(data).hexdigest()
        for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{number}: missing nonempty prompt")
            prompts.add(prompt)
    if not prompts:
        raise ValueError("No prompts found in supplied manifests")
    return sorted(prompts), manifest_hashes


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_texts(prompts, checkpoint_path, device="cpu", batch_size=32, audio_model="HTSAT-base"):
    """Real LAION CLAP text branch only; never consumes clean target audio."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("Prompts must be nonempty strings")
    if not prompts:
        raise ValueError("At least one text prompt is required")
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    try:
        import laion_clap
    except ImportError as error:
        raise RuntimeError("Real text encoding requires optional laion-clap in a compatible CLAP environment") from error
    clap_version = version("laion-clap")
    encoder = laion_clap.CLAP_Module(enable_fusion=False, amodel=audio_model,
                                    tmodel="roberta", device=device)
    encoder.load_ckpt(str(checkpoint), verbose=False)
    encoder.eval()
    encoder.requires_grad_(False)
    vectors = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            subset = prompts[start:start + batch_size]
            values = encoder.get_text_embedding(subset, use_tensor=True)
            values = torch.as_tensor(values).detach().float().cpu()
            if values.ndim != 2 or values.shape[0] != len(subset):
                raise ValueError("CLAP returned invalid text embedding shape")
            if not torch.isfinite(values).all() or (values.norm(dim=-1) <= 1e-12).any():
                raise ValueError("CLAP returned a zero or nonfinite text vector")
            values = F.normalize(values, dim=-1)
            vectors.update(zip(subset, values.unbind(0)))
    dimension = next(iter(vectors.values())).numel()
    checkpoint_hash = _file_hash(checkpoint)
    identity = (f"laion-clap:{clap_version}:{audio_model}:roberta:no-fusion:"
                f"sha256={checkpoint_hash}:l2")
    return {"encoder_id": identity, "query_dim": dimension, "vectors": vectors,
            "is_mock": False, "normalization": "l2",
            "metadata": {"package": "laion-clap", "package_version": clap_version,
                         "audio_model": audio_model, "text_model": "roberta",
                         "checkpoint_sha256": checkpoint_hash,
                         "tokenizer_max_length": 77, "encoder_frozen": True}}


def mock_cache(prompts, query_dim=512, seed=0):
    """Explicitly labelled deterministic random fixture, NOT a text encoder."""
    if query_dim < 1:
        raise ValueError("query_dim must be positive")
    vectors = {}
    for prompt in sorted(set(prompts)):
        digest = hashlib.sha256((str(seed) + "\0" + prompt).encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
        vectors[prompt] = F.normalize(torch.randn(query_dim, generator=generator), dim=0)
    return {"encoder_id": f"MOCK-NO-SEMANTICS:sha256-rng:{query_dim}:seed={seed}:l2",
            "query_dim": query_dim, "vectors": vectors, "is_mock": True,
            "normalization": "l2"}
