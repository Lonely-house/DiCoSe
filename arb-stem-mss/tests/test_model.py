import json
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from arb_stem_mss.model import TextConditionedDET

ROOT = Path(__file__).resolve().parents[1]


def tiny_model():
    config = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())["model"]
    return TextConditionedDET(config), config


def test_real_dicose_unet_padding_and_query_gradients():
    model, config = tiny_model()
    assert model.__class__.__module__ == "arb_stem_mss.model"
    assert any(module.__class__.__name__ == "UNet1d" for module in model.modules())
    assert model.required_multiple > 1
    mixture = torch.randn(2, 1, model.required_multiple * 2 + 3) * 0.03
    query = torch.randn(2, config["query_dim"], requires_grad=True)
    prediction = model(mixture, query)
    assert prediction.shape == mixture.shape
    prediction.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all() and query.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        same_mix = mixture[:1].repeat(2, 1, 1)
        outputs = model(same_mix, torch.randn(2, config["query_dim"]))
    assert not torch.allclose(outputs[0], outputs[1])


def test_wrong_query_dimension_rejected():
    model, config = tiny_model()
    with pytest.raises((ValueError, RuntimeError)):
        model(torch.zeros(1, 1, 101), torch.ones(1, config["query_dim"] + 1))


def test_vendored_sources_match_pinned_provenance():
    provenance = json.loads((ROOT / "UPSTREAM.json").read_text())["dicose"]
    assert provenance["commit"] == "a1dc0a41ad2b1829674a60ea74e74edfd1509083"
    for filename, metadata in provenance["files"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == metadata["sha256"]
        assert metadata["modified"] is False
