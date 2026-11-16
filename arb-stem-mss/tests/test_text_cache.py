import pytest
import torch

from arb_stem_mss.text import load_embedding_cache


def test_mock_cache_requires_explicit_permission(query_fixture):
    with pytest.raises(ValueError, match="[Mm]ock"):
        load_embedding_cache(query_fixture["cache"])
    accepted = load_embedding_cache(query_fixture["cache"], allow_mock=True,
                                    expected_dim=8, expected_encoder_id="synthetic-test-fixture")
    assert accepted["is_mock"]


def test_wrong_encoder_or_dimension_is_rejected(query_fixture):
    for expected in ({"expected_dim": 512}, {"expected_encoder_id": "different-checkpoint"}):
        with pytest.raises(ValueError):
            load_embedding_cache(query_fixture["cache"], allow_mock=True, **expected)
