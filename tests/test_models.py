from __future__ import annotations

import pytest

from genescope.models import registry


def test_nucleotide_transformer_registry_metadata() -> None:
    specs = {spec.key: spec for spec in registry.available_models()}
    spec = specs["nucleotide-transformer"]

    assert spec.model_id == "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    assert spec.adapter == "nucleotide-transformer-v2"
    assert "masked-LM" in spec.notes


def test_get_model_routes_to_dedicated_adapter(monkeypatch) -> None:
    captured = {}

    class FakeAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(registry, "NucleotideTransformerSequenceModel", FakeAdapter)

    model = registry.get_model(
        "nucleotide-transformer",
        batch_size=2,
        max_length=128,
        revision="abc123",
    )

    assert isinstance(model, FakeAdapter)
    assert captured == {
        "name": "nucleotide-transformer",
        "model_id": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "batch_size": 2,
        "max_length": 128,
        "revision": "abc123",
    }


def test_get_model_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        registry.get_model("does-not-exist")
