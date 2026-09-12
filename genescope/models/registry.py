"""Model registry for GeneScope."""

from __future__ import annotations

from dataclasses import dataclass

from .hf import HuggingFaceSequenceModel


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_id: str
    notes: str


_MODELS: dict[str, ModelSpec] = {
    "nucleotide-transformer": ModelSpec(
        key="nucleotide-transformer",
        display_name="Nucleotide Transformer",
        model_id="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        notes="50M multi-species Nucleotide Transformer checkpoint.",
    ),
}


def available_models() -> list[ModelSpec]:
    """Return registered model specifications."""
    return list(_MODELS.values())


def get_model(name: str, **kwargs):
    """Instantiate a registered GeneScope model backend."""
    try:
        spec = _MODELS[name]
    except KeyError as exc:
        options = ", ".join(sorted(_MODELS))
        raise ValueError(f"Unknown model '{name}'. Available models: {options}") from exc

    return HuggingFaceSequenceModel(
        name=spec.key,
        model_id=spec.model_id,
        **kwargs,
    )
