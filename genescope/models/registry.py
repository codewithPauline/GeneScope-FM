"""Model registry for GeneScope."""

from __future__ import annotations

from dataclasses import dataclass

from .hf import HuggingFaceSequenceModel
from .nucleotide_transformer import MODEL_ID, REVISION, NucleotideTransformerSequenceModel


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_id: str
    notes: str
    adapter: str = "huggingface"
    revision: str | None = None


_MODELS: dict[str, ModelSpec] = {
    "nucleotide-transformer": ModelSpec(
        key="nucleotide-transformer",
        display_name="Nucleotide Transformer v2 50M multi-species",
        model_id=MODEL_ID,
        notes="Pinned 50M masked-LM checkpoint; custom code requires explicit opt-in.",
        revision=REVISION,
        adapter="nucleotide-transformer-v2",
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

    if kwargs.get("revision") is None:
        kwargs["revision"] = spec.revision
    if spec.adapter == "nucleotide-transformer-v2":
        return NucleotideTransformerSequenceModel(name=spec.key, model_id=spec.model_id, **kwargs)
    return HuggingFaceSequenceModel(name=spec.key, model_id=spec.model_id, **kwargs)
