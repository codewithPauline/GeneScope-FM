"""Foundation-model backends exposed by GeneScope."""

from .registry import ModelSpec, available_models, get_model

__all__ = ["ModelSpec", "available_models", "get_model"]
