"""Base interfaces for GeneScope model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SequenceModel(ABC):
    """Abstract interface implemented by every GeneScope sequence model."""

    name: str

    @abstractmethod
    def embed(self, sequences: list[str]) -> np.ndarray:
        """Return one embedding vector per input sequence."""
        raise NotImplementedError
