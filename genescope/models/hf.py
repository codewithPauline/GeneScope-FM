"""Hugging Face model adapter for genomic foundation models."""

from __future__ import annotations

import numpy as np

from .base import SequenceModel


class HuggingFaceSequenceModel(SequenceModel):
    """Generic Hugging Face encoder with masked mean pooling.

    This adapter intentionally keeps model-specific logic thin. Models with unusual
    tokenization, trust-remote-code requirements, or custom pooling should receive
    dedicated adapters rather than being forced through this implementation.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        batch_size: int = 8,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without AI extra
            raise ImportError(
                "Foundation-model inference requires the optional AI dependencies. "
                "Install them with: pip install -e '.[ai]'"
            ) from exc

        self.name = name
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            raise ValueError("At least one sequence is required for embedding.")

        outputs: list[np.ndarray] = []
        torch = self._torch

        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                model_output = self.model(**encoded)
                hidden = model_output.last_hidden_state
                mask = encoded.get("attention_mask")

                if mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    expanded_mask = mask.unsqueeze(-1).expand(hidden.size()).float()
                    summed = (hidden * expanded_mask).sum(dim=1)
                    counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
                    pooled = summed / counts

            outputs.append(pooled.detach().cpu().numpy())

        return np.vstack(outputs)
