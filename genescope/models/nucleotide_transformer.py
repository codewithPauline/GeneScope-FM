"""Validated adapter for the Nucleotide Transformer v2 family."""

from __future__ import annotations

import numpy as np

from .base import SequenceModel


class NucleotideTransformerSequenceModel(SequenceModel):
    """Sequence embedding adapter for InstaDeep Nucleotide Transformer v2.

    The adapter follows the checkpoint model card: it loads the masked-language-model
    architecture with ``trust_remote_code=True``, requests hidden states, constructs the
    attention mask from the tokenizer pad token, and mean-pools the final hidden layer.

    ``max_length`` is measured in tokenizer tokens, not nucleotide bases. The v2 50M
    multi-species checkpoint uses a 6-mer tokenizer when possible.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        batch_size: int = 8,
        max_length: int | None = None,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - only when AI extra is absent
            raise ImportError(
                "Nucleotide Transformer inference requires the optional AI dependencies. "
                "Install them with: pip install -e '.[ai]'"
            ) from exc

        self.name = name
        self.model_id = model_id
        self.batch_size = batch_size
        self.revision = revision
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        load_kwargs = {"trust_remote_code": True}
        if revision is not None:
            load_kwargs["revision"] = revision

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()

        tokenizer_limit = int(self.tokenizer.model_max_length)
        self.max_length = (
            tokenizer_limit if max_length is None else min(max_length, tokenizer_limit)
        )
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2 tokenizer tokens.")

    def embed(self, sequences: list[str]) -> np.ndarray:
        """Return one mean-pooled final-layer embedding per sequence."""
        if not sequences:
            raise ValueError("At least one sequence is required for embedding.")

        outputs: list[np.ndarray] = []
        torch = self._torch

        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            encoded = self.tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

            with torch.inference_mode():
                model_output = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    encoder_attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden = model_output.hidden_states[-1]
                expanded_mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * expanded_mask).sum(dim=1)
                counts = expanded_mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts

            outputs.append(pooled.detach().cpu().numpy())

        return np.vstack(outputs)

    def provenance(self) -> dict[str, object]:
        """Return model settings useful for reproducibility reports."""
        return {
            "backend": "nucleotide-transformer-v2",
            "model_id": self.model_id,
            "revision": self.revision,
            "max_length_tokens": self.max_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "pooling": "masked-mean-final-hidden-state",
            "trust_remote_code": True,
        }
