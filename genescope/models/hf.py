"""Hugging Face inference with explicit sequence handling and pooling semantics."""

from __future__ import annotations

import warnings
from importlib.metadata import version

import numpy as np

from ..explore import validate_matrix
from ..io import normalize_sequence
from .base import SequenceModel


def mean_pool(hidden, attention_mask, special_tokens_mask):
    """Average attended sequence tokens, excluding padding and special tokens."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden states and attention mask have incompatible shapes.")
    if special_tokens_mask.shape != attention_mask.shape:
        raise ValueError("Special-token and attention masks must have the same shape.")
    mask = attention_mask.bool() & ~special_tokens_mask.bool()
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every sequence must retain at least one non-special token.")
    # Float32 accumulation also supports inference with lower-precision model weights.
    return (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


class HuggingFaceSequenceModel(SequenceModel):
    """Encoder adapter; masked-LM checkpoints select their final hidden layer.

    No truncation occurs unless length_policy='truncate'. Remote model code is
    disabled unless the caller explicitly enables it. Dedicated registered
    adapters provide checkpoint-specific loader choices and context limits.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        batch_size: int = 8,
        max_length: int = 512,
        device: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        length_policy: str = "error",
        masked_lm: bool = False,
        context_limit: int | None = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 2:
            raise ValueError("max_length must be an integer of at least 2 tokens.")
        if length_policy not in {"error", "truncate"}:
            raise ValueError("length_policy must be error or truncate.")
        if context_limit is not None and max_length > context_limit:
            raise ValueError(f"max_length exceeds this adapter's {context_limit}-token limit.")
        try:
            import torch
            from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Inference requires: pip install -e '.[ai]'") from exc

        self.name, self.model_id, self.revision = name, str(model_id), revision
        self.batch_size, self.max_length = batch_size, max_length
        self.length_policy, self.masked_lm = length_policy, masked_lm
        self.trust_remote_code = trust_remote_code
        self.sequence_stats: list[dict] = []
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        loading = dict(
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if trust_remote_code:
            loading["code_revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **loading)
        tokenizer_limit = self.tokenizer.model_max_length
        if max_length > tokenizer_limit:
            raise ValueError(f"max_length exceeds the tokenizer limit of {tokenizer_limit} tokens.")
        factory = AutoModelForMaskedLM if masked_lm else AutoModel
        self.model = factory.from_pretrained(model_id, use_safetensors=True, **loading)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, sequences: list[str]) -> np.ndarray:
        self.sequence_stats = []
        if not sequences:
            raise ValueError("At least one sequence is required for embedding.")
        normalized = [normalize_sequence(sequence) for sequence in sequences]
        # Inspect all rows before inference, so an over-length row cannot silently
        # produce a partly processed dataset. These are token counts, not base counts.
        full = self.tokenizer(normalized, truncation=False, padding=False)
        counts = [len(ids) for ids in full["input_ids"]]
        long_rows = [i for i, count in enumerate(counts) if count > self.max_length]
        if long_rows and self.length_policy == "error":
            raise ValueError(
                f"{len(long_rows)} sequence(s) exceed max_length={self.max_length} tokens; "
                f"first is row {long_rows[0] + 1} ({counts[long_rows[0]]} tokens). "
                "Increase --max-length within the model limit, or explicitly use "
                "--length-policy truncate. No embeddings were generated."
            )
        if long_rows:
            warnings.warn(
                f"Truncating {len(long_rows)} sequence(s) to {self.max_length} tokens; "
                "affected rows are recorded in provenance.",
                UserWarning,
                stacklevel=2,
            )
        unk = self.tokenizer.unk_token_id
        if unk is not None and any(unk in ids for ids in full["input_ids"]):
            raise ValueError("The tokenizer produced unknown tokens for the supplied DNA.")

        outputs, stats = [], []
        torch = self._torch
        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=self.length_policy == "truncate",
                return_attention_mask=True,
                **({"max_length": self.max_length} if self.length_policy == "truncate" else {}),
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            # EsmTokenizer can report an extra EOS mask entry when eos_token=None.
            # Derive the mask from actual IDs, preserving exact tensor alignment.
            special_ids = torch.tensor(self.tokenizer.all_special_ids, device=self.device)
            special = torch.isin(encoded["input_ids"], special_ids)
            attention = encoded["attention_mask"]
            with torch.inference_mode():
                result = self.model(
                    **encoded, output_hidden_states=self.masked_lm, return_dict=True
                )
                hidden = result.hidden_states[-1] if self.masked_lm else result.last_hidden_state
                pooled = mean_pool(hidden, attention, special)
            outputs.append(pooled.detach().cpu().numpy())
            retained = attention.sum(dim=1).tolist()
            biological = (attention.bool() & ~special.bool()).sum(dim=1).tolist()
            for offset, sequence in enumerate(batch):
                row = start + offset
                stats.append(
                    {
                        "row": row + 1,
                        "original_bases": len(sequence),
                        "original_tokens": counts[row],
                        "retained_tokens": retained[offset],
                        "pooled_tokens": biological[offset],
                        "truncated": counts[row] > retained[offset],
                    }
                )
        matrix = validate_matrix(np.vstack(outputs))
        self.sequence_stats = stats
        return matrix

    def provenance(self) -> dict:
        """Describe this adapter and its most recent successful embedding operation."""
        return {
            "kind": "foundation_model",
            "backend": self.name,
            "model_id": self.model_id,
            "requested_revision": self.revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_class": type(self.model).__name__,
            "loader": "AutoModelForMaskedLM" if self.masked_lm else "AutoModel",
            "remote_code_enabled": self.trust_remote_code,
            "code_revision": self.revision if self.trust_remote_code else None,
            "weights_format": "safetensors",
            "device": str(self.device),
            "model_dtype": str(next(self.model.parameters()).dtype),
            "batch_size": self.batch_size,
            "max_length_tokens_including_special": self.max_length,
            "length_policy": self.length_policy,
            "pooling": "mean of final-layer attended non-special tokens",
            "sequence_normalization": "remove whitespace; uppercase; validate A/C/G/T/N",
            "software": {
                name: version(name)
                for name in [
                    "torch",
                    "transformers",
                    "tokenizers",
                    "huggingface-hub",
                    "safetensors",
                ]
            },
            "sequence_stats": self.sequence_stats,
        }
