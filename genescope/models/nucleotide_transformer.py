"""Pinned Nucleotide Transformer v2 50M adapter."""

import re

from .hf import HuggingFaceSequenceModel

MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"


class NucleotideTransformerSequenceModel(HuggingFaceSequenceModel):
    """Use the checkpoint's custom masked-LM architecture, never a generic ESM encoder.

    The 1,000-token cap follows the documented training context rather than the
    tokenizer's larger 2,048-token setting. Includes the leading CLS token.
    """

    def __init__(
        self,
        name: str = "nucleotide-transformer",
        model_id: str = MODEL_ID,
        *,
        revision: str = REVISION,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        if not trust_remote_code:
            raise ValueError(
                "Nucleotide Transformer requires its custom model code. "
                "Review the pinned checkpoint and explicitly pass "
                "--allow-remote-code (Python: trust_remote_code=True)."
            )
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a full immutable 40-character checkpoint SHA.")
        if model_id != MODEL_ID:
            raise ValueError("This adapter is validated only for the registered v2 50M checkpoint.")
        kwargs.setdefault("max_length", 1000)
        super().__init__(
            name=name,
            model_id=model_id,
            revision=revision,
            trust_remote_code=True,
            masked_lm=True,
            context_limit=1000,
            **kwargs,
        )

    def provenance(self) -> dict:
        return {
            **super().provenance(),
            "backend": "nucleotide-transformer-v2",
            "model_license": "CC-BY-NC-SA-4.0",
            "adapter_context_limit_tokens": 1000,
        }
