"""Exercise actual torch/transformers inference with small locally saved models."""

from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import EsmConfig, EsmForMaskedLM, EsmModel, EsmTokenizer  # noqa: E402

from genescope.models.hf import HuggingFaceSequenceModel, mean_pool  # noqa: E402


@pytest.fixture
def model_dir(tmp_path):
    vocabulary = ["<unk>", "<pad>", "<mask>", "<cls>", "<eos>", "A", "C", "G", "T", "N"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(vocabulary) + "\n")
    # Reproduce the real checkpoint's null-EOS EsmTokenizer behavior.
    tokenizer = EsmTokenizer(vocab_file=str(vocab), eos_token=None, model_max_length=32)
    tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(42)
    config = EsmConfig(
        vocab_size=10,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
        pad_token_id=1,
        mask_token_id=2,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        token_dropout=False,
    )
    EsmModel(config).save_pretrained(tmp_path)
    return tmp_path


def test_pooling_excludes_padding_and_special_tokens():
    hidden = torch.tensor([[[100.0, 200.0], [2.0, 4.0], [4.0, 8.0], [999.0, 999.0]]])
    pooled = mean_pool(hidden, torch.tensor([[1, 1, 1, 0]]), torch.tensor([[1, 0, 0, 1]]))
    torch.testing.assert_close(pooled, torch.tensor([[3.0, 6.0]]))
    with pytest.raises(ValueError, match="at least one"):
        mean_pool(hidden, torch.ones((1, 4)), torch.ones((1, 4)))
    with pytest.raises(ValueError, match="same shape"):
        mean_pool(hidden, torch.ones((1, 4)), torch.ones((1, 5)))


def test_actual_inference_padding_batching_and_null_eos(model_dir):
    backend = HuggingFaceSequenceModel(
        "tiny", str(model_dir), max_length=16, device="cpu", batch_size=2, local_files_only=True
    )
    sequences = ["ACG", "ACGTNNNNACGT"]
    batched = backend.embed(sequences)
    stats = backend.sequence_stats
    single = np.vstack([backend.embed([sequence]) for sequence in sequences])
    np.testing.assert_allclose(batched, single, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(backend.embed(["ac g\n"]), single[:1], rtol=1e-5, atol=1e-6)
    assert [s["pooled_tokens"] for s in stats] == [3, 12]
    assert not any(s["truncated"] for s in stats)
    assert batched.shape == (2, 8)
    assert backend.provenance()["weights_format"] == "safetensors"


def test_length_error_happens_before_any_forward_pass(model_dir):
    backend = HuggingFaceSequenceModel(
        "tiny", str(model_dir), max_length=5, batch_size=1, local_files_only=True
    )
    with patch.object(backend.model, "forward", side_effect=AssertionError("Forward must not run")):
        with pytest.raises(ValueError, match="first is row 2"):
            backend.embed(["AC", "A" * 8])
    assert backend.sequence_stats == []


def test_explicit_truncation_matches_the_retained_prefix(model_dir):
    backend = HuggingFaceSequenceModel(
        "tiny", str(model_dir), max_length=5, length_policy="truncate", local_files_only=True
    )
    with pytest.warns(UserWarning, match="Truncating 1"):
        truncated = backend.embed(["A" * 8])
    stats = backend.sequence_stats[0]
    assert (stats["original_tokens"], stats["retained_tokens"], stats["pooled_tokens"]) == (9, 5, 4)
    assert stats["truncated"] is True
    np.testing.assert_allclose(truncated, backend.embed(["AAAA"]))


def test_masked_lm_uses_final_hidden_layer(model_dir):
    config = EsmConfig.from_pretrained(model_dir)
    EsmForMaskedLM(config).save_pretrained(model_dir)
    backend = HuggingFaceSequenceModel(
        "tiny-lm", str(model_dir), max_length=16, masked_lm=True, local_files_only=True
    )
    result = backend.embed(["ACG"])
    inputs = backend.tokenizer(["ACG"], return_tensors="pt")
    with torch.inference_mode():
        hidden = backend.model(**inputs, output_hidden_states=True).hidden_states[-1]
    # CLS is position 0; the remaining positions correspond to DNA letters.
    np.testing.assert_allclose(result, hidden[:, 1:, :].mean(dim=1).numpy(), atol=1e-6)
    assert result.shape[1] == config.hidden_size  # Not vocabulary-sized masked-LM logits.


@pytest.mark.parametrize(
    "kwargs", [{"batch_size": 0}, {"max_length": 1}, {"length_policy": "silent"}]
)
def test_bad_inference_options_rejected_before_loading(kwargs):
    with pytest.raises(ValueError):
        HuggingFaceSequenceModel("tiny", "nonexistent-model", **kwargs)
