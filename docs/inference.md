# Nucleotide Transformer inference

GeneScope has a dedicated adapter for
[`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species).
The adapter loads the checkpoint's custom masked-language-model architecture from
revision `81b29e5786726d891dbf929404ef20adca5b36f1`, requests the final hidden layer,
and produces one 512-dimensional vector per sequence.

## Run it

For CPU inference, from the repository root:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ai]"
genescope embed examples/sequences.fasta \
  --model nucleotide-transformer --allow-remote-code \
  --device cpu --batch-size 2 --max-length 128 --output nt_embeddings.csv
genescope explore nt_embeddings.csv --output nt_report
```

The first embedding run downloads the checkpoint. Later runs reuse the Hugging Face
cache; `--local-files-only` requires a complete existing cache. GPU users can install
the appropriate PyTorch build for their system and select `--device cuda`.
The compatibility environment uses Transformers 4.48.3 and PyTorch 2.6.0 on CPU.
CUDA execution has not been validated by the committed integration check.

`--allow-remote-code` explicitly enables the checkpoint's custom Python model code.
Review the [pinned source](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species/tree/81b29e5786726d891dbf929404ef20adca5b36f1)
before opting in. Both weights and custom model code use the same immutable revision.
`--revision` accepts a full 40-character commit SHA; an override is not automatically
covered by the validation evidence for the default pin. Weights load in safetensors
format. The Nucleotide Transformer weights have their own CC-BY-NC-SA-4.0 license,
as stated on the model card; GeneScope's MIT license covers GeneScope code.

## Tokens and length handling

The checkpoint uses 6-mer tokens when possible and individual bases otherwise.
GeneScope accepts A/C/G/T/N, strips whitespace, and uppercases input. Unknown
tokenizer outputs are rejected. The default and maximum adapter budget is **1,000
tokens including the leading CLS token**. It is not a fixed number of bases.
This cap follows the model card's training context, even though the tokenizer
configuration permits 2,048 tokens.

Over-length inputs raise an error before any forward pass. To deliberately retain
only the tokenized prefix, select:

```bash
genescope embed sequences.fasta --allow-remote-code \
  --max-length 128 --length-policy truncate --output prefix_embeddings.csv
```

Truncation emits a warning and records affected rows. No automatic windowing,
chunk averaging, or full-sequence representation of the omitted bases is implied.

## Pooling

The sequence vector is the arithmetic mean of final-layer hidden states for
attended, non-special tokens. Padding, CLS, and any other special-token IDs are
excluded. It is a mean over **tokens**, not a mean weighted by the number of bases
represented by each token.

The supported EsmTokenizer configuration has `eos_token=None`. In the compatibility
environment, its generated special-token mask can nevertheless contain an extra
EOS entry. GeneScope derives the mask from actual token IDs so it always aligns
with the model input. Tests reproduce this configuration and compare sequences
processed alone with the same sequences in a padded batch.

**Behavior change from the initial adapter:** CLS is no longer included in the
mean, and long inputs are no longer silently truncated. Previously generated
vectors should be regenerated before combining them with new exports.

## Provenance sidecar

Each CLI export writes `embeddings.csv.provenance.json` alongside the CSV. It records:

- A schema version, the exact CSV SHA-256, dimensions, and package versions.
- The generating model, requested/resolved revision, code revision, tokenizer,
  loader, model dtype, device, token budget, batch size, and pooling rule.
- The source FASTA SHA-256 and per-sequence original base count, original/retained
  token counts, pooled token count, and truncation flag.

K-mer CLI exports use the same format with `generation.kind = "kmer_baseline"` and
their feature settings. The explorer validates the CSV hash before attaching a
sidecar to its report. A missing sidecar is allowed for external embeddings; a
present but invalid or mismatched sidecar raises an error. Hash agreement checks
file consistency, not authenticity of self-reported model claims.

This replaces the initial unversioned sidecar with a versioned envelope containing
`generation`. Old sidecars lack a CSV hash; regenerate them from inference rather
than treating them as verified provenance. CSV-only exports remain supported.

## Reproduce the integration check

```bash
python examples/validate_checkpoint.py --allow-remote-code --output checkpoint_validation
```

This loads the actual pretrained checkpoint and uses four toy sequences, including
N bases, different lengths, and a single base. It checks finite 512-dimensional
embeddings, repeat inference, batch/padding consistency, strict length errors,
explicit truncation against the retained prefix, export, and HTML reporting.
The comparison tolerances are `rtol=1e-4` and `atol=1e-5`; bitwise equivalence across
hardware or dependency versions is not claimed.

The resulting directory contains the input FASTA, embeddings, provenance, an
exploration report, and `validation.json`. A committed result is kept in
[`validation/nucleotide-transformer.json`](validation/nucleotide-transformer.json).
These are software integration checks, not evidence of biological prediction accuracy.

Routine CI runs small local encoder/masked-LM tests without downloading pretrained
weights. To run the real-checkpoint workflow on GitHub, select **Actions → Foundation
model smoke test → Run workflow**. Its results are saved as a workflow artifact.

The next scientific milestone is evaluation on a labeled biological dataset with
sequence-aware train/test splits and the k-mer baseline as a comparator.
