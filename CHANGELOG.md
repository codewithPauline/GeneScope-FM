# Changelog

## 0.2.1

- Add a dedicated Nucleotide Transformer v2 adapter instead of routing the checkpoint
  through the generic Hugging Face encoder.
- Follow the checkpoint's masked-language-model loading and final-hidden-state mean
  pooling behavior with explicit attention masking.
- Make the CLI default to 1,000 tokenizer tokens for the v2 50M checkpoint and clarify
  that this limit is measured in tokens rather than nucleotide bases.
- Add optional model revision pinning and write a provenance JSON sidecar for embedding
  runs.
- Add registry tests and a manual GitHub Actions smoke workflow for real checkpoint
  inference without forcing large model downloads into routine CI.
- Pin checkpoint and custom code to an immutable revision; require explicit custom-code opt-in.
- Exclude special tokens from pooling and handle the null-EOS tokenizer mask edge case.
- Reject over-length input before inference; allow warned, recorded prefix truncation explicitly.
- Link versioned provenance to CSV content hashes and carry it into exploration reports.
- Test actual local encoders and masked-LM inference in routine CI; provide a reproducible
  real-checkpoint validation script and record its results.

## 0.2.0

- Add model-independent embedding exploration: PCA, cosine similarity, nearest
  neighbors, and numerical representation diagnostics.
- Add optional seeded UMAP with explicit neighborhood settings.
- Export a portable HTML report, SVG plot, analysis tables, and input-hashed
  reproducibility summary.
- Validate embedding CSV schemas and preserve string IDs such as `001` and `NA`.
- Add an offline k-mer baseline and reproducible synthetic demonstration.
- Test numerical results, metadata alignment, malformed inputs, CLI execution,
  report escaping, output protection, and optional UMAP reproducibility.

## 0.1.0

- Initial FASTA reader, embedding export, model registry, generic Hugging Face
  adapter, CLI, MIT license, and Python CI.
