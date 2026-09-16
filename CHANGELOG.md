# Changelog

## 0.3.1

- Add `genescope split-groups` to apply prespecified chromosome, homology, or locus
  assignments, retaining every row and rejecting cross-split sequence overlap.
- Export content-linked split provenance, sequence FASTA, group counts, and class
  counts. Group annotations remain externally supplied, not independently verified.
- Use paired whole-group bootstrap intervals whenever groups are supplied; reject
  insufficient independent class support and report discarded single-class draws.
- Export per-group test diagnostics; leave undefined single-class metrics blank.
- Add an offline synthetic grouped demo, CI coverage, and a grouped-evaluation guide.
- Benchmark results use schema version 2. Ungrouped point estimates and bootstrap
  calculations are unchanged; the v0.3 promoter pilot remains the measured result.

## 0.3.0

- Add `genescope benchmark` for frozen embeddings, 3-mers, and GC/length controls.
- Fit scaling and logistic regression on training rows only; select regularization
  on validation AUROC; export held-out metrics and paired bootstrap intervals.
- Reject exact/reverse-complement duplicates, shared exact 50-base windows across
  splits, and overlapping optional groups. Document remaining homology limitations.
- Add exact input-sequence hashes to new embedding sidecars and require them when
  matching benchmark sequences to vectors.
- Add a checksum-pinned public promoter data preparer and a real NT v2 50M pilot
  with portable results, predictions, source manifests, and interpretation limits.
- Add offline regression tests and a dedicated optional-dependency CI job.

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
