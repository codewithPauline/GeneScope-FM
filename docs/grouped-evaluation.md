# Evaluation with biological groups

GeneScope v0.3.1 can prepare prespecified chromosome, homology-cluster, or locus
holdouts and estimate uncertainty by resampling whole test groups. This is a
workflow capability, not a new biological benchmark result. The existing promoter
pilot has no chromosome metadata and retains its original sequence-level analysis.

## Prepare the inputs before fitting models

Supply `annotated_sequences.csv` with exactly four columns:

| Column | Meaning |
| --- | --- |
| `sequence_id` | Unique FASTA-safe ID; no whitespace or `>` |
| `sequence` | DNA using A/C/G/T/N; case and sequence whitespace are normalized |
| `label` | Binary 0 or 1; 1 is positive |
| `group` | Externally defined chromosome, homology cluster, locus, or other unit |

Supply `group_assignments.csv` with exactly two columns, `group` and `split`.
Every dataset group must appear exactly once, assigned to `train`, `validation`,
or `test`. Extra or missing groups are errors. String IDs such as `001` and `NA`
are preserved. Group names are case-sensitive and cannot have surrounding spaces;
normalize chromosome aliases such as `chr1` and `1` yourself.

```bash
pip install -e ".[benchmark]"
genescope split-groups annotated_sequences.csv group_assignments.csv \
  --group-kind chromosome \
  --group-source "Your assembly accession and annotation release; documented locus mapping" \
  --output grouped_dataset
```

Use `--group-kind homology` for externally computed clusters, with the clustering
program/version, identity and coverage thresholds, strand handling, and source
accession in `--group-source`. `locus` and `other` are also supported. GeneScope
records these declarations but does not verify their biological correctness.

Preparation retains every row and applies the assignments without randomized
splitting, class balancing, or performance-based selection. Rows are sorted by ID,
so input row ordering does not change the prepared data. It rejects exact and
reverse-complement duplicate sequences and cross-split exact 50-base matches,
even when the supplied group names differ. This screen does not establish
approximate homology isolation or absence of genomic interval overlap.

Each split needs at least two sequences per class. Group uncertainty requires at
least two distinct **test groups containing each class**; these can be mixed-class
groups. Do not subdivide genuine groups just to satisfy that minimum. A single
held-out chromosome does not provide enough independent units for this interval.

The output directory contains:

| File | Purpose |
| --- | --- |
| `dataset.csv` | Validated sequence/label/split/group table |
| `sequences.fasta` | Exact sequences for embedding generation |
| `split_provenance.json` | Source and assignment hashes, biological-group declaration, complete group map, split counts, and overlap audit |

A new directory is required; existing results are protected. The manifest links
the prepared dataset by SHA-256. Editing the dataset invalidates that link, so
prepare it again after changes. Hashes detect stale files, not fabricated metadata.

## Generate representations and evaluate

Use the [pinned inference workflow](inference.md):

```bash
# Install the AI extra and CPU PyTorch as described in the inference guide.
genescope embed grouped_dataset/sequences.fasta \
  --model nucleotide-transformer --allow-remote-code --device cpu \
  --output grouped_dataset/nt_embeddings.csv

genescope benchmark grouped_dataset/dataset.csv grouped_dataset/nt_embeddings.csv \
  --output grouped_dataset/evaluation \
  --description "Describe the biological dataset, group definitions, and holdout design."
```

The evaluator retains training-only scaling and fitting, validation-only selection
of regularization, and a fixed 0.5 decision threshold. It verifies any adjacent
split manifest and copies it into the report. It checks embedding DNA hashes
against the dataset as before. Supplying a `group` column always activates group
uncertainty, including for manually prepared datasets without a split manifest.
No sequence-bootstrap fallback is silently used when there are too few groups.

### What the uncertainty interval means

For each of 1,000 draws, sample the same number of test groups as observed, with
replacement. Include **all sequences** from each selected group, repeating them
when that group is sampled multiple times. Use the same draw for every
representation and for their paired differences. Report percentile 95% intervals.

A draw containing only one class cannot define AUROC; it is discarded. The report
records the number discarded. Sampling stops with an error if 20,000 attempts
cannot produce 1,000 valid draws. This procedure conditions on draws containing
both classes and does not hold class counts fixed. Independent test groups are
assumed; dependence between distinct supplied groups remains a limitation.

Point estimates still pool test sequences: large groups contribute more rows.
These are not equal-weight averages of per-group scores. Confidence intervals
condition on the fitted classifiers and do not include training or split-selection
variation, model pretraining, or group-annotation uncertainty. A warning is shown
when fewer than ten groups support either class; ten is a caution threshold, not
a guarantee of reliable inference.

### Inspect individual groups

Grouped reports export `group_metrics.csv` with group size, class counts, balanced
accuracy, AUROC, and Brier score for each representation. AUROC and balanced
accuracy are left blank for single-class groups, with an explicit status column;
Brier score is still available. `predictions.csv` retains each sequence's group.
These diagnostics help identify uneven performance but must not be used to
retroactively choose favorable test groups.

## Run a fully offline software demonstration

```bash
python examples/prepare_grouped_demo.py --output .cache/grouped-demo
genescope baseline .cache/grouped-demo/sequences.fasta \
  --output .cache/grouped-demo/kmer.csv
genescope benchmark .cache/grouped-demo/dataset.csv .cache/grouped-demo/kmer.csv \
  --output .cache/grouped-demo/report \
  --description "Synthetic software demonstration; arbitrary groups and k-mer features."
```

Open `.cache/grouped-demo/report/report.html`. The demonstration contains 96
synthetic 100-base sequences in 12 arbitrary groups, assigned 6/3/3 to
training/validation/test. The labels deliberately encode GC differences. It uses
3-mers as both the supplied representation and the built-in comparator, so their
paired differences should be zero. It downloads no data or model weights and
provides **no evidence of biological generalization or foundation-model quality**.

## Designing a defensible biological follow-up

- Recover coordinates and genome assembly from the original source. Join those
  annotations through an auditable sequence match, not row position or guessed IDs.
- Fix test groups before measuring performance. Preserve any established official
  holdout; replacing it creates a different benchmark that must be labeled as such.
- For approximate-homology holdouts, document the alignment/clustering method and
  thresholds. If chromosome and homology constraints are both required, form groups
  that satisfy both (for example, connected components of their relationships).
  Concatenating the two IDs does not ensure either form of separation.
- Resolve any cross-split overlaps by a documented design before model fitting;
  do not repeatedly alter assignments based on test scores.
- Prespecify multiple split maps to study sensitivity. Report all of them; overlapping
  test sets are not independent replicates and their intervals cannot simply be pooled.

The next measured milestone is a larger dataset with verified locus metadata or
documented homology clusters, evaluated using this workflow. Foundation-model
pretraining overlap remains a separate issue even with perfect downstream splits.
