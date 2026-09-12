# Exploring sequence representations

The exploration workflow accepts the same CSV schema produced by `genescope embed`.
It can also analyze conventional features produced by `genescope baseline`.
Those representations must be interpreted according to how they were generated.

## Run an offline example

From the repository root, after `pip install -e .`:

```bash
genescope baseline examples/synthetic/sequences.fasta --k 3 --output kmer_embeddings.csv
genescope explore kmer_embeddings.csv \
  --metadata examples/synthetic/labels.csv \
  --neighbors 3 --output my_report
```

Open `my_report/report.html` in a browser. The report works offline, with an embedded
SVG plot and point tooltips. The input has 24 synthetic 240-base sequences sampled
from three nucleotide-composition distributions (seed 42), not real genomic loci.
The k-mer baseline counts overlapping, strand-specific words; it is not a trained
model and does not merge reverse complements. Windows containing N are skipped,
and frequencies are normalized by the number of valid windows. Sequences with no
valid windows are rejected. `k` is limited to 1–6 to bound feature expansion.

Regenerate the input with `python examples/generate_demo.py`. The committed example
outputs in [`demo/`](demo/) were produced by these commands, using that input.
Visible separation reflects the deliberately different base compositions; it is
not evidence of foundation-model performance or biological discovery.

To rebuild the committed report and its README preview:

```bash
genescope baseline examples/synthetic/sequences.fasta --k 3 --output docs/demo/kmer_embeddings.csv
genescope explore docs/demo/kmer_embeddings.csv --metadata examples/synthetic/labels.csv \
  --neighbors 3 --output docs/demo --overwrite \
  --description "Synthetic DNA · strand-specific 3-mer frequencies · no foundation model used."
python examples/render_preview.py
```

`--description` adds your dataset and representation context to the report and summary.

## Input contract

```csv
sequence_id,sequence_length,embedding_0000,embedding_0001
001,240,0.15,0.45
NA,240,0.35,0.20
```

- `sequence_id`: required, unique, nonempty string. IDs like `001` and `NA` stay intact.
- `sequence_length`: optional positive integer.
- `embedding_0000`, `embedding_0001`, …: required finite numeric columns,
  consecutively numbered from zero. Column order is normalized when loading.
- Unexpected columns and duplicate headers are rejected. Put labels in a separate
  metadata CSV with `sequence_id,label`; rows are joined by ID, not row position.
  Every analyzed sequence needs a label when metadata is supplied; extra metadata
  records are ignored. Labels never influence fitting or neighbor rankings.

At least two sequences and one feature are required. Cosine analysis rejects
zero vectors because their direction is undefined. The dense report is limited
to 5,000 sequences; both similarity calculation and neighbor sorting grow
quadratically. Subsample larger datasets deliberately and retain the selected IDs.
This version is intended for exploratory datasets, not genome-scale search.

## Analysis choices

**PCA:** mean-centered, unscaled features; full singular-value decomposition. The
report returns up to two components, bounded by `min(n_sequences - 1, n_features)`.
A two-sequence or one-feature dataset has one component and a zero display-only
vertical axis. Explained variance uses the total variance across all components.
Constant input returns zero coordinates and zero variance ratios. Component signs
are fixed, although degenerate eigenspaces and different numerical libraries may
produce different orientations. Feature scaling is deliberately not automatic.

**Similarity and neighbors:** cosine similarity on the original input features,
not PCA/UMAP coordinates. A self match is removed by row index, while a different
sequence with an identical vector remains a legitimate match. Ties retain input
row order. The requested neighbor count is clipped to `n_sequences - 1` and both
requested and actual values are recorded. Cosine values are not probabilities,
alignment identities, or evolutionary distances.

**Diagnostics:** centered numerical rank, constant feature count, duplicate row
count, and zero-vector count. These can flag representation collapse or suspicious
exports; they do not measure predictive performance. Group separation alone may
reflect sequence length, GC content, repeats, shared ancestry, or data leakage.

## Optional UMAP

```bash
pip install -e ".[explore]"
genescope explore kmer_embeddings.csv \
  --metadata examples/synthetic/labels.csv \
  --method umap --umap-neighbors 8 --min-dist 0.1 --seed 42 \
  --output my_umap_report
```

UMAP needs at least four sequences with variation and
`2 <= --umap-neighbors < n_sequences`. It uses Euclidean distance, random
initialization, two components, and one worker with an explicit seed. Results
should repeat within the same software environment; a seed does not guarantee
identical results across dependency versions. PCA outputs are also included for
comparison. UMAP distances and apparent clusters require cautious interpretation.
See the official [UMAP parameter guide](https://umap-learn.readthedocs.io/en/latest/parameters.html)
and [reproducibility notes](https://umap-learn.readthedocs.io/en/latest/reproducibility.html).

## Exported files

| File | Contents |
| --- | --- |
| `report.html` | Portable report with tooltips, label legend, and first 100 neighbor rows |
| `projection.svg` | Standalone projection for embedding in other documents |
| `projection.csv` | Chosen projection coordinates with labels |
| `pca.csv` | PCA coordinates, also written for UMAP runs |
| `cosine_similarity.csv` | Full matrix with sequence IDs on both axes |
| `neighbors.csv` | All non-self neighbor rankings and cosine similarities |
| `summary.json` | Input hashes, versions, settings, variance ratios, and diagnostics |

Existing nonempty output directories require `--overwrite`; only named output
files are replaced. Inputs cannot occupy one of the output file paths. The summary
records the input and metadata SHA-256 values, but a CSV alone cannot prove which
model/checkpoint produced it. Retain the original model ID, checkpoint revision,
tokenization, pooling, sequence-length handling, and inference settings alongside
your embeddings. The current generic Hugging Face adapter remains experimental;
real checkpoint integration and provenance-aware inference are the next milestone.

## Python API

```python
from genescope.embeddings import load_embeddings
from genescope.explore import project_pca, nearest_neighbors
from genescope.report import explore_embeddings

frame = load_embeddings("kmer_embeddings.csv")
matrix = frame.filter(regex=r"^embedding_\d+$").to_numpy()
pca = project_pca(matrix)
neighbors = nearest_neighbors(matrix, frame.sequence_id.tolist(), k=3)
report = explore_embeddings("kmer_embeddings.csv", "python_report", neighbors=3)
```

Future supervised evaluations should split related sequences before fitting any
preprocessing and use held-out biological labels. This exploratory command fits
the supplied dataset for visualization; it does not perform predictive evaluation.
