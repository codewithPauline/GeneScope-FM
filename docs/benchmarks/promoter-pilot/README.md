# Human non-TATA promoter pilot

Frozen Nucleotide Transformer v2 50M versus GC/length and 3-mer controls, using
600 training, 200 validation, and 200 held-out test sequences. Each split is
balanced by class. This is an overlap-screened subset of Genomic Benchmarks.

![Held-out classification results](metrics.svg)

NT's balanced accuracy is 80.5%, compared with 77.0% for 3-mers. The paired 95%
interval for that difference is −3.0 to +10.0 percentage points and includes zero.
The pilot does not establish a reliable advantage or a full-benchmark result.

See the [methods and reproduction guide](../../benchmarking.md) for the source,
selection procedure, classifier settings, uncertainty, and remaining limitations.

| File | Contents |
| --- | --- |
| [report.html](report.html) | Portable report; download the folder and open locally |
| [results.json](results.json) | Scores, intervals, validation search, protocol, and environment |
| [predictions.csv](predictions.csv) | Each test sequence's label and three model probabilities |
| [split_manifest.csv](split_manifest.csv) | Sample IDs, labels, splits, lengths, and DNA hashes |
| [source_manifest.json](source_manifest.json) | Pinned dataset files, checksums, selection, and exclusions |
| [embedding_provenance.json](embedding_provenance.json) | Model revision, inference settings, sequence hashes, and token handling |

Raw DNA, embeddings, and model weights are not bundled. The reproduction guide
downloads the pinned sources and regenerates them.
