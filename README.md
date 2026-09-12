# GeneScope

> **See what foundation models see in DNA.**

**GeneScope-FM** is an open-source AI-for-biology framework for exploring, benchmarking, predicting, and interpreting DNA sequences with genomic foundation models.

GeneScope is built around a simple question:

> **What biological information is encoded inside a genomic foundation model's representation of DNA?**

Rather than treating foundation models as black-box predictors, GeneScope turns their sequence representations into objects researchers can inspect, compare, visualize, and evaluate.

## Why GeneScope?

Traditional sequence analysis often begins with hand-designed features such as GC content, k-mer frequencies, motif counts, alignments, or variant summaries. Genomic foundation models instead learn high-dimensional representations directly from sequence.

GeneScope provides the research layer for interrogating those learned representations.

```text
DNA / FASTA
    |
    v
Genomic foundation model
    |
    v
Sequence embeddings
    |
    +--> Explore representation space
    +--> Compare models
    +--> Train downstream predictors
    +--> Explain predictions
    +--> Benchmark accuracy, calibration and compute
```

## Project goals

GeneScope is being developed to support five core capabilities:

1. **Embed** — convert DNA sequences into reusable foundation-model embeddings.
2. **Explore** — inspect learned representation space with dimensionality reduction, similarity analysis and clustering.
3. **Predict** — use embeddings for downstream biological classification and prediction tasks.
4. **Explain** — identify sequence regions that influence model predictions.
5. **Compare** — benchmark multiple genomic foundation models on the same biological problem.

The long-term goal is model-agnostic biological sequence intelligence: one interface, multiple foundation-model backends.

## Initial model architecture

GeneScope uses a lightweight adapter layer so supported models can be switched without rewriting the analysis pipeline.

```python
from genescope.models import get_model

model = get_model("nucleotide-transformer")
embeddings = model.embed(["ACGTACGTACGT"])
```

The first release is structured to support Hugging Face-compatible genomic models, with planned adapters for:

- Nucleotide Transformer
- DNABERT-family models
- HyenaDNA-style models
- future biological sequence foundation models

> Model support will be added only when the inference behavior has been tested and documented. GeneScope will not silently pretend incompatible models share identical tokenization or sequence-length behavior.

## Installation

Clone the repository and install the development package:

```bash
git clone https://github.com/codewithPauline/GeneScope-FM.git
cd GeneScope-FM
pip install -e .
```

For foundation-model inference:

```bash
pip install -e ".[ai]"
```

## CLI

GeneScope exposes a command-line interface designed for biological workflows.

```bash
genescope models
```

Inspect FASTA input:

```bash
genescope inspect examples/sequences.fasta
```

Generate embeddings:

```bash
genescope embed examples/sequences.fasta \
  --model nucleotide-transformer \
  --output embeddings.csv
```

The current embedding command is wired to the model-adapter system. Foundation-model execution requires the optional AI dependencies and a compatible model backend.

## Python API

```python
from genescope.io import read_fasta
from genescope.models import get_model

records = read_fasta("examples/sequences.fasta")
model = get_model("nucleotide-transformer")

matrix = model.embed([record.sequence for record in records])
print(matrix.shape)
```

## What GeneScope will benchmark

A serious foundation-model framework should answer more than "which model achieved the highest accuracy?" GeneScope is being designed to compare models across:

- downstream predictive performance
- biological clustering quality
- nearest-neighbor structure
- calibration and uncertainty
- runtime
- memory requirements
- sequence-length constraints
- robustness to sequence perturbation
- interpretability

A major future workflow will compare learned embeddings against conventional biological representations such as k-mer features.

## Scientific direction

GeneScope is intentionally organism-independent. Human, plant, microbial, vertebrate, and other nucleotide sequences can share the same interface.

One flagship research direction will ask whether genomic foundation-model embeddings contain **evolutionary signal**: whether learned sequence representations recover meaningful biological relationships that can be compared with conventional genetic or evolutionary distances.

That case study will complement, rather than define, the framework.

## Repository structure

```text
GeneScope-FM/
├── genescope/
│   ├── cli.py
│   ├── io.py
│   ├── embeddings.py
│   └── models/
│       ├── base.py
│       ├── hf.py
│       └── registry.py
├── examples/
├── tests/
├── .github/workflows/
├── pyproject.toml
└── README.md
```

## Development roadmap

**v0.1 — Foundation**

- FASTA ingestion and validation
- clean Python API
- CLI
- model registry
- Hugging Face adapter
- embedding export
- unit tests and CI

**v0.2 — Explore**

- PCA and UMAP
- similarity matrices
- nearest-neighbor search
- representation-quality metrics

**v0.3 — Predict**

- downstream classifiers
- train/validation/test evaluation
- k-mer baselines
- calibration metrics

**v0.4 — Explain**

- sequence attribution
- Integrated Gradients / saliency backends
- sequence-level visualization

**v0.5 — Compare**

- multi-model benchmarking
- runtime and memory profiling
- standardized benchmark reports

## Philosophy

GeneScope is not meant to be a thin wrapper around pretrained models. The project is centered on **evaluation, biological interpretation, reproducibility, and transparent model comparison**.

If a foundation model learns a useful representation of DNA, GeneScope should help us measure what it learned, where it works, where it fails, and why.

## Author

**Pauline Owusu-Ansah**  
Ph.D. researcher in computational and evolutionary biology.

## License

Released under the MIT License. See `LICENSE`.
