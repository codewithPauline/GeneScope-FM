"""Command-line interface for GeneScope."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .baselines import kmer_frequencies
from .embeddings import save_embeddings
from .io import read_fasta
from .models import available_models, get_model
from .report import explore_embeddings

app = typer.Typer(
    help="GeneScope: explore genomic foundation-model representations of DNA.",
    no_args_is_help=True,
)


@app.command("models")
def models_command() -> None:
    """List registered foundation-model backends."""

    for spec in available_models():
        typer.echo(f"{spec.key}\t{spec.display_name}\t{spec.model_id}")


@app.command("inspect")
def inspect_command(fasta: Path) -> None:
    """Validate a FASTA file and summarize its sequences."""

    records = read_fasta(fasta)
    lengths = [len(record.sequence) for record in records]
    typer.echo(f"records: {len(records)}")
    typer.echo(f"min_length: {min(lengths)}")
    typer.echo(f"max_length: {max(lengths)}")
    typer.echo(f"mean_length: {sum(lengths) / len(lengths):.1f}")


@app.command("embed")
def embed_command(
    fasta: Path,
    model: str = typer.Option(
        "nucleotide-transformer",
        "--model",
        help="Registered genomic foundation model.",
    ),
    output: Path = typer.Option(
        Path("embeddings.csv"),
        "--output",
        "-o",
        help="CSV output path.",
    ),
    batch_size: int = typer.Option(8, min=1),
    max_length: int = typer.Option(
        1000,
        min=8,
        help="Maximum tokenizer tokens, not nucleotide bases.",
    ),
    revision: str | None = typer.Option(
        None,
        help="Optional Hugging Face model revision/commit for reproducible inference.",
    ),
) -> None:
    """Generate one foundation-model embedding per FASTA record."""

    try:
        records = read_fasta(fasta)
        backend = get_model(
            model,
            batch_size=batch_size,
            max_length=max_length,
            revision=revision,
        )
        matrix = backend.embed([record.sequence for record in records])
        saved = save_embeddings(records, matrix, output)
    except (ValueError, OSError, ImportError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    provenance = getattr(backend, "provenance", None)
    if callable(provenance):
        provenance_path = saved.with_suffix(saved.suffix + ".provenance.json")
        provenance_path.write_text(
            json.dumps(provenance(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        typer.echo(f"Provenance: {provenance_path}")

    typer.echo(
        f"Saved {matrix.shape[0]} sequence embeddings ({matrix.shape[1]} dimensions) to {saved}"
    )


@app.command("baseline")
def baseline_command(
    fasta: Path,
    output: Path = typer.Option(Path("kmer_embeddings.csv"), "--output", "-o"),
    k: int = typer.Option(3, min=1, max=6),
) -> None:
    """Export conventional k-mer frequencies (offline; not foundation-model embeddings)."""
    try:
        records = read_fasta(fasta)
        matrix = kmer_frequencies([record.sequence for record in records], k)
        saved = save_embeddings(records, matrix, output)
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Saved {len(records)} k-mer baseline vectors ({matrix.shape[1]} features) to {saved}"
    )
    typer.echo("Conventional sequence features; no foundation model was used.")


@app.command("explore")
def explore_command(
    embeddings: Path,
    output: Path = typer.Option(Path("genescope_report"), "--output", "-o"),
    metadata: Path | None = typer.Option(None, help="CSV with sequence_id and label columns."),
    neighbors: int = typer.Option(5, min=1, help="Clipped to n_sequences - 1 if needed."),
    method: str = typer.Option("pca", help="pca or umap; UMAP needs the explore extra."),
    seed: int = typer.Option(42, min=0, max=2**32 - 1),
    umap_neighbors: int = typer.Option(15, min=2),
    min_dist: float = typer.Option(0.1, min=0, max=1),
    overwrite: bool = typer.Option(False, help="Replace generated report files in the directory."),
    description: str | None = typer.Option(
        None, help="Dataset/representation context for the report."
    ),
) -> None:
    """Explore exported embeddings and write a portable HTML report plus CSV results."""
    try:
        report = explore_embeddings(
            embeddings,
            output,
            metadata=metadata,
            neighbors=neighbors,
            method=method,
            random_state=seed,
            umap_neighbors=umap_neighbors,
            min_dist=min_dist,
            overwrite=overwrite,
            description=description,
        )
    except (ValueError, OSError, ImportError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Report: {report}")
    typer.echo(
        "Saved projection, PCA, cosine similarities, neighbors, and reproducibility summary."
    )


if __name__ == "__main__":
    app()
