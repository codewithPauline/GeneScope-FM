"""Command-line interface for GeneScope."""

from __future__ import annotations

from pathlib import Path

import typer

from .embeddings import save_embeddings
from .io import read_fasta
from .models import available_models, get_model

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
    max_length: int = typer.Option(512, min=8),
) -> None:
    """Generate one foundation-model embedding per FASTA record."""

    records = read_fasta(fasta)
    backend = get_model(model, batch_size=batch_size, max_length=max_length)
    matrix = backend.embed([record.sequence for record in records])
    saved = save_embeddings(records, matrix, output)
    typer.echo(
        f"Saved {matrix.shape[0]} sequence embeddings "
        f"({matrix.shape[1]} dimensions) to {saved}"
    )


if __name__ == "__main__":
    app()
