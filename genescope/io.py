"""Input utilities for nucleotide sequence data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_VALID_DNA = set("ACGTN")


@dataclass(frozen=True)
class SequenceRecord:
    """A minimal nucleotide-sequence record."""

    identifier: str
    sequence: str


def normalize_sequence(sequence: str) -> str:
    """Normalize DNA sequence text and validate nucleotide symbols."""

    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("Sequence is empty.")

    invalid = sorted(set(normalized) - _VALID_DNA)
    if invalid:
        raise ValueError(
            "Unsupported nucleotide symbols: " + ", ".join(invalid) + ". "
            "GeneScope currently accepts A, C, G, T, and N."
        )
    return normalized


def read_fasta(path: str | Path) -> list[SequenceRecord]:
    """Read a FASTA file into validated ``SequenceRecord`` objects."""

    fasta_path = Path(path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    records: list[SequenceRecord] = []
    current_id: str | None = None
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal current_id, sequence_parts
        if current_id is None:
            return
        records.append(
            SequenceRecord(
                identifier=current_id,
                sequence=normalize_sequence("".join(sequence_parts)),
            )
        )
        sequence_parts = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_id = line[1:].strip()
                if not current_id:
                    raise ValueError("Encountered a FASTA header without an identifier.")
            else:
                if current_id is None:
                    raise ValueError("FASTA sequence encountered before the first header.")
                sequence_parts.append(line)

    flush()

    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}.")

    identifiers = [record.identifier for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("FASTA identifiers must be unique.")

    return records
