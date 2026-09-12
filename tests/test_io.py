from pathlib import Path

import pytest

from genescope.io import normalize_sequence, read_fasta


def test_normalize_sequence_accepts_standard_dna():
    assert normalize_sequence("acgt n\n") == "ACGTN"


def test_normalize_sequence_rejects_invalid_symbols():
    with pytest.raises(ValueError, match="Unsupported nucleotide symbols"):
        normalize_sequence("ACGTX")


def test_read_fasta(tmp_path: Path):
    fasta = tmp_path / "example.fasta"
    fasta.write_text(">seq1\nACGTACGT\n>seq2\nNNNNACGT\n", encoding="utf-8")

    records = read_fasta(fasta)

    assert [record.identifier for record in records] == ["seq1", "seq2"]
    assert [record.sequence for record in records] == ["ACGTACGT", "NNNNACGT"]


def test_read_fasta_rejects_duplicate_ids(tmp_path: Path):
    fasta = tmp_path / "duplicate.fasta"
    fasta.write_text(">seq1\nACGT\n>seq1\nACGT\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identifiers must be unique"):
        read_fasta(fasta)
