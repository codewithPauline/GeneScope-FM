"""Utilities for turning model embeddings into analysis-ready tables."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .explore import validate_matrix
from .io import SequenceRecord


def load_embeddings(path: str | Path) -> pd.DataFrame:
    """Read exported CSV without coercing IDs such as '001' or 'NA'."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Embedding CSV column names must be unique.")
    frame = pd.read_csv(path, dtype={"sequence_id": str}, keep_default_na=False)
    if "sequence_id" not in frame:
        raise ValueError("Embedding CSV requires a sequence_id column.")
    ids = frame["sequence_id"]
    if ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Sequence IDs must be nonempty and unique.")
    columns = [column for column in frame if re.fullmatch(r"embedding_\d{4,}", column)]
    if not columns:
        raise ValueError("Embedding CSV requires embedding_0000, embedding_0001, ... columns.")
    expected = [f"embedding_{index:04d}" for index in range(len(columns))]
    if set(columns) != set(expected):
        raise ValueError("Embedding columns must be numbered consecutively from embedding_0000.")
    unknown = set(frame.columns) - set(columns) - {"sequence_id", "sequence_length"}
    if unknown:
        raise ValueError("Unexpected embedding CSV columns: " + ", ".join(sorted(unknown)))
    try:
        matrix = frame[expected].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Embedding features must be numeric.") from exc
    validate_matrix(matrix)
    frame[expected] = matrix
    if "sequence_length" in frame:
        lengths = pd.to_numeric(frame["sequence_length"], errors="coerce")
        if not (np.isfinite(lengths) & (lengths > 0) & (lengths % 1 == 0)).all():
            raise ValueError("Sequence lengths must be positive integers.")
    metadata = [column for column in ("sequence_id", "sequence_length") if column in frame]
    return frame[metadata + expected]


def embeddings_to_frame(
    records: list[SequenceRecord],
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Create a tidy dataframe with one row per sequence embedding."""

    embeddings = validate_matrix(embeddings)
    if len(records) != embeddings.shape[0]:
        raise ValueError("Number of sequence records does not match embedding rows.")
    ids = [record.identifier for record in records]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("Sequence IDs must be nonempty and unique.")
    if any(not record.sequence for record in records):
        raise ValueError("Sequences must not be empty.")

    columns = [f"embedding_{index:04d}" for index in range(embeddings.shape[1])]
    frame = pd.DataFrame(embeddings, columns=columns)
    frame.insert(0, "sequence_id", [record.identifier for record in records])
    frame.insert(1, "sequence_length", [len(record.sequence) for record in records])
    return frame


def save_embeddings(
    records: list[SequenceRecord],
    embeddings: np.ndarray,
    output: str | Path,
) -> Path:
    """Save embeddings as CSV and return the resolved output path."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_to_frame(records, embeddings).to_csv(output_path, index=False)
    return output_path
