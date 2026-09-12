"""Utilities for turning model embeddings into analysis-ready tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import SequenceRecord


def embeddings_to_frame(
    records: list[SequenceRecord],
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Create a tidy dataframe with one row per sequence embedding."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix.")
    if len(records) != embeddings.shape[0]:
        raise ValueError("Number of sequence records does not match embedding rows.")

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
