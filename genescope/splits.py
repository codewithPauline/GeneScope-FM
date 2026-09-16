"""Prepare auditable binary splits from externally defined biological groups."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from . import __version__
from .benchmark import SPLITS, audit_splits, load_dataset, sequence_hash, validate_test_groups
from .io import normalize_sequence

GROUP_KINDS = ("chromosome", "homology", "locus", "other")


def read_table(path: str | Path, columns: set[str]) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)) or set(header) != columns:
        raise ValueError(f"Expected exactly these unique columns: {', '.join(sorted(columns))}.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty or frame.isna().any().any():
        raise ValueError("Input table must be nonempty with no missing fields.")
    for column in columns - {"sequence"}:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{column} values must be nonempty.")
        if not frame[column].eq(frame[column].str.strip()).all():
            raise ValueError(f"{column} values must not have surrounding whitespace.")
    return frame


def prepare_group_splits(
    dataset: str | Path,
    assignments: str | Path,
    output: str | Path,
    *,
    group_kind: str,
    group_source: str,
) -> Path:
    """Apply a complete, prespecified group-to-split map without dropping rows.

    Groups and their biological provenance are supplied by the caller; this does
    not infer chromosomes, run homology clustering, or validate those annotations.
    Cross-split sequence overlap remains a hard error even for distinct groups.
    """
    if group_kind not in GROUP_KINDS:
        raise ValueError(f"group_kind must be one of {GROUP_KINDS}.")
    if not group_source.strip():
        raise ValueError("group_source must describe the assembly/annotation or clustering method.")
    output = Path(output)
    if output.exists():
        raise ValueError("Split output already exists; choose a new directory.")
    frame = read_table(dataset, {"sequence_id", "sequence", "label", "group"})
    mapping = read_table(assignments, {"group", "split"})
    if frame.sequence_id.duplicated().any():
        raise ValueError("Sequence IDs must be unique.")
    if frame.sequence_id.str.contains(r"\s|>", regex=True).any():
        raise ValueError("Sequence IDs must be FASTA-safe: no whitespace or > characters.")
    if mapping.group.duplicated().any():
        raise ValueError("Each group must have exactly one split assignment.")
    if set(mapping.group) != set(frame.group):
        raise ValueError(
            "Assignment groups must match dataset groups exactly; no missing or extra groups."
        )
    if set(mapping.split) != set(SPLITS):
        raise ValueError("Assignments must include exactly train, validation, and test.")
    frame["sequence"] = frame.sequence.map(normalize_sequence)
    frame["split"] = frame.group.map(mapping.set_index("group").split)
    frame = frame[["sequence_id", "sequence", "label", "split", "group"]]
    frame = frame.sort_values("sequence_id", kind="stable").reset_index(drop=True)
    # Validate with the same schema and scientific checks used by the evaluator.
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "splits"
        staging.mkdir()
        path = staging / "dataset.csv"
        frame.to_csv(path, index=False)
        frame = load_dataset(path)
        audit = audit_splits(frame)
        test = frame.loc[frame.split == "test"]
        validate_test_groups(test.label.to_numpy(), test.group.to_numpy())
        fasta = "".join(f">{row.sequence_id}\n{row.sequence}\n" for row in frame.itertuples())
        (staging / "sequences.fasta").write_text(fasta, encoding="ascii")
        manifest = {
            "schema_version": 1,
            "genescope_version": __version__,
            "group_kind": group_kind,
            "group_source": group_source.strip(),
            "group_annotations_verified": False,
            "policy": "Prespecified complete group assignment; no resampling, balancing, row exclusion, or model-based split selection.",
            "input_csv_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
            "assignments_csv_sha256": hashlib.sha256(Path(assignments).read_bytes()).hexdigest(),
            "dataset_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fasta_sha256": sequence_hash(fasta),
            "group_assignments": mapping.sort_values("group").to_dict("records"),
            "splits": {
                split: {
                    "n": int((frame.split == split).sum()),
                    "groups": int(frame.loc[frame.split == split, "group"].nunique()),
                    "class_counts": {
                        str(c): int(((frame.split == split) & (frame.label == c)).sum())
                        for c in (0, 1)
                    },
                }
                for split in SPLITS
            },
            "audit": audit,
        }
        (staging / "split_provenance.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return output / "dataset.csv"


def load_split_provenance(dataset: str | Path, frame: pd.DataFrame) -> dict | None:
    """Verify an adjacent split-preparation manifest before displaying its claims."""
    path = Path(dataset).with_name("split_provenance.json")
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError("Invalid split provenance JSON.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported split provenance schema.")
    digest = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
    if manifest.get("dataset_csv_sha256") != digest:
        raise ValueError("Split provenance hash does not match dataset; prepare the splits again.")
    if "group" not in frame or manifest.get("group_kind") not in GROUP_KINDS:
        raise ValueError("Split provenance requires groups and a valid group kind.")
    source = manifest.get("group_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Split provenance requires a nonempty group source.")
    actual = frame[["group", "split"]].drop_duplicates().sort_values("group").to_dict("records")
    if manifest.get("group_assignments") != actual:
        raise ValueError("Split provenance group assignments do not match the dataset.")
    return manifest
