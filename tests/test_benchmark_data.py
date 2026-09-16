import hashlib
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genescope.benchmark import audit_splits

PREPARER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples" / "prepare_promoter_benchmark.py")
)


def source_frames():
    rng = np.random.default_rng(13)
    frames = {}
    for split, n in (("train", 60), ("test", 20)):
        frames[split] = pd.DataFrame(
            {
                "sequence_id": [f"gb_{split}_{i:06d}" for i in range(n)],
                "sequence": ["".join(rng.choice(list("ACGT"), 251)) for _ in range(n)],
                "label": [0, 1] * (n // 2),
            }
        )
    # Force both exact and reverse-strand partial contamination into the source.
    frames["train"].loc[0, "sequence"] = frames["test"].loc[0, "sequence"]
    reverse = frames["test"].loc[1, "sequence"].translate(str.maketrans("ACGT", "TGCA"))[::-1]
    frames["train"].loc[1, "sequence"] = "A" * 100 + reverse[:50] + "C" * 101
    return frames


def test_sampling_reproducible_balanced_and_respects_official_partitions():
    select = PREPARER["select_pilot"]
    frames = source_frames()
    kwargs = {"train_per_class": 5, "validation_per_class": 3, "test_per_class": 3}
    sample, manifest = select(frames, **kwargs)
    repeated, repeated_manifest = select({k: v.iloc[::-1] for k, v in frames.items()}, **kwargs)
    pd.testing.assert_frame_equal(sample, repeated)
    assert manifest == repeated_manifest
    assert manifest["exclusions"]["official_train_matching_official_test"] == 2
    assert not set(sample.sequence_id) & {"gb_train_000000", "gb_train_000001"}
    assert sample.loc[sample.split == "test", "sequence_id"].str.startswith("gb_test_").all()
    assert sample.loc[sample.split != "test", "sequence_id"].str.startswith("gb_train_").all()
    assert sample.groupby(["split", "label"]).size().to_dict() == {
        ("train", 0): 5,
        ("train", 1): 5,
        ("validation", 0): 3,
        ("validation", 1): 3,
        ("test", 0): 3,
        ("test", 1): 3,
    }
    audit_splits(sample)
    changed, _ = select(frames, seed=43, **kwargs)
    assert changed.sequence_id.tolist() != sample.sequence_id.tolist()


def test_selected_validation_windows_exclude_training_candidates():
    select = PREPARER["select_pilot"]
    frames = source_frames()
    kwargs = {"train_per_class": 5, "validation_per_class": 3, "test_per_class": 3}
    original, _ = select(frames, **kwargs)
    validation = original.loc[original.split == "validation"].iloc[0]
    training = original.loc[original.split == "train"].iloc[0]
    mask = frames["train"].sequence_id == training.sequence_id
    frames["train"].loc[mask, "sequence"] = validation.sequence[:50] + training.sequence[50:]
    selected, manifest = select(frames, **kwargs)
    assert training.sequence_id not in set(selected.sequence_id)
    assert validation.sequence_id in set(selected.sequence_id)
    assert manifest["exclusions"]["training_candidates_matching_selected_validation"] > 0
    audit_splits(selected)


def test_sampling_fails_instead_of_silently_reducing_requested_size():
    with pytest.raises(ValueError, match="Insufficient"):
        PREPARER["select_pilot"](
            source_frames(), train_per_class=200, validation_per_class=3, test_per_class=3
        )
    with pytest.raises(ValueError, match="at least two"):
        PREPARER["select_pilot"](source_frames(), test_per_class=1)


def test_source_hash_is_checked_before_reading_cached_parquet(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    fetch = PREPARER["fetch_source"]
    files = {}
    for split, frame in source_frames().items():
        path = tmp_path / f"{split}.parquet"
        frame.rename(columns={"sequence": "seq"})[["seq", "label"]].to_parquet(path)
        files[split] = (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setitem(fetch.__globals__, "FILES", files)
    frames, manifest = fetch(tmp_path)
    assert len(frames["train"]) == 60
    assert manifest["test"]["class_counts"] == {"0": 10, "1": 10}
    (tmp_path / "train.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Cached source hash mismatch"):
        fetch(tmp_path)
