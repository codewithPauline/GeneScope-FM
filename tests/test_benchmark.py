import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from genescope.benchmark import (
    align_embeddings,
    audit_splits,
    binary_metrics,
    bootstrap_metrics,
    canonical_sequence,
    fit_probe,
    load_dataset,
    overlap_words,
    sequence_hash,
)
from genescope.cli import app
from genescope.embeddings import save_embeddings
from genescope.io import SequenceRecord
from genescope.provenance import provenance_path


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(71)
    frame = pd.DataFrame(
        {
            "sequence_id": ["001", "NA"] + [f"seq_{i}" for i in range(2, 24)],
            "sequence": ["".join(rng.choice(list("ACGT"), 80)) for _ in range(24)],
            "label": [0, 1] * 12,
            "split": ["train"] * 12 + ["validation"] * 6 + ["test"] * 6,
        }
    )
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)
    return frame, path


def export_vectors(frame, path):
    rng = np.random.default_rng(9)
    features = rng.normal(size=(len(frame), 4))
    features[:, 0] += frame.label.to_numpy() * 2
    # Reverse file order to test joins rather than positional alignment.
    records = [SequenceRecord(row.sequence_id, row.sequence) for row in frame.itertuples()]
    save_embeddings(records[::-1], features[::-1], path, provenance={"kind": "test_fixture"})
    return features


def test_dataset_preserves_string_ids_and_normalizes_dna(dataset):
    frame, path = dataset
    frame.loc[0, "sequence"] = "acgt\n" + frame.loc[0, "sequence"]
    frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded.sequence_id.tolist()[:2] == ["001", "NA"]
    assert loaded.sequence.iloc[0].startswith("ACGT")
    assert audit_splits(loaded)["shared_windows_across_splits"] == {
        "train/validation": 0,
        "train/test": 0,
        "validation/test": 0,
    }


@pytest.mark.parametrize(
    "change", ["id", "label", "split", "single_class", "group", "bad_dna", "header"]
)
def test_malformed_dataset_rejected(dataset, change):
    frame, path = dataset
    if change == "id":
        frame.loc[1, "sequence_id"] = frame.loc[0, "sequence_id"]
    elif change == "label":
        frame.loc[0, "label"] = 2
    elif change == "split":
        frame.loc[0, "split"] = "other"
    elif change == "single_class":
        frame.loc[frame.split == "test", "label"] = 0
    elif change == "group":
        frame["group"] = ""
    elif change == "bad_dna":
        frame.loc[0, "sequence"] = "ACXG"
    frame.to_csv(path, index=False)
    if change == "header":
        path.write_text(path.read_text().replace("sequence_id,sequence,", "sequence,sequence,"))
    with pytest.raises(ValueError):
        load_dataset(path)


def test_duplicate_reverse_complement_and_partial_overlap_rejected(dataset):
    frame, _ = dataset
    original = frame.loc[0, "sequence"]
    reverse = original.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    frame.loc[23, "sequence"] = reverse
    with pytest.raises(ValueError, match="reverse-complement"):
        audit_splits(frame)
    frame.loc[23, "sequence"] = "AC" + reverse[:50] + "GT"
    with pytest.raises(ValueError, match="Shared exact 50-base"):
        audit_splits(frame)


def test_cross_split_groups_and_short_sequence_audit(dataset):
    frame, _ = dataset
    frame["group"] = frame.split
    assert audit_splits(frame)["group_disjoint"] is True
    frame.loc[23, "group"] = "train"
    with pytest.raises(ValueError, match="Group overlap"):
        audit_splits(frame)
    frame = frame.drop(columns="group")
    frame.loc[0, "sequence"] = "ACGT"
    assert audit_splits(frame)["sequences_without_eligible_overlap_windows"] == 1
    assert overlap_words("N" * 100) == set()
    assert canonical_sequence("TTTT") == "AAAA"


def test_embedding_alignment_checks_dna_and_csv_hashes(dataset, tmp_path):
    frame, _ = dataset
    path = tmp_path / "vectors.csv"
    expected = export_vectors(frame, path)
    actual, _ = align_embeddings(frame, path)
    np.testing.assert_allclose(actual, expected)
    frame.loc[0, "sequence"] = "A" * 80  # Same ID and length, different DNA.
    with pytest.raises(ValueError, match="sequence IDs and DNA"):
        align_embeddings(frame, path)
    frame.loc[0, "sequence"] = "ACGT"
    path.write_text(path.read_text().replace("seq_23", "changed_id"))
    with pytest.raises(ValueError, match="hash does not match"):
        align_embeddings(frame, path)


def test_missing_and_duplicate_sequence_provenance_rejected(dataset, tmp_path):
    frame, _ = dataset
    path = tmp_path / "vectors.csv"
    export_vectors(frame, path)
    sidecar = provenance_path(path)
    manifest = json.loads(sidecar.read_text())
    manifest["input_sequences"][0] = manifest["input_sequences"][1]
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sequence IDs and DNA"):
        align_embeddings(frame, path)
    sidecar.unlink()
    with pytest.raises(ValueError, match="re-export"):
        align_embeddings(frame, path)


def test_scaler_and_selection_never_use_test_features_or_labels(dataset, tmp_path):
    pytest.importorskip("sklearn")
    frame, _ = dataset
    features = export_vectors(frame, tmp_path / "features.csv")
    labels, splits = frame.label.to_numpy(), frame.split.to_numpy()
    fitted, selection = fit_probe(features, labels, splits)
    np.testing.assert_allclose(fitted[0].mean_, features[splits == "train"].mean(axis=0))
    features[splits == "test"] = 1e9
    labels[splits == "test"] = 1 - labels[splits == "test"]
    changed, changed_selection = fit_probe(features, labels, splits)
    assert selection == changed_selection
    np.testing.assert_array_equal(fitted[0].mean_, changed[0].mean_)
    np.testing.assert_array_equal(fitted[1].coef_, changed[1].coef_)


def test_metrics_known_values_and_paired_bootstrap():
    pytest.importorskip("sklearn")
    labels = np.array([0, 1, 0, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(labels, probability)
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["auroc"] == 0.75
    assert metrics["brier_score"] == pytest.approx(0.325)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    probabilities = {"3mer": probability, "frozen_embeddings": probability.copy()}
    first = bootstrap_metrics(labels, probabilities, repetitions=25)
    assert first == bootstrap_metrics(labels, probabilities, repetitions=25)
    assert first[1] == {"balanced_accuracy": [0.0, 0.0], "auroc": [0.0, 0.0]}


def test_benchmark_cli_exports_predictions_and_protects_results(dataset, tmp_path):
    pytest.importorskip("sklearn")
    frame, path = dataset
    vectors = tmp_path / "vectors.csv"
    export_vectors(frame, vectors)
    output = tmp_path / "report"
    runner = CliRunner()
    args = [
        "benchmark",
        str(path),
        str(vectors),
        "--output",
        str(output),
        "--description",
        "<script>example</script>",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    summary = json.loads((output / "results.json").read_text())
    predictions = pd.read_csv(output / "predictions.csv", dtype={"sequence_id": str})
    manifest = pd.read_csv(output / "split_manifest.csv", dtype={"sequence_id": str})
    assert (
        predictions.sequence_id.tolist() == frame.loc[frame.split == "test", "sequence_id"].tolist()
    )
    assert manifest.sequence_sha256.tolist() == frame.sequence.map(sequence_hash).tolist()
    for name in summary["representations"]:
        assert binary_metrics(
            predictions.label.to_numpy(), predictions[f"{name}_probability"].to_numpy()
        ) == pytest.approx(summary["representations"][name]["test"])
    assert "Frozen NT embeddings" not in (output / "metrics.svg").read_text()
    assert "results.json" in (output / "report.html").read_text()
    assert "&lt;script&gt;example&lt;/script&gt;" in (output / "report.html").read_text()
    assert "<script>example</script>" not in (output / "report.html").read_text()
    original = (output / "results.json").read_bytes()
    refused = runner.invoke(app, args)
    assert refused.exit_code == 1 and "already exists" in refused.output
    assert (output / "results.json").read_bytes() == original
