import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from genescope.benchmark import bootstrap_metrics, load_dataset, validate_test_groups
from genescope.cli import app
from genescope.splits import load_split_provenance, prepare_group_splits


@pytest.fixture
def grouped(tmp_path):
    rng = np.random.default_rng(314)
    frame = pd.DataFrame(
        {
            "sequence_id": ["001", "NA"] + [f"seq_{i:03}" for i in range(2, 48)],
            "sequence": ["".join(rng.choice(list("ACGT"), 80)) for _ in range(48)],
            "label": [0, 1] * 24,
            "group": [f"cluster_{i // 4:02}" for i in range(48)],
        }
    )
    mapping = pd.DataFrame(
        {
            "group": [f"cluster_{i:02}" for i in range(12)],
            "split": ["train"] * 4 + ["validation"] * 4 + ["test"] * 4,
        }
    )
    data, assignments = tmp_path / "input.csv", tmp_path / "assignments.csv"
    frame.to_csv(data, index=False)
    mapping.to_csv(assignments, index=False)
    return frame, mapping, data, assignments


def prepare(grouped, output):
    return prepare_group_splits(
        grouped[2],
        grouped[3],
        output,
        group_kind="homology",
        group_source="Synthetic groups; no real homology inferred.",
    )


def test_preparation_preserves_rows_ids_and_provenance_under_reordering(grouped, tmp_path):
    path = prepare(grouped, tmp_path / "one")
    frame = load_dataset(path)
    assert len(frame) == 48
    assert frame.sequence_id.tolist()[:2] == ["001", "NA"]
    assert frame.groupby("group").split.nunique().eq(1).all()
    manifest = load_split_provenance(path, frame)
    assert manifest["splits"]["test"] == {"n": 16, "groups": 4, "class_counts": {"0": 8, "1": 8}}
    assert manifest["group_annotations_verified"] is False
    assert manifest["audit"]["approximate_homology_checked"] is False
    grouped[0].iloc[::-1].to_csv(grouped[2], index=False)
    grouped[1].iloc[::-1].to_csv(grouped[3], index=False)
    other = prepare(grouped, tmp_path / "two")
    assert path.read_bytes() == other.read_bytes()
    assert (path.parent / "sequences.fasta").read_bytes() == (
        other.parent / "sequences.fasta"
    ).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        prepare(grouped, path.parent)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate",
        "split",
        "whitespace",
        "blank",
        "id",
        "fasta_id",
        "header",
        "label",
    ],
)
def test_bad_assignments_and_inputs_fail_without_partial_output(grouped, tmp_path, change):
    frame, mapping, data, assignments = grouped
    if change == "missing":
        mapping = mapping.iloc[1:]
    elif change == "extra":
        mapping.loc[len(mapping)] = ["extra", "train"]
    elif change == "duplicate":
        mapping.loc[len(mapping)] = mapping.iloc[0]
    elif change == "split":
        mapping.loc[0, "split"] = "testing"
    elif change == "whitespace":
        frame.loc[0, "group"] += " "
    elif change == "blank":
        frame.loc[0, "group"] = ""
    elif change == "id":
        frame.loc[1, "sequence_id"] = frame.loc[0, "sequence_id"]
    elif change == "fasta_id":
        frame.loc[0, "sequence_id"] = "two words"
    elif change == "header":
        frame["unexpected"] = "x"
    elif change == "label":
        frame.loc[0, "label"] = 2
    frame.to_csv(data, index=False)
    mapping.to_csv(assignments, index=False)
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        prepare(grouped, output)
    assert not output.exists()


def test_distinct_chromosome_names_do_not_override_sequence_overlap(grouped, tmp_path):
    frame, _, data, _ = grouped
    reverse = frame.loc[0, "sequence"].translate(str.maketrans("ACGT", "TGCA"))[::-1]
    frame.loc[47, "sequence"] = "A" * 15 + reverse[:50] + "C" * 15
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError, match="Shared exact 50-base"):
        prepare(grouped, tmp_path / "bad")


@pytest.mark.parametrize("change", ["dataset", "assignments", "json", "kind", "source"])
def test_stale_or_invalid_split_provenance_is_rejected(grouped, tmp_path, change):
    path = prepare(grouped, tmp_path / "prepared")
    frame = load_dataset(path)
    sidecar = path.with_name("split_provenance.json")
    manifest = json.loads(sidecar.read_text())
    if change == "dataset":
        path.write_text(path.read_text() + "\n")
    elif change == "assignments":
        manifest["group_assignments"][0]["split"] = "test"
    elif change == "kind":
        manifest["group_kind"] = "guessed"
    elif change == "source":
        manifest["group_source"] = ""
    if change == "json":
        sidecar.write_text("not json")
    else:
        sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="[Pp]rovenance"):
        load_split_provenance(path, frame)


def test_whole_group_bootstrap_keeps_correlated_sequences_together():
    pytest.importorskip("sklearn")
    labels = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.9, 0.1])
    groups = np.array(["a", "a", "b", "b"])
    kwargs = {"repetitions": 100, "return_diagnostics": True}
    first = bootstrap_metrics(labels, {"3mer": p, "frozen_embeddings": p}, groups=groups, **kwargs)
    assert first[0]["3mer"]["balanced_accuracy"] == [0.0, 1.0]
    assert first[1] == {"balanced_accuracy": [0.0, 0.0], "auroc": [0.0, 0.0]}
    assert first[2]["unit"] == "group" and first[2]["test_groups"] == 2
    # Replicating every observation within each group adds no independent evidence.
    duplicated = bootstrap_metrics(
        np.repeat(labels, 20),
        {"3mer": np.repeat(p, 20), "frozen_embeddings": np.repeat(p, 20)},
        groups=np.repeat(groups, 20),
        **kwargs,
    )
    assert duplicated == first
    independent = bootstrap_metrics(
        np.repeat(labels, 20),
        {"3mer": np.repeat(p, 20), "frozen_embeddings": np.repeat(p, 20)},
        **kwargs,
    )
    low, high = independent[0]["3mer"]["balanced_accuracy"]
    assert low > 0.0 and high < 1.0


def test_single_class_group_draws_are_counted_and_paired():
    pytest.importorskip("sklearn")
    labels = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    args = {"3mer": p, "frozen_embeddings": p}
    kwargs = {"groups": ["a", "b", "c", "d"], "repetitions": 100, "return_diagnostics": True}
    first = bootstrap_metrics(labels, args, **kwargs)
    assert first == bootstrap_metrics(labels, args, **kwargs)
    assert first[2]["discarded_single_class_draws"] > 0
    assert first[2]["groups_containing_class"] == {"0": 2, "1": 2}
    assert first[0]["3mer"]["auroc"] == [1.0, 1.0]


@pytest.mark.parametrize(
    "groups", [["one"] * 4, ["a", "a", "b", "c"], ["a", "b"], ["a", "b", "c", " "]]
)
def test_insufficient_or_misaligned_test_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        validate_test_groups(np.array([0, 0, 1, 1]), groups)


def test_grouped_cli_to_report_and_single_class_diagnostics(grouped, tmp_path):
    pytest.importorskip("sklearn")
    frame, mapping, data, assignments = grouped
    # Four test groups each contain one class, two groups per class.
    frame.loc[frame.group.isin(["cluster_08", "cluster_09"]), "label"] = 0
    frame.loc[frame.group.isin(["cluster_10", "cluster_11"]), "label"] = 1
    frame.to_csv(data, index=False)
    output = tmp_path / "grouped"
    runner = CliRunner()
    response = runner.invoke(
        app,
        [
            "split-groups",
            str(data),
            str(assignments),
            "--group-kind",
            "homology",
            "--group-source",
            "Synthetic <groups>",
            "--output",
            str(output),
        ],
    )
    assert response.exit_code == 0, response.output
    vectors = output / "vectors.csv"
    response = runner.invoke(
        app, ["baseline", str(output / "sequences.fasta"), "--output", str(vectors)]
    )
    assert response.exit_code == 0, response.output
    report = output / "report"
    response = runner.invoke(
        app, ["benchmark", str(output / "dataset.csv"), str(vectors), "--output", str(report)]
    )
    assert response.exit_code == 0, response.output
    result = json.loads((report / "results.json").read_text())
    assert result["schema_version"] == 2
    assert result["protocol"]["bootstrap"]["unit"] == "group"
    assert result["splits"]["test"]["groups"] == 4
    assert result["split_provenance"]["group_kind"] == "homology"
    assert result["frozen_minus_3mer"]["auroc"]["ci95"] == [0.0, 0.0]
    assert any("Fewer than ten" in item for item in result["limitations"])
    per_group = pd.read_csv(report / "group_metrics.csv")
    assert len(per_group) == 12
    assert per_group.auroc.isna().all() and per_group.balanced_accuracy.isna().all()
    assert per_group.brier_score.notna().all()
    assert "group" in pd.read_csv(report / "predictions.csv")
    html = (report / "report.html").read_text()
    assert "Synthetic &lt;groups&gt;" in html and "Synthetic <groups>" not in html
    assert "group_metrics.csv" in html
    assert "95% group bootstrap" in (report / "metrics.svg").read_text()
    # Changed split tables must not silently retain their old provenance.
    altered = output / "dataset.csv"
    altered.write_text(altered.read_text() + "\n")
    response = runner.invoke(
        app, ["benchmark", str(altered), str(vectors), "--output", str(output / "stale")]
    )
    assert response.exit_code == 1 and "hash does not match" in response.output
