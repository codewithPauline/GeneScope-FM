import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from genescope.cli import app
from genescope.embeddings import load_embeddings, save_embeddings
from genescope.io import SequenceRecord
from genescope.report import explore_embeddings


@pytest.fixture
def embeddings(tmp_path):
    path = tmp_path / "embeddings.csv"
    records = [SequenceRecord(i, "ACGT") for i in ["001", "NA", "c"]]
    save_embeddings(records, np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), path)
    return path


def test_roundtrip_ids_and_report_outputs(embeddings, tmp_path):
    metadata = tmp_path / "labels.csv"
    metadata.write_text("sequence_id,label\nc,third\nNA,second\n001,<script>bad</script>\n")
    report = explore_embeddings(embeddings, tmp_path / "report", metadata=metadata)
    assert load_embeddings(embeddings).sequence_id.tolist() == ["001", "NA", "c"]
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["input_sha256"] == hashlib.sha256(embeddings.read_bytes()).hexdigest()
    assert summary["neighbors_returned_per_sequence"] == 2
    assert summary["labels_used_for_fit"] is False
    projection = pd.read_csv(report.parent / "projection.csv", keep_default_na=False)
    assert projection.label.tolist() == ["<script>bad</script>", "second", "third"]
    assert "<script>bad</script>" not in report.read_text()
    assert "&lt;script&gt;bad&lt;/script&gt;" in report.read_text()
    similarity = pd.read_csv(report.parent / "cosine_similarity.csv", index_col=0)
    np.testing.assert_allclose(similarity.to_numpy()[0], [1, 1 / np.sqrt(2), 0])
    assert len(list(report.parent.iterdir())) == 7


def test_labels_do_not_change_projection_or_neighbors(embeddings, tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("sequence_id,label\n001,X\nNA,Y\nc,Y\n")
    first = explore_embeddings(embeddings, tmp_path / "a")
    second = explore_embeddings(embeddings, tmp_path / "b", metadata=labels)
    for name in ["pca.csv", "neighbors.csv", "cosine_similarity.csv"]:
        assert (first.parent / name).read_bytes() == (second.parent / name).read_bytes()


def test_report_rejects_missing_labels_and_existing_output(embeddings, tmp_path):
    metadata = tmp_path / "labels.csv"
    metadata.write_text("sequence_id,label\n001,A\n")
    with pytest.raises(ValueError, match="missing"):
        explore_embeddings(embeddings, tmp_path / "bad", metadata=metadata)
    assert not (tmp_path / "bad").exists()
    report = explore_embeddings(embeddings, tmp_path / "report")
    with pytest.raises(ValueError, match="not empty"):
        explore_embeddings(embeddings, report.parent)
    assert explore_embeddings(embeddings, report.parent, overwrite=True) == report


@pytest.mark.parametrize(
    "body",
    [
        "sequence_id,embedding_0000\na,nan\n",
        "sequence_id,embedding_0000\na,inf\n",
        "sequence_id,embedding_0000\na,1\na,2\n",
        "sequence_id,embedding_0000\n,1\n",
        "sequence_id,embedding_0001\na,1\n",
        "sequence_id,embedding_0000,embedding_0000\na,1,2\n",
        "sequence_id,embedding_0000,label\na,1,2\n",
        "sequence_id,sequence_length,embedding_0000\na,0,1\n",
        "sequence_id,embedding_0000\n",
    ],
)
def test_malformed_csv_rejected(tmp_path, body):
    path = tmp_path / "bad.csv"
    path.write_text(body)
    with pytest.raises(ValueError):
        load_embeddings(path)


def test_cli_offline_workflow_and_errors(tmp_path):
    runner = CliRunner()
    fasta = tmp_path / "toy.fasta"
    fasta.write_text(">a\nAAAAC\n>b\nCCCCA\n>c\nACGTA\n")
    vectors = tmp_path / "vectors.csv"
    result = runner.invoke(app, ["baseline", str(fasta), "--k", "2", "-o", str(vectors)])
    assert result.exit_code == 0, result.output
    assert "no foundation model" in result.output
    out = tmp_path / "analysis"
    result = runner.invoke(app, ["explore", str(vectors), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "report.html").exists()
    result = runner.invoke(app, ["explore", str(vectors), "--method", "bogus"])
    assert result.exit_code == 1
    assert "Projection method" in result.output


def test_two_sequences_and_one_feature_produce_a_report(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_text("sequence_id,embedding_0000\na,1\nb,2\n")
    report = explore_embeddings(path, tmp_path / "tiny-report")
    assert "No second component" in report.read_text()


def test_report_wont_overwrite_source(tmp_path):
    path = tmp_path / "projection.csv"
    path.write_text("sequence_id,embedding_0000\na,1\nb,2\n")
    with pytest.raises(ValueError, match="overwrite an input"):
        explore_embeddings(path, tmp_path, overwrite=True)
    assert "embedding_0000" in path.read_text()
