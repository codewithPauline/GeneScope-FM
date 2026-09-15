import json

import numpy as np
import pytest

from genescope.embeddings import save_embeddings
from genescope.io import SequenceRecord
from genescope.provenance import load_provenance, provenance_path
from genescope.report import explore_embeddings


def export(path, provenance=None):
    return save_embeddings(
        [SequenceRecord("a", "ACGT"), SequenceRecord("b", "GGGG")],
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        path,
        provenance=provenance,
    )


def test_provenance_roundtrip_and_report_attachment(tmp_path):
    path = export(tmp_path / "vectors.csv", {"kind": "kmer_baseline", "k": 1})
    manifest = load_provenance(path)
    assert manifest["generation"]["k"] == 1
    assert (manifest["n_sequences"], manifest["n_features"]) == (2, 2)
    report = explore_embeddings(path, tmp_path / "report")
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["embedding_provenance"] == manifest
    assert "kmer_baseline" in report.read_text()


def test_modified_csv_cannot_inherit_original_provenance(tmp_path):
    path = export(tmp_path / "vectors.csv", {"kind": "foundation_model"})
    path.write_text(path.read_text().replace("1.0", "9.0"))
    with pytest.raises(ValueError, match="hash does not match"):
        load_provenance(path)
    with pytest.raises(ValueError, match="hash does not match"):
        explore_embeddings(path, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_plain_export_removes_stale_model_claim(tmp_path):
    path = export(tmp_path / "vectors.csv", {"model_id": "previous-model"})
    export(path)
    assert load_provenance(path) is None
    assert not provenance_path(path).exists()


def test_invalid_generation_metadata_does_not_damage_existing_csv(tmp_path):
    path = export(tmp_path / "vectors.csv")
    original = path.read_bytes()
    with pytest.raises(ValueError):
        export(path, {"bad": float("nan")})
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "body",
    ["broken json", "[]", '{"schema_version": 9}', '{"schema_version": 1, "generation": []}'],
)
def test_malformed_sidecar_rejected(tmp_path, body):
    path = export(tmp_path / "vectors.csv")
    provenance_path(path).write_text(body)
    with pytest.raises(ValueError):
        load_provenance(path)
