import numpy as np
import pytest

from genescope.explore import (
    cosine_similarity,
    nearest_neighbors,
    project_pca,
    project_umap,
    representation_diagnostics,
    validate_matrix,
)


def test_pca_known_axes_and_reconstruction():
    # Orthogonal axes with variances 8:2; a translated input must give the same scores.
    matrix = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    result = project_pca(matrix + [12, -7])
    np.testing.assert_allclose(result.explained_variance_ratio, [0.8, 0.2])
    np.testing.assert_allclose(result.coordinates, matrix)
    np.testing.assert_allclose(
        result.coordinates @ result.components + result.mean, matrix + [12, -7]
    )


def test_pca_constant_and_rank_one_input():
    result = project_pca(np.ones((4, 3)))
    assert not result.coordinates.any()
    assert not result.explained_variance_ratio.any()
    result = project_pca(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    np.testing.assert_allclose(result.explained_variance_ratio, [1, 0], atol=1e-15)
    with pytest.raises(ValueError, match="n_components"):
        project_pca(np.ones((2, 3)))


@pytest.mark.parametrize(
    "matrix", [[], [[1, float("nan")]], [[float("inf")]], [[1 + 2j]], [["a"]], np.ones((1, 1, 1))]
)
def test_invalid_matrix_rejected(matrix):
    with pytest.raises(ValueError):
        validate_matrix(matrix)


def test_cosine_known_vectors_and_scale_invariance():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-2.0, 0.0], [1.0, 1.0]])
    actual = cosine_similarity(matrix)
    assert actual[0, 1] == 0
    assert actual[0, 2] == -1
    np.testing.assert_allclose(actual[0, 3], 1 / np.sqrt(2))
    np.testing.assert_allclose(actual, actual.T)
    np.testing.assert_allclose(actual, cosine_similarity(matrix * 1e200))
    with pytest.raises(ValueError, match="zero-norm"):
        cosine_similarity(np.array([[0.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="limited"):
        cosine_similarity(matrix, max_sequences=3)


def test_neighbors_exclude_self_retain_duplicates_and_tie_order():
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    table = nearest_neighbors(matrix, ["a", "b", "c", "d"], k=3)
    assert not (table.sequence_id == table.neighbor_id).any()
    assert table.loc[table.sequence_id == "a", "neighbor_id"].tolist() == ["b", "d", "c"]
    assert table.loc[table.sequence_id == "d", "neighbor_id"].tolist() == ["a", "b", "c"]
    with pytest.raises(ValueError, match="unique"):
        nearest_neighbors(matrix, ["a"] * 4)


def test_diagnostics_identifies_collapse():
    result = representation_diagnostics(np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 0.0]]))
    assert result["centered_rank"] == 1
    assert result["constant_features"] == 1
    assert result["duplicate_embedding_rows"] == 1
    assert result["zero_norm_rows"] == 1


def test_umap_validation_before_optional_import():
    with pytest.raises(ValueError, match="at least four"):
        project_umap(np.ones((3, 2)))
    with pytest.raises(ValueError, match="n_neighbors"):
        project_umap(np.ones((4, 2)), n_neighbors=15)


def test_optional_umap_real_execution_and_seed():
    pytest.importorskip("umap")
    matrix = np.random.default_rng(3).normal(size=(20, 5))
    first = project_umap(matrix, n_neighbors=4, random_state=9)
    second = project_umap(matrix, n_neighbors=4, random_state=9)
    assert first.shape == (20, 2)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
