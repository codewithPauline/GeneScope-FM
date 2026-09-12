"""Model-independent analysis of sequence representations in their original space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def validate_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Require a nonempty, finite, real-valued matrix."""
    raw = np.asarray(embeddings)
    if raw.ndim != 2 or min(raw.shape) == 0:
        raise ValueError("Embeddings must be a nonempty two-dimensional matrix.")
    if raw.dtype.kind not in "iuf":
        raise ValueError("Embeddings must contain real numeric values.")
    matrix = raw.astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must contain only finite values (no NaN or infinity).")
    return matrix


@dataclass(frozen=True)
class PCAResult:
    coordinates: np.ndarray
    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray


def project_pca(embeddings: np.ndarray, n_components: int = 2) -> PCAResult:
    """Mean-center, without feature scaling, and project using a full SVD.

    Component signs are fixed by each loading's largest absolute entry. Degenerate
    eigenspaces may still differ between numerical libraries. Constant input has
    zero coordinates and zero explained-variance ratios, rather than NaNs.
    """
    matrix = validate_matrix(embeddings)
    if matrix.shape[0] < 2:
        raise ValueError("PCA requires at least two sequences.")
    if not 1 <= n_components <= min(matrix.shape[0] - 1, matrix.shape[1]):
        raise ValueError("n_components must be between 1 and min(n_sequences - 1, n_features).")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, loadings = np.linalg.svd(centered, full_matrices=False)
    components = loadings[:n_components].copy()
    indices = np.abs(components).argmax(axis=1)
    signs = np.sign(components[np.arange(n_components), indices])
    components *= np.where(signs == 0, 1, signs)[:, None]
    total = np.square(singular_values).sum()
    ratios = (
        np.square(singular_values[:n_components]) / total if total > 0 else np.zeros(n_components)
    )
    return PCAResult(centered @ components.T, components, mean, ratios)


def cosine_similarity(embeddings: np.ndarray, *, max_sequences: int = 5000) -> np.ndarray:
    """Return dense cosine similarities; reject zero vectors (undefined cosine).

    The default size cap bounds the n-by-n allocation. These are representation
    similarities, not evolutionary distances or probabilities.
    """
    matrix = validate_matrix(embeddings)
    if matrix.shape[0] > max_sequences:
        raise ValueError(f"Dense similarity is limited to {max_sequences} sequences.")
    # Scale each row first to avoid overflow/underflow when computing its norm.
    scale = np.abs(matrix).max(axis=1, keepdims=True)
    if (scale == 0).any():
        raise ValueError("Cosine similarity is undefined for zero-norm embedding rows.")
    unit = matrix / scale
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def nearest_neighbors(embeddings: np.ndarray, sequence_ids: list[str], k: int = 5) -> pd.DataFrame:
    """Rank non-self neighbors by cosine similarity; ties retain input row order."""
    matrix = validate_matrix(embeddings)
    if len(sequence_ids) != len(matrix) or len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("Provide one unique sequence ID per embedding row.")
    if not all(isinstance(identifier, str) and identifier.strip() for identifier in sequence_ids):
        raise ValueError("Sequence IDs must be nonempty strings.")
    if not 1 <= k < len(matrix):
        raise ValueError("k must be between 1 and n_sequences - 1.")
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    ids = np.asarray(sequence_ids)
    return pd.DataFrame(
        {
            "sequence_id": np.repeat(ids, k),
            "rank": np.tile(np.arange(1, k + 1), len(ids)),
            "neighbor_id": ids[order].ravel(),
            "cosine_similarity": np.take_along_axis(similarity, order, axis=1).ravel(),
        }
    )


def representation_diagnostics(embeddings: np.ndarray) -> dict:
    """Descriptive numerical checks, not evidence of biological predictive quality."""
    matrix = validate_matrix(embeddings)
    centered = matrix - matrix.mean(axis=0)
    return {
        "n_sequences": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "centered_rank": int(np.linalg.matrix_rank(centered)),
        "constant_features": int(np.count_nonzero(np.ptp(matrix, axis=0) == 0)),
        "duplicate_embedding_rows": int(len(matrix) - len(np.unique(matrix, axis=0))),
        "zero_norm_rows": int(np.count_nonzero(np.all(matrix == 0, axis=1))),
    }


def project_umap(
    embeddings: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Optional seeded 2D UMAP, using Euclidean distance and random initialization."""
    matrix = validate_matrix(embeddings)
    if len(matrix) < 4:
        raise ValueError("UMAP requires at least four sequences in GeneScope.")
    if not 2 <= n_neighbors < len(matrix):
        raise ValueError("n_neighbors must be between 2 and n_sequences - 1.")
    if not 0 <= min_dist <= 1:
        raise ValueError("min_dist must be between 0 and 1.")
    if np.all(matrix == matrix[0]):
        raise ValueError("UMAP requires variation between embedding rows.")
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("UMAP requires: pip install 'genescope-fm[explore]'") from exc
    return UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        n_jobs=1,
        init="random",
    ).fit_transform(matrix)
