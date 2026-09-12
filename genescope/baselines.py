"""Conventional DNA features for offline exploration and future model comparisons."""

from itertools import product

import numpy as np

from .io import normalize_sequence


def kmer_frequencies(sequences: list[str], k: int = 3) -> np.ndarray:
    """Overlapping, strand-specific A/C/G/T k-mers in lexicographic order.

    Windows containing N are excluded from both counts and denominator. Each row
    sums to one. A sequence without any valid windows raises rather than yielding
    an unusable zero vector. This is a baseline, not a foundation model.
    """
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 6:
        raise ValueError("k must be an integer between 1 and 6.")
    if not sequences:
        raise ValueError("At least one sequence is required.")
    vocabulary = {"".join(word): index for index, word in enumerate(product("ACGT", repeat=k))}
    matrix = np.zeros((len(sequences), len(vocabulary)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        sequence = normalize_sequence(sequence)
        for start in range(len(sequence) - k + 1):
            word = sequence[start : start + k]
            if "N" not in word:
                matrix[row, vocabulary[word]] += 1
        total = matrix[row].sum()
        if total == 0:
            raise ValueError(f"Sequence at row {row + 1} has no valid {k}-mer windows.")
        matrix[row] /= total
    return matrix
