import numpy as np
import pytest

from genescope.baselines import kmer_frequencies


def test_kmer_counts_overlap_and_skip_ambiguous_windows():
    matrix = kmer_frequencies(["AAAANAC", "AC"], k=2)
    # AA, AC, AG, AT, CA, ... in lexicographic order: three AA and one AC.
    np.testing.assert_allclose(matrix[0, :2], [0.75, 0.25])
    assert matrix[1, 1] == 1
    np.testing.assert_allclose(matrix.sum(axis=1), 1)


@pytest.mark.parametrize(
    "sequences,k", [(["NNNN"], 2), (["AC"], 3), ([], 2), (["AX"], 1), (["ACGT"], 0), (["ACGT"], 7)]
)
def test_unusable_kmer_input_rejected(sequences, k):
    with pytest.raises(ValueError):
        kmer_frequencies(sequences, k)
