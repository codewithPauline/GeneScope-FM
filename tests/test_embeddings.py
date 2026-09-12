import numpy as np

from genescope.embeddings import embeddings_to_frame
from genescope.io import SequenceRecord


def test_embeddings_to_frame():
    records = [
        SequenceRecord("seq1", "ACGT"),
        SequenceRecord("seq2", "AACCGG"),
    ]
    matrix = np.array([[0.1, 0.2], [0.3, 0.4]])

    frame = embeddings_to_frame(records, matrix)

    assert list(frame.columns) == [
        "sequence_id",
        "sequence_length",
        "embedding_0000",
        "embedding_0001",
    ]
    assert frame["sequence_id"].tolist() == ["seq1", "seq2"]
    assert frame["sequence_length"].tolist() == [4, 6]
