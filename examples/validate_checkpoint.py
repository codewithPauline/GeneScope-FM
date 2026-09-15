"""Validate the real pinned NT checkpoint on toy DNA; not a biological benchmark."""

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

from genescope.embeddings import save_embeddings
from genescope.io import SequenceRecord
from genescope.models import get_model
from genescope.provenance import load_provenance
from genescope.report import explore_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-remote-code",
        action="store_true",
        help="Explicitly allow the pinned checkpoint's custom model code.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("checkpoint_validation"))
    args = parser.parse_args()
    if not args.allow_remote_code:
        parser.error("Review the pinned checkpoint, then specify --allow-remote-code.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory.")
    records = [
        SequenceRecord("toy_short", "ACGTACGTACGT"),
        SequenceRecord("toy_ambiguous", "ACGTNNNNACGTACGTACGT"),
        SequenceRecord("toy_long", "TGCA" * 30),
        SequenceRecord("toy_single_base", "A"),
    ]
    sequences = [record.sequence for record in records]
    started = time.perf_counter()
    backend = get_model(
        "nucleotide-transformer",
        trust_remote_code=True,
        device="cpu",
        batch_size=4,
        max_length=128,
        local_files_only=args.local_files_only,
    )
    batched = backend.embed(sequences)
    details = backend.provenance()
    for record, stats in zip(records, details["sequence_stats"], strict=True):
        stats["sequence_id"] = record.identifier
    assert batched.shape == (4, 512), batched.shape
    assert np.isfinite(batched).all()
    assert np.all(np.linalg.norm(batched, axis=1) > 0)
    assert details["resolved_revision"] == details["requested_revision"]
    repeat = backend.embed(sequences)
    single = np.vstack([backend.embed([sequence]) for sequence in sequences])
    np.testing.assert_allclose(batched, repeat, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(batched, single, rtol=1e-4, atol=1e-5)

    # For unambiguous DNA, seven 6-mers + CLS fill an eight-token budget.
    long_dna = "ACGT" * 90
    backend.max_length = 8
    try:
        backend.embed([long_dna])
    except ValueError as exc:
        assert "exceed max_length" in str(exc)
    else:
        raise AssertionError("Over-length DNA was silently accepted.")
    backend.length_policy = "truncate"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        truncated = backend.embed([long_dna])
    truncation_stats = backend.sequence_stats[0]
    assert any("Truncating" in str(w.message) for w in caught)
    assert truncation_stats["truncated"] and truncation_stats["retained_tokens"] == 8
    np.testing.assert_allclose(truncated, backend.embed([long_dna[:42]]), rtol=1e-4, atol=1e-5)

    args.output.mkdir(parents=True, exist_ok=True)
    fasta = "".join(f">{r.identifier}\n{r.sequence}\n" for r in records)
    (args.output / "sequences.fasta").write_text(fasta, encoding="utf-8")
    details["input_fasta_sha256"] = hashlib.sha256(fasta.encode()).hexdigest()
    csv = save_embeddings(records, batched, args.output / "embeddings.csv", provenance=details)
    report = explore_embeddings(
        csv,
        args.output / "report",
        neighbors=2,
        description="Real pinned foundation-model inference on four toy "
        "DNA sequences; an integration check, not a biological benchmark.",
    )
    result = {
        "status": "passed",
        "purpose": "checkpoint integration; no biological accuracy claim",
        "checks": [
            "real safetensors checkpoint load",
            "four finite 512-dimensional vectors",
            "repeat inference",
            "batch-size and padding invariance",
            "strict over-length rejection",
            "explicit truncation equals retained prefix",
            "content-linked export",
            "HTML exploration report",
        ],
        "repeat_max_absolute_difference": float(np.max(np.abs(batched - repeat))),
        "batch_max_absolute_difference": float(np.max(np.abs(batched - single))),
        "comparison_tolerances": {"rtol": 1e-4, "atol": 1e-5},
        "truncation_example": truncation_stats,
        "elapsed_seconds_including_loading": time.perf_counter() - started,
        "provenance": load_provenance(csv),
    }
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "shape": list(batched.shape),
                "batch_max_absolute_difference": result["batch_max_absolute_difference"],
                "report": str(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
