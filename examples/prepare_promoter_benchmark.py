"""Download pinned public data and prepare a small, overlap-screened promoter pilot.

No source DNA is bundled with GeneScope. Run from an installed checkout with the
benchmark extra. The official test partition is never used for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from genescope.benchmark import audit_splits, canonical_sequence, overlap_words, sequence_hash
from genescope.io import normalize_sequence

REPOSITORY = "katarinagresova/Genomic_Benchmarks_human_nontata_promoters"
REVISION = "e0003669df0a180c8570ebe091ed91f20fa080ec"
FILES = {
    "train": (
        "train-00000-of-00001-aeb55728756a44ae.parquet",
        "75e934078d6b880464c6a60f2e5ffd5b894204a11f32aac8c9607adcae625afa",
    ),
    "test": (
        "test-00000-of-00001-d7614d9116c62f44.parquet",
        "baac2b635666547c56c6482d6c7dd53def995cdcc4acca367f980c82d15c9f39",
    ),
}


def fetch_source(cache: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    cache.mkdir(parents=True, exist_ok=True)
    frames, manifest = {}, {}
    for split, (filename, expected) in FILES.items():
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/data/{filename}"
        path = cache / filename
        if not path.exists():
            print(f"Downloading pinned {split} partition...", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Downloaded source hash mismatch: {filename}")
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Cached source hash mismatch: {filename}")
        frame = pd.read_parquet(path).rename(columns={"seq": "sequence"})
        if set(frame.columns) != {"sequence", "label"} or set(frame.label) != {0, 1}:
            raise ValueError("Unexpected source schema.")
        frame["sequence"] = frame.sequence.map(normalize_sequence)
        if not frame.sequence.str.len().eq(251).all():
            raise ValueError("Expected fixed 251-base source sequences.")
        frame["sequence_id"] = [f"gb_{split}_{row:06d}" for row in range(len(frame))]
        frames[split] = frame
        manifest[split] = {
            "url": url,
            "sha256": expected,
            "n": len(frame),
            "class_counts": {str(c): int(sum(frame.label == c)) for c in (0, 1)},
        }
    return frames, manifest


def select_pilot(
    frames, *, train_per_class=300, validation_per_class=100, test_per_class=100, seed=42
):
    """Deterministic class-balanced selection, without consulting prediction outcomes.

    Screen the entire official training pool against ALL official test sequences.
    Choose validation first, then exclude its exact windows from training. Exact
    duplicates (either strand) are also excluded within the selected sample.
    """
    if any(n < 2 for n in (train_per_class, validation_per_class, test_per_class)):
        raise ValueError("Each split needs at least two sequences per class.")

    def order(row):
        return sequence_hash(f"{seed}:{row['sequence_id']}")

    training = sorted(frames["train"].to_dict("records"), key=order)
    testing = sorted(frames["test"].to_dict("records"), key=order)
    blocked_words, blocked_sequences = set(), set()
    for row in testing:
        blocked_words.update(overlap_words(row["sequence"]))
        blocked_sequences.add(canonical_sequence(row["sequence"]))
    eligible, exclusions = [], {"official_train_matching_official_test": 0}
    for row in training:
        sequence = row["sequence"]
        if canonical_sequence(sequence) in blocked_sequences or not blocked_words.isdisjoint(
            overlap_words(sequence)
        ):
            exclusions["official_train_matching_official_test"] += 1
        else:
            eligible.append(row)
    used_sequences = set()

    def take(rows, n, split, forbidden_words=None):
        selected, counts, screened = [], {0: 0, 1: 0}, 0
        for row in rows:
            label, sequence = row["label"], row["sequence"]
            if counts[label] == n:
                continue
            canonical = canonical_sequence(sequence)
            if canonical in used_sequences:
                continue
            if forbidden_words is not None and not forbidden_words.isdisjoint(
                overlap_words(sequence)
            ):
                screened += 1
                continue
            selected.append({**row, "split": split})
            used_sequences.add(canonical)
            counts[label] += 1
            if all(count == n for count in counts.values()):
                break
        if any(count != n for count in counts.values()):
            raise ValueError(f"Insufficient overlap-screened {split} sequences: {counts}")
        return selected, screened

    test, _ = take(testing, test_per_class, "test")
    validation, _ = take(eligible, validation_per_class, "validation")
    validation_words = set()
    for row in validation:
        validation_words.update(overlap_words(row["sequence"]))
    train, screened = take(eligible, train_per_class, "train", validation_words)
    exclusions["training_candidates_matching_selected_validation"] = screened
    exclusions["eligible_official_training_rows"] = len(eligible)
    sample = pd.DataFrame(train + validation + test)[["sequence_id", "sequence", "label", "split"]]
    audit = audit_splits(sample)
    return sample, {"exclusions": exclusions, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("promoter_pilot"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/genescope-promoters"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=300)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory.")
    frames, files = fetch_source(args.cache)
    print(
        "Screening the complete official training pool against the official test partition...",
        flush=True,
    )
    sample, selection = select_pilot(
        frames,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )
    csv_content = sample.to_csv(index=False).encode("utf-8")
    fasta = "".join(f">{row.sequence_id}\n{row.sequence}\n" for row in sample.itertuples())
    manifest = {
        "schema_version": 1,
        "dataset": REPOSITORY,
        "revision": REVISION,
        "source": "https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks",
        "citation": "Gresova et al. (2023), Genomic benchmarks: a collection of datasets for genomic sequence classification. BMC Genomic Data 24, 25.",
        "label_mapping": {"0": "negative", "1": "positive (non-TATA promoter)"},
        "files": files,
        "selection": {
            "seed": args.seed,
            "order": "ascending SHA256 of seed:sequence_id; IDs use pinned parquet row indices",
            "per_class": {
                "train": args.train_per_class,
                "validation": args.validation_per_class,
                "test": args.test_per_class,
            },
            "policy": "Official test retained for testing. All official test sequences exclude matching training rows by canonical full sequence or shared exact 50-base window. Validation sampled before training; its windows exclude matching training candidates. Selected canonical sequences unique. Balanced class quotas fixed before inference.",
            **selection,
        },
        "dataset_csv_sha256": hashlib.sha256(csv_content).hexdigest(),
        "fasta_sha256": hashlib.sha256(fasta.encode("ascii")).hexdigest(),
        "license_note": "The pinned Hugging Face card supplies no dataset license declaration. Raw source sequences are downloaded by the user and are not redistributed in this repository. The upstream code repository's Apache-2.0 license does not establish a separate dataset license.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        (staging / "dataset.csv").write_bytes(csv_content)
        (staging / "sequences.fasta").write_text(fasta, encoding="ascii")
        (staging / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(args.output)
    print(f"Prepared {len(sample)} sequences in {args.output}", flush=True)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
