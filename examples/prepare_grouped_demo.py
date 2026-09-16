"""Prepare an offline synthetic grouped-split smoke test, not biological evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from genescope.splits import prepare_group_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/grouped-demo"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory.")
    rng = np.random.default_rng(314)
    rows = []
    for group in range(12):
        for member in range(8):
            label = member % 2
            probabilities = [0.35, 0.15, 0.15, 0.35] if label == 0 else [0.15, 0.35, 0.35, 0.15]
            rows.append(
                {
                    "sequence_id": f"synthetic_{group:02}_{member:02}",
                    "sequence": "".join(rng.choice(list("ACGT"), 100, p=probabilities)),
                    "label": label,
                    "group": f"synthetic_group_{group:02}",
                }
            )
    mapping = pd.DataFrame(
        {
            "group": [f"synthetic_group_{i:02}" for i in range(12)],
            "split": ["train"] * 6 + ["validation"] * 3 + ["test"] * 3,
        }
    )
    with TemporaryDirectory() as temporary:
        source, assignments = Path(temporary) / "input.csv", Path(temporary) / "assignments.csv"
        pd.DataFrame(rows).to_csv(source, index=False)
        mapping.to_csv(assignments, index=False)
        path = prepare_group_splits(
            source,
            assignments,
            args.output,
            group_kind="other",
            group_source="Synthetic software demonstration, NumPy seed 314; groups are arbitrary and have no biological meaning.",
        )
        # Preserve the exact source tables alongside the prepared outputs.
        (args.output / "input.csv").write_bytes(source.read_bytes())
        (args.output / "assignments.csv").write_bytes(assignments.read_bytes())
    print(f"Prepared synthetic grouped demonstration: {path}")
    print("This verifies software plumbing, not chromosome or homology generalization.")


if __name__ == "__main__":
    main()
