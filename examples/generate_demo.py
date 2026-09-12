"""Regenerate the synthetic, composition-based demonstration input (seed 42)."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/synthetic"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    fasta, labels = [], ["sequence_id,label"]
    groups = {
        "AT_rich": [0.4, 0.1, 0.1, 0.4],
        "GC_rich": [0.1, 0.4, 0.4, 0.1],
        "Balanced": [0.25, 0.25, 0.25, 0.25],
    }
    for group, probabilities in groups.items():
        for replicate in range(1, 9):
            identifier = f"synthetic_{group}_{replicate:02d}"
            sequence = "".join(rng.choice(list("ACGT"), size=240, p=probabilities))
            fasta.append(f">{identifier}\n{sequence}\n")
            labels.append(f"{identifier},{group}")
    (args.output / "sequences.fasta").write_text("".join(fasta), encoding="utf-8")
    (args.output / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
