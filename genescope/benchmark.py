"""Binary frozen-feature evaluation with explicit splits and sequence-overlap checks."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import warnings
from html import escape
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from . import __version__
from .baselines import kmer_frequencies
from .embeddings import load_embeddings
from .io import normalize_sequence
from .provenance import load_provenance

SPLITS = ("train", "validation", "test")
OVERLAP_BASES = 50
C_GRID = (0.01, 0.1, 1.0, 10.0)
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
LIMITATIONS = [
    "Exact/reverse-complement duplicates and shared exact 50-base windows are screened. "
    "This does not exclude approximate homology or genomic interval overlap.",
    "Absent chromosome/locus groups, this is not a chromosome-held-out evaluation.",
    "Foundation-model pretraining overlap with these genomes is not ruled out.",
    "Bootstrap intervals condition on this fitted model and test class counts, assume "
    "independent sequences, and exclude training, split-selection, and pretraining uncertainty.",
    "Brier score and log loss describe probabilities; no probability calibration is fitted.",
]


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def canonical_sequence(sequence: str) -> str:
    return min(sequence, sequence.translate(COMPLEMENT)[::-1])


def overlap_words(sequence: str, k: int = OVERLAP_BASES) -> set[str]:
    """Exact windows on either strand; N-containing windows do not count as matches."""
    return {
        canonical_sequence(word)
        for start in range(len(sequence) - k + 1)
        if "N" not in (word := sequence[start : start + k])
    }


def audit_splits(frame: pd.DataFrame) -> dict:
    """Reject repeated sequences, cross-split exact windows, and cross-split groups."""
    canonical = frame.sequence.map(canonical_sequence)
    if canonical.duplicated().any():
        raise ValueError("Duplicate or reverse-complement sequences found; deduplicate first.")
    words = {}
    for split in SPLITS:
        words[split] = set()
        for seq in frame.loc[frame.split == split, "sequence"]:
            words[split].update(overlap_words(seq))
    overlaps = {}
    for left, right in combinations(SPLITS, 2):
        count = len(words[left] & words[right])
        overlaps[f"{left}/{right}"] = count
        if count:
            raise ValueError(f"Shared exact {OVERLAP_BASES}-base windows across {left}/{right}.")
        if "group" in frame:
            groups = set(frame.loc[frame.split == left, "group"])
            if groups & set(frame.loc[frame.split == right, "group"]):
                raise ValueError(f"Group overlap across {left}/{right}.")
    return {
        "duplicate_or_reverse_complement_sequences": 0,
        "overlap_window_bases": OVERLAP_BASES,
        "shared_windows_across_splits": overlaps,
        "sequences_without_eligible_overlap_windows": sum(
            not overlap_words(s) for s in frame.sequence
        ),
        "group_disjoint": True if "group" in frame else None,
        "approximate_homology_checked": False,
    }


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read sequence_id, sequence, binary label, split, and optional group columns."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    required = {"sequence_id", "sequence", "label", "split"}
    if len(header) != len(set(header)) or not required <= set(header):
        raise ValueError("Dataset needs unique sequence_id, sequence, label, split columns.")
    if set(header) - required - {"group"}:
        raise ValueError("Only the optional group column may accompany the required columns.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty or frame.sequence_id.str.strip().eq("").any():
        raise ValueError("Dataset and sequence IDs must be nonempty.")
    if frame.sequence_id.duplicated().any():
        raise ValueError("Sequence IDs must be unique.")
    if set(frame.label) != {"0", "1"}:
        raise ValueError("Labels must be exactly 0 and 1; 1 is the positive class.")
    if set(frame.split) != set(SPLITS):
        raise ValueError("Splits must be exactly train, validation, and test.")
    if "group" in frame:
        if frame.group.str.strip().eq("").any():
            raise ValueError("Group values must be nonempty when provided.")
        if not frame.group.eq(frame.group.str.strip()).all():
            raise ValueError("Group values must not have surrounding whitespace.")
    frame["sequence"] = frame.sequence.map(normalize_sequence)
    frame["label"] = frame.label.astype(int)
    for split in SPLITS:
        counts = frame.loc[frame.split == split, "label"].value_counts()
        if set(counts.index) != {0, 1} or counts.min() < 2:
            raise ValueError(f"{split} needs at least two sequences of each class.")
    return frame


def align_embeddings(frame: pd.DataFrame, path: str | Path) -> tuple[np.ndarray, dict]:
    """Check CSV provenance and exact input sequences, then join features by ID."""
    vectors = load_embeddings(path)
    provenance = load_provenance(path)
    if not provenance or not isinstance(provenance.get("input_sequences"), list):
        raise ValueError("Benchmark requires sequence-linked provenance; re-export with GeneScope.")
    expected = dict(zip(frame.sequence_id, frame.sequence.map(sequence_hash), strict=True))
    manifest = provenance["input_sequences"]
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("sequence_id"), str)
        or not isinstance(row.get("sequence_sha256"), str)
        for row in manifest
    ):
        raise ValueError("Invalid input_sequences provenance.")
    actual = {row["sequence_id"]: row["sequence_sha256"] for row in manifest}
    if len(manifest) != len(expected) or actual != expected:
        raise ValueError("Embedding provenance does not match the dataset's sequence IDs and DNA.")
    if set(vectors.sequence_id) != set(frame.sequence_id):
        raise ValueError("Embedding and dataset sequence IDs must match exactly.")
    vectors = vectors.set_index("sequence_id").loc[frame.sequence_id]
    if "sequence_length" in vectors and not np.array_equal(
        vectors.sequence_length.to_numpy(dtype=int), frame.sequence.str.len().to_numpy()
    ):
        raise ValueError("Embedding sequence lengths do not match the dataset.")
    return vectors.filter(regex=r"^embedding_\d+$").to_numpy(), provenance


def fit_probe(features: np.ndarray, labels: np.ndarray, splits: np.ndarray):
    """Select C on validation AUROC; fit every scaler/classifier on train only.

    The selected training fit is retained (no train+validation refit). Returning it
    makes preprocessing and coefficients inspectable without reading test labels.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, validation = splits == "train", splits == "validation"
    candidates = []
    best_score, best_model, selected_c = -np.inf, None, None
    for c in C_GRID:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, solver="lbfgs", max_iter=2000, tol=1e-6),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(features[train], labels[train])
            except ConvergenceWarning as exc:
                raise ValueError(
                    "Logistic regression did not converge; benchmark aborted."
                ) from exc
        score = float(
            roc_auc_score(labels[validation], model.predict_proba(features[validation])[:, 1])
        )
        candidates.append({"C": c, "validation_auroc": score})
        if score > best_score:  # Ascending grid gives smaller C on an exact tie.
            best_score, best_model, selected_c = score, model, c
    return best_model, {"selected_C": selected_c, "candidates": candidates}


def binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    prediction = probability >= 0.5
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "brier_score": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
    }


def validate_test_groups(labels, groups) -> dict:
    """Require enough independent test units for a paired group bootstrap."""
    labels, groups = np.asarray(labels), np.asarray(groups, dtype=str)
    if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups):
        raise ValueError("Test groups must be a one-dimensional array aligned with labels.")
    if any(not group.strip() or group != group.strip() for group in groups):
        raise ValueError("Test groups must be nonempty with no surrounding whitespace.")
    counts = {str(c): int(len(np.unique(groups[labels == c]))) for c in (0, 1)}
    if min(counts.values()) < 2:
        raise ValueError(
            "Group bootstrap needs at least two distinct test groups containing each class. "
            "Add independent held-out groups; do not split a biological group to meet this rule."
        )
    return {"test_groups": int(len(np.unique(groups))), "groups_containing_class": counts}


def bootstrap_metrics(
    labels, probabilities, *, seed=42, repetitions=1000, groups=None, return_diagnostics=False
):
    """Paired percentile intervals, resampling whole groups when supplied.

    Otherwise use the original class-stratified sequence bootstrap. Group draws
    retain every member and multiplicity of sampled groups. Single-class draws
    cannot define AUROC and are discarded, with a bounded retry budget.
    """
    from sklearn.metrics import roc_auc_score

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer.")
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(labels) != {0, 1}:
        raise ValueError("Bootstrap labels must contain both binary classes.")
    probabilities = {name: np.asarray(p) for name, p in probabilities.items()}
    if not {"3mer", "frozen_embeddings"} <= probabilities.keys():
        raise ValueError("Paired bootstrap needs 3mer and frozen_embeddings probabilities.")
    for p in probabilities.values():
        if p.shape != labels.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Probabilities must be finite, in [0, 1], and aligned with labels.")
    diagnostics = {
        "method": "paired class-stratified percentile",
        "unit": "sequence",
        "repetitions": repetitions,
        "discarded_single_class_draws": 0,
    }
    members = None
    if groups is not None:
        diagnostics.update(validate_test_groups(labels, groups))
        groups = np.asarray(groups, dtype=str)
        members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
        diagnostics.update(method="paired whole-group percentile", unit="group")
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(labels == label) for label in (0, 1)]
    samples = {
        name: {metric: [] for metric in ("balanced_accuracy", "auroc")} for name in probabilities
    }
    accepted, attempted = 0, 0
    while accepted < repetitions:
        attempted += 1
        if attempted > repetitions * 20:
            raise ValueError(
                "Insufficient two-class group bootstrap draws; more test groups needed."
            )
        if members is None:
            rows = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in classes])
        else:
            selected = rng.integers(0, len(members), size=len(members))
            rows = np.concatenate([members[i] for i in selected])
        truth = labels[rows]
        if len(np.unique(truth)) != 2:
            diagnostics["discarded_single_class_draws"] += 1
            continue
        accepted += 1
        for name, probability in probabilities.items():
            p = probability[rows]
            accuracy = np.mean([np.mean((p[truth == c] >= 0.5) == c) for c in (0, 1)])
            samples[name]["balanced_accuracy"].append(float(accuracy))
            samples[name]["auroc"].append(float(roc_auc_score(truth, p)))
    intervals = {
        name: {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in metrics.items()}
        for name, metrics in samples.items()
    }
    differences = {
        key: np.quantile(
            np.array(samples["frozen_embeddings"][key]) - np.array(samples["3mer"][key]),
            [0.025, 0.975],
        ).tolist()
        for key in ("balanced_accuracy", "auroc")
    }
    if return_diagnostics:
        return intervals, differences, diagnostics
    return intervals, differences


def render_benchmark_svg(results: dict) -> str:
    names = {
        "gc_length": "GC + length",
        "3mer": "3-mer frequencies",
        "frozen_embeddings": "Frozen NT embeddings",
    }
    # Generic external embeddings should not inherit a model-specific label.
    if results["model_id"] != "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species":
        names["frozen_embeddings"] = "Frozen embeddings"
    unit = results["protocol"]["bootstrap"].get("unit", "sequence")
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430" viewBox="0 0 900 430" role="img" aria-labelledby="title desc">',
        '<title id="title">Held-out binary classification results</title>',
        '<desc id="desc">Scores on a zero to one axis. Error bars are 95 percent bootstrap intervals; the resampling unit is stated below.</desc>',
        '<rect width="900" height="430" rx="16" fill="#f5f8fc"/>',
        '<g font-family="Arial, sans-serif" fill="#14243a">',
        '<text x="28" y="38" font-size="24" font-weight="bold">GeneScope-FM · Held-out evaluation</text>',
        f'<text x="28" y="65" font-size="14">{results["splits"]["test"]["n"]} test sequences · threshold 0.5 · 95% {unit} bootstrap intervals</text>',
    ]
    colors = ("#0f766e", "#3569b4")
    for tick in np.linspace(0, 1, 6):
        x = 230 + 590 * tick
        svg.append(f'<path d="M{x:.1f} 105V340" stroke="#dbe3ed"/>')
        svg.append(
            f'<text x="{x:.1f}" y="362" text-anchor="middle" font-size="12">{tick:.1f}</text>'
        )
    for row, (name, result) in enumerate(results["representations"].items()):
        y = 122 + row * 80
        svg.append(f'<text x="28" y="{y + 20}" font-size="15">{escape(names[name])}</text>')
        for index, metric in enumerate(("balanced_accuracy", "auroc")):
            score = result["test"][metric]
            low, high = result["test_ci95"][metric]
            yy = y + index * 28
            xlo, xhi = 230 + 590 * low, 230 + 590 * high
            svg.extend(
                [
                    f'<rect x="230" y="{yy}" width="{590 * score:.2f}" height="18" rx="3" fill="{colors[index]}"/>',
                    f'<path d="M{xlo:.2f} {yy + 5}v8m0 -4H{xhi:.2f}m0 -4v8" stroke="#14243a" stroke-width="1.5"/>',
                    f'<text x="866" y="{yy + 14}" font-size="13" text-anchor="end">{score:.3f}</text>',
                ]
            )
    svg.extend(
        [
            '<rect x="230" y="390" width="14" height="14" fill="#0f766e"/><text x="251" y="402" font-size="13">Balanced accuracy</text>',
            '<rect x="440" y="390" width="14" height="14" fill="#3569b4"/><text x="461" y="402" font-size="13">AUROC</text>',
            "</g></svg>",
        ]
    )
    return "\n".join(svg)


def run_benchmark(
    dataset: str | Path,
    embeddings: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    description: str = "User-supplied binary sequence dataset.",
) -> Path:
    """Evaluate GC/length, 3-mers, and supplied frozen embeddings on identical splits."""
    try:
        import sklearn
    except ImportError as exc:
        raise ImportError('Install benchmark dependencies: pip install -e ".[benchmark]"') from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    output = Path(output)
    if output.exists():
        raise ValueError("Benchmark output already exists; choose a new directory.")
    frame = load_dataset(dataset)
    audit = audit_splits(frame)
    from .splits import load_split_provenance

    split_provenance = load_split_provenance(dataset, frame)
    test_groups = frame.loc[frame.split == "test", "group"].to_numpy() if "group" in frame else None
    if test_groups is not None:
        validate_test_groups(frame.loc[frame.split == "test", "label"].to_numpy(), test_groups)
    matrix, provenance = align_embeddings(frame, embeddings)
    sequences = frame.sequence.tolist()
    gc_length = np.array(
        [[(s.count("G") + s.count("C")) / max(1, len(s) - s.count("N")), len(s)] for s in sequences]
    )
    representations = {
        "gc_length": gc_length,
        "3mer": kmer_frequencies(sequences, 3),
        "frozen_embeddings": matrix,
    }
    labels, splits = frame.label.to_numpy(), frame.split.to_numpy()
    test = splits == "test"
    probabilities, evaluation = {}, {}
    for name, features in representations.items():
        model, selection = fit_probe(features, labels, splits)
        probability = model.predict_proba(features[test])[:, 1]
        probabilities[name] = probability
        evaluation[name] = {
            "n_features": features.shape[1],
            **selection,
            "test": binary_metrics(labels[test], probability),
        }
    intervals, differences, bootstrap = bootstrap_metrics(
        labels[test], probabilities, seed=seed, groups=test_groups, return_diagnostics=True
    )
    limitations = list(LIMITATIONS)
    if test_groups is not None:
        limitations[1] = (
            "Supplied groups are disjoint across splits; their biological annotations and "
            "independence are not verified by GeneScope. Group names alone do not establish "
            "chromosome or homology isolation."
        )
        limitations[3] = (
            "Whole-group bootstrap intervals assume independent test groups and condition on "
            "the fitted classifiers. Single-class draws are discarded because AUROC is undefined. "
            "Intervals exclude training, split-selection, and pretraining uncertainty."
        )
        if bootstrap["test_groups"] < 10 or min(bootstrap["groups_containing_class"].values()) < 10:
            limitations.append(
                "Fewer than ten independent test groups support at least one class; "
                "group-bootstrap intervals may be unstable. Additional sequences within "
                "the same groups do not replace independent groups."
            )
    for name, values in intervals.items():
        evaluation[name]["test_ci95"] = values
    results = {
        "schema_version": 2,
        "genescope_version": __version__,
        "description": description,
        "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "embedding_csv_sha256": provenance["csv_sha256"],
        "model_id": provenance["generation"].get("model_id"),
        "model_revision": provenance["generation"].get("resolved_revision")
        or provenance["generation"].get("requested_revision"),
        "splits": {
            split: {
                "n": int(sum(splits == split)),
                "groups": int(frame.loc[frame.split == split, "group"].nunique())
                if "group" in frame
                else None,
                "class_counts": {
                    str(c): int(sum((splits == split) & (labels == c))) for c in (0, 1)
                },
            }
            for split in SPLITS
        },
        "protocol": {
            "classifier": "L2 logistic regression; lbfgs; max_iter=2000; tol=1e-6",
            "preprocessing": "StandardScaler fitted on train only, separately for each representation",
            "selection": "Maximum validation AUROC; smaller C wins exact ties; no refit",
            "C_grid": list(C_GRID),
            "positive_label": 1,
            "threshold": 0.5,
            "bootstrap": {
                "seed": seed,
                **bootstrap,
                "confidence": 0.95,
            },
        },
        "audit": audit,
        "split_provenance": split_provenance,
        "representations": evaluation,
        "frozen_minus_3mer": {
            metric: {
                "difference": evaluation["frozen_embeddings"]["test"][metric]
                - evaluation["3mer"]["test"][metric],
                "ci95": differences[metric],
            }
            for metric in differences
        },
        "limitations": limitations,
        "environment": {
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
            "scipy": version("scipy"),
        },
    }
    manifest = frame.drop(columns="sequence").copy()
    manifest["sequence_length"] = frame.sequence.str.len()
    manifest["sequence_sha256"] = frame.sequence.map(sequence_hash)
    prediction_columns = ["sequence_id", "label"] + (["group"] if "group" in frame else [])
    predictions = frame.loc[test, prediction_columns].copy()
    group_rows = []
    if test_groups is not None:
        for group in sorted(set(test_groups)):
            mask = test_groups == group
            truth = labels[test][mask]
            for name, probability in probabilities.items():
                p = probability[mask]
                both_classes = len(np.unique(truth)) == 2
                metrics = binary_metrics(truth, p) if both_classes else {}
                group_rows.append(
                    {
                        "group": group,
                        "representation": name,
                        "n": int(mask.sum()),
                        "negative": int((truth == 0).sum()),
                        "positive": int((truth == 1).sum()),
                        "balanced_accuracy": metrics.get("balanced_accuracy"),
                        "auroc": metrics.get("auroc"),
                        "brier_score": float(np.mean((p - truth) ** 2)),
                        "status": "both_classes"
                        if both_classes
                        else "single_class_metrics_undefined",
                    }
                )
    for name, probability in probabilities.items():
        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_prediction"] = (probability >= 0.5).astype(int)
    svg = render_benchmark_svg(results)
    rows = "".join(
        f"<tr><th>{escape(name)}</th><td>{values['selected_C']}</td>"
        + "".join(
            f"<td>{values['test'][metric]:.4f}</td>"
            for metric in ("balanced_accuracy", "auroc", "average_precision", "mcc", "brier_score")
        )
        + "</tr>"
        for name, values in evaluation.items()
    )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>GeneScope-FM benchmark</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#14243a}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#0f766e}</style>"
        "<h1>GeneScope-FM · Binary benchmark</h1><p>"
        + escape(description)
        + "</p><p>Identical sequence splits, frozen features, and a shared logistic-regression protocol. "
        "C is selected by validation AUROC; the test set is evaluated after selection.</p>"
        + (
            f"<p>Uncertainty resamples <strong>{bootstrap['unit']}s</strong>. "
            + (f"Held-out groups: {bootstrap['test_groups']}. " if test_groups is not None else "")
            + "Point estimates pool test sequences; larger groups contribute more sequences.</p>"
        )
        + (
            f"<p>Declared group kind: {escape(split_provenance['group_kind'])}. "
            f"Source: {escape(split_provenance['group_source'])}. "
            "Annotations are supplied by the user and are not independently verified.</p>"
            if split_provenance
            else ""
        )
        + ('<p><a href="group_metrics.csv">Per-group diagnostics</a></p>' if group_rows else "")
        + svg
        + '<div style="overflow-x:auto"><table><tr><th>Representation</th><th>C</th><th>Balanced accuracy</th><th>AUROC</th><th>Average precision</th><th>MCC</th><th>Brier ↓</th></tr>'
        + rows
        + "</table></div><p>Higher is better except Brier score (lower is better). "
        "Scores apply to these test sequences and this sampling protocol.</p><h2>Interpretation limits</h2><ul>"
        + "".join(f"<li>{escape(item)}</li>" for item in limitations)
        + '</ul><p>Download <a href="results.json">full results</a>, <a href="predictions.csv">test predictions</a>, '
        '<a href="split_manifest.csv">sequence hashes and splits</a>, or <a href="embedding_provenance.json">embedding provenance</a>.</p>'
        "<details><summary>Protocol and overlap audit</summary><pre>"
        + escape(
            json.dumps(
                {"protocol": results["protocol"], "audit": audit, "splits": results["splits"]},
                indent=2,
            )
        )
        + "</pre></details></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        (staging / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (staging / "embedding_provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        manifest.to_csv(staging / "split_manifest.csv", index=False)
        predictions.to_csv(staging / "predictions.csv", index=False)
        if group_rows:
            pd.DataFrame(group_rows).to_csv(staging / "group_metrics.csv", index=False)
        if split_provenance:
            (staging / "split_provenance.json").write_text(
                json.dumps(split_provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        (staging / "metrics.svg").write_text(svg, encoding="utf-8")
        (staging / "report.html").write_text(page, encoding="utf-8")
        staging.rename(output)
    return output / "report.html"
