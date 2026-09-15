"""Portable exploration reports with no browser-side dependencies or network calls."""

from __future__ import annotations

import hashlib
import json
import platform
from html import escape
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .embeddings import load_embeddings
from .explore import (
    cosine_similarity,
    nearest_neighbors,
    project_pca,
    project_umap,
    representation_diagnostics,
)
from .provenance import load_provenance

PALETTE = ["#0f766e", "#d97706", "#6366f1", "#db2777", "#2563eb", "#64748b"]


def _labels(path: Path | None, ids: list[str]) -> list[str]:
    if path is None:
        return ["Unlabeled"] * len(ids)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"sequence_id", "label"}.issubset(frame.columns):
        raise ValueError("Metadata CSV requires sequence_id and label columns.")
    if frame.sequence_id.duplicated().any() or frame.sequence_id.str.strip().eq("").any():
        raise ValueError("Metadata sequence IDs must be nonempty and unique.")
    frame = frame.set_index("sequence_id")
    missing = set(ids) - set(frame.index)
    if missing:
        raise ValueError(f"Metadata is missing {len(missing)} sequence ID(s).")
    labels = frame.loc[ids, "label"].tolist()
    if any(not label.strip() for label in labels):
        raise ValueError("Metadata labels must be nonempty.")
    return labels


def _scatter_svg(
    coordinates: np.ndarray, ids: list[str], labels: list[str], axes: list[str]
) -> str:
    groups = list(dict.fromkeys(labels))
    colors = {label: PALETTE[index % len(PALETTE)] for index, label in enumerate(groups)}
    xy = np.zeros((len(ids), 2))
    xy[:, : min(2, coordinates.shape[1])] = coordinates[:, :2]
    low, high = xy.min(axis=0), xy.max(axis=0)
    span = high - low
    position = np.full_like(xy, 0.5)
    np.divide(xy - low, span, out=position, where=span != 0)
    position = position * [650, -280] + [135, 365]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 460" role="img" '
        'aria-label="Sequence representation projection">',
        '<rect width="880" height="460" rx="18" fill="#f8fafc"/>',
        '<text x="48" y="38" font-family="sans-serif" font-size="17" fill="#0f172a">'
        "Sequence representation space</text>",
        '<path d="M115 70V385H815" fill="none" stroke="#94a3b8"/>',
    ]
    for dimension in range(2):
        ticks = (
            np.linspace(low[dimension], high[dimension], 5) if span[dimension] else [low[dimension]]
        )
        for tick in ticks:
            fraction = (tick - low[dimension]) / span[dimension] if span[dimension] else 0.5
            if dimension == 0:
                x, y = 135 + 650 * fraction, 402
                anchor = "middle"
            else:
                x, y = 106, 369 - 280 * fraction
                anchor = "end"
            svg.append(
                f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
                f'font-family="sans-serif" font-size="10" fill="#64748b">{tick:.3g}</text>'
            )
    for index, (x, y) in enumerate(position):
        title = escape(
            f"{ids[index]} | {labels[index]} | x={xy[index, 0]:.5g}, y={xy[index, 1]:.5g}"
        )
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" '
            f'fill="{colors[labels[index]]}" fill-opacity="0.8" '
            f'stroke="white" stroke-width="1.5"><title>{title}</title></circle>'
        )
    svg.extend(
        [
            f'<text x="440" y="422" text-anchor="middle" font-family="sans-serif" '
            f'font-size="13" fill="#475569">{escape(axes[0])}</text>',
            f'<text transform="translate(24 230) rotate(-90)" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" fill="#475569">{escape(axes[1])}</text>',
            "</svg>",
        ]
    )
    return "".join(svg)


def explore_embeddings(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    metadata: str | Path | None = None,
    neighbors: int = 5,
    method: str = "pca",
    random_state: int = 42,
    umap_neighbors: int = 15,
    min_dist: float = 0.1,
    overwrite: bool = False,
    description: str | None = None,
) -> Path:
    """Write coordinates, similarities, neighbors, diagnostics, and an HTML report.

    Labels are joined by ID and only color the plot; they never influence the
    projection. All neighbor calculations use the original representation.
    """
    input_path, output_dir = Path(input_path), Path(output_dir)
    metadata_path = Path(metadata) if metadata is not None else None
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        if not overwrite:
            raise ValueError("Output directory is not empty; choose a new path or use --overwrite.")
    if method not in {"pca", "umap"}:
        raise ValueError("Projection method must be pca or umap.")
    frame = load_embeddings(input_path)
    provenance = load_provenance(input_path)
    ids = frame.sequence_id.tolist()
    features = frame.filter(regex=r"^embedding_\d+$").to_numpy(dtype=float)
    if len(ids) < 2:
        raise ValueError("Exploration requires at least two sequences.")
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1.")
    labels = _labels(metadata_path, ids)
    # Validate the dense-size bound and cosine norms before expensive decomposition.
    similarity = cosine_similarity(features)
    k = min(neighbors, len(ids) - 1)
    neighbor_table = nearest_neighbors(features, ids, k)
    pca = project_pca(features, min(2, len(ids) - 1, features.shape[1]))
    ratios = pca.explained_variance_ratio.tolist()
    pca_columns = [f"PC{index + 1}" for index in range(len(ratios))]
    pca_frame = pd.DataFrame(pca.coordinates, columns=pca_columns)
    pca_frame.insert(0, "sequence_id", ids)
    if method == "pca":
        coordinates = pca.coordinates
        axes = [f"PC{index + 1} ({ratio:.1%} variance)" for index, ratio in enumerate(ratios)]
        if len(axes) == 1:
            axes.append("No second component (displayed at zero)")
        projection_frame = pca_frame
    else:
        coordinates = project_umap(
            features, n_neighbors=umap_neighbors, min_dist=min_dist, random_state=random_state
        )
        axes = ["UMAP 1", "UMAP 2"]
        projection_frame = pd.DataFrame(coordinates, columns=["UMAP1", "UMAP2"])
        projection_frame.insert(0, "sequence_id", ids)
    summary = {
        "genescope_version": __version__,
        "input_file": input_path.name,
        "embedding_provenance": provenance,
        "description": description,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        if metadata_path
        else None,
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
        },
        "projection": method,
        "pca_centered": True,
        "pca_feature_scaling": False,
        "pca_explained_variance_ratio": ratios,
        "similarity_metric": "cosine",
        "neighbor_space": "original embedding dimensions",
        "neighbors_requested": neighbors,
        "neighbors_returned_per_sequence": k,
        "neighbor_tie_break": "input row order",
        "labels_used_for_fit": False,
        "diagnostics": representation_diagnostics(features),
        "interpretation": "Exploratory representation geometry; not proof of biological "
        "function, ancestry, evolutionary distance, or predictive performance. "
        "Generation metadata, when present, is linked to the CSV by its content hash; "
        "it is self-reported and not independently authenticated.",
    }
    if method == "umap":
        summary["umap"] = {
            "random_state": random_state,
            "n_neighbors": umap_neighbors,
            "min_dist": min_dist,
            "metric": "euclidean",
            "init": "random",
            "n_jobs": 1,
            "version": version("umap-learn"),
        }
    svg = _scatter_svg(coordinates, ids, labels, axes)
    generation = provenance["generation"] if provenance else {}
    representation = str(generation.get("kind", "external embeddings; provenance unavailable"))
    if generation.get("model_id"):
        representation += " · " + str(generation["model_id"])
    groups = list(dict.fromkeys(labels))
    legend = " ".join(
        f'<span style="--color:{PALETTE[i % len(PALETTE)]}">{escape(label)}</span>'
        for i, label in enumerate(groups)
    )
    table = neighbor_table.head(100).to_html(index=False, float_format=lambda x: f"{x:.4f}")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeneScope · Representation explorer</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#edf2f5;color:#162e3b;
font:16px/1.6 system-ui,sans-serif}} main{{max-width:1100px;margin:auto;padding:48px 24px}}
.eyebrow{{letter-spacing:.18em;font-size:12px;font-weight:750;color:#0f766e}}
h1{{font-size:clamp(32px,5vw,52px);letter-spacing:-.04em;line-height:1.12;margin:12px 0}}
h2{{font-size:22px}} .muted{{color:#526673}} .cards{{display:flex;gap:14px;flex-wrap:wrap;
margin:28px 0}} .card{{flex:1;min-width:180px;background:white;padding:20px;border-radius:14px}}
.card strong{{display:block;font-size:30px;line-height:1.25}} section{{background:white;
padding:26px;border-radius:18px;margin:20px 0}} svg{{display:block;width:100%;height:auto}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}} .legend span:before{{content:'';
display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--color);margin-right:7px}}
.scroll{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{text-align:left;padding:10px 12px;border-bottom:1px solid #e2e8f0}}
th{{background:#f8fafc}} code{{overflow-wrap:anywhere}} a{{color:#0f766e}}
</style></head><body><main>
<div class="eyebrow">GENESCOPE-FM / EXPLORE</div>
<h1>See the structure in<br>sequence representations.</h1>
<p class="muted">{escape(input_path.name)} · {method.upper()} projection ·
cosine neighbors in the original feature space</p>
<p>{escape(description or "")}</p>
<p class="muted">Representation: {escape(representation)}</p>
<div class="cards"><div class="card"><strong>{len(ids):,}</strong>DNA sequences</div>
<div class="card"><strong>{features.shape[1]:,}</strong>representation dimensions</div>
<div class="card"><strong>{sum(ratios):.1%}</strong>variance in {len(ratios)} PCA components</div>
<div class="card"><strong>{k}</strong>neighbors per sequence</div></div>
<section>{svg}<div class="legend">{legend}</div>
<p class="muted">Hover over a point to inspect its ID and coordinates. Labels color the plot;
they are never used to fit it. Two-dimensional proximity can distort the original geometry.</p>
</section><section><h2>Closest sequence representations</h2>
<p class="muted">Self matches are excluded. Ties follow input order. Showing the first
{min(100, len(neighbor_table))} of {len(neighbor_table):,} rows; all rows are in neighbors.csv.</p>
<div class="scroll">{table}</div></section>
<section><h2>Read the result in context</h2><p>{escape(summary["interpretation"])}</p>
<p>Input SHA-256: <code>{summary["input_sha256"]}</code></p>
<p class="muted">GeneScope-FM {__version__} · Created by Pauline Owusu-Ansah ·
See summary.json for settings, numerical diagnostics, and software versions.</p></section>
</main></body></html>"""
    payloads = {
        "pca.csv": pca_frame.to_csv(index=False),
        "projection.csv": projection_frame.assign(label=labels).to_csv(index=False),
        "cosine_similarity.csv": pd.DataFrame(similarity, index=ids, columns=ids).to_csv(
            index_label="sequence_id"
        ),
        "neighbors.csv": neighbor_table.to_csv(index=False),
        "summary.json": json.dumps(summary, indent=2, allow_nan=False) + "\n",
        "projection.svg": svg,
        "report.html": html,
    }
    for source in [input_path, metadata_path]:
        if source is not None and source.resolve() in {
            (output_dir / filename).resolve() for filename in payloads
        }:
            raise ValueError("Output would overwrite an input file; choose a different directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in payloads.items():
        (output_dir / filename).write_text(contents, encoding="utf-8")
    return output_dir / "report.html"
