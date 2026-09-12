"""Build the README's vector preview from the committed analysis outputs."""

import json
from html import escape
from pathlib import Path

import pandas as pd

from genescope.report import PALETTE

root = Path(__file__).resolve().parents[1]
demo = root / "docs" / "demo"
summary = json.loads((demo / "summary.json").read_text())
projection = (
    (demo / "projection.svg")
    .read_text()
    .replace("<svg xmlns=", '<svg x="34" y="310" width="1032" height="540" xmlns=', 1)
)
labels = pd.read_csv(demo / "projection.csv")["label"].drop_duplicates().tolist()
legend = "".join(
    f'<circle cx="{64 + i * 210}" cy="868" r="5" fill="{PALETTE[i % len(PALETTE)]}"/>'
    f'<text x="{78 + i * 210}" y="873" font-size="14" fill="#475569">{escape(label)}</text>'
    for i, label in enumerate(labels)
)
values = [
    (str(summary["diagnostics"]["n_sequences"]), "synthetic DNA sequences"),
    (str(summary["diagnostics"]["n_features"]), "3-mer frequency features"),
    (f"{sum(summary['pca_explained_variance_ratio']):.1%}", "variance in two PCs"),
]
cards = "".join(
    f'<rect x="{44 + i * 340}" y="218" width="326" height="84" rx="12" fill="white"/>'
    f'<text x="{66 + i * 340}" y="254" font-size="30" font-weight="700">{value}</text>'
    f'<text x="{66 + i * 340}" y="280" font-size="13" fill="#526673">{label}</text>'
    for i, (value, label) in enumerate(values)
)
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 944"
role="img" aria-label="GeneScope synthetic k-mer analysis preview">
<rect width="1100" height="944" rx="20" fill="#edf2f5"/>
<g font-family="DejaVu Sans, sans-serif" fill="#162e3b">
<text x="46" y="52" font-size="13" fill="#0f766e" letter-spacing="3">GENESCOPE-FM / EXPLORE</text>
<text x="44" y="111" font-size="43" font-weight="700">See the structure in DNA.</text>
<text x="46" y="154" font-size="17" fill="#526673">Reproducible projections. Full-space neighbors. Portable reports.</text>
<text x="46" y="186" font-size="14" fill="#526673">Synthetic demonstration · Conventional k-mer baseline · No foundation model used</text>
{cards}{projection}{legend}
<text x="46" y="919" font-size="12" fill="#526673">Pauline Owusu-Ansah · AI for biology · MIT licensed</text>
</g></svg>"""
destination = root / "docs" / "assets" / "explorer-demo.svg"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(svg, encoding="utf-8")
