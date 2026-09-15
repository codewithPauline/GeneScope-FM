"""Content-linked provenance for exported sequence representations."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from . import __version__


def provenance_path(csv_path: str | Path) -> Path:
    return Path(str(csv_path) + ".provenance.json")


def build_provenance(csv_bytes: bytes, shape: tuple[int, int], details: dict) -> dict:
    """Link user-supplied generation details to the exact exported CSV bytes."""
    return {
        "schema_version": 1,
        "genescope_version": __version__,
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "n_sequences": int(shape[0]),
        "n_features": int(shape[1]),
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
        },
        "generation": details,
    }


def load_provenance(csv_path: str | Path) -> dict | None:
    """Validate the CSV hash before using a sidecar; missing provenance is allowed.

    This checks file consistency, not authenticity of the generation claims.
    """
    sidecar = provenance_path(csv_path)
    if not sidecar.exists():
        return None
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid provenance JSON: {sidecar.name}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported provenance schema; expected schema_version=1.")
    if not isinstance(manifest.get("generation"), dict):
        raise ValueError("Provenance must contain a generation object.")
    actual = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    if manifest.get("csv_sha256") != actual:
        raise ValueError(
            "Provenance hash does not match the embedding CSV. "
            "The file was changed or paired with the wrong sidecar."
        )
    return manifest
