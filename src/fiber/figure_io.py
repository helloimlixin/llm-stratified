"""Shared figure saving helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def companion_pdf_path(path: str | Path) -> Path:
    """Return the vector PDF companion path for a saved figure."""
    return Path(path).with_suffix(".pdf")


def save_figure(fig: Any, out_path: str | Path, *, dpi: int = 200, **savefig_kwargs: Any) -> Path:
    """Save a Matplotlib figure and a same-stem PDF companion.

    The requested path is preserved, usually a PNG used by W&B and quick
    previews.  When that path is not already a PDF, we also save a vector PDF
    next to it so paper figures keep crisp text, axes, and colorbars when
    zoomed.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, **savefig_kwargs)
    if path.suffix.lower() != ".pdf":
        fig.savefig(companion_pdf_path(path), dpi=dpi, format="pdf", **savefig_kwargs)
    return path
