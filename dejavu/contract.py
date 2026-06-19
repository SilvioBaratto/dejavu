"""Load the bundled service-agreement text, with an optional path override."""

from __future__ import annotations

import importlib.resources
import pathlib


def load_contract(path: str | None = None) -> str:
    """Return contract text; reads the bundled .txt file when *path* is None."""
    if path is None:
        return _read_bundled()
    return _read_override(path)


def _read_bundled() -> str:
    resource = importlib.resources.files("dejavu").joinpath("data/contract.txt")
    return resource.read_text(encoding="utf-8")


def _read_override(path: str) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Contract file not found: {path}")
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Contract file is empty: {path}")
    return text
