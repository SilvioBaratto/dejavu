"""Load the fixed question set, with an optional path override."""

from __future__ import annotations

import importlib.resources
import json
import pathlib

QUESTION_COUNT = 10


def load_questions(path: str | pathlib.Path | None = None) -> list[str]:
    """Return the ordered question list; reads the bundled JSON when *path* is None."""
    if path is None:
        return _read_bundled()
    return _read_override(path)


def _read_bundled() -> list[str]:
    resource = importlib.resources.files("dejavu").joinpath("data/questions.json")
    return _validate(json.loads(resource.read_text(encoding="utf-8")))


def _read_override(path: str | pathlib.Path) -> list[str]:
    return _validate(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def _validate(questions: object) -> list[str]:
    if not isinstance(questions, list):
        raise ValueError(
            f"questions file must contain a JSON array, got {type(questions).__name__}"
        )
    if len(questions) != QUESTION_COUNT:
        raise ValueError(
            f"expected exactly {QUESTION_COUNT} questions, got {len(questions)}"
        )
    for i, item in enumerate(questions):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"question {i} must be a non-empty string, got {item!r}")
    return questions
