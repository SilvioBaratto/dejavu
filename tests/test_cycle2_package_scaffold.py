"""
Tests for issue #4: scaffold dejavu package and enable data-file bundling.

Criteria tested (per oracle report):
  AC-1  [UNIT]  import dejavu succeeds and resolves to dejavu/__init__.py
  AC-2  [UNIT]  packages.find.include lists dejavu* alongside baml_client*
  AC-3  [UNIT]  package-data declares dejavu data/*.txt and data/*.json globs
  AC-4  [UNIT]  baml_client/ excludes in [tool.ruff] and [tool.pyright] are unchanged

Skipped (not verifiable per oracle):
  All tests pass        — boilerplate suite gate; no per-criterion assertion
  SOLID, clean code     — subjective prose; no concrete runtime assertion

No property-based tests: none of the AC-1..AC-4 criteria imply an invariant over a
range of inputs — they are structural/configuration point-checks, not functions of
arbitrary input data.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _toml() -> dict:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


# ── AC-1  import dejavu succeeds and resolves to dejavu/__init__.py ──────────


class TestDejavuImport:
    def test_when_dejavu_is_imported_then_no_exception_is_raised(self):
        import dejavu  # noqa: F401

    def test_when_dejavu_spec_is_resolved_then_origin_ends_with_init_py(self):
        spec = importlib.util.find_spec("dejavu")
        assert spec is not None, "importlib could not locate the dejavu package"
        assert spec.origin is not None, "dejavu spec has no origin (namespace package?)"
        assert spec.origin.endswith("__init__.py"), (
            f"dejavu resolves to {spec.origin!r}, expected a path ending '__init__.py'"
        )


# ── AC-2  packages.find.include ──────────────────────────────────────────────


class TestPackagesFindInclude:
    def test_when_pyproject_is_parsed_then_include_contains_dejavu_star(self):
        include: list[str] = _toml()["tool"]["setuptools"]["packages"]["find"][
            "include"
        ]
        assert "dejavu*" in include, (
            f"'dejavu*' not found in [tool.setuptools.packages.find].include: {include}"
        )

    def test_when_pyproject_is_parsed_then_include_still_contains_baml_client_star(
        self,
    ):
        include: list[str] = _toml()["tool"]["setuptools"]["packages"]["find"][
            "include"
        ]
        assert "baml_client*" in include, (
            "'baml_client*' missing from "
            f"[tool.setuptools.packages.find].include: {include}"
        )

    def test_when_pyproject_is_parsed_then_both_globs_coexist_in_include(self):
        include: list[str] = _toml()["tool"]["setuptools"]["packages"]["find"][
            "include"
        ]
        assert "dejavu*" in include and "baml_client*" in include, (
            "Expected both 'dejavu*' and 'baml_client*' in include list, "
            f"got: {include}"
        )


# ── AC-3  [tool.setuptools.package-data] globs ───────────────────────────────


class TestPackageData:
    def test_when_pyproject_is_parsed_then_package_data_table_declares_dejavu_key(self):
        pkg_data: dict = _toml()["tool"]["setuptools"]["package-data"]
        assert "dejavu" in pkg_data, (
            "'dejavu' key missing from "
            f"[tool.setuptools.package-data]: {list(pkg_data)}"
        )

    def test_when_pyproject_is_parsed_then_dejavu_package_data_includes_txt_glob(self):
        globs: list[str] = _toml()["tool"]["setuptools"]["package-data"]["dejavu"]
        assert "data/*.txt" in globs, (
            f"'data/*.txt' not declared in dejavu package-data globs: {globs}"
        )

    def test_when_pyproject_is_parsed_then_dejavu_package_data_includes_json_glob(self):
        globs: list[str] = _toml()["tool"]["setuptools"]["package-data"]["dejavu"]
        assert "data/*.json" in globs, (
            f"'data/*.json' not declared in dejavu package-data globs: {globs}"
        )


# ── AC-4  baml_client/ excludes unchanged in both [tool.ruff] and [tool.pyright]


class TestLinterExcludes:
    def test_when_pyproject_is_parsed_then_ruff_still_excludes_baml_client(self):
        excludes: list[str] = _toml()["tool"]["ruff"].get("exclude", [])
        assert "baml_client/" in excludes, (
            f"baml_client/ missing from [tool.ruff].exclude: {excludes}"
        )

    def test_when_pyproject_is_parsed_then_pyright_still_excludes_baml_client(self):
        excludes: list[str] = _toml()["tool"]["pyright"].get("exclude", [])
        assert "baml_client/" in excludes, (
            f"baml_client/ missing from [tool.pyright].exclude: {excludes}"
        )
