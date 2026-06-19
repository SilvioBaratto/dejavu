"""
Source-blind example tests for Issue #13:
  feat: declare runtime dependencies and add dejavu console-script entry point

Derived solely from acceptance criteria and requirements.md.
No implementation source was read; assumed non-existent at authoring time (Red phase of TDD).

Criteria tested:
  AC-1  [UNIT] [project].dependencies includes baml-py, typer, rich, anthropic, python-dotenv
  AC-2  [UNIT] [project.scripts] defines dejavu = "dejavu.cli:app"
  AC-3  [UNIT] All existing packaging / pytest / ruff / pyright sections are unchanged
  AC-5  [UNIT] pip install -e . (or a build) succeeds — verified via importlib.metadata

Skipped:
  AC-4  A new test asserts the five deps are declared and the console-script entry exists
        — satisfied by the existence of this very file (this IS that test)
  AC-6  All tests pass — boilerplate suite gate; no per-criterion assertion
  AC-7  SOLID, clean code — subjective prose; not runtime-verifiable

Design choices (where spec is ambiguous):
  - Dependency names are matched case-insensitively with hyphen/underscore normalisation,
    since PEP 503 treats baml-py and baml_py as equivalent.  The criteria text uses the
    canonical hyphenated names; the helper `_dep_includes` normalises both sides.
  - Version specifiers (>=, ==, etc.) and extras ([...]) are stripped before comparison
    so pinned or bounded declarations still satisfy the criteria.
  - The console-script entry-point test reads from importlib.metadata, which is only
    populated after `pip install -e .`.  This is the minimal proof that AC-5 holds.
  - pytest.ini_options are re-verified here even though test_cycle2_package_scaffold.py
    may cover subsets: the issue requires ALL existing sections to be unchanged, so an
    independent check of asyncio_mode and the integration marker is warranted.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _toml() -> dict:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _dep_includes(deps: list[str], name: str) -> bool:
    """Return True if *name* (normalised) appears as the package name of any dep."""
    needle = name.lower().replace("-", "_")
    for dep in deps:
        raw_name = re.split(r"[>=<!;\[\s]", dep)[0].strip()
        if raw_name.lower().replace("-", "_") == needle:
            return True
    return False


# ---------------------------------------------------------------------------
# AC-1: [project].dependencies includes all five runtime packages
# ---------------------------------------------------------------------------


class TestDependenciesDeclared:
    """AC-1: every required runtime package must appear in [project].dependencies."""

    def _deps(self) -> list[str]:
        return _toml()["project"]["dependencies"]

    def test_when_pyproject_is_parsed_then_baml_py_is_in_dependencies(self):
        assert _dep_includes(self._deps(), "baml-py"), (
            "'baml-py' (or 'baml_py') not found in [project].dependencies"
        )

    def test_when_pyproject_is_parsed_then_typer_is_in_dependencies(self):
        assert _dep_includes(self._deps(), "typer"), (
            "'typer' not found in [project].dependencies"
        )

    def test_when_pyproject_is_parsed_then_rich_is_in_dependencies(self):
        assert _dep_includes(self._deps(), "rich"), (
            "'rich' not found in [project].dependencies"
        )

    def test_when_pyproject_is_parsed_then_anthropic_is_in_dependencies(self):
        assert _dep_includes(self._deps(), "anthropic"), (
            "'anthropic' not found in [project].dependencies"
        )

    def test_when_pyproject_is_parsed_then_python_dotenv_is_in_dependencies(self):
        assert _dep_includes(self._deps(), "python-dotenv"), (
            "'python-dotenv' (or 'python_dotenv') not found in [project].dependencies"
        )

    def test_when_pyproject_is_parsed_then_dependencies_list_is_not_empty(self):
        deps = self._deps()
        assert isinstance(deps, list) and len(deps) > 0, (
            "[project].dependencies must be a non-empty list"
        )


# ---------------------------------------------------------------------------
# AC-2: [project.scripts] defines dejavu = "dejavu.cli:app"
# ---------------------------------------------------------------------------


class TestConsoleScriptEntry:
    """AC-2: [project.scripts] must declare the dejavu entrypoint."""

    def _scripts(self) -> dict:
        return _toml()["project"]["scripts"]

    def test_when_pyproject_is_parsed_then_project_scripts_table_exists(self):
        data = _toml()["project"]
        assert "scripts" in data, (
            "[project.scripts] table is missing from pyproject.toml"
        )

    def test_when_pyproject_is_parsed_then_dejavu_key_is_in_scripts(self):
        assert "dejavu" in self._scripts(), (
            "'dejavu' key not found in [project.scripts]"
        )

    def test_when_pyproject_is_parsed_then_dejavu_script_value_is_dejavu_cli_app(self):
        value = self._scripts()["dejavu"]
        assert value == "dejavu.cli:app", (
            f"[project.scripts].dejavu must equal 'dejavu.cli:app', got {value!r}"
        )


# ---------------------------------------------------------------------------
# AC-3: All pre-existing packaging / pytest / ruff / pyright sections unchanged
# ---------------------------------------------------------------------------


class TestExistingPackagingSectionsUnchanged:
    """AC-3 (packaging): setuptools packages and package-data globs must be preserved."""

    def test_when_pyproject_is_parsed_then_packages_find_include_still_has_dejavu_star(
        self,
    ):
        include: list[str] = _toml()["tool"]["setuptools"]["packages"]["find"][
            "include"
        ]
        assert "dejavu*" in include, (
            f"'dejavu*' missing from [tool.setuptools.packages.find].include: {include}"
        )

    def test_when_pyproject_is_parsed_then_packages_find_include_still_has_baml_client_star(
        self,
    ):
        include: list[str] = _toml()["tool"]["setuptools"]["packages"]["find"][
            "include"
        ]
        assert "baml_client*" in include, (
            f"'baml_client*' missing from [tool.setuptools.packages.find].include: {include}"
        )

    def test_when_pyproject_is_parsed_then_package_data_dejavu_still_declares_txt_glob(
        self,
    ):
        globs: list[str] = _toml()["tool"]["setuptools"]["package-data"]["dejavu"]
        assert "data/*.txt" in globs, (
            f"'data/*.txt' missing from [tool.setuptools.package-data].dejavu: {globs}"
        )

    def test_when_pyproject_is_parsed_then_package_data_dejavu_still_declares_json_glob(
        self,
    ):
        globs: list[str] = _toml()["tool"]["setuptools"]["package-data"]["dejavu"]
        assert "data/*.json" in globs, (
            f"'data/*.json' missing from [tool.setuptools.package-data].dejavu: {globs}"
        )


class TestExistingLinterSectionsUnchanged:
    """AC-3 (linters): ruff and pyright exclusions for baml_client/ must be preserved."""

    def test_when_pyproject_is_parsed_then_ruff_still_excludes_baml_client(self):
        excludes: list[str] = _toml()["tool"]["ruff"].get("exclude", [])
        assert "baml_client/" in excludes, (
            f"'baml_client/' missing from [tool.ruff].exclude: {excludes}"
        )

    def test_when_pyproject_is_parsed_then_pyright_still_excludes_baml_client(self):
        excludes: list[str] = _toml()["tool"]["pyright"].get("exclude", [])
        assert "baml_client/" in excludes, (
            f"'baml_client/' missing from [tool.pyright].exclude: {excludes}"
        )


class TestExistingPytestSectionUnchanged:
    """AC-3 (pytest): asyncio_mode and the integration marker must still be present."""

    def _pytest_opts(self) -> dict:
        return _toml()["tool"]["pytest"]["ini_options"]

    def test_when_pyproject_is_parsed_then_pytest_asyncio_mode_is_still_auto(self):
        mode = self._pytest_opts().get("asyncio_mode")
        assert mode == "auto", (
            f"[tool.pytest.ini_options].asyncio_mode must be 'auto', got {mode!r}"
        )

    def test_when_pyproject_is_parsed_then_pytest_integration_marker_is_still_declared(
        self,
    ):
        markers: list[str] = self._pytest_opts().get("markers", [])
        has_integration = any(m.startswith("integration") for m in markers)
        assert has_integration, (
            f"'integration' marker missing from [tool.pytest.ini_options].markers: {markers}"
        )


# ---------------------------------------------------------------------------
# AC-5: pip install -e . (or a build) succeeds
# Verified by asking importlib.metadata for the installed package + entry-point.
# ---------------------------------------------------------------------------


class TestPackageInstallable:
    """AC-5: the package must be discoverable by importlib.metadata after pip install -e ."""

    def test_when_metadata_is_queried_then_dejavu_package_is_found(self):
        from importlib.metadata import metadata, PackageNotFoundError

        try:
            m = metadata("dejavu")
        except PackageNotFoundError:
            raise AssertionError(
                "Package 'dejavu' is not installed. Run `pip install -e .` first."
            )
        assert m["Name"].lower() == "dejavu", (
            f"Installed package name mismatch: {m['Name']!r}"
        )

    def test_when_entry_points_are_queried_then_dejavu_console_script_is_registered(
        self,
    ):
        from importlib.metadata import entry_points

        eps = entry_points(group="console_scripts")
        dejavu_ep = next((ep for ep in eps if ep.name == "dejavu"), None)
        assert dejavu_ep is not None, (
            "'dejavu' console-script entry point not found. "
            "Ensure [project.scripts] is declared and `pip install -e .` has been run."
        )

    def test_when_dejavu_console_script_is_registered_then_it_points_to_dejavu_cli_app(
        self,
    ):
        from importlib.metadata import entry_points

        eps = entry_points(group="console_scripts")
        dejavu_ep = next((ep for ep in eps if ep.name == "dejavu"), None)
        assert dejavu_ep is not None, "'dejavu' entry point not registered"
        assert dejavu_ep.value == "dejavu.cli:app", (
            f"'dejavu' console script must point to 'dejavu.cli:app', got {dejavu_ep.value!r}"
        )
