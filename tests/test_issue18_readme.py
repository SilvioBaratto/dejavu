"""
Source-blind example tests for issue #18 — criterion 1.

Verifiable criterion:
  README.md covers: the intern / 50-page-contract analogy (in English),
  what dejavu does (two panels: uncached vs rolling cache), install + .env
  setup, CLI usage with all flags, a sample-run description, and the
  Collector-read vs raw-write attribution note.

These tests fail today (no README.md exists) and pass once the criterion is met.
"""

import pathlib


PROJECT_ROOT = pathlib.Path(__file__).parent.parent
README = PROJECT_ROOT / "README.md"


def _readme_text():
    return README.read_text(encoding="utf-8")


def _readme_lower():
    return _readme_text().lower()


def test_when_readme_is_present_then_file_exists_at_project_root():
    assert README.exists(), "README.md not found at project root"


def test_when_readme_is_read_then_intern_contract_analogy_is_present():
    """Criterion: intern / 50-page-contract analogy must appear in English."""
    text = _readme_lower()
    assert "intern" in text, "README must contain the intern analogy"
    assert "contract" in text, "README must mention the 50-page contract"


def test_when_readme_is_read_then_two_panel_description_is_present():
    """Criterion: dejavu does two panels — uncached vs rolling cache."""
    text = _readme_lower()
    assert "uncached" in text, "README must describe the uncached panel"
    assert "cache" in text, "README must describe the rolling-cache panel"
    assert "panel" in text, "README must reference the two-panel layout"


def test_when_readme_is_read_then_install_section_is_present():
    """Criterion: install + .env setup is documented."""
    text = _readme_lower()
    assert "install" in text, "README must include install instructions"
    assert ".env" in text, "README must cover .env setup"


def test_when_readme_is_read_then_all_cli_flags_are_documented():
    """Criterion: CLI usage with all flags is present."""
    text = _readme_text()
    required_flags = [
        "--model",
        "--doc",
        "--questions-file",
        "--ttl",
        "--max-tokens",
        "--delay",
    ]
    missing = [f for f in required_flags if f not in text]
    assert not missing, f"CLI flags missing from README: {missing}"


def test_when_readme_is_read_then_sample_run_description_is_present():
    """Criterion: a sample-run description is included."""
    text = _readme_lower()
    has_sample = any(kw in text for kw in ("sample", "example", "```"))
    assert has_sample, (
        "README must include a sample-run description "
        "(look for 'sample', 'example', or a fenced code block)"
    )


def test_when_readme_is_read_then_collector_attribution_note_is_present():
    """
    Criterion: Collector-read vs raw-write attribution note.

    Cache reads come from the BAML Collector (cached_input_tokens);
    cache writes come from the raw Anthropic HTTP response
    (cache_creation_input_tokens).  Both sources must be mentioned.
    """
    text = _readme_lower()
    assert "collector" in text, (
        "README must attribute cache-read tokens to the BAML Collector"
    )
    has_raw_write_ref = any(
        kw in text
        for kw in (
            "raw",
            "cache_creation",
            "http_response",
            "cache_creation_input_tokens",
        )
    )
    assert has_raw_write_ref, (
        "README must explain that cache-write tokens come from the raw Anthropic "
        "usage (cache_creation_input_tokens / http_response)"
    )
