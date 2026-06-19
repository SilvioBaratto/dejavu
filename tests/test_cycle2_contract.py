"""
Source-blind tests for Issue #5 — bundled contract + load_contract hook.

Authored exclusively from acceptance criteria; no implementation source was
read.  Every test is in the Red phase: it must FAIL before the implementation
exists and PASS only once the criterion is genuinely satisfied.
"""

import os
import pathlib
import re
import tempfile

import pytest
from hypothesis import given, settings, strategies as st

from dejavu.contract import load_contract


# ── Criterion: byte-identical across calls ────────────────────────────────────


def test_when_load_contract_called_twice_then_strings_are_equal():
    assert load_contract() == load_contract()


def test_when_load_contract_called_twice_then_utf8_encodings_are_equal():
    assert load_contract().encode("utf-8") == load_contract().encode("utf-8")


# ── Criterion: token count exceeds the 1 024-token cache minimum ─────────────


def test_when_contract_is_tokenized_then_count_exceeds_floor():
    """Must be well above 1 024; assert a comfortable floor of 5 000 tokens."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(load_contract()))
    assert token_count > 5_000, (
        f"Token count {token_count} is below the 5 000-token floor required "
        "for reliable prompt-cache hits."
    )


# ── Criterion: contract covers all required sections ─────────────────────────

REQUIRED_SECTION_KEYWORDS = [
    # parties & recitals
    "parties",
    "recital",
    # definitions
    "definitions",
    # scope of services
    "scope",
    # fees & payment terms
    "fees",
    "payment",
    # term & renewal
    "renewal",
    # termination
    "termination",
    # confidentiality
    "confidentiality",
    # IP & license grants
    "license",
    # warranties
    "warrant",
    # limitation of liability
    "limitation of liability",
    # indemnification
    "indemnif",
    # governing law
    "governing law",
    # notices
    "notice",
]


@pytest.mark.parametrize("keyword", REQUIRED_SECTION_KEYWORDS)
def test_when_contract_is_loaded_then_required_section_keyword_is_present(keyword):
    assert keyword in load_contract().lower(), (
        f"Required section keyword '{keyword}' is missing from the bundled contract."
    )


def test_when_contract_is_loaded_then_explicit_dollar_cap_is_present():
    """Limitation-of-liability clause must state a concrete dollar-amount cap."""
    cap_re = re.compile(r"\$[\d,]+(?:\.\d+)?")
    assert cap_re.search(load_contract()), (
        "No explicit dollar-amount liability cap found in the bundled contract. "
        "Issue #6 questions require a single verifiable answer."
    )


def test_when_contract_is_loaded_then_named_parties_are_present():
    """Parties must be concrete named entities, not generic labels like 'Party A'."""
    entity_re = re.compile(
        r"[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)* (?:Inc\.|LLC|Ltd\.|Corp\.|Co\.)",
        re.MULTILINE,
    )
    assert entity_re.search(load_contract()), (
        "No concrete named parties (e.g. 'Acme Corp.') found in the bundled contract."
    )


def test_when_contract_is_loaded_then_specific_date_is_present():
    """Contract must cite at least one concrete date for verifiable answers."""
    month_names = (
        "January|February|March|April|May|June|"
        "July|August|September|October|November|December"
    )
    date_re = re.compile(
        rf"(?:{month_names})\s+\d{{1,2}},\s+\d{{4}}"
        r"|\d{4}-\d{2}-\d{2}"
    )
    assert date_re.search(load_contract()), (
        "No specific date found in the bundled contract."
    )


# ── Criterion: load_contract(path) reads a valid non-empty file ───────────────


def test_when_load_contract_given_valid_path_then_returns_file_content():
    body = "Service Agreement\n\nThis is a test clause with substance."
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(body)
        path = fh.name
    try:
        result = load_contract(path)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == body
    finally:
        os.unlink(path)


# ── Criterion: load_contract(path) raises a clear error on missing file ───────


def test_when_load_contract_given_missing_path_then_raises_error():
    with pytest.raises(Exception):
        load_contract("/tmp/__no_such_dejavu_contract_xyz_99999__.txt")


# ── Criterion: load_contract(path) raises a clear error on empty file ─────────


def test_when_load_contract_given_empty_file_then_raises_error():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("")
        path = fh.name
    try:
        with pytest.raises(Exception):
            load_contract(path)
    finally:
        os.unlink(path)


# ── Criterion: contract.py stays well under the size guideline ────────────────


def test_when_contract_module_is_inspected_then_line_count_is_under_guideline():
    """
    The agreement text must live in the .txt data file, not in contract.py.
    Guideline: module well under 500 lines.
    """
    import dejavu.contract as _mod

    source = pathlib.Path(_mod.__file__).read_text(encoding="utf-8")
    line_count = len(source.splitlines())
    assert line_count < 500, (
        f"dejavu/contract.py has {line_count} lines; "
        "embed the agreement text in dejavu/data/contract.txt instead."
    )


# ── Property-based tests ──────────────────────────────────────────────────────


@given(st.text(min_size=1).filter(lambda s: s.strip() != ""))
@settings(max_examples=25)
def test_when_load_contract_given_any_non_empty_text_file_then_no_error_is_raised(
    content,
):
    """
    Never-raises-for-valid-input invariant: load_contract(path) must accept any
    file with non-whitespace content, regardless of that content.  A whitespace-
    only file is treated as empty and is covered by the empty-file test, matching
    the criterion 'reads and validates the given file (non-empty)'.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(content)
        path = fh.name
    try:
        load_contract(path)  # must not raise
    finally:
        os.unlink(path)
