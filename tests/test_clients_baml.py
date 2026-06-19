"""
Source-blind tests for issue #1: Anthropic BAML client definitions.

Derived from acceptance criteria only. Parses baml_src/clients.baml
as text — no generated baml_client/ or Python implementation is imported.

No property-based tests: all three verifiable criteria describe specific
named configuration values, not invariants over a domain.
"""

import re
import pathlib


CLIENTS_BAML = pathlib.Path(__file__).parent.parent / "baml_src" / "clients.baml"


def _file_content() -> str:
    return CLIENTS_BAML.read_text(encoding="utf-8")


def _extract_client_block(content: str, client_name: str) -> str | None:
    """
    Return the full text of a client<llm> NAME { ... } block, handling
    nested braces.
    """
    pattern = rf"client<llm>\s+{re.escape(client_name)}\s*\{{"
    start_match = re.search(pattern, content)
    if not start_match:
        return None
    brace_start = content.index("{", start_match.start())
    depth = 0
    for i in range(brace_start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[start_match.start() : i + 1]
    return None


# ---------------------------------------------------------------------------
# Criterion: Three Anthropic clients (Opus48, Fable5, Mythos5) must exist
# ---------------------------------------------------------------------------


def test_when_clients_baml_is_read_then_opus48_client_block_is_defined():
    content = _file_content()
    assert _extract_client_block(content, "Opus48") is not None, (
        "client<llm> Opus48 block must exist in clients.baml"
    )


def test_when_clients_baml_is_read_then_fable5_client_block_is_defined():
    content = _file_content()
    assert _extract_client_block(content, "Fable5") is not None, (
        "client<llm> Fable5 block must exist in clients.baml"
    )


def test_when_clients_baml_is_read_then_mythos5_client_block_is_defined():
    content = _file_content()
    assert _extract_client_block(content, "Mythos5") is not None, (
        "client<llm> Mythos5 block must exist in clients.baml"
    )


def test_when_opus48_block_is_examined_then_model_is_claude_opus_4_8():
    block = _extract_client_block(_file_content(), "Opus48")
    assert block is not None, "client<llm> Opus48 block must exist"
    assert "claude-opus-4-8" in block, (
        "Opus48 client must reference model string 'claude-opus-4-8'"
    )


def test_when_opus48_block_is_examined_then_provider_is_anthropic():
    block = _extract_client_block(_file_content(), "Opus48")
    assert block is not None, "client<llm> Opus48 block must exist"
    assert "provider anthropic" in block, (
        "Opus48 client must declare 'provider anthropic'"
    )


def test_when_fable5_block_is_examined_then_provider_is_anthropic():
    block = _extract_client_block(_file_content(), "Fable5")
    assert block is not None, "client<llm> Fable5 block must exist"
    assert "provider anthropic" in block, (
        "Fable5 client must declare 'provider anthropic'"
    )


def test_when_mythos5_block_is_examined_then_provider_is_anthropic():
    block = _extract_client_block(_file_content(), "Mythos5")
    assert block is not None, "client<llm> Mythos5 block must exist"
    assert "provider anthropic" in block, (
        "Mythos5 client must declare 'provider anthropic'"
    )


# ---------------------------------------------------------------------------
# Criterion: allowed_role_metadata ["cache_control"] must be set on Opus48
# ---------------------------------------------------------------------------


def test_when_opus48_block_is_examined_then_allowed_role_metadata_is_declared():
    """
    Without allowed_role_metadata BAML silently strips cache_control from every
    message, making ChatCached indistinguishable from ChatUncached and breaking
    the entire cost-gap demo invisibly.
    """
    block = _extract_client_block(_file_content(), "Opus48")
    assert block is not None, "client<llm> Opus48 block must exist"
    assert "allowed_role_metadata" in block, "Opus48 must declare allowed_role_metadata"


def test_when_opus48_block_is_examined_then_allowed_role_metadata_contains_cache_control():  # noqa: E501
    """
    Chosen interpretation: allowed_role_metadata is a list literal whose
    content includes the string 'cache_control'.
    """
    block = _extract_client_block(_file_content(), "Opus48")
    assert block is not None, "client<llm> Opus48 block must exist"
    meta_match = re.search(r"allowed_role_metadata\s*\[([^\]]*)\]", block)
    assert meta_match is not None, (
        "allowed_role_metadata must be a bracket-list in the Opus48 block"
    )
    assert "cache_control" in meta_match.group(1), (
        'allowed_role_metadata list must include "cache_control"'
    )


# ---------------------------------------------------------------------------
# Criterion: Mythos5 is defined but NOT used as any default client
# ---------------------------------------------------------------------------


def test_when_clients_baml_is_read_then_mythos5_is_not_set_as_default_client():
    """
    Mythos5 is limited-availability and must not be the default target;
    Opus48 is the required default for all chat functions.
    """
    content = _file_content()
    assert not re.search(r"default_client\s+Mythos5", content), (
        "Mythos5 must not appear as a default_client"
    )


def test_when_clients_baml_is_read_then_mythos5_is_not_primary_strategy_in_any_client():
    """
    Mythos5 must not be the leading entry in any round-robin or fallback
    strategy list, which would make it the effective default for that client.
    """
    content = _file_content()
    assert not re.search(r"strategy\s*\[\s*Mythos5", content), (
        "Mythos5 must not be the first strategy in any round-robin or fallback client"
    )
