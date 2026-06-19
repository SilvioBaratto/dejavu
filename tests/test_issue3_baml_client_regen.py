"""
Source-blind tests for issue #3: regenerate baml_client/ and verify live
Claude Opus 4.8 access.

All tests are derived solely from acceptance criteria and requirements.md.
No implementation source was read; no baml_client/ internals are assumed
beyond what the criteria explicitly state must exist.

Criteria tested (per oracle report):
  AC-1  [UNIT]  baml_client/ contains expected generated output
                (proxy for "baml-cli generate runs cleanly")
  AC-2  [UNIT]  b.ChatUncached and b.ChatCached are importable on b;
                types.Message is importable from baml_client.types
  AC-3  [UNIT]  Real call to b.ChatUncached against claude-opus-4-8
                returns a non-empty string
  AC-4  [UNIT]  Real call to b.ChatCached with one-turn history
                returns a non-empty string
  AC-5  [UNIT]  Both chat functions route to claude-opus-4-8, so
                Mythos5 unavailability cannot cause AC-3 or AC-4 to fail

Skipped (not runtime-verifiable per oracle):
  AC-6  All tests pass    — boilerplate suite gate; no per-criterion assertion
  AC-7  SOLID/clean code  — subjective prose; no concrete runtime assertion

No property-based tests:
  AC-3 and AC-4 imply a never-raises invariant over (document, history,
  question) triples, but exercising with Hypothesis would fire one real API
  call per generated example — prohibitively expensive and non-deterministic.
  The remaining criteria describe concrete named values, not invariants.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from baml_client import b
from baml_client import types
from baml_client.types import Message


# ---------------------------------------------------------------------------
# Shared test fixtures (inline, no fixtures file needed)
# ---------------------------------------------------------------------------

_BAML_CLIENT_DIR = pathlib.Path(__file__).parent.parent / "baml_client"

_DOC = (
    "SERVICE AGREEMENT\n"
    "Parties: Client Corp ('Client') and Provider Inc ('Provider').\n"
    "Payment: net-30. Term: 12 months. Termination: 30-day written notice.\n"
    "Liability cap: $50 000."
)
_Q = "What are the payment terms stated in this agreement?"

# One-turn history: one complete exchange (user + assistant).
# AC-4 requires "a one-turn history" be passed to ChatCached.
_H1 = [
    Message(role="user", content="Who are the parties to this agreement?"),
    Message(role="assistant", content="Client Corp and Provider Inc."),
]

_NO_API_KEY = not os.environ.get("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# AC-1  baml_client/ contains expected generated artefacts
#       Proxy assertion: if baml-cli generate ran cleanly these files must exist.
# ---------------------------------------------------------------------------


class TestBamlClientDirectoryIsPopulated:
    """AC-1: baml-cli generate regenerates baml_client/ cleanly."""

    def test_when_baml_client_dir_is_inspected_then_init_py_exists(self):
        assert (_BAML_CLIENT_DIR / "__init__.py").is_file(), (
            "baml_client/__init__.py must exist after baml-cli generate"
        )

    def test_when_baml_client_dir_is_inspected_then_sync_client_py_exists(self):
        assert (_BAML_CLIENT_DIR / "sync_client.py").is_file(), (
            "baml_client/sync_client.py must exist after baml-cli generate"
        )

    def test_when_baml_client_dir_is_inspected_then_async_client_py_exists(self):
        assert (_BAML_CLIENT_DIR / "async_client.py").is_file(), (
            "baml_client/async_client.py must exist after baml-cli generate"
        )

    def test_when_baml_client_dir_is_inspected_then_types_py_exists(self):
        assert (_BAML_CLIENT_DIR / "types.py").is_file(), (
            "baml_client/types.py must exist after baml-cli generate"
        )

    def test_when_sync_client_is_read_then_chat_uncached_symbol_is_present(self):
        """Generated sync client exposes ChatUncached as a callable."""
        content = (_BAML_CLIENT_DIR / "sync_client.py").read_text(encoding="utf-8")
        assert "ChatUncached" in content, (
            "sync_client.py must contain 'ChatUncached' (from chat.baml)"
        )

    def test_when_sync_client_is_read_then_chat_cached_symbol_is_present(self):
        """Generated sync client exposes ChatCached as a callable."""
        content = (_BAML_CLIENT_DIR / "sync_client.py").read_text(encoding="utf-8")
        assert "ChatCached" in content, (
            "sync_client.py must contain 'ChatCached' (from chat.baml)"
        )


# ---------------------------------------------------------------------------
# AC-2  b.ChatUncached, b.ChatCached, and types.Message are importable
# ---------------------------------------------------------------------------


class TestBamlClientImports:
    """AC-2: b exposes ChatUncached/ChatCached; types.Message exists."""

    def test_when_b_is_imported_then_chat_uncached_is_accessible(self):
        assert hasattr(b, "ChatUncached"), (
            "b.ChatUncached must be accessible after 'from baml_client import b'"
        )

    def test_when_b_is_imported_then_chat_cached_is_accessible(self):
        assert hasattr(b, "ChatCached"), (
            "b.ChatCached must be accessible after 'from baml_client import b'"
        )

    def test_when_types_is_imported_then_message_class_exists(self):
        assert hasattr(types, "Message"), (
            "types.Message must exist after 'from baml_client import types'"
        )

    def test_when_message_is_imported_directly_then_it_is_a_class(self):
        assert isinstance(Message, type), "baml_client.types.Message must be a class"

    def test_when_b_chat_uncached_is_inspected_then_it_is_callable(self):
        assert callable(b.ChatUncached), "b.ChatUncached must be callable"

    def test_when_b_chat_cached_is_inspected_then_it_is_callable(self):
        assert callable(b.ChatCached), "b.ChatCached must be callable"


# ---------------------------------------------------------------------------
# AC-3  Real call to b.ChatUncached returns successfully (non-empty string)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_NO_API_KEY, reason="ANTHROPIC_API_KEY not set — live call skipped")
def test_when_chat_uncached_is_called_live_then_a_non_empty_string_is_returned():
    """
    AC-3: b.ChatUncached against claude-opus-4-8 returns a response (HTTP 200 implies
    a parsed, non-empty string from BAML — BAML raises on non-2xx responses).

    Chosen interpretation: 'returns successfully (200)' is satisfied when BAML returns
    a non-empty string without raising an exception.
    """
    result = b.ChatUncached(document=_DOC, history=[], question=_Q)
    assert isinstance(result, str), "b.ChatUncached must return a str"
    assert len(result) > 0, "b.ChatUncached must return a non-empty string"


# ---------------------------------------------------------------------------
# AC-4  Real call to b.ChatCached with one-turn history returns successfully
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_NO_API_KEY, reason="ANTHROPIC_API_KEY not set — live call skipped")
def test_when_chat_cached_is_called_live_with_one_turn_history_then_a_non_empty_string_is_returned():  # noqa: E501
    """
    AC-4: b.ChatCached with one full exchange in history against claude-opus-4-8
    returns a response without raising.

    'One-turn history' is interpreted as one user message + one assistant message —
    the minimum history from which the rolling cache_control breakpoint is meaningful.
    """
    result = b.ChatCached(document=_DOC, history=_H1, question=_Q)
    assert isinstance(result, str), "b.ChatCached must return a str"
    assert len(result) > 0, "b.ChatCached must return a non-empty string"


# ---------------------------------------------------------------------------
# AC-5  Mythos5 unavailability does not fail this issue
#       Verified by confirming both functions route to claude-opus-4-8 (not Mythos5).
# ---------------------------------------------------------------------------


class TestMythos5UnavailabilityDoesNotBreakIssue:
    """
    AC-5: Mythos5 being uncallable must not cause this issue to fail.

    Chosen approach: assert that both chat functions dispatch to claude-opus-4-8
    by inspecting the dry-run request body via b.request.*. Because neither
    function routes through Mythos5, Mythos5 unavailability cannot affect AC-3 or AC-4.
    """

    def test_when_chat_uncached_request_is_inspected_then_model_is_claude_opus_4_8(
        self,
    ):
        req = b.request.ChatUncached(document=_DOC, history=[], question=_Q)
        body = req.body.json()
        assert body.get("model") == "claude-opus-4-8", (
            "ChatUncached must target claude-opus-4-8, not any Mythos5 model"
        )

    def test_when_chat_cached_request_is_inspected_then_model_is_claude_opus_4_8(self):
        req = b.request.ChatCached(document=_DOC, history=_H1, question=_Q)
        body = req.body.json()
        assert body.get("model") == "claude-opus-4-8", (
            "ChatCached must target claude-opus-4-8, not any Mythos5 model"
        )

    def test_when_chat_uncached_request_is_inspected_then_model_is_not_mythos5(self):
        req = b.request.ChatUncached(document=_DOC, history=[], question=_Q)
        body = req.body.json()
        model: str = body.get("model", "")
        assert "mythos" not in model.lower(), (
            "ChatUncached must not route through any Mythos5 model"
        )

    def test_when_chat_cached_request_is_inspected_then_model_is_not_mythos5(self):
        req = b.request.ChatCached(document=_DOC, history=_H1, question=_Q)
        body = req.body.json()
        model: str = body.get("model", "")
        assert "mythos" not in model.lower(), (
            "ChatCached must not route through any Mythos5 model"
        )
