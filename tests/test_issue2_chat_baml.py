"""
Tests for issue #2: ChatUncached and ChatCached BAML functions with
rolling cache_control.

All prompt-rendering tests call `b.request.*` — no API tokens are spent.

Criteria tested (per oracle report):
  AC-1  [UNIT]  Message class with role/content string fields; history is Message[]
  AC-2  [UNIT]  ChatUncached renders with NO cache_control metadata anywhere
  AC-3  [UNIT]  ChatCached renders with a rolling ephemeral cache_control on history
  AC-4  [UNIT]  Both functions pin the SAME client (Sonnet46)
  AC-5  [T3]    Prompts are byte-identical except for the cache_control kwarg
  AC-6  [UNIT]  Empty history does not crash either function

Skipped (not runtime-verifiable):
  AC-7  All tests pass              — boilerplate suite gate; no per-criterion assertion
  AC-8  SOLID / clean code          — subjective prose; no concrete runtime assertion
"""

from __future__ import annotations

import dataclasses
import json

from hypothesis import given, settings, strategies as st

from baml_client import b
from baml_client.types import Message


# ──────────────────────────────────────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────────────────────────────────────

_DOC = (
    "SERVICE AGREEMENT\n"
    "This Agreement is entered into between Client Corp ('Client') and "
    "Provider Inc ('Provider').\n"
    "Payment terms: net-30. Termination: 30-day written notice. "
    "Liability cap: $100 000."
)
_Q = "What are the payment terms stated in the agreement?"

# Two-turn history: user message is the "last history message" ChatCached should mark.
_H2 = [
    Message(role="user", content="What parties are named in this agreement?"),
    Message(role="assistant", content="Client Corp and Provider Inc."),
]

# Empty history: the rolling marker guard must survive this.
_H0: list[Message] = []


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers — source-blind; handles dataclasses and Pydantic v1/v2
# ──────────────────────────────────────────────────────────────────────────────


def _as_dict(obj: object) -> object:
    """Recursively convert obj to plain dicts/lists for comparison and serialisation."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump()  # type: ignore
    if hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()  # type: ignore
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(i) for i in obj]
    if hasattr(obj, "__dict__"):
        return {k: _as_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def _serialise(obj: object) -> str:
    return json.dumps(_as_dict(obj), default=str)


def _body(req: object) -> dict:
    """Return the JSON body dict from a baml_py HTTPRequest."""
    # HTTPRequest exposes .body.json() as the parsed request body dict.
    return req.body.json()  # type: ignore[union-attr]


def _has_cache_control(req: object) -> bool:
    # Detect a real cache_control *key* in the request structure, not a literal
    # "cache_control" substring (a question whose text is "cache_control" must
    # not register as a marker). Stripping the key changes the body iff present.
    body = _body(req)
    return _strip_cache_control(body) != body


def _strip_cache_control(obj: object) -> object:
    """Return a deep copy of obj with every 'cache_control' key removed."""
    if isinstance(obj, dict):
        return {
            k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"
        }
    if isinstance(obj, list):
        return [_strip_cache_control(i) for i in obj]
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# AC-1  Message class — role: string, content: string; history is Message[]
# ──────────────────────────────────────────────────────────────────────────────


class TestMessageClass:
    def test_when_message_is_constructed_then_role_field_is_a_string(self):
        msg = Message(role="user", content="hello")
        assert isinstance(msg.role, str)
        assert msg.role == "user"

    def test_when_message_is_constructed_then_content_field_is_a_string(self):
        msg = Message(role="assistant", content="world")
        assert isinstance(msg.content, str)
        assert msg.content == "world"

    def test_when_history_is_a_list_of_messages_then_all_elements_are_message_instances(
        self,
    ):
        history: list[Message] = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
        ]
        assert all(isinstance(m, Message) for m in history)


# ──────────────────────────────────────────────────────────────────────────────
# AC-2  ChatUncached — NO cache_control anywhere in the rendered request
# ──────────────────────────────────────────────────────────────────────────────


class TestChatUncachedNoCacheControl:
    def test_when_rendered_with_non_empty_history_then_no_cache_control_is_present(
        self,
    ):
        req = b.request.ChatUncached(document=_DOC, history=_H2, question=_Q)
        assert not _has_cache_control(req)

    def test_when_rendered_with_empty_history_then_no_cache_control_is_present(self):
        req = b.request.ChatUncached(document=_DOC, history=_H0, question=_Q)
        assert not _has_cache_control(req)


# ──────────────────────────────────────────────────────────────────────────────
# AC-3  ChatCached — rolling ephemeral cache_control on the last history message
# ──────────────────────────────────────────────────────────────────────────────


class TestChatCachedHasRollingCacheControl:
    def test_when_rendered_with_non_empty_history_then_cache_control_is_present(self):
        req = b.request.ChatCached(document=_DOC, history=_H2, question=_Q)
        assert _has_cache_control(req)

    def test_when_rendered_with_non_empty_history_then_cache_control_type_is_ephemeral(
        self,
    ):
        req = b.request.ChatCached(document=_DOC, history=_H2, question=_Q)
        assert "ephemeral" in _serialise(req)

    def test_when_rendered_with_empty_history_then_no_exception_is_raised(self):
        """Rolling marker guard: {% if history %} prevents crash on empty."""
        b.request.ChatCached(document=_DOC, history=_H0, question=_Q)


# ──────────────────────────────────────────────────────────────────────────────
# AC-4  Both functions pin the SAME client (Sonnet46 from issue #1)
# ──────────────────────────────────────────────────────────────────────────────


class TestBothFunctionsPinSameClient:
    def test_when_both_functions_are_rendered_then_they_reference_the_same_client(self):
        # HTTPRequest has no .client attribute; the client is reflected by
        # the target model in the request body. Both must be identical.
        req_u = b.request.ChatUncached(document=_DOC, history=_H2, question=_Q)
        req_c = b.request.ChatCached(document=_DOC, history=_H2, question=_Q)
        assert _body(req_u)["model"] == _body(req_c)["model"]

    def test_when_chat_uncached_is_rendered_then_client_is_sonnet46(self):
        # Sonnet46 is configured with model "claude-sonnet-4-6".
        req = b.request.ChatUncached(document=_DOC, history=_H2, question=_Q)
        assert _body(req)["model"] == "claude-sonnet-4-6"

    def test_when_chat_cached_is_rendered_then_client_is_sonnet46(self):
        req = b.request.ChatCached(document=_DOC, history=_H2, question=_Q)
        assert _body(req)["model"] == "claude-sonnet-4-6"


# ──────────────────────────────────────────────────────────────────────────────
# AC-5  [T3]  Prompts are byte-identical except for the cache_control kwarg
# ──────────────────────────────────────────────────────────────────────────────


class TestPromptsIdenticalExceptCacheControl:
    def test_when_same_inputs_are_given_then_requests_are_equal_after_stripping_cache_control(  # noqa: E501
        self,
    ):
        req_u = b.request.ChatUncached(document=_DOC, history=_H2, question=_Q)
        req_c = b.request.ChatCached(document=_DOC, history=_H2, question=_Q)
        # Compare the HTTP body dicts with every cache_control key removed.
        stripped_u = _strip_cache_control(_body(req_u))
        stripped_c = _strip_cache_control(_body(req_c))
        assert stripped_u == stripped_c


# ──────────────────────────────────────────────────────────────────────────────
# AC-6  Empty history guard (each function tested independently)
# ──────────────────────────────────────────────────────────────────────────────


class TestEmptyHistoryGuard:
    def test_when_chat_uncached_receives_empty_history_then_no_exception_is_raised(
        self,
    ):
        b.request.ChatUncached(document=_DOC, history=_H0, question=_Q)

    def test_when_chat_cached_receives_empty_history_then_no_exception_is_raised(self):
        b.request.ChatCached(document=_DOC, history=_H0, question=_Q)


# ──────────────────────────────────────────────────────────────────────────────
# Hypothesis — property-based tests
#
# AC-2 implies the invariant: ChatUncached NEVER produces cache_control for any input.
# AC-3 / AC-6 together imply: ChatCached accepts any valid history without raising.
# AC-2 / AC-6 together imply: ChatUncached accepts any valid history without raising.
# These are "never-raises-for-valid-input" and "invariant-holds-for-all-inputs" shapes.
# ──────────────────────────────────────────────────────────────────────────────

_message_st = st.builds(
    Message,
    role=st.sampled_from(["user", "assistant"]),
    content=st.text(min_size=1, max_size=200),
)
_history_st = st.lists(_message_st, min_size=0, max_size=6)
_question_st = st.text(min_size=1, max_size=200)


@given(history=_history_st, question=_question_st)
@settings(max_examples=30)
def test_when_any_valid_history_is_given_then_chat_uncached_does_not_raise(
    history: list[Message], question: str
) -> None:
    """Never-raises invariant: ChatUncached accepts any (history, question)."""
    b.request.ChatUncached(document=_DOC, history=history, question=question)


@given(history=_history_st, question=_question_st)
@settings(max_examples=30)
def test_when_any_valid_history_is_given_then_chat_cached_does_not_raise(
    history: list[Message], question: str
) -> None:
    """Never-raises invariant: ChatCached accepts any valid (history, question) pair."""
    b.request.ChatCached(document=_DOC, history=history, question=question)


@given(history=_history_st, question=_question_st)
@settings(max_examples=30)
def test_when_any_valid_inputs_are_given_then_chat_uncached_never_has_cache_control(
    history: list[Message], question: str
) -> None:
    """Invariant: ChatUncached produces no cache_control marker for any inputs."""
    req = b.request.ChatUncached(document=_DOC, history=history, question=question)
    assert not _has_cache_control(req)
