"""
Source-blind tests for Issue #8: per-turn metrics parser.

All fixtures are derived solely from acceptance criteria and requirements.md.
No implementation source was read; no dejavu/metrics.py internals are assumed
beyond what the criteria explicitly state must exist.

Criteria tested (per oracle report):
  AC-1  [UNIT] TurnMetrics frozen dataclass with 5 fields + turn_metrics() callable
  AC-2  [UNIT] Reads + tokens from Collector.usage; latency from FunctionLog.timing
  AC-3  [UNIT] Cache writes from raw HTTP body, not typed usage
  AC-4  [UNIT] No cache write → cache_creation_input_tokens == 0 (non-exceptional path)
  AC-5  [UNIT] Degrades: missing http_response, empty calls, non-JSON, last=None
  AC-6  [LIVE] Real ChatCached: write turn cache_creation>0; read turn cached>0
  AC-7  [UNIT] dejavu re-exports TurnMetrics / turn_metrics; appends to __all__

Skipped (per oracle):
  AC-8  All tests pass    — boilerplate suite gate; no per-criterion assertion
  AC-9  SOLID/clean code  — subjective prose; not runtime-verifiable
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from hypothesis import given, strategies as st


# ---------------------------------------------------------------------------
# Stub helpers — inferred solely from spec, never from implementation source
#
# Collector interface (from spec):
#   .usage.input_tokens
#   .usage.output_tokens
#   .usage.cached_input_tokens
#   .last                              → FunctionLog | None
#   .last.timing.duration_ms
#   .last.calls                        → list
#   .last.calls[-1].http_response      → object | None
#   .last.calls[-1].http_response.body.json()  → dict with "usage" key
# ---------------------------------------------------------------------------


def _make_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def _make_body_ok(usage_dict: dict) -> SimpleNamespace:
    """Body whose .json() returns {'usage': usage_dict}."""
    captured = {"usage": usage_dict}
    return SimpleNamespace(json=lambda: captured)


def _make_body_invalid_json() -> SimpleNamespace:
    def _raise():
        raise ValueError("body is not JSON")

    return SimpleNamespace(json=_raise)


def _make_collector(
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens_raw: int = 0,
    duration_ms: float = 0.0,
    http_response: str = "ok",  # "ok" | "missing" | "invalid_json"
    calls: str = "one",  # "one" | "empty"
    last: str = "present",  # "present" | "none"
) -> SimpleNamespace:
    """
    Build a minimal Collector stub from spec-derived fields only.

    Choices rationale (from criterion text):
      - http_response="missing" → simulates missing http_response (AC-5)
      - calls="empty"           → simulates empty calls list (AC-5)
      - http_response="invalid_json" → simulates non-JSON body (AC-5)
      - last="none"             → simulates collector.last is None (AC-5)
    """
    usage = _make_usage(input_tokens, output_tokens, cached_input_tokens)

    if last == "none":
        return SimpleNamespace(usage=usage, last=None)

    timing = SimpleNamespace(duration_ms=duration_ms)

    if calls == "empty":
        function_log = SimpleNamespace(timing=timing, calls=[])
        return SimpleNamespace(usage=usage, last=function_log)

    if http_response == "missing":
        call = SimpleNamespace(http_response=None)
    elif http_response == "invalid_json":
        call = SimpleNamespace(
            http_response=SimpleNamespace(body=_make_body_invalid_json())
        )
    else:  # "ok"
        raw_usage = {"cache_creation_input_tokens": cache_creation_input_tokens_raw}
        call = SimpleNamespace(
            http_response=SimpleNamespace(body=_make_body_ok(raw_usage))
        )

    function_log = SimpleNamespace(timing=timing, calls=[call])
    return SimpleNamespace(usage=usage, last=function_log)


# ---------------------------------------------------------------------------
# AC-1: TurnMetrics frozen dataclass shape + turn_metrics() callable
# ---------------------------------------------------------------------------


class TestTurnMetricsShape:
    """AC-1: TurnMetrics exposes five required fields and must be frozen."""

    def test_when_turn_metrics_is_created_then_input_tokens_field_is_accessible(self):
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=20,
            cache_creation_input_tokens=30,
            duration_ms=100.0,
        )
        assert tm.input_tokens == 10

    def test_when_turn_metrics_is_created_then_output_tokens_field_is_accessible(self):
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=20,
            cache_creation_input_tokens=30,
            duration_ms=100.0,
        )
        assert tm.output_tokens == 5

    def test_when_turn_metrics_is_created_then_cached_input_tokens_field_is_accessible(
        self,
    ):
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=20,
            cache_creation_input_tokens=30,
            duration_ms=100.0,
        )
        assert tm.cached_input_tokens == 20

    def test_when_turn_metrics_is_created_then_cache_creation_field_is_accessible(
        self,
    ):
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=20,
            cache_creation_input_tokens=30,
            duration_ms=100.0,
        )
        assert tm.cache_creation_input_tokens == 30

    def test_when_turn_metrics_is_created_then_duration_ms_field_is_accessible(self):
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=20,
            cache_creation_input_tokens=30,
            duration_ms=100.0,
        )
        assert tm.duration_ms == 100.0

    def test_when_turn_metrics_is_mutated_then_error_is_raised(self):
        """TurnMetrics must be frozen — mutation must raise an error."""
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            duration_ms=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            tm.input_tokens = 99  # type: ignore[misc]

    def test_when_metrics_module_is_imported_then_turn_metrics_function_is_callable(
        self,
    ):
        from dejavu import metrics

        assert callable(metrics.turn_metrics)


# ---------------------------------------------------------------------------
# AC-2: reads + tokens from Collector.usage; latency from FunctionLog.timing
# ---------------------------------------------------------------------------


class TestReadSourcesFromTypedUsage:
    """AC-2: input/output/cached from Collector.usage; latency from timing."""

    def test_when_collector_has_typed_usage_then_input_tokens_come_from_usage(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(input_tokens=123)
        assert turn_metrics(collector).input_tokens == 123

    def test_when_collector_has_typed_usage_then_output_tokens_come_from_usage(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(output_tokens=77)
        assert turn_metrics(collector).output_tokens == 77

    def test_when_collector_has_typed_usage_then_cached_input_tokens_come_from_usage(
        self,
    ):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(cached_input_tokens=200)
        assert turn_metrics(collector).cached_input_tokens == 200

    def test_when_function_log_has_timing_then_duration_ms_comes_from_timing(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(duration_ms=1234.5)
        assert turn_metrics(collector).duration_ms == 1234.5


# ---------------------------------------------------------------------------
# AC-3: cache writes from raw HTTP body, not typed usage
# ---------------------------------------------------------------------------


class TestWriteSourceFromRawHttpBody:
    """AC-3: cache_creation comes from raw log.calls[-1].http_response.body.json()."""

    def test_when_raw_body_has_cache_creation_then_result_reflects_http_value(
        self,
    ):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(cache_creation_input_tokens_raw=350)
        assert turn_metrics(collector).cache_creation_input_tokens == 350

    def test_when_raw_http_body_has_large_cache_creation_tokens_then_result_is_exact(
        self,
    ):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(cache_creation_input_tokens_raw=18_432)
        assert turn_metrics(collector).cache_creation_input_tokens == 18_432

    def test_when_raw_http_body_has_zero_cache_creation_tokens_then_result_is_zero(
        self,
    ):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(cache_creation_input_tokens_raw=0)
        assert turn_metrics(collector).cache_creation_input_tokens == 0


# ---------------------------------------------------------------------------
# AC-4: no cache write → cache_creation_input_tokens == 0 (non-exceptional path)
# ---------------------------------------------------------------------------


class TestNoCacheWritePath:
    """AC-4: key absent in raw HTTP body is a first-class, non-exceptional zero."""

    def test_when_http_body_usage_omits_cache_creation_key_then_result_is_zero(self):
        """Key absent (not value 0) must yield 0 without exception."""
        from dejavu.metrics import turn_metrics

        body_without_key = SimpleNamespace(
            json=lambda: {"usage": {"cache_read_input_tokens": 50}}
        )
        call = SimpleNamespace(http_response=SimpleNamespace(body=body_without_key))
        function_log = SimpleNamespace(
            timing=SimpleNamespace(duration_ms=100.0),
            calls=[call],
        )
        collector = SimpleNamespace(
            usage=_make_usage(input_tokens=10, output_tokens=5, cached_input_tokens=50),
            last=function_log,
        )
        assert turn_metrics(collector).cache_creation_input_tokens == 0

    def test_when_http_body_usage_omits_cache_creation_key_then_no_exception_is_raised(
        self,
    ):
        from dejavu.metrics import turn_metrics

        body_without_key = SimpleNamespace(json=lambda: {"usage": {}})
        call = SimpleNamespace(http_response=SimpleNamespace(body=body_without_key))
        function_log = SimpleNamespace(
            timing=SimpleNamespace(duration_ms=0.0), calls=[call]
        )
        collector = SimpleNamespace(
            usage=_make_usage(input_tokens=5, output_tokens=3),
            last=function_log,
        )
        turn_metrics(collector)  # must not raise — this is a normal path, not an error


# ---------------------------------------------------------------------------
# AC-5: graceful degradation for all four degenerate collector states
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """AC-5: writes/latency default to 0; no crash for all four degenerate states."""

    def test_when_http_response_is_missing_then_cache_creation_defaults_to_zero(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(http_response="missing")
        assert turn_metrics(collector).cache_creation_input_tokens == 0

    def test_when_http_response_is_missing_then_no_exception_is_raised(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(http_response="missing")
        turn_metrics(collector)  # must not raise

    def test_when_calls_list_is_empty_then_cache_creation_defaults_to_zero(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(calls="empty")
        assert turn_metrics(collector).cache_creation_input_tokens == 0

    def test_when_calls_list_is_empty_then_no_exception_is_raised(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(calls="empty")
        turn_metrics(collector)  # must not raise

    def test_when_body_is_invalid_json_then_cache_creation_defaults_to_zero(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(http_response="invalid_json")
        assert turn_metrics(collector).cache_creation_input_tokens == 0

    def test_when_body_is_invalid_json_then_no_exception_is_raised(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(http_response="invalid_json")
        turn_metrics(collector)  # must not raise

    def test_when_collector_last_is_none_then_cache_creation_defaults_to_zero(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(last="none")
        assert turn_metrics(collector).cache_creation_input_tokens == 0

    def test_when_collector_last_is_none_then_duration_ms_defaults_to_zero(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(last="none")
        assert turn_metrics(collector).duration_ms == 0

    def test_when_collector_last_is_none_then_no_exception_is_raised(self):
        from dejavu.metrics import turn_metrics

        collector = _make_collector(last="none")
        turn_metrics(collector)  # must not raise


# ---------------------------------------------------------------------------
# AC-7: dejavu/__init__.py re-exports TurnMetrics / turn_metrics
# ---------------------------------------------------------------------------


class TestInitReexports:
    """AC-7: Public metrics symbols importable from top-level dejavu package."""

    def test_when_TurnMetrics_is_imported_from_dejavu_then_it_is_accessible(self):
        from dejavu import TurnMetrics as TM  # noqa: F401

        assert TM is not None

    def test_when_turn_metrics_is_imported_from_dejavu_then_it_is_callable(self):
        from dejavu import turn_metrics as f  # noqa: F401

        assert callable(f)

    def test_when_dejavu_all_is_inspected_then_TurnMetrics_is_listed(self):
        import dejavu

        assert "TurnMetrics" in dejavu.__all__

    def test_when_dejavu_all_is_inspected_then_turn_metrics_is_listed(self):
        import dejavu

        assert "turn_metrics" in dejavu.__all__


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
#
# AC-4 invariant:  for ALL valid token counts, a body whose usage dict omits
#   cache_creation_input_tokens always yields result == 0 (never an exception,
#   never a non-zero value).
#
# AC-5 invariant:  for ANY valid usage, collector.last=None never raises and
#   always returns TurnMetrics with duration_ms == 0 and cache_creation == 0.
# ---------------------------------------------------------------------------

_NON_NEG_INT = st.integers(min_value=0, max_value=10_000_000)


@given(
    input_tokens=_NON_NEG_INT,
    output_tokens=_NON_NEG_INT,
    cached_input_tokens=_NON_NEG_INT,
)
def test_when_http_body_always_omits_cache_creation_key_then_result_is_always_zero(
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> None:
    """
    AC-4 invariant: regardless of how many input/output/cached tokens are present,
    a usage dict that lacks cache_creation_input_tokens must yield 0 — never an
    error, never a non-zero value.  Derived from: "key absent → 0 as a first-class,
    non-exceptional path."
    """
    from dejavu.metrics import turn_metrics

    body = SimpleNamespace(json=lambda: {"usage": {}})
    call = SimpleNamespace(http_response=SimpleNamespace(body=body))
    function_log = SimpleNamespace(
        timing=SimpleNamespace(duration_ms=0.0),
        calls=[call],
    )
    collector = SimpleNamespace(
        usage=_make_usage(input_tokens, output_tokens, cached_input_tokens),
        last=function_log,
    )
    assert turn_metrics(collector).cache_creation_input_tokens == 0


@given(
    input_tokens=_NON_NEG_INT,
    output_tokens=_NON_NEG_INT,
    cached_input_tokens=_NON_NEG_INT,
)
def test_when_collector_last_is_none_with_any_usage_then_no_exception_is_raised(
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> None:
    """
    AC-5 invariant: collector.last=None must never raise for any valid usage.
    Derived from: "collector.last is None degrades gracefully (writes/latency
    default to 0; no crash)."
    """
    from dejavu.metrics import turn_metrics

    collector = SimpleNamespace(
        usage=_make_usage(input_tokens, output_tokens, cached_input_tokens),
        last=None,
    )
    turn_metrics(
        collector
    )  # must not raise for any combination of non-negative token counts


@given(
    input_tokens=_NON_NEG_INT,
    output_tokens=_NON_NEG_INT,
    cached_input_tokens=_NON_NEG_INT,
)
def test_when_collector_last_is_none_then_cache_creation_and_latency_are_zero(
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> None:
    """
    AC-5 invariant: for any valid usage, last=None always produces
    cache_creation_input_tokens == 0 and duration_ms == 0.
    """
    from dejavu.metrics import turn_metrics

    collector = SimpleNamespace(
        usage=_make_usage(input_tokens, output_tokens, cached_input_tokens),
        last=None,
    )
    result = turn_metrics(collector)
    assert result.cache_creation_input_tokens == 0
    assert result.duration_ms == 0


# ---------------------------------------------------------------------------
# AC-6: Live, skipif-gated test against real Anthropic API
#
# Two separate test functions — one for the write half, one for the read half —
# so a skip or failure is attributed to the right assertion without masking the
# other.  The document is generated inline (~2 700 tokens) to exceed Anthropic's
# 1 024-token cache minimum without depending on dejavu.contract (implementation
# source that must not be read at this phase).
# ---------------------------------------------------------------------------

_LIVE_DOC = (
    "MASTER SERVICES AGREEMENT\n\n"
    "This Master Services Agreement ('Agreement') is entered into between "
    "Acme Corporation ('Client') and Provider Inc ('Provider').  "
    "The parties agree to the following terms and conditions governing the "
    "provision of professional services.\n\n"
    + (
        "Article content clause definitions obligations representations warranties. "
        * 600
    )
    # ~39k chars ≈ ~9.7k tokens — well above Anthropic's 1024-token cache minimum
)


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — live test skipped in CI",
)
def test_live_when_first_chat_cached_turn_then_cache_creation_is_positive():
    """
    AC-6 (write half): the first ChatCached call against a fresh document prefix
    must produce cache_creation_input_tokens > 0, proving turn_metrics reads
    cache_creation from the raw HTTP body (not from typed usage).
    """
    from baml_client import b  # type: ignore[import]
    from baml_py import Collector  # type: ignore[import]

    from dejavu.metrics import turn_metrics

    collector = Collector()
    b.ChatCached(
        document=_LIVE_DOC,
        history=[],
        question="What is the full legal name of the client party in this agreement?",
        baml_options={"collector": collector},
    )
    result = turn_metrics(collector)
    assert result.cache_creation_input_tokens > 0, (
        f"First ChatCached call must write to cache; "
        f"got cache_creation_input_tokens={result.cache_creation_input_tokens}"
    )


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — live test skipped in CI",
)
def test_live_when_second_chat_cached_turn_fires_then_cached_input_tokens_is_positive():
    """
    AC-6 (read half): a second ChatCached call that reuses the same document prefix
    must produce cached_input_tokens > 0, proving the cache was hit and that
    turn_metrics reads cached_input_tokens from the typed Collector.usage.
    """
    from baml_client import b  # type: ignore[import]
    from baml_py import Collector  # type: ignore[import]

    from dejavu.metrics import turn_metrics

    # Warm the cache with an identical first call
    warm_up = Collector()
    b.ChatCached(
        document=_LIVE_DOC,
        history=[],
        question="What is the full legal name of the client party in this agreement?",
        baml_options={"collector": warm_up},
    )

    # Second call reusing the same large document — cache prefix should now be warm
    read_collector = Collector()
    b.ChatCached(
        document=_LIVE_DOC,
        history=[],
        question="What is the full legal name of the provider party in this agreement?",
        baml_options={"collector": read_collector},
    )
    result = turn_metrics(read_collector)
    assert result.cached_input_tokens > 0, (
        f"Second ChatCached call with same document prefix must hit cache; "
        f"got cached_input_tokens={result.cached_input_tokens}"
    )
