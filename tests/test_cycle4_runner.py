"""
Source-blind tests for Issue #9: priced per-turn result model and assembly.

All fixtures are derived solely from acceptance criteria and requirements.md.
No implementation source (dejavu/runner.py) was read; the file was assumed
not to exist at authoring time (Red phase of TDD).

Criteria tested:
  AC-1  [UNIT] Frozen dataclasses SideResult, TurnResult, TokenEvent defined in runner.py
  AC-2  [UNIT] Side enum exposes UNCACHED and CACHED members
  AC-3  [UNIT] RunConfig frozen dataclass with correct fields and defaults
  AC-4  [UNIT] _build_uncached_result wires turn_metrics then price_uncached_turn
  AC-5  [UNIT] _build_cached_result wires turn_metrics then price_cached_turn
  AC-6  [UNIT] _fresh_question_tokens == max(0, input - cached - creation)
  AC-7  [UNIT] Public symbols re-exported from dejavu.__init__ and in __all__

Skipped:
  AC-8  All tests pass — boilerplate suite gate; no per-criterion assertion
  AC-9  SOLID/clean code — not runtime-verifiable
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from hypothesis import given, strategies as st


# ---------------------------------------------------------------------------
# Stub helpers — derived solely from requirements.md (not from source)
#
# Collector interface (requirements.md §5):
#   .usage.input_tokens
#   .usage.output_tokens
#   .usage.cached_input_tokens
#   .last.timing.duration_ms
#   .last.calls[-1].http_response.body.json()  → dict with usage data
# ---------------------------------------------------------------------------


def _make_collector_stub(
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    duration_ms: float = 100.0,
) -> SimpleNamespace:
    """Build a minimal Collector stub from requirements.md field descriptions."""
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )
    raw_body = {"usage": {"cache_creation_input_tokens": cache_creation_input_tokens}}
    body = SimpleNamespace(json=lambda: raw_body)
    http_response = SimpleNamespace(body=body)
    call = SimpleNamespace(http_response=http_response)
    timing = SimpleNamespace(duration_ms=duration_ms)
    log = SimpleNamespace(timing=timing, calls=[call])
    return SimpleNamespace(usage=usage, last=log)


# ---------------------------------------------------------------------------
# AC-1a: SideResult frozen dataclass — text, metrics, cost_usd
# ---------------------------------------------------------------------------


class TestSideResultDataclass:
    """AC-1a: SideResult is a frozen dataclass with text, metrics, cost_usd."""

    def _tm(self):
        from dejavu.metrics import TurnMetrics

        return TurnMetrics(
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            duration_ms=100.0,
        )

    def test_when_side_result_is_created_then_text_field_is_accessible(self):
        from dejavu.runner import SideResult

        r = SideResult(text="hello", metrics=self._tm(), cost_usd=0.01)
        assert r.text == "hello"

    def test_when_side_result_is_created_then_metrics_field_is_accessible(self):
        from dejavu.runner import SideResult

        tm = self._tm()
        r = SideResult(text="hello", metrics=tm, cost_usd=0.01)
        assert r.metrics is tm

    def test_when_side_result_is_created_then_cost_usd_field_is_accessible(self):
        from dejavu.runner import SideResult

        r = SideResult(text="hello", metrics=self._tm(), cost_usd=0.01)
        assert r.cost_usd == pytest.approx(0.01)

    def test_when_side_result_is_mutated_then_error_is_raised(self):
        from dejavu.runner import SideResult

        r = SideResult(text="hello", metrics=self._tm(), cost_usd=0.01)
        with pytest.raises((AttributeError, TypeError)):
            r.text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AC-1b: TurnResult frozen dataclass — index, question, uncached, cached
# ---------------------------------------------------------------------------


class TestTurnResultDataclass:
    """AC-1b: TurnResult is a frozen dataclass with index, question, uncached, cached."""

    def _side(self, text: str = "answer"):
        from dejavu.runner import SideResult
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            duration_ms=0.0,
        )
        return SideResult(text=text, metrics=tm, cost_usd=0.0)

    def test_when_turn_result_is_created_then_index_field_is_accessible(self):
        from dejavu.runner import TurnResult

        r = TurnResult(
            index=3, question="q?", uncached=self._side(), cached=self._side()
        )
        assert r.index == 3

    def test_when_turn_result_is_created_then_question_field_is_accessible(self):
        from dejavu.runner import TurnResult

        r = TurnResult(
            index=0,
            question="What is the fee?",
            uncached=self._side(),
            cached=self._side(),
        )
        assert r.question == "What is the fee?"

    def test_when_turn_result_is_created_then_uncached_field_is_accessible(self):
        from dejavu.runner import TurnResult

        u = self._side("u")
        r = TurnResult(index=0, question="q?", uncached=u, cached=self._side())
        assert r.uncached is u

    def test_when_turn_result_is_created_then_cached_field_is_accessible(self):
        from dejavu.runner import TurnResult

        c = self._side("c")
        r = TurnResult(index=0, question="q?", uncached=self._side(), cached=c)
        assert r.cached is c

    def test_when_turn_result_is_mutated_then_error_is_raised(self):
        from dejavu.runner import TurnResult

        r = TurnResult(
            index=0, question="q?", uncached=self._side(), cached=self._side()
        )
        with pytest.raises((AttributeError, TypeError)):
            r.index = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AC-1c: TokenEvent frozen dataclass — index, side, partial
# ---------------------------------------------------------------------------


class TestTokenEventDataclass:
    """AC-1c: TokenEvent is a frozen dataclass with index, side, partial."""

    def test_when_token_event_is_created_then_index_field_is_accessible(self):
        from dejavu.runner import TokenEvent, Side

        e = TokenEvent(index=2, side=Side.UNCACHED, partial="tok")
        assert e.index == 2

    def test_when_token_event_is_created_then_side_field_is_accessible(self):
        from dejavu.runner import TokenEvent, Side

        e = TokenEvent(index=2, side=Side.CACHED, partial="tok")
        assert e.side == Side.CACHED

    def test_when_token_event_is_created_then_partial_field_is_accessible(self):
        from dejavu.runner import TokenEvent, Side

        e = TokenEvent(index=2, side=Side.UNCACHED, partial="hello world")
        assert e.partial == "hello world"

    def test_when_token_event_is_mutated_then_error_is_raised(self):
        from dejavu.runner import TokenEvent, Side

        e = TokenEvent(index=0, side=Side.UNCACHED, partial="tok")
        with pytest.raises((AttributeError, TypeError)):
            e.partial = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AC-2: Side enum exposes UNCACHED and CACHED
# ---------------------------------------------------------------------------


class TestSideEnum:
    """AC-2: Side enum has UNCACHED and CACHED members, and they are distinct."""

    def test_when_side_enum_is_imported_then_uncached_member_is_accessible(self):
        from dejavu.runner import Side

        assert hasattr(Side, "UNCACHED")

    def test_when_side_enum_is_imported_then_cached_member_is_accessible(self):
        from dejavu.runner import Side

        assert hasattr(Side, "CACHED")

    def test_when_side_uncached_and_cached_are_compared_then_they_are_not_equal(self):
        from dejavu.runner import Side

        assert Side.UNCACHED != Side.CACHED


# ---------------------------------------------------------------------------
# AC-3: RunConfig frozen dataclass with correct fields and defaults
# ---------------------------------------------------------------------------


class TestRunConfigDataclass:
    """AC-3: RunConfig is a frozen dataclass with documented fields and defaults."""

    def test_when_run_config_is_created_with_defaults_then_model_is_opus_48(self):
        from dejavu.runner import RunConfig
        from dejavu.pricing import Model

        assert RunConfig().model == Model.OPUS_48

    def test_when_run_config_is_created_with_defaults_then_doc_is_none(self):
        from dejavu.runner import RunConfig

        assert RunConfig().doc is None

    def test_when_run_config_is_created_with_defaults_then_questions_file_is_none(self):
        from dejavu.runner import RunConfig

        assert RunConfig().questions_file is None

    def test_when_run_config_is_created_with_defaults_then_ttl_is_5m(self):
        from dejavu.runner import RunConfig

        assert RunConfig().ttl == "5m"

    def test_when_run_config_is_created_with_defaults_then_max_tokens_is_400(self):
        from dejavu.runner import RunConfig

        assert RunConfig().max_tokens == 400

    def test_when_run_config_is_created_with_defaults_then_temperature_is_0_0(self):
        from dejavu.runner import RunConfig

        assert RunConfig().temperature == pytest.approx(0.0)

    def test_when_run_config_is_created_with_defaults_then_delay_is_0_0(self):
        from dejavu.runner import RunConfig

        assert RunConfig().delay == pytest.approx(0.0)

    def test_when_run_config_is_mutated_then_error_is_raised(self):
        from dejavu.runner import RunConfig

        cfg = RunConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_tokens = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AC-4: _build_uncached_result wires turn_metrics then price_uncached_turn
# ---------------------------------------------------------------------------


class TestBuildUncachedResult:
    """AC-4: _build_uncached_result returns SideResult with correctly wired cost."""

    def test_when_build_uncached_result_is_called_then_returned_type_is_side_result(
        self,
    ):
        from dejavu.runner import _build_uncached_result, SideResult
        from dejavu.pricing import Model

        collector = _make_collector_stub(input_tokens=100, output_tokens=50)
        result = _build_uncached_result("answer", collector, Model.OPUS_48)
        assert isinstance(result, SideResult)

    def test_when_build_uncached_result_is_called_then_text_is_preserved(self):
        from dejavu.runner import _build_uncached_result
        from dejavu.pricing import Model

        collector = _make_collector_stub(input_tokens=100, output_tokens=50)
        result = _build_uncached_result("the uncached answer", collector, Model.OPUS_48)
        assert result.text == "the uncached answer"

    def test_when_build_uncached_result_is_called_then_metrics_input_tokens_match_collector(
        self,
    ):
        from dejavu.runner import _build_uncached_result
        from dejavu.pricing import Model

        collector = _make_collector_stub(input_tokens=200, output_tokens=30)
        result = _build_uncached_result("answer", collector, Model.OPUS_48)
        assert result.metrics.input_tokens == 200

    def test_when_build_uncached_result_is_called_then_metrics_output_tokens_match_collector(
        self,
    ):
        from dejavu.runner import _build_uncached_result
        from dejavu.pricing import Model

        collector = _make_collector_stub(input_tokens=200, output_tokens=30)
        result = _build_uncached_result("answer", collector, Model.OPUS_48)
        assert result.metrics.output_tokens == 30

    def test_when_build_uncached_result_is_called_then_cost_matches_price_uncached_turn(
        self,
    ):
        """
        Wiring check: cost_usd must equal what price_uncached_turn produces for the
        same input and output token counts. Derives expected from Cycle-3 public API.
        """
        from dejavu.runner import _build_uncached_result
        from dejavu.metrics import turn_metrics
        from dejavu.pricing import Model, price_uncached_turn

        collector = _make_collector_stub(input_tokens=1000, output_tokens=200)
        result = _build_uncached_result("answer", collector, Model.OPUS_48)

        m = turn_metrics(collector)
        expected = price_uncached_turn(
            model=Model.OPUS_48,
            input_tokens=m.input_tokens,
            output_tokens=m.output_tokens,
        )
        assert result.cost_usd == pytest.approx(expected)

    def test_when_build_uncached_result_is_called_with_fable_model_then_cost_matches(
        self,
    ):
        """Model is forwarded to price_uncached_turn — different model → different rate."""
        from dejavu.runner import _build_uncached_result
        from dejavu.metrics import turn_metrics
        from dejavu.pricing import Model, price_uncached_turn

        collector = _make_collector_stub(input_tokens=500, output_tokens=100)
        result = _build_uncached_result("answer", collector, Model.FABLE_5)

        m = turn_metrics(collector)
        expected = price_uncached_turn(
            model=Model.FABLE_5,
            input_tokens=m.input_tokens,
            output_tokens=m.output_tokens,
        )
        assert result.cost_usd == pytest.approx(expected)


# ---------------------------------------------------------------------------
# AC-5: _build_cached_result wires turn_metrics then price_cached_turn
# ---------------------------------------------------------------------------


class TestBuildCachedResult:
    """AC-5: _build_cached_result returns SideResult with correctly wired cached cost."""

    def test_when_build_cached_result_is_called_then_returned_type_is_side_result(self):
        from dejavu.runner import _build_cached_result, SideResult
        from dejavu.pricing import Model

        collector = _make_collector_stub(
            input_tokens=30,
            output_tokens=20,
            cached_input_tokens=5000,
            cache_creation_input_tokens=100,
        )
        result = _build_cached_result("cached", collector, Model.OPUS_48)
        assert isinstance(result, SideResult)

    def test_when_build_cached_result_is_called_then_text_is_preserved(self):
        from dejavu.runner import _build_cached_result
        from dejavu.pricing import Model

        collector = _make_collector_stub(
            input_tokens=30,
            output_tokens=20,
            cached_input_tokens=5000,
            cache_creation_input_tokens=0,
        )
        result = _build_cached_result("the cached answer", collector, Model.OPUS_48)
        assert result.text == "the cached answer"

    def test_when_build_cached_result_is_called_then_cost_matches_price_cached_turn(
        self,
    ):
        """
        Wiring check: cost_usd must equal what price_cached_turn produces for the
        same cache_read, cache_creation, fresh_question, and output token counts.
        fresh_question_tokens is m.input_tokens (Anthropic already excludes cache
        reads/writes from that field).
        """
        from dejavu.runner import _build_cached_result
        from dejavu.metrics import turn_metrics
        from dejavu.pricing import Model, price_cached_turn

        collector = _make_collector_stub(
            input_tokens=40,
            output_tokens=20,
            cached_input_tokens=6000,
            cache_creation_input_tokens=200,
        )
        result = _build_cached_result("cached", collector, Model.OPUS_48)

        m = turn_metrics(collector)
        expected = price_cached_turn(
            model=Model.OPUS_48,
            tier="5m",
            cache_read_tokens=m.cached_input_tokens or 0,
            cache_creation_tokens=m.cache_creation_input_tokens or 0,
            fresh_question_tokens=m.input_tokens or 0,
            output_tokens=m.output_tokens or 0,
        )
        assert result.cost_usd == pytest.approx(expected)

    def test_when_build_cached_result_is_called_with_1h_tier_then_cost_matches(self):
        """tier parameter is forwarded to price_cached_turn."""
        from dejavu.runner import _build_cached_result
        from dejavu.metrics import turn_metrics
        from dejavu.pricing import Model, price_cached_turn

        collector = _make_collector_stub(
            input_tokens=40,
            output_tokens=20,
            cached_input_tokens=6000,
            cache_creation_input_tokens=200,
        )
        result = _build_cached_result("cached", collector, Model.OPUS_48, tier="1h")

        m = turn_metrics(collector)
        expected = price_cached_turn(
            model=Model.OPUS_48,
            tier="1h",
            cache_read_tokens=m.cached_input_tokens or 0,
            cache_creation_tokens=m.cache_creation_input_tokens or 0,
            fresh_question_tokens=m.input_tokens or 0,
            output_tokens=m.output_tokens or 0,
        )
        assert result.cost_usd == pytest.approx(expected)

    def test_when_build_cached_result_is_called_with_fable_model_then_cost_matches(
        self,
    ):
        """Model is forwarded to price_cached_turn — different model → different rate."""
        from dejavu.runner import _build_cached_result
        from dejavu.metrics import turn_metrics
        from dejavu.pricing import Model, price_cached_turn

        collector = _make_collector_stub(
            input_tokens=30,
            output_tokens=15,
            cached_input_tokens=3000,
            cache_creation_input_tokens=50,
        )
        result = _build_cached_result("cached", collector, Model.FABLE_5)

        m = turn_metrics(collector)
        expected = price_cached_turn(
            model=Model.FABLE_5,
            tier="5m",
            cache_read_tokens=m.cached_input_tokens or 0,
            cache_creation_tokens=m.cache_creation_input_tokens or 0,
            fresh_question_tokens=m.input_tokens or 0,
            output_tokens=m.output_tokens or 0,
        )
        assert result.cost_usd == pytest.approx(expected)


# ---------------------------------------------------------------------------
# AC-6: _fresh_question_tokens == max(0, input - cached - creation)
# ---------------------------------------------------------------------------


class TestFreshQuestionTokens:
    """AC-6: _fresh_question_tokens returns max(0, input_tokens - cached - creation)."""

    def test_when_no_cached_or_creation_tokens_then_fresh_equals_input_tokens(self):
        from dejavu.runner import _fresh_question_tokens
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=100,
            output_tokens=20,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            duration_ms=0.0,
        )
        assert _fresh_question_tokens(tm) == 100

    def test_when_cached_input_tokens_are_present_then_fresh_equals_input_tokens(self):
        """
        Anthropic excludes cache reads from input_tokens already — input_tokens IS
        the post-breakpoint (fresh question) count.  No subtraction should occur.
        Corrected per reviewer comment on issue #9.
        """
        from dejavu.runner import _fresh_question_tokens
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=100,
            output_tokens=20,
            cached_input_tokens=70,
            cache_creation_input_tokens=0,
            duration_ms=0.0,
        )
        assert _fresh_question_tokens(tm) == 100

    def test_when_cache_creation_tokens_are_present_then_fresh_equals_input_tokens(
        self,
    ):
        """
        Anthropic excludes cache-creation tokens from input_tokens already.
        No subtraction should occur.  Corrected per reviewer comment on issue #9.
        """
        from dejavu.runner import _fresh_question_tokens
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=100,
            output_tokens=20,
            cached_input_tokens=0,
            cache_creation_input_tokens=60,
            duration_ms=0.0,
        )
        assert _fresh_question_tokens(tm) == 100

    def test_when_large_cached_tokens_present_then_fresh_still_equals_input_tokens(
        self,
    ):
        """
        On a warm cached turn input_tokens is small (~30) and cached_input_tokens
        is large (~6000).  The old formula would have clamped to 0; the correct
        formula returns input_tokens directly.
        """
        from dejavu.runner import _fresh_question_tokens
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=30,
            output_tokens=20,
            cached_input_tokens=6000,
            cache_creation_input_tokens=0,
            duration_ms=0.0,
        )
        assert _fresh_question_tokens(tm) == 30

    def test_when_all_token_counts_are_zero_then_fresh_question_tokens_is_zero(self):
        from dejavu.runner import _fresh_question_tokens
        from dejavu.metrics import TurnMetrics

        tm = TurnMetrics(
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            duration_ms=0.0,
        )
        assert _fresh_question_tokens(tm) == 0


_NON_NEG_INT = st.integers(min_value=0, max_value=10_000_000)


@given(
    input_tokens=_NON_NEG_INT,
    output_tokens=_NON_NEG_INT,
    cached_input_tokens=_NON_NEG_INT,
    cache_creation_input_tokens=_NON_NEG_INT,
)
def test_when_fresh_question_tokens_called_with_any_valid_counts_then_result_equals_input_tokens(
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
    cache_creation_input_tokens: int,
) -> None:
    """
    AC-6 invariant (corrected per reviewer comment on issue #9):
    Anthropic's usage.input_tokens already excludes cache reads and cache-creation
    tokens — the three counts are mutually exclusive.  So _fresh_question_tokens
    must return input_tokens directly for all non-negative token counts.
    """
    from dejavu.runner import _fresh_question_tokens
    from dejavu.metrics import TurnMetrics

    tm = TurnMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        duration_ms=0.0,
    )
    assert _fresh_question_tokens(tm) == input_tokens


# ---------------------------------------------------------------------------
# AC-7: Public symbols re-exported from dejavu.__init__ and in __all__
# ---------------------------------------------------------------------------


class TestInitReexports:
    """AC-7: Runner public symbols are importable from the top-level dejavu package."""

    def test_when_side_is_imported_from_dejavu_then_it_is_accessible(self):
        from dejavu import Side  # noqa: F401

        assert Side is not None

    def test_when_side_result_is_imported_from_dejavu_then_it_is_accessible(self):
        from dejavu import SideResult  # noqa: F401

        assert SideResult is not None

    def test_when_turn_result_is_imported_from_dejavu_then_it_is_accessible(self):
        from dejavu import TurnResult  # noqa: F401

        assert TurnResult is not None

    def test_when_token_event_is_imported_from_dejavu_then_it_is_accessible(self):
        from dejavu import TokenEvent  # noqa: F401

        assert TokenEvent is not None

    def test_when_run_config_is_imported_from_dejavu_then_it_is_accessible(self):
        from dejavu import RunConfig  # noqa: F401

        assert RunConfig is not None

    def test_when_dejavu_all_is_inspected_then_side_is_listed(self):
        import dejavu

        assert "Side" in dejavu.__all__

    def test_when_dejavu_all_is_inspected_then_side_result_is_listed(self):
        import dejavu

        assert "SideResult" in dejavu.__all__

    def test_when_dejavu_all_is_inspected_then_turn_result_is_listed(self):
        import dejavu

        assert "TurnResult" in dejavu.__all__

    def test_when_dejavu_all_is_inspected_then_token_event_is_listed(self):
        import dejavu

        assert "TokenEvent" in dejavu.__all__

    def test_when_dejavu_all_is_inspected_then_run_config_is_listed(self):
        import dejavu

        assert "RunConfig" in dejavu.__all__
