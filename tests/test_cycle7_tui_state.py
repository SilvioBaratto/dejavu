"""
Source-blind tests for issue #14: pure TUI render-state core with running costs.

All tests derived solely from acceptance criteria and requirements.md.
No implementation source was read; no dejavu/tui.py internals are assumed
beyond what the criteria explicitly state must exist (Red phase of TDD).

Criteria tested:
  AC-1  [UNIT] PanelState holds the seven required fields
  AC-2  [UNIT] RenderState.apply_token sets the matching panel's live_partial
  AC-3  [UNIT] RenderState.apply_turn updates transcripts, costs, turn metadata, strip, partial
  AC-4  [UNIT] CostStripState.multiple(), saved(), multiple_text() with divide-by-zero guard
  AC-5  [UNIT] _fmt_usd and _fmt_int produce fixed-precision, right-aligned output
  AC-6  [PROPERTY] multiple() >= 1 and saved() >= 0 when cached_total <= uncached_total

Skipped (per oracle):
  AC-7  All tests pass     — boilerplate suite gate; no per-criterion assertion
  AC-8  SOLID/clean code   — subjective prose; not runtime-verifiable
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings as h_settings, strategies as st

from dejavu.tui import (
    CostStripState,
    PanelState,
    RenderState,
    _fmt_int,
    _fmt_usd,
)


# ---------------------------------------------------------------------------
# Fixtures — built from criteria and requirements.md; no source was read
# ---------------------------------------------------------------------------


def _make_turn_result(
    *,
    index: int = 0,
    question: str = "What is the termination fee?",
    uncached_text: str = "The termination fee is $500.",
    cached_text: str = "The fee is $500.",
    uncached_cost: float = 0.10,
    cached_cost: float = 0.01,
    cache_read_tokens: int = 5_000,
):
    """Build a minimal TurnResult fixture from criteria-stated fields only."""
    from dejavu.metrics import TurnMetrics
    from dejavu.runner import SideResult, TurnResult

    uncached_metrics = TurnMetrics(
        input_tokens=1_000,
        output_tokens=80,
        cached_input_tokens=0,
        cache_creation_input_tokens=0,
        duration_ms=200.0,
    )
    cached_metrics = TurnMetrics(
        input_tokens=40,
        output_tokens=80,
        cached_input_tokens=cache_read_tokens,
        cache_creation_input_tokens=0,
        duration_ms=180.0,
    )
    uncached = SideResult(
        text=uncached_text, metrics=uncached_metrics, cost_usd=uncached_cost
    )
    cached = SideResult(text=cached_text, metrics=cached_metrics, cost_usd=cached_cost)
    return TurnResult(index=index, question=question, uncached=uncached, cached=cached)


# ---------------------------------------------------------------------------
# AC-1: PanelState holds the seven required fields with correct initial values
# ---------------------------------------------------------------------------


class TestPanelStateFields:
    """AC-1: PanelState exposes all seven fields with sensible initial values."""

    def test_when_panel_state_is_created_then_title_field_is_accessible(self):
        p = PanelState(title="Uncached")
        assert p.title == "Uncached"

    def test_when_panel_state_is_created_then_running_cost_starts_at_zero(self):
        p = PanelState(title="Cached")
        assert p.running_cost == pytest.approx(0.0)

    def test_when_panel_state_is_created_then_turn_number_is_zero(self):
        p = PanelState(title="Uncached")
        assert p.turn_number == 0

    def test_when_panel_state_is_created_then_this_turn_cache_read_is_zero(self):
        p = PanelState(title="Uncached")
        assert p.this_turn_cache_read == 0

    def test_when_panel_state_is_created_then_last_turn_delta_is_zero(self):
        p = PanelState(title="Uncached")
        assert p.last_turn_delta == pytest.approx(0.0)

    def test_when_panel_state_is_created_then_transcript_is_empty(self):
        p = PanelState(title="Uncached")
        assert len(p.transcript) == 0

    def test_when_panel_state_is_created_then_live_partial_is_empty_string(self):
        p = PanelState(title="Uncached")
        assert p.live_partial == ""

    def test_when_transcript_entry_is_inspected_then_it_has_role_attribute(self):
        """Criterion states 'role, text entries' — role attribute must exist after populate."""
        state = RenderState()
        state.apply_turn(_make_turn_result())
        entry = state.uncached.transcript[0]
        assert hasattr(entry, "role"), "transcript entry must expose a .role attribute"

    def test_when_transcript_entry_is_inspected_then_it_has_text_attribute(self):
        state = RenderState()
        state.apply_turn(_make_turn_result())
        entry = state.uncached.transcript[0]
        assert hasattr(entry, "text"), "transcript entry must expose a .text attribute"


# ---------------------------------------------------------------------------
# AC-2: RenderState.apply_token sets the matching panel's live_partial
# ---------------------------------------------------------------------------


class TestApplyToken:
    """AC-2: apply_token(event) routes the partial to the correct panel only."""

    def test_when_apply_token_with_uncached_event_then_uncached_live_partial_is_set(
        self,
    ):
        from dejavu.runner import Side, TokenEvent

        state = RenderState()
        event = TokenEvent(index=0, side=Side.UNCACHED, partial="streaming text so far")
        state.apply_token(event)
        assert state.uncached.live_partial == "streaming text so far"

    def test_when_apply_token_with_cached_event_then_cached_live_partial_is_set(self):
        from dejavu.runner import Side, TokenEvent

        state = RenderState()
        event = TokenEvent(index=0, side=Side.CACHED, partial="cached stream text")
        state.apply_token(event)
        assert state.cached.live_partial == "cached stream text"

    def test_when_apply_token_uncached_then_cached_panel_live_partial_is_unaffected(
        self,
    ):
        from dejavu.runner import Side, TokenEvent

        state = RenderState()
        state.apply_token(TokenEvent(index=0, side=Side.UNCACHED, partial="left only"))
        assert state.cached.live_partial == ""

    def test_when_apply_token_cached_then_uncached_panel_live_partial_is_unaffected(
        self,
    ):
        from dejavu.runner import Side, TokenEvent

        state = RenderState()
        state.apply_token(TokenEvent(index=0, side=Side.CACHED, partial="right only"))
        assert state.uncached.live_partial == ""

    def test_when_apply_token_called_twice_then_live_partial_equals_latest_partial(
        self,
    ):
        """
        event.partial IS the cumulative stream text; apply_token sets (not appends).
        A second call with a longer partial overwrites the first.
        """
        from dejavu.runner import Side, TokenEvent

        state = RenderState()
        state.apply_token(TokenEvent(index=0, side=Side.UNCACHED, partial="Hello"))
        state.apply_token(
            TokenEvent(index=0, side=Side.UNCACHED, partial="Hello, world!")
        )
        assert state.uncached.live_partial == "Hello, world!"

    def test_when_apply_token_empty_partial_then_live_partial_is_empty(self):
        from dejavu.runner import Side, TokenEvent

        state = RenderState()
        state.apply_token(TokenEvent(index=0, side=Side.UNCACHED, partial=""))
        assert state.uncached.live_partial == ""


# ---------------------------------------------------------------------------
# AC-3: RenderState.apply_turn — comprehensive state update
# ---------------------------------------------------------------------------


class TestApplyTurn:
    """AC-3: apply_turn updates transcripts, costs, turn metadata, cost strip, and clears partial."""

    # --- Transcript: question appended to both panels ---

    def test_when_apply_turn_then_question_text_is_in_uncached_transcript(self):
        state = RenderState()
        result = _make_turn_result(question="What is the termination fee?")
        state.apply_turn(result)
        texts = [e.text for e in state.uncached.transcript]
        assert "What is the termination fee?" in texts

    def test_when_apply_turn_then_question_text_is_in_cached_transcript(self):
        state = RenderState()
        result = _make_turn_result(question="What is the liability cap?")
        state.apply_turn(result)
        texts = [e.text for e in state.cached.transcript]
        assert "What is the liability cap?" in texts

    def test_when_apply_turn_then_question_has_user_role_in_uncached_transcript(self):
        state = RenderState()
        result = _make_turn_result(question="What is the payment term?")
        state.apply_turn(result)
        matches = [
            e
            for e in state.uncached.transcript
            if e.text == "What is the payment term?"
        ]
        assert matches, "question must appear in uncached transcript"
        assert matches[0].role == "user"

    def test_when_apply_turn_then_question_has_user_role_in_cached_transcript(self):
        state = RenderState()
        result = _make_turn_result(question="What is the notice period?")
        state.apply_turn(result)
        matches = [
            e for e in state.cached.transcript if e.text == "What is the notice period?"
        ]
        assert matches, "question must appear in cached transcript"
        assert matches[0].role == "user"

    # --- Transcript: answers appended with assistant role ---

    def test_when_apply_turn_then_uncached_answer_is_in_uncached_transcript(self):
        state = RenderState()
        result = _make_turn_result(uncached_text="Termination requires 30 days notice.")
        state.apply_turn(result)
        texts = [e.text for e in state.uncached.transcript]
        assert "Termination requires 30 days notice." in texts

    def test_when_apply_turn_then_cached_answer_is_in_cached_transcript(self):
        state = RenderState()
        result = _make_turn_result(cached_text="Notice period is thirty days.")
        state.apply_turn(result)
        texts = [e.text for e in state.cached.transcript]
        assert "Notice period is thirty days." in texts

    def test_when_apply_turn_then_uncached_answer_has_assistant_role(self):
        state = RenderState()
        result = _make_turn_result(uncached_text="The fee is $500.")
        state.apply_turn(result)
        matches = [e for e in state.uncached.transcript if e.text == "The fee is $500."]
        assert matches, "uncached answer must appear in uncached transcript"
        assert matches[0].role == "assistant"

    def test_when_apply_turn_then_cached_answer_has_assistant_role(self):
        state = RenderState()
        result = _make_turn_result(cached_text="The cap is $1,000,000.")
        state.apply_turn(result)
        matches = [
            e for e in state.cached.transcript if e.text == "The cap is $1,000,000."
        ]
        assert matches, "cached answer must appear in cached transcript"
        assert matches[0].role == "assistant"

    def test_when_apply_turn_then_each_panel_transcript_has_two_entries(self):
        """One question (user) + one answer (assistant) = 2 entries per panel per turn."""
        state = RenderState()
        state.apply_turn(_make_turn_result())
        assert len(state.uncached.transcript) == 2
        assert len(state.cached.transcript) == 2

    def test_when_apply_turn_called_twice_then_each_panel_transcript_has_four_entries(
        self,
    ):
        state = RenderState()
        state.apply_turn(_make_turn_result(index=0))
        state.apply_turn(_make_turn_result(index=1))
        assert len(state.uncached.transcript) == 4
        assert len(state.cached.transcript) == 4

    # --- Running costs ---

    def test_when_apply_turn_then_uncached_cost_is_added_to_uncached_running_cost(self):
        state = RenderState()
        state.apply_turn(_make_turn_result(uncached_cost=0.10))
        assert state.uncached.running_cost == pytest.approx(0.10)

    def test_when_apply_turn_then_cached_cost_is_added_to_cached_running_cost(self):
        state = RenderState()
        state.apply_turn(_make_turn_result(cached_cost=0.01))
        assert state.cached.running_cost == pytest.approx(0.01)

    def test_when_apply_turn_called_twice_then_uncached_running_cost_accumulates(self):
        state = RenderState()
        state.apply_turn(_make_turn_result(uncached_cost=0.10))
        state.apply_turn(_make_turn_result(uncached_cost=0.12))
        assert state.uncached.running_cost == pytest.approx(0.22)

    def test_when_apply_turn_called_twice_then_cached_running_cost_accumulates(self):
        state = RenderState()
        state.apply_turn(_make_turn_result(cached_cost=0.01))
        state.apply_turn(_make_turn_result(cached_cost=0.015))
        assert state.cached.running_cost == pytest.approx(0.025)

    # --- Turn number ---

    def test_when_apply_turn_then_turn_number_is_set_to_result_index(self):
        """
        Chosen interpretation: turn_number == TurnResult.index (0-based).
        If the implementation is 1-based, update accordingly.
        """
        state = RenderState()
        result = _make_turn_result(index=3)
        state.apply_turn(result)
        assert state.uncached.turn_number == result.index

    def test_when_apply_turn_then_cached_panel_turn_number_is_also_set(self):
        state = RenderState()
        result = _make_turn_result(index=5)
        state.apply_turn(result)
        assert state.cached.turn_number == result.index

    # --- this_turn_cache_read ---

    def test_when_apply_turn_then_cached_panel_cache_read_is_set_from_cached_metrics(
        self,
    ):
        """Criterion: 'sets this_turn_cache_read from the cached side's reads'."""
        state = RenderState()
        result = _make_turn_result(cache_read_tokens=7_500)
        state.apply_turn(result)
        assert state.cached.this_turn_cache_read == 7_500

    def test_when_apply_turn_then_cache_read_reflects_turn_value_not_accumulation(self):
        """this_turn_cache_read is per-turn, not cumulative — second turn overwrites."""
        state = RenderState()
        state.apply_turn(_make_turn_result(cache_read_tokens=5_000))
        state.apply_turn(_make_turn_result(cache_read_tokens=8_000))
        assert state.cached.this_turn_cache_read == 8_000

    # --- last_turn_delta ---

    def test_when_apply_turn_then_last_turn_delta_is_set_on_uncached_panel(self):
        """
        Chosen interpretation: last_turn_delta is the per-panel per-turn cost.
        The 'delta' from zero for the uncached panel = result.uncached.cost_usd.
        """
        state = RenderState()
        result = _make_turn_result(uncached_cost=0.10, cached_cost=0.01)
        state.apply_turn(result)
        assert state.uncached.last_turn_delta == pytest.approx(0.10)

    def test_when_apply_turn_then_last_turn_delta_is_set_on_cached_panel(self):
        state = RenderState()
        result = _make_turn_result(uncached_cost=0.10, cached_cost=0.01)
        state.apply_turn(result)
        assert state.cached.last_turn_delta == pytest.approx(0.01)

    def test_when_apply_turn_called_twice_then_last_turn_delta_reflects_latest_turn(
        self,
    ):
        """last_turn_delta is per-turn (not cumulative) — second turn overwrites first."""
        state = RenderState()
        state.apply_turn(_make_turn_result(uncached_cost=0.10))
        state.apply_turn(_make_turn_result(uncached_cost=0.15))
        assert state.uncached.last_turn_delta == pytest.approx(0.15)

    # --- Cost-strip totals ---

    def test_when_apply_turn_then_cost_strip_uncached_total_is_updated(self):
        state = RenderState()
        state.apply_turn(_make_turn_result(uncached_cost=0.10, cached_cost=0.01))
        assert state.cost_strip.uncached_total == pytest.approx(0.10)

    def test_when_apply_turn_then_cost_strip_cached_total_is_updated(self):
        state = RenderState()
        state.apply_turn(_make_turn_result(uncached_cost=0.10, cached_cost=0.01))
        assert state.cost_strip.cached_total == pytest.approx(0.01)

    def test_when_apply_turn_called_twice_then_cost_strip_totals_accumulate(self):
        state = RenderState()
        state.apply_turn(_make_turn_result(uncached_cost=0.10, cached_cost=0.01))
        state.apply_turn(_make_turn_result(uncached_cost=0.12, cached_cost=0.015))
        assert state.cost_strip.uncached_total == pytest.approx(0.22)
        assert state.cost_strip.cached_total == pytest.approx(0.025)

    # --- live_partial cleared ---

    def test_when_apply_turn_then_uncached_live_partial_is_cleared(self):
        from dejavu.runner import Side, TokenEvent

        state = RenderState()
        state.apply_token(
            TokenEvent(index=0, side=Side.UNCACHED, partial="in progress")
        )
        state.apply_turn(_make_turn_result())
        assert state.uncached.live_partial == ""

    def test_when_apply_turn_then_cached_live_partial_is_cleared(self):
        from dejavu.runner import Side, TokenEvent

        state = RenderState()
        state.apply_token(TokenEvent(index=0, side=Side.CACHED, partial="streaming"))
        state.apply_turn(_make_turn_result())
        assert state.cached.live_partial == ""


# ---------------------------------------------------------------------------
# AC-4: CostStripState.multiple(), saved(), multiple_text() + divide-by-zero guard
# ---------------------------------------------------------------------------


class TestCostStripState:
    """AC-4: CostStripState computes multiple, saved, and formatted text correctly."""

    def test_when_cached_total_is_zero_then_multiple_does_not_raise(self):
        """Divide-by-zero guard: must return a float, never raise ZeroDivisionError."""
        cs = CostStripState(uncached_total=1.0, cached_total=0.0)
        result = cs.multiple()
        assert isinstance(result, float)

    def test_when_cached_total_equals_uncached_total_then_multiple_is_one(self):
        cs = CostStripState(uncached_total=2.0, cached_total=2.0)
        assert cs.multiple() == pytest.approx(1.0)

    def test_when_uncached_is_ten_times_cached_then_multiple_is_ten(self):
        cs = CostStripState(uncached_total=1.0, cached_total=0.1)
        assert cs.multiple() == pytest.approx(10.0)

    def test_when_uncached_is_five_times_cached_then_multiple_is_five(self):
        cs = CostStripState(uncached_total=5.0, cached_total=1.0)
        assert cs.multiple() == pytest.approx(5.0)

    def test_when_both_totals_are_zero_then_multiple_does_not_raise(self):
        cs = CostStripState(uncached_total=0.0, cached_total=0.0)
        result = cs.multiple()
        assert isinstance(result, float)

    def test_when_multiple_is_called_then_it_returns_a_float(self):
        cs = CostStripState(uncached_total=3.0, cached_total=1.0)
        assert isinstance(cs.multiple(), float)

    def test_when_saved_is_called_then_it_returns_uncached_minus_cached(self):
        cs = CostStripState(uncached_total=1.0, cached_total=0.3)
        assert cs.saved() == pytest.approx(0.7)

    def test_when_both_totals_are_equal_then_saved_is_zero(self):
        cs = CostStripState(uncached_total=1.5, cached_total=1.5)
        assert cs.saved() == pytest.approx(0.0)

    def test_when_both_totals_are_zero_then_saved_is_zero(self):
        cs = CostStripState(uncached_total=0.0, cached_total=0.0)
        assert cs.saved() == pytest.approx(0.0)

    def test_when_saved_is_called_then_it_returns_a_float(self):
        cs = CostStripState(uncached_total=3.0, cached_total=1.0)
        assert isinstance(cs.saved(), float)

    def test_when_multiple_text_is_called_then_it_returns_a_string(self):
        cs = CostStripState(uncached_total=5.0, cached_total=1.0)
        assert isinstance(cs.multiple_text(), str)

    def test_when_multiple_text_is_called_then_the_multiple_value_is_represented(self):
        """multiple() == 5.0 → multiple_text() must include '5' in some form."""
        cs = CostStripState(uncached_total=5.0, cached_total=1.0)
        assert "5" in cs.multiple_text()

    def test_when_multiple_text_is_called_with_ten_times_multiple_then_ten_is_in_text(
        self,
    ):
        cs = CostStripState(uncached_total=10.0, cached_total=1.0)
        assert "10" in cs.multiple_text()

    def test_when_multiple_text_called_with_zero_cached_then_it_does_not_raise(self):
        """Divide-by-zero guard also covers multiple_text()."""
        cs = CostStripState(uncached_total=1.0, cached_total=0.0)
        result = cs.multiple_text()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AC-5: Formatting helpers — fixed-precision, right-aligned output for legibility
# ---------------------------------------------------------------------------


class TestFormattingHelpers:
    """AC-5: _fmt_usd and _fmt_int produce fixed-precision, right-aligned strings."""

    # --- _fmt_usd ---

    def test_when_fmt_usd_is_called_then_result_is_a_string(self):
        assert isinstance(_fmt_usd(1.23), str)

    def test_when_fmt_usd_is_called_then_result_contains_dollar_sign(self):
        assert "$" in _fmt_usd(1.23)

    def test_when_fmt_usd_is_called_then_result_has_no_trailing_whitespace(self):
        """Right-aligned: padding goes on the left, not the right."""
        result = _fmt_usd(1.23)
        assert result == result.rstrip()

    def test_when_fmt_usd_is_called_with_zero_then_result_contains_zero(self):
        assert "0" in _fmt_usd(0.0)

    def test_when_fmt_usd_is_called_with_different_values_then_decimal_places_are_consistent(
        self,
    ):
        """Fixed precision: both small and large values must have the same decimal count."""

        def _dec_places(s: str) -> int:
            return len(s.split(".")[-1]) if "." in s else 0

        a = _fmt_usd(0.001)
        b = _fmt_usd(999.5)
        assert _dec_places(a) == _dec_places(b), (
            f"_fmt_usd must have fixed decimal precision: {a!r} vs {b!r}"
        )

    def test_when_fmt_usd_is_called_with_different_magnitudes_then_output_width_is_equal(
        self,
    ):
        """Right-aligned: strings are padded to a fixed column width."""
        a = _fmt_usd(0.0001)
        b = _fmt_usd(1_234.56)
        assert len(a) == len(b), (
            f"_fmt_usd must produce fixed-width right-aligned output: "
            f"{a!r} ({len(a)} chars) vs {b!r} ({len(b)} chars)"
        )

    def test_when_fmt_usd_is_called_with_small_value_then_decimal_places_are_at_least_two(
        self,
    ):
        """Legibility for small amounts requires at least 2 decimal places."""

        def _dec_places(s: str) -> int:
            return len(s.split(".")[-1]) if "." in s else 0

        assert _dec_places(_fmt_usd(0.001)) >= 2

    # --- _fmt_int ---

    def test_when_fmt_int_is_called_then_result_is_a_string(self):
        assert isinstance(_fmt_int(1_000), str)

    def test_when_fmt_int_is_called_then_result_has_no_trailing_whitespace(self):
        result = _fmt_int(1_234)
        assert result == result.rstrip()

    def test_when_fmt_int_is_called_with_zero_then_result_contains_zero(self):
        assert "0" in _fmt_int(0)

    def test_when_fmt_int_is_called_with_value_then_all_digits_are_represented(self):
        """The numeric value must survive into the output string."""
        result = _fmt_int(12345)
        digits_only = "".join(c for c in result if c.isdigit())
        assert "12345" in digits_only, (
            f"_fmt_int(12345) must contain the digits 12345, got {result!r}"
        )

    def test_when_fmt_int_is_called_with_different_values_then_output_width_is_equal(
        self,
    ):
        """Right-aligned: strings are padded to a fixed column width."""
        a = _fmt_int(0)
        b = _fmt_int(50_000)
        assert len(a) == len(b), (
            f"_fmt_int must produce fixed-width right-aligned output: "
            f"{a!r} ({len(a)} chars) vs {b!r} ({len(b)} chars)"
        )


# ---------------------------------------------------------------------------
# AC-6: Property tests — multiple() >= 1 and saved() >= 0 when cached <= uncached
# ---------------------------------------------------------------------------

_NON_NEG_FLOAT = st.floats(
    min_value=0.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)


@given(cached_total=_NON_NEG_FLOAT, extra=_NON_NEG_FLOAT)
@h_settings(max_examples=300)
def test_when_cached_total_lte_uncached_total_then_multiple_is_at_least_one(
    cached_total: float,
    extra: float,
) -> None:
    """
    AC-6 invariant: multiple() >= 1.0 for all cached_total <= uncached_total.
    uncached_total = cached_total + extra guarantees the ordering condition.
    The divide-by-zero guard must also satisfy this (return >= 1.0 when cached == 0).
    """
    uncached_total = cached_total + extra
    cs = CostStripState(uncached_total=uncached_total, cached_total=cached_total)
    assert cs.multiple() >= 1.0


@given(cached_total=_NON_NEG_FLOAT, extra=_NON_NEG_FLOAT)
@h_settings(max_examples=300)
def test_when_cached_total_lte_uncached_total_then_saved_is_non_negative(
    cached_total: float,
    extra: float,
) -> None:
    """
    AC-6 invariant: saved() >= 0.0 for all cached_total <= uncached_total.
    uncached_total = cached_total + extra guarantees cached_total <= uncached_total.
    """
    uncached_total = cached_total + extra
    cs = CostStripState(uncached_total=uncached_total, cached_total=cached_total)
    assert cs.saved() >= 0.0
