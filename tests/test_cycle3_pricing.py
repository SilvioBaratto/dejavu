"""
Source-blind tests for issue #7: Anthropic pricing engine.

All tests are derived solely from acceptance criteria and requirements.md.
No implementation source was read; no dejavu/pricing.py internals are assumed
beyond what the criteria explicitly state must exist.

Criteria tested (per oracle report):
  AC-1  [UNIT] Model str-Enum and frozen Price dataclass exposed from pricing.py
  AC-2  [UNIT] Price table reproduces every spec cell; Mythos row is the Fable row
  AC-3  [UNIT] Tier selector picks correct write column; default tier is "5m"
  AC-4  [UNIT] price_uncached_turn returns input×base + output×output per-MTok
  AC-5  [UNIT] price_cached_turn returns four-term rolling formula for both tiers
  AC-6  [UNIT] MTOK == 1_000_000; sub-MTok > 0.0; zero-token == 0.0; no rounding
  AC-7  [UNIT] dejavu/__init__.py re-exports pricing public API; appends to __all__

Skipped (per oracle):
  AC-8  All tests pass        — boilerplate suite gate; no per-criterion assertion
  AC-9  SOLID/clean code      — subjective prose; not runtime-verifiable
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings as h_settings, strategies as st

import dejavu
from dejavu.pricing import (
    MTOK,
    PRICES,
    Model,
    Price,
    get_price,
    price_cached_turn,
    price_uncached_turn,
    write_price,
)

# ---------------------------------------------------------------------------
# Module-level helpers (derived from criteria, not from implementation)
# ---------------------------------------------------------------------------

_OPUS = Model("opus-4.8")
_FABLE = Model("fable-5")
_MYTHOS = Model("mythos-5")


# ---------------------------------------------------------------------------
# AC-1: Model str-Enum with three members + frozen Price with five fields
# ---------------------------------------------------------------------------


class TestModelEnum:
    """AC-1: Model is a str-Enum whose values are the three model slugs."""

    def test_when_opus_48_slug_is_used_then_model_member_is_returned(self):
        assert Model("opus-4.8") is not None

    def test_when_fable_5_slug_is_used_then_model_member_is_returned(self):
        assert Model("fable-5") is not None

    def test_when_mythos_5_slug_is_used_then_model_member_is_returned(self):
        assert Model("mythos-5") is not None

    def test_when_model_member_is_inspected_then_it_is_a_str_subclass(self):
        for m in Model:
            assert isinstance(m, str), f"{m!r} must be an instance of str"

    def test_when_model_values_are_listed_then_all_three_slugs_are_present(self):
        values = {m.value for m in Model}
        assert {"opus-4.8", "fable-5", "mythos-5"}.issubset(values)

    def test_when_invalid_slug_is_used_then_value_error_is_raised(self):
        with pytest.raises(ValueError):
            Model("not-a-real-model")


class TestPriceDataclass:
    """AC-1: Price is a frozen dataclass with the five pricing fields."""

    def test_when_price_row_is_retrieved_then_base_field_exists(self):
        p = get_price(_OPUS)
        assert hasattr(p, "base")

    def test_when_price_row_is_retrieved_then_write_5m_field_exists(self):
        p = get_price(_OPUS)
        assert hasattr(p, "write_5m")

    def test_when_price_row_is_retrieved_then_write_1h_field_exists(self):
        p = get_price(_OPUS)
        assert hasattr(p, "write_1h")

    def test_when_price_row_is_retrieved_then_read_field_exists(self):
        p = get_price(_OPUS)
        assert hasattr(p, "read")

    def test_when_price_row_is_retrieved_then_output_field_exists(self):
        p = get_price(_OPUS)
        assert hasattr(p, "output")

    def test_when_price_row_is_mutated_then_frozen_error_is_raised(self):
        """Price must be frozen (immutable), not a plain mutable object."""
        p = get_price(_OPUS)
        with pytest.raises((AttributeError, TypeError)):
            p.base = 0.0  # type: ignore[misc]

    def test_when_price_instance_is_checked_then_it_is_a_price(self):
        p = get_price(_OPUS)
        assert isinstance(p, Price)


# ---------------------------------------------------------------------------
# AC-2: Price table cell values + MYTHOS_5 shares the same row as FABLE_5
# ---------------------------------------------------------------------------


class TestPriceTableValues:
    """AC-2: Every spec cell reproduced exactly for all three models."""

    # --- Opus 4.8 ---

    def test_when_opus_base_is_retrieved_then_it_equals_5_00(self):
        assert get_price(_OPUS).base == pytest.approx(5.00)

    def test_when_opus_write_5m_is_retrieved_then_it_equals_6_25(self):
        assert get_price(_OPUS).write_5m == pytest.approx(6.25)

    def test_when_opus_write_1h_is_retrieved_then_it_equals_10_00(self):
        assert get_price(_OPUS).write_1h == pytest.approx(10.00)

    def test_when_opus_read_is_retrieved_then_it_equals_0_50(self):
        assert get_price(_OPUS).read == pytest.approx(0.50)

    def test_when_opus_output_is_retrieved_then_it_equals_25_00(self):
        assert get_price(_OPUS).output == pytest.approx(25.00)

    # --- Fable 5 ---

    def test_when_fable_base_is_retrieved_then_it_equals_10_00(self):
        assert get_price(_FABLE).base == pytest.approx(10.00)

    def test_when_fable_write_5m_is_retrieved_then_it_equals_12_50(self):
        assert get_price(_FABLE).write_5m == pytest.approx(12.50)

    def test_when_fable_write_1h_is_retrieved_then_it_equals_20_00(self):
        assert get_price(_FABLE).write_1h == pytest.approx(20.00)

    def test_when_fable_read_is_retrieved_then_it_equals_1_00(self):
        assert get_price(_FABLE).read == pytest.approx(1.00)

    def test_when_fable_output_is_retrieved_then_it_equals_50_00(self):
        assert get_price(_FABLE).output == pytest.approx(50.00)

    # --- Mythos 5 (must mirror Fable 5 exactly) ---

    def test_when_mythos_base_is_retrieved_then_it_equals_10_00(self):
        assert get_price(_MYTHOS).base == pytest.approx(10.00)

    def test_when_mythos_write_5m_is_retrieved_then_it_equals_12_50(self):
        assert get_price(_MYTHOS).write_5m == pytest.approx(12.50)

    def test_when_mythos_write_1h_is_retrieved_then_it_equals_20_00(self):
        assert get_price(_MYTHOS).write_1h == pytest.approx(20.00)

    def test_when_mythos_read_is_retrieved_then_it_equals_1_00(self):
        assert get_price(_MYTHOS).read == pytest.approx(1.00)

    def test_when_mythos_output_is_retrieved_then_it_equals_50_00(self):
        assert get_price(_MYTHOS).output == pytest.approx(50.00)

    # --- Identity: Mythos and Fable share the same row instance ---

    def test_when_mythos_and_fable_rows_are_compared_then_they_are_equal(self):
        """Value equality: every cell is numerically identical."""
        assert PRICES[_MYTHOS] == PRICES[_FABLE]

    def test_when_mythos_and_fable_rows_are_compared_then_they_are_the_same_object(
        self,
    ):
        """Instance identity: both keys point to the same Price object.

        Chosen interpretation: the spec says 'point both at the same Price row
        so the values cannot drift'.  Object identity (is) is the only check
        that fails when a future edit accidentally gives Mythos its own literal
        row — value equality (==) would silently pass even then.
        """
        assert PRICES[_MYTHOS] is PRICES[_FABLE]


# ---------------------------------------------------------------------------
# AC-3: Tier selector — correct write column; default is "5m"
# ---------------------------------------------------------------------------


class TestTierSelector:
    """AC-3: write_price selects write_5m or write_1h; default tier is '5m'."""

    def test_when_tier_is_5m_then_write_price_returns_write_5m(self):
        assert write_price(_OPUS, "5m") == pytest.approx(get_price(_OPUS).write_5m)

    def test_when_tier_is_1h_then_write_price_returns_write_1h(self):
        assert write_price(_OPUS, "1h") == pytest.approx(get_price(_OPUS).write_1h)

    def test_when_tier_is_5m_then_fable_write_price_returns_12_50(self):
        assert write_price(_FABLE, "5m") == pytest.approx(12.50)

    def test_when_tier_is_1h_then_fable_write_price_returns_20_00(self):
        assert write_price(_FABLE, "1h") == pytest.approx(20.00)

    def test_when_no_tier_is_given_then_default_is_5m(self):
        """Default tier must be '5m': write_price(m) == write_price(m, '5m')."""
        assert write_price(_OPUS) == pytest.approx(write_price(_OPUS, "5m"))

    def test_when_no_tier_is_given_for_fable_then_default_is_5m(self):
        assert write_price(_FABLE) == pytest.approx(write_price(_FABLE, "5m"))

    def test_when_invalid_tier_is_given_then_error_is_raised(self):
        with pytest.raises((ValueError, KeyError, TypeError)):
            write_price(_OPUS, "invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC-4: price_uncached_turn = input×base + output×output_price  (per-MTok)
# ---------------------------------------------------------------------------


class TestPriceUncachedTurn:
    """AC-4: per-MTok formula: input_tokens×base + output_tokens×output."""

    def test_when_1m_input_tokens_opus_then_cost_equals_base_price(self):
        # 1_000_000 input, 0 output → 1.0 MTok × $5.00 = $5.00
        cost = price_uncached_turn(model=_OPUS, input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(5.00)

    def test_when_1m_output_tokens_opus_then_cost_equals_output_price(self):
        # 0 input, 1_000_000 output → 1.0 MTok × $25.00 = $25.00
        cost = price_uncached_turn(model=_OPUS, input_tokens=0, output_tokens=1_000_000)
        assert cost == pytest.approx(25.00)

    def test_when_mixed_tokens_opus_then_cost_matches_hand_computed_value(self):
        # 1 000 input, 500 output (Opus 4.8)
        # = 1000/1_000_000 × 5.00  +  500/1_000_000 × 25.00
        # = 0.005 + 0.0125 = 0.0175
        cost = price_uncached_turn(model=_OPUS, input_tokens=1_000, output_tokens=500)
        assert cost == pytest.approx(0.0175)

    def test_when_mixed_tokens_fable_then_cost_matches_hand_computed_value(self):
        # 2 000 input, 100 output (Fable 5)
        # = 2000/1_000_000 × 10.00  +  100/1_000_000 × 50.00
        # = 0.02 + 0.005 = 0.025
        cost = price_uncached_turn(model=_FABLE, input_tokens=2_000, output_tokens=100)
        assert cost == pytest.approx(0.025)

    def test_when_zero_tokens_opus_then_cost_is_zero(self):
        cost = price_uncached_turn(model=_OPUS, input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AC-5: price_cached_turn — four-term rolling formula, both tiers
# ---------------------------------------------------------------------------


class TestPriceCachedTurn:
    """AC-5: read×read + creation×write + fresh×base + output×output rate."""

    # Shared fixture values for the hand-computed checks:
    #   cache_read=50_000, cache_creation=5_000, fresh=500, output=300
    _CR = 50_000
    _CW = 5_000
    _FQ = 500
    _OUT = 300

    def test_when_cached_turn_5m_opus_then_cost_matches_hand_computed_value(self):
        # Opus 4.8 / 5m tier:
        # = 50000/1e6×0.50 + 5000/1e6×6.25 + 500/1e6×5.00 + 300/1e6×25.00
        # = 0.025 + 0.03125 + 0.0025 + 0.0075
        # = 0.06625
        cost = price_cached_turn(
            model=_OPUS,
            tier="5m",
            cache_read_tokens=self._CR,
            cache_creation_tokens=self._CW,
            fresh_question_tokens=self._FQ,
            output_tokens=self._OUT,
        )
        assert cost == pytest.approx(0.06625)

    def test_when_cached_turn_1h_opus_then_cost_matches_hand_computed_value(self):
        # Opus 4.8 / 1h tier (write_1h=10.00 instead of 6.25):
        # = 50000/1e6×0.50 + 5000/1e6×10.00 + 500/1e6×5.00 + 300/1e6×25.00
        # = 0.025 + 0.05 + 0.0025 + 0.0075
        # = 0.085
        cost = price_cached_turn(
            model=_OPUS,
            tier="1h",
            cache_read_tokens=self._CR,
            cache_creation_tokens=self._CW,
            fresh_question_tokens=self._FQ,
            output_tokens=self._OUT,
        )
        assert cost == pytest.approx(0.085)

    def test_when_cached_turn_5m_fable_then_cost_matches_hand_computed_value(self):
        # Fable 5 / 5m tier (read=1.00, write_5m=12.50, base=10.00, output=50.00):
        # cache_read=10_000, cache_creation=2_000, fresh=200, output=100
        # = 10000/1e6×1.00 + 2000/1e6×12.50 + 200/1e6×10.00 + 100/1e6×50.00
        # = 0.01 + 0.025 + 0.002 + 0.005
        # = 0.042
        cost = price_cached_turn(
            model=_FABLE,
            tier="5m",
            cache_read_tokens=10_000,
            cache_creation_tokens=2_000,
            fresh_question_tokens=200,
            output_tokens=100,
        )
        assert cost == pytest.approx(0.042)

    def test_when_cached_turn_1h_fable_then_cost_matches_hand_computed_value(self):
        # Fable 5 / 1h tier (write_1h=20.00):
        # = 10000/1e6×1.00 + 2000/1e6×20.00 + 200/1e6×10.00 + 100/1e6×50.00
        # = 0.01 + 0.04 + 0.002 + 0.005
        # = 0.057
        cost = price_cached_turn(
            model=_FABLE,
            tier="1h",
            cache_read_tokens=10_000,
            cache_creation_tokens=2_000,
            fresh_question_tokens=200,
            output_tokens=100,
        )
        assert cost == pytest.approx(0.057)

    def test_when_all_tokens_are_zero_then_cached_turn_cost_is_zero(self):
        cost = price_cached_turn(
            model=_OPUS,
            tier="5m",
            cache_read_tokens=0,
            cache_creation_tokens=0,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost == pytest.approx(0.0)

    def test_when_only_cache_read_tokens_then_cost_uses_read_rate(self):
        # 1_000_000 cache_read tokens, all others 0, Opus 4.8
        # = 1_000_000/1_000_000 × 0.50 = 0.50
        cost = price_cached_turn(
            model=_OPUS,
            tier="5m",
            cache_read_tokens=1_000_000,
            cache_creation_tokens=0,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost == pytest.approx(0.50)

    def test_when_only_cache_creation_tokens_5m_then_cost_uses_write_5m_rate(self):
        # 1_000_000 cache_creation tokens, all others 0, Opus 4.8, 5m tier
        # = 1_000_000/1_000_000 × 6.25 = 6.25
        cost = price_cached_turn(
            model=_OPUS,
            tier="5m",
            cache_read_tokens=0,
            cache_creation_tokens=1_000_000,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost == pytest.approx(6.25)

    def test_when_only_cache_creation_tokens_1h_then_cost_uses_write_1h_rate(self):
        # 1_000_000 cache_creation tokens, all others 0, Opus 4.8, 1h tier
        # = 1_000_000/1_000_000 × 10.00 = 10.00
        cost = price_cached_turn(
            model=_OPUS,
            tier="1h",
            cache_read_tokens=0,
            cache_creation_tokens=1_000_000,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost == pytest.approx(10.00)


# ---------------------------------------------------------------------------
# AC-6: MTOK constant; sub-MTok > 0.0; zero-token == 0.0; no rounding
# ---------------------------------------------------------------------------


class TestMTokDivision:
    """AC-6: MTOK == 1_000_000; sub-MTok costs are strictly positive; no floor."""

    def test_when_mtok_constant_is_inspected_then_it_equals_one_million(self):
        assert MTOK == 1_000_000

    def test_when_single_input_token_is_priced_uncached_then_cost_is_strictly_positive(
        self,
    ):
        """Sub-MTok input must yield a non-floored, strictly positive cost."""
        cost = price_uncached_turn(model=_OPUS, input_tokens=1, output_tokens=0)
        assert cost > 0.0

    def test_when_single_output_token_is_priced_uncached_then_cost_is_strictly_positive(
        self,
    ):
        cost = price_uncached_turn(model=_OPUS, input_tokens=0, output_tokens=1)
        assert cost > 0.0

    def test_when_zero_tokens_are_priced_uncached_then_cost_is_exactly_zero(self):
        cost = price_uncached_turn(model=_OPUS, input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_when_single_cache_read_token_is_priced_then_cost_is_strictly_positive(
        self,
    ):
        cost = price_cached_turn(
            model=_OPUS,
            tier="5m",
            cache_read_tokens=1,
            cache_creation_tokens=0,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost > 0.0

    def test_when_single_cache_creation_token_is_priced_then_cost_is_strictly_positive(
        self,
    ):
        cost = price_cached_turn(
            model=_OPUS,
            tier="5m",
            cache_read_tokens=0,
            cache_creation_tokens=1,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost > 0.0


# ---------------------------------------------------------------------------
# Property-based tests — AC-6 monotonicity invariants
# ---------------------------------------------------------------------------

_NON_NEG_INT = st.integers(min_value=0, max_value=10_000_000)
_POS_INT = st.integers(min_value=1, max_value=10_000_000)


@given(input_tokens=_NON_NEG_INT, output_tokens=_NON_NEG_INT)
def test_when_non_negative_tokens_then_uncached_cost_is_non_negative(
    input_tokens: int,
    output_tokens: int,
) -> None:
    """price_uncached_turn >= 0.0 for all non-negative token counts (Opus)."""
    cost = price_uncached_turn(
        model=_OPUS, input_tokens=input_tokens, output_tokens=output_tokens
    )
    assert cost >= 0.0


@given(input_tokens=_POS_INT)
def test_when_positive_input_tokens_are_given_then_uncached_cost_is_strictly_positive(
    input_tokens: int,
) -> None:
    """Any positive input_tokens must produce a strictly positive cost — no floor."""
    cost = price_uncached_turn(model=_OPUS, input_tokens=input_tokens, output_tokens=0)
    assert cost > 0.0


@given(output_tokens=_POS_INT)
def test_when_positive_output_tokens_are_given_then_uncached_cost_is_strictly_positive(
    output_tokens: int,
) -> None:
    """Any positive output_tokens must produce a strictly positive cost — no floor."""
    cost = price_uncached_turn(model=_OPUS, input_tokens=0, output_tokens=output_tokens)
    assert cost > 0.0


@given(
    cache_read_tokens=_NON_NEG_INT,
    cache_creation_tokens=_NON_NEG_INT,
    fresh_question_tokens=_NON_NEG_INT,
    output_tokens=_NON_NEG_INT,
)
@h_settings(max_examples=50)
def test_when_any_non_negative_cached_tokens_are_given_then_cached_cost_is_non_negative(
    cache_read_tokens: int,
    cache_creation_tokens: int,
    fresh_question_tokens: int,
    output_tokens: int,
) -> None:
    """price_cached_turn >= 0.0 for all non-negative token counts."""
    cost = price_cached_turn(
        model=_OPUS,
        tier="5m",
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        fresh_question_tokens=fresh_question_tokens,
        output_tokens=output_tokens,
    )
    assert cost >= 0.0


@given(more_input=_POS_INT, baseline_input=_NON_NEG_INT, output_tokens=_NON_NEG_INT)
def test_when_input_tokens_increase_then_uncached_cost_is_monotonically_non_decreasing(
    more_input: int,
    baseline_input: int,
    output_tokens: int,
) -> None:
    """More input tokens must never produce a lower cost (monotonicity)."""
    cost_low = price_uncached_turn(
        model=_OPUS, input_tokens=baseline_input, output_tokens=output_tokens
    )
    cost_high = price_uncached_turn(
        model=_OPUS,
        input_tokens=baseline_input + more_input,
        output_tokens=output_tokens,
    )
    assert cost_high >= cost_low


# ---------------------------------------------------------------------------
# AC-7: dejavu/__init__.py re-exports the public pricing API
# ---------------------------------------------------------------------------


class TestInitReexports:
    """AC-7: Public pricing symbols importable from top-level dejavu package."""

    def test_when_model_is_imported_from_dejavu_then_it_is_accessible(self):
        from dejavu import Model as M  # noqa: F401 (import is the assertion)

        assert M is not None

    def test_when_price_is_imported_from_dejavu_then_it_is_accessible(self):
        from dejavu import Price as P  # noqa: F401

        assert P is not None

    def test_when_price_uncached_turn_is_imported_from_dejavu_then_it_is_callable(self):
        from dejavu import price_uncached_turn as f  # noqa: F401

        assert callable(f)

    def test_when_price_cached_turn_is_imported_from_dejavu_then_it_is_callable(self):
        from dejavu import price_cached_turn as f  # noqa: F401

        assert callable(f)

    def test_when_dejavu_all_is_inspected_then_model_is_listed(self):
        assert "Model" in dejavu.__all__

    def test_when_dejavu_all_is_inspected_then_price_is_listed(self):
        assert "Price" in dejavu.__all__

    def test_when_dejavu_all_is_inspected_then_price_uncached_turn_is_listed(self):
        assert "price_uncached_turn" in dejavu.__all__

    def test_when_dejavu_all_is_inspected_then_price_cached_turn_is_listed(self):
        assert "price_cached_turn" in dejavu.__all__
