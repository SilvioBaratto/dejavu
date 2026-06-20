"""
Source-blind tests for issue #7: Anthropic pricing engine.

All tests are derived solely from acceptance criteria and requirements.md.
No implementation source was read; no dejavu/pricing.py internals are assumed
beyond what the criteria explicitly state must exist.

Criteria tested (per oracle report):
  AC-1  [UNIT] Model str-Enum and frozen Price dataclass exposed from pricing.py
  AC-2  [UNIT] Price table reproduces every spec cell for the single Sonnet 4.6 model
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

_SONNET = Model("sonnet-4.6")


# ---------------------------------------------------------------------------
# AC-1: Model str-Enum with one member + frozen Price with five fields
# ---------------------------------------------------------------------------


class TestModelEnum:
    """AC-1: Model is a str-Enum whose value is the single model slug."""

    def test_when_sonnet_46_slug_is_used_then_model_member_is_returned(self):
        assert Model("sonnet-4.6") is not None

    def test_when_model_member_is_inspected_then_it_is_a_str_subclass(self):
        for m in Model:
            assert isinstance(m, str), f"{m!r} must be an instance of str"

    def test_when_model_values_are_listed_then_the_single_slug_is_present(self):
        values = {m.value for m in Model}
        assert values == {"sonnet-4.6"}

    def test_when_invalid_slug_is_used_then_value_error_is_raised(self):
        with pytest.raises(ValueError):
            Model("not-a-real-model")


class TestPriceDataclass:
    """AC-1: Price is a frozen dataclass with the five pricing fields."""

    def test_when_price_row_is_retrieved_then_base_field_exists(self):
        p = get_price(_SONNET)
        assert hasattr(p, "base")

    def test_when_price_row_is_retrieved_then_write_5m_field_exists(self):
        p = get_price(_SONNET)
        assert hasattr(p, "write_5m")

    def test_when_price_row_is_retrieved_then_write_1h_field_exists(self):
        p = get_price(_SONNET)
        assert hasattr(p, "write_1h")

    def test_when_price_row_is_retrieved_then_read_field_exists(self):
        p = get_price(_SONNET)
        assert hasattr(p, "read")

    def test_when_price_row_is_retrieved_then_output_field_exists(self):
        p = get_price(_SONNET)
        assert hasattr(p, "output")

    def test_when_price_row_is_mutated_then_frozen_error_is_raised(self):
        """Price must be frozen (immutable), not a plain mutable object."""
        p = get_price(_SONNET)
        with pytest.raises((AttributeError, TypeError)):
            p.base = 0.0  # type: ignore[misc]

    def test_when_price_instance_is_checked_then_it_is_a_price(self):
        p = get_price(_SONNET)
        assert isinstance(p, Price)


# ---------------------------------------------------------------------------
# AC-2: Price table cell values for the single Sonnet 4.6 model
# ---------------------------------------------------------------------------


class TestPriceTableValues:
    """AC-2: Every spec cell reproduced exactly for the single model."""

    # --- Sonnet 4.6 ---

    def test_when_sonnet_base_is_retrieved_then_it_equals_3_00(self):
        assert get_price(_SONNET).base == pytest.approx(3.00)

    def test_when_sonnet_write_5m_is_retrieved_then_it_equals_3_75(self):
        assert get_price(_SONNET).write_5m == pytest.approx(3.75)

    def test_when_sonnet_write_1h_is_retrieved_then_it_equals_6_00(self):
        assert get_price(_SONNET).write_1h == pytest.approx(6.00)

    def test_when_sonnet_read_is_retrieved_then_it_equals_0_30(self):
        assert get_price(_SONNET).read == pytest.approx(0.30)

    def test_when_sonnet_output_is_retrieved_then_it_equals_15_00(self):
        assert get_price(_SONNET).output == pytest.approx(15.00)

    # --- The price table holds exactly the single Sonnet 4.6 row ---

    def test_when_prices_keys_are_listed_then_only_sonnet_46_is_present(self):
        assert set(PRICES.keys()) == {_SONNET}


# ---------------------------------------------------------------------------
# AC-3: Tier selector — correct write column; default is "5m"
# ---------------------------------------------------------------------------


class TestTierSelector:
    """AC-3: write_price selects write_5m or write_1h; default tier is '5m'."""

    def test_when_tier_is_5m_then_write_price_returns_write_5m(self):
        assert write_price(_SONNET, "5m") == pytest.approx(get_price(_SONNET).write_5m)

    def test_when_tier_is_1h_then_write_price_returns_write_1h(self):
        assert write_price(_SONNET, "1h") == pytest.approx(get_price(_SONNET).write_1h)

    def test_when_tier_is_5m_then_sonnet_write_price_returns_3_75(self):
        assert write_price(_SONNET, "5m") == pytest.approx(3.75)

    def test_when_tier_is_1h_then_sonnet_write_price_returns_6_00(self):
        assert write_price(_SONNET, "1h") == pytest.approx(6.00)

    def test_when_no_tier_is_given_then_default_is_5m(self):
        """Default tier must be '5m': write_price(m) == write_price(m, '5m')."""
        assert write_price(_SONNET) == pytest.approx(write_price(_SONNET, "5m"))

    def test_when_invalid_tier_is_given_then_error_is_raised(self):
        with pytest.raises((ValueError, KeyError, TypeError)):
            write_price(_SONNET, "invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC-4: price_uncached_turn = input×base + output×output_price  (per-MTok)
# ---------------------------------------------------------------------------


class TestPriceUncachedTurn:
    """AC-4: per-MTok formula: input_tokens×base + output_tokens×output."""

    def test_when_1m_input_tokens_sonnet_then_cost_equals_base_price(self):
        # 1_000_000 input, 0 output → 1.0 MTok × $3.00 = $3.00
        cost = price_uncached_turn(
            model=_SONNET, input_tokens=1_000_000, output_tokens=0
        )
        assert cost == pytest.approx(3.00)

    def test_when_1m_output_tokens_sonnet_then_cost_equals_output_price(self):
        # 0 input, 1_000_000 output → 1.0 MTok × $15.00 = $15.00
        cost = price_uncached_turn(
            model=_SONNET, input_tokens=0, output_tokens=1_000_000
        )
        assert cost == pytest.approx(15.00)

    def test_when_mixed_tokens_sonnet_then_cost_matches_hand_computed_value(self):
        # 1 000 input, 500 output (Sonnet 4.6)
        # = 1000/1_000_000 × 3.00  +  500/1_000_000 × 15.00
        # = 0.003 + 0.0075 = 0.0105
        cost = price_uncached_turn(model=_SONNET, input_tokens=1_000, output_tokens=500)
        assert cost == pytest.approx(0.0105)

    def test_when_other_mixed_tokens_sonnet_then_cost_matches_hand_computed_value(self):
        # 2 000 input, 100 output (Sonnet 4.6)
        # = 2000/1_000_000 × 3.00  +  100/1_000_000 × 15.00
        # = 0.006 + 0.0015 = 0.0075
        cost = price_uncached_turn(model=_SONNET, input_tokens=2_000, output_tokens=100)
        assert cost == pytest.approx(0.0075)

    def test_when_zero_tokens_sonnet_then_cost_is_zero(self):
        cost = price_uncached_turn(model=_SONNET, input_tokens=0, output_tokens=0)
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

    def test_when_cached_turn_5m_sonnet_then_cost_matches_hand_computed_value(self):
        # Sonnet 4.6 / 5m tier:
        # = 50000/1e6×0.30 + 5000/1e6×3.75 + 500/1e6×3.00 + 300/1e6×15.00
        # = 0.015 + 0.01875 + 0.0015 + 0.0045
        # = 0.03975
        cost = price_cached_turn(
            model=_SONNET,
            tier="5m",
            cache_read_tokens=self._CR,
            cache_creation_tokens=self._CW,
            fresh_question_tokens=self._FQ,
            output_tokens=self._OUT,
        )
        assert cost == pytest.approx(0.03975)

    def test_when_cached_turn_1h_sonnet_then_cost_matches_hand_computed_value(self):
        # Sonnet 4.6 / 1h tier (write_1h=6.00 instead of 3.75):
        # = 50000/1e6×0.30 + 5000/1e6×6.00 + 500/1e6×3.00 + 300/1e6×15.00
        # = 0.015 + 0.03 + 0.0015 + 0.0045
        # = 0.051
        cost = price_cached_turn(
            model=_SONNET,
            tier="1h",
            cache_read_tokens=self._CR,
            cache_creation_tokens=self._CW,
            fresh_question_tokens=self._FQ,
            output_tokens=self._OUT,
        )
        assert cost == pytest.approx(0.051)

    def test_when_cached_turn_5m_smaller_counts_then_cost_matches_hand_computed_value(
        self,
    ):
        # Sonnet 4.6 / 5m tier (read=0.30, write_5m=3.75, base=3.00, output=15.00):
        # cache_read=10_000, cache_creation=2_000, fresh=200, output=100
        # = 10000/1e6×0.30 + 2000/1e6×3.75 + 200/1e6×3.00 + 100/1e6×15.00
        # = 0.003 + 0.0075 + 0.0006 + 0.0015
        # = 0.0126
        cost = price_cached_turn(
            model=_SONNET,
            tier="5m",
            cache_read_tokens=10_000,
            cache_creation_tokens=2_000,
            fresh_question_tokens=200,
            output_tokens=100,
        )
        assert cost == pytest.approx(0.0126)

    def test_when_cached_turn_1h_smaller_counts_then_cost_matches_hand_computed_value(
        self,
    ):
        # Sonnet 4.6 / 1h tier (write_1h=6.00):
        # = 10000/1e6×0.30 + 2000/1e6×6.00 + 200/1e6×3.00 + 100/1e6×15.00
        # = 0.003 + 0.012 + 0.0006 + 0.0015
        # = 0.0171
        cost = price_cached_turn(
            model=_SONNET,
            tier="1h",
            cache_read_tokens=10_000,
            cache_creation_tokens=2_000,
            fresh_question_tokens=200,
            output_tokens=100,
        )
        assert cost == pytest.approx(0.0171)

    def test_when_all_tokens_are_zero_then_cached_turn_cost_is_zero(self):
        cost = price_cached_turn(
            model=_SONNET,
            tier="5m",
            cache_read_tokens=0,
            cache_creation_tokens=0,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost == pytest.approx(0.0)

    def test_when_only_cache_read_tokens_then_cost_uses_read_rate(self):
        # 1_000_000 cache_read tokens, all others 0, Sonnet 4.6
        # = 1_000_000/1_000_000 × 0.30 = 0.30
        cost = price_cached_turn(
            model=_SONNET,
            tier="5m",
            cache_read_tokens=1_000_000,
            cache_creation_tokens=0,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost == pytest.approx(0.30)

    def test_when_only_cache_creation_tokens_5m_then_cost_uses_write_5m_rate(self):
        # 1_000_000 cache_creation tokens, all others 0, Sonnet 4.6, 5m tier
        # = 1_000_000/1_000_000 × 3.75 = 3.75
        cost = price_cached_turn(
            model=_SONNET,
            tier="5m",
            cache_read_tokens=0,
            cache_creation_tokens=1_000_000,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost == pytest.approx(3.75)

    def test_when_only_cache_creation_tokens_1h_then_cost_uses_write_1h_rate(self):
        # 1_000_000 cache_creation tokens, all others 0, Sonnet 4.6, 1h tier
        # = 1_000_000/1_000_000 × 6.00 = 6.00
        cost = price_cached_turn(
            model=_SONNET,
            tier="1h",
            cache_read_tokens=0,
            cache_creation_tokens=1_000_000,
            fresh_question_tokens=0,
            output_tokens=0,
        )
        assert cost == pytest.approx(6.00)


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
        cost = price_uncached_turn(model=_SONNET, input_tokens=1, output_tokens=0)
        assert cost > 0.0

    def test_when_single_output_token_is_priced_uncached_then_cost_is_strictly_positive(
        self,
    ):
        cost = price_uncached_turn(model=_SONNET, input_tokens=0, output_tokens=1)
        assert cost > 0.0

    def test_when_zero_tokens_are_priced_uncached_then_cost_is_exactly_zero(self):
        cost = price_uncached_turn(model=_SONNET, input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_when_single_cache_read_token_is_priced_then_cost_is_strictly_positive(
        self,
    ):
        cost = price_cached_turn(
            model=_SONNET,
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
            model=_SONNET,
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
    """price_uncached_turn >= 0.0 for all non-negative token counts (Sonnet)."""
    cost = price_uncached_turn(
        model=_SONNET, input_tokens=input_tokens, output_tokens=output_tokens
    )
    assert cost >= 0.0


@given(input_tokens=_POS_INT)
def test_when_positive_input_tokens_are_given_then_uncached_cost_is_strictly_positive(
    input_tokens: int,
) -> None:
    """Any positive input_tokens must produce a strictly positive cost — no floor."""
    cost = price_uncached_turn(model=_SONNET, input_tokens=input_tokens, output_tokens=0)
    assert cost > 0.0


@given(output_tokens=_POS_INT)
def test_when_positive_output_tokens_are_given_then_uncached_cost_is_strictly_positive(
    output_tokens: int,
) -> None:
    """Any positive output_tokens must produce a strictly positive cost — no floor."""
    cost = price_uncached_turn(model=_SONNET, input_tokens=0, output_tokens=output_tokens)
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
        model=_SONNET,
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
        model=_SONNET, input_tokens=baseline_input, output_tokens=output_tokens
    )
    cost_high = price_uncached_turn(
        model=_SONNET,
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
