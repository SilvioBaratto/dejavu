"""Anthropic price table and per-turn cost calculators (3 models, 2 cache tiers)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

MTOK: int = 1_000_000

Tier = Literal["5m", "1h"]

_VALID_TIERS: frozenset[str] = frozenset({"5m", "1h"})


class Model(str, Enum):
    """Anthropic model identifiers; values double as future CLI keys."""

    OPUS_48 = "opus-4.8"
    FABLE_5 = "fable-5"
    MYTHOS_5 = "mythos-5"


@dataclass(frozen=True)
class Price:
    """USD per MTok pricing row for one model."""

    base: float
    write_5m: float
    write_1h: float
    read: float
    output: float


_OPUS_ROW = Price(base=5.00, write_5m=6.25, write_1h=10.00, read=0.50, output=25.00)
_FABLE_ROW = Price(base=10.00, write_5m=12.50, write_1h=20.00, read=1.00, output=50.00)

PRICES: dict[Model, Price] = {
    Model.OPUS_48: _OPUS_ROW,
    Model.FABLE_5: _FABLE_ROW,
    Model.MYTHOS_5: _FABLE_ROW,  # same instance — cannot drift
}


def _per_mtok(tokens: int, usd_per_mtok: float) -> float:
    """Convert a raw token count to USD using per-MTok rate."""
    return tokens / MTOK * usd_per_mtok


def get_price(model: Model) -> Price:
    """Return the Price row for *model*."""
    return PRICES[model]


def write_price(model: Model, tier: Tier = "5m") -> float:
    """Return the cache-write USD/MTok rate for *model* and *tier*."""
    if tier not in _VALID_TIERS:
        raise ValueError(f"Unknown tier {tier!r}; expected '5m' or '1h'")
    p = get_price(model)
    return p.write_5m if tier == "5m" else p.write_1h


def price_uncached_turn(
    *,
    model: Model,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Cost of one uncached turn: input×base + output×output_rate (per-MTok)."""
    p = get_price(model)
    return _per_mtok(input_tokens, p.base) + _per_mtok(output_tokens, p.output)


def price_cached_turn(
    *,
    model: Model,
    tier: Tier = "5m",
    cache_read_tokens: int,
    cache_creation_tokens: int,
    fresh_question_tokens: int,
    output_tokens: int,
) -> float:
    """Cost of one cached turn using the rolling breakpoint formula (per-MTok)."""
    p = get_price(model)
    w = write_price(model, tier)
    return (
        _per_mtok(cache_read_tokens, p.read)
        + _per_mtok(cache_creation_tokens, w)
        + _per_mtok(fresh_question_tokens, p.base)
        + _per_mtok(output_tokens, p.output)
    )
