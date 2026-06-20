"""Anthropic price table and per-turn cost calculators (1 model, 2 cache tiers)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

MTOK: int = 1_000_000

Tier = Literal["5m", "1h"]

_VALID_TIERS: frozenset[str] = frozenset({"5m", "1h"})


class Model(str, Enum):
    """Anthropic model identifiers; values double as future CLI keys."""

    SONNET_46 = "sonnet-4.6"


@dataclass(frozen=True)
class Price:
    """USD per MTok pricing row for one model."""

    base: float
    write_5m: float
    write_1h: float
    read: float
    output: float


_SONNET_ROW = Price(base=3.00, write_5m=3.75, write_1h=6.00, read=0.30, output=15.00)

PRICES: dict[Model, Price] = {
    Model.SONNET_46: _SONNET_ROW,
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
