"""Per-turn metrics record parsed from a BAML Collector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurnMetrics:
    """Immutable snapshot of one conversation turn's token and latency usage."""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_creation_input_tokens: int
    duration_ms: float


def turn_metrics(collector: Any) -> TurnMetrics:
    """Parse a BAML Collector into a frozen TurnMetrics record."""
    usage = _typed_usage(collector)
    log = _last_log(collector)
    return TurnMetrics(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_creation_input_tokens=_cache_write_tokens(log),
        duration_ms=_duration_ms(log),
    )


def _typed_usage(collector: Any) -> Any:
    return collector.usage


def _last_log(collector: Any) -> Any | None:
    return collector.last


def _duration_ms(log: Any | None) -> float:
    if log is None:
        return 0.0
    return log.timing.duration_ms


def _raw_usage(log: Any | None) -> dict[str, Any]:
    """Walk raw HTTP response to Anthropic's usage dict; returns {} on any failure.

    Cache writes live only in the raw body — BAML typed usage omits them.
    """
    if log is None or not log.calls:
        return {}
    http_resp = log.calls[-1].http_response
    if http_resp is None:
        return {}
    try:
        return http_resp.body.json().get("usage", {})
    except Exception:
        return {}


def _cache_write_tokens(log: Any | None) -> int:
    return _raw_usage(log).get("cache_creation_input_tokens", 0)
