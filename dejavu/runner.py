"""Priced per-turn results, assembly, ClientRegistry builder, and stream consumer."""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Callable

import baml_py

from baml_client.async_client import b
from baml_client.types import Message
from dejavu.contract import load_contract
from dejavu.conversation import load_questions
from dejavu.metrics import TurnMetrics, turn_metrics
from dejavu.pricing import Model, Tier, price_cached_turn, price_uncached_turn

_MODEL_IDS: dict[Model, str] = {
    Model.SONNET_46: "claude-sonnet-4-6",
}


def _to_model(model: Any) -> Model:
    """Coerce any value to a Model enum; fall back to SONNET_46 for unknowns."""
    if isinstance(model, Model):
        return model
    try:
        return Model(str(model))
    except ValueError:
        return Model.SONNET_46


def _client_options(model: Model, max_tokens: int, temperature: float) -> dict:
    """Return options dict for BAML LLM client registration."""
    return {
        "model": _MODEL_IDS[model],
        "api_key": os.environ["ANTHROPIC_API_KEY"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "allowed_role_metadata": ["cache_control"],
    }


def build_registry(
    model: Model,
    *,
    max_tokens: int,
    temperature: float,
) -> baml_py.ClientRegistry:
    """Build a BAML ClientRegistry with one primary Anthropic client."""
    cr = baml_py.ClientRegistry()
    cr.add_llm_client(
        name="DejavuPrimary",
        provider="anthropic",
        options=_client_options(model, max_tokens, temperature),
    )
    cr.set_primary("DejavuPrimary")
    return cr


class Side(str, Enum):
    """Which session side produced this result."""

    UNCACHED = "uncached"
    CACHED = "cached"


@dataclass(frozen=True)
class SideResult:
    """Priced result for one side of a conversation turn."""

    text: str
    metrics: TurnMetrics
    cost_usd: float

    @property
    def cost(self) -> float:
        return self.cost_usd

    @property
    def input_tokens(self) -> int | None:
        return self.metrics.input_tokens

    @property
    def output_tokens(self) -> int | None:
        return self.metrics.output_tokens

    @property
    def cached_input_tokens(self) -> int | None:
        return self.metrics.cached_input_tokens

    @property
    def duration_ms(self) -> float:
        return self.metrics.duration_ms


@dataclass(frozen=True)
class TurnResult:
    """Both sides' results for one conversation turn."""

    index: int
    question: str
    uncached: SideResult
    cached: SideResult

    @property
    def turn_number(self) -> int:
        """1-based turn number (index + 1)."""
        return self.index + 1


@dataclass(frozen=True)
class TokenEvent:
    """A streamed token partial from one side of a turn."""

    index: int
    side: Side
    partial: str


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for the dual-session runner."""

    model: Model = Model.SONNET_46
    doc: str | None = None
    questions_file: str | None = None
    ttl: Tier = "5m"
    max_tokens: int = 400
    temperature: float = 0.0
    delay: float = 0.0


def _fresh_question_tokens(metrics: TurnMetrics) -> int:
    """Return the fresh (uncached) input token count for a cached turn.

    Anthropic's usage.input_tokens already excludes cache reads and cache-creation
    tokens — it counts only the post-breakpoint tokens. So input_tokens IS the
    fresh question count; no subtraction needed.  None coerced to 0 for robustness
    against partial/failed collectors (baml_py.Usage fields are Optional[int]).
    """
    return metrics.input_tokens or 0


def _build_uncached_result(text: str, collector: object, model: Any) -> SideResult:
    """Assemble a SideResult for the uncached side of a turn."""
    m = turn_metrics(collector)
    cost = price_uncached_turn(
        model=_to_model(model),
        input_tokens=m.input_tokens or 0,
        output_tokens=m.output_tokens or 0,
    )
    return SideResult(text=text, metrics=m, cost_usd=cost)


def _build_cached_result(
    text: str,
    collector: object,
    model: Any,
    tier: Any = "5m",
) -> SideResult:
    """Assemble a SideResult for the cached side of a turn."""
    m = turn_metrics(collector)
    effective_tier: Tier = tier if tier in ("5m", "1h") else "5m"
    cost = price_cached_turn(
        model=_to_model(model),
        tier=effective_tier,
        cache_read_tokens=m.cached_input_tokens or 0,
        cache_creation_tokens=m.cache_creation_input_tokens or 0,
        fresh_question_tokens=_fresh_question_tokens(m),
        output_tokens=m.output_tokens or 0,
    )
    return SideResult(text=text, metrics=m, cost_usd=cost)


async def _emit(on_token: Callable, event: TokenEvent) -> None:
    """Call on_token(event); await the result if it is a coroutine."""
    result = on_token(event)
    if inspect.isawaitable(result):
        await result


async def run_side(
    *,
    index: int,
    side: Side,
    stream_fn: Callable,
    document: str,
    history: list,
    question: str,
    registry: Any,
    model: Model,
    on_token: Callable,
    build_result: Callable,
) -> SideResult:
    """Stream a BAML fn end-to-end, emit TokenEvents, return a priced SideResult."""
    collector = baml_py.Collector()
    baml_options = {"collector": collector, "client_registry": registry}
    stream = stream_fn(
        document=document, history=history, question=question, baml_options=baml_options
    )
    async for partial in stream:
        await _emit(on_token, TokenEvent(index=index, side=side, partial=partial))
    text = await stream.get_final_response()
    return build_result(text, collector, model)


# ---------------------------------------------------------------------------
# Session-level orchestration (Issue #12)
# ---------------------------------------------------------------------------


def _noop_token(_: TokenEvent) -> None:
    pass


def _safe_ttl(cfg: Any) -> Tier:
    t = getattr(cfg, "ttl", "5m")
    return t if t in ("5m", "1h") else "5m"


def _safe_build_registry(cfg: Any) -> baml_py.ClientRegistry | None:
    """Return a live ClientRegistry, or None in headless/test mode (no API key)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    m = _to_model(getattr(cfg, "model", Model.SONNET_46))
    max_t = int(getattr(cfg, "max_tokens", 400))
    temp = float(getattr(cfg, "temperature", 0.0))
    return build_registry(m, max_tokens=max_t, temperature=temp)


def _opt_path(cfg: Any, attr: str) -> str | None:
    """Return a string config attribute, or None if the value is not a string."""
    v = getattr(cfg, attr, None)
    return v if isinstance(v, str) else None


def _session_setup(cfg: Any) -> tuple:
    """Load document + questions, build registry, resolve model + ttl from cfg."""
    document = load_contract(_opt_path(cfg, "doc"))
    questions = load_questions(_opt_path(cfg, "questions_file"))
    registry = _safe_build_registry(cfg)
    model = _to_model(getattr(cfg, "model", Model.SONNET_46))
    return document, questions, registry, model, _safe_ttl(cfg)


def _append_history(history: list, question: str, answer: str) -> None:
    """Append one user + one assistant Message to the shared history in place."""
    history.extend(
        [
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ]
    )


async def run_turn(
    index: int,
    question: str,
    document: str,
    history: list,
    registry: Any,
    model: Any,
    ttl: Tier,
    on_token: Callable,
) -> TurnResult:
    """Fire uncached and cached sides concurrently and return a paired TurnResult."""
    shared: dict = dict(
        index=index,
        document=document,
        history=history,
        question=question,
        registry=registry,
        model=model,
        on_token=on_token,
    )
    u, c = await asyncio.gather(
        run_side(
            **shared,
            side=Side.UNCACHED,
            stream_fn=b.stream.ChatUncached,
            build_result=_build_uncached_result,
        ),
        run_side(
            **shared,
            side=Side.CACHED,
            stream_fn=b.stream.ChatCached,
            build_result=lambda t, col, m: _build_cached_result(t, col, m, tier=ttl),
        ),
    )
    return TurnResult(index=index, question=question, uncached=u, cached=c)


async def run_session(
    cfg: RunConfig, *, on_token: Callable | None = _noop_token
) -> AsyncIterator[TurnResult]:
    """Async-generate one TurnResult per question, streaming tokens to on_token."""
    sink = on_token if on_token is not None else _noop_token
    document, questions, registry, model, ttl = _session_setup(cfg)
    history: list = []
    for i, question in enumerate(questions):
        result = await run_turn(
            i, question, document, list(history), registry, model, ttl, sink
        )
        yield result
        _append_history(history, question, result.uncached.text)
        if cfg.delay:
            await asyncio.sleep(cfg.delay)


# ---------------------------------------------------------------------------
# Resilience (Issue #15) — re-exported for backward-compatible imports
# ---------------------------------------------------------------------------
from dejavu.resilience import _backoff_delay, _is_transient, resilient_session  # noqa: E402,F401
