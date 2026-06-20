"""Typer CLI: flags, .env loading, worst-case cost guard, and run_tui launcher."""

from __future__ import annotations

import asyncio
import os
from enum import Enum
from typing import Optional

import typer
from dotenv import load_dotenv

from dejavu.contract import load_contract
from dejavu.conversation import load_questions
from dejavu.pricing import Model, Tier, price_uncached_turn
from dejavu.runner import RunConfig
from dejavu.tui import run_tui

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN: int = 4
N_TURNS: int = 10

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="dejavu — prompt-caching cost simulator (side-by-side live demo)"
)


class _Ttl(str, Enum):
    """Valid cache TTL choices for --ttl."""

    min5 = "5m"
    h1 = "1h"


# ---------------------------------------------------------------------------
# Small single-purpose helpers
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load .env so ANTHROPIC_API_KEY reaches os.environ."""
    load_dotenv()


def _build_config(
    model: str = "sonnet-4.6",
    doc: Optional[str] = None,
    questions_file: Optional[str] = None,
    ttl: str = "5m",
    max_tokens: int = 400,
    temperature: float = 0.0,
    delay: float = 0.0,
) -> RunConfig:
    """Map CLI flag values to a frozen RunConfig value object."""
    tier: Tier = "1h" if ttl == "1h" else "5m"
    return RunConfig(
        model=Model(model),
        doc=doc,
        questions_file=questions_file,
        ttl=tier,
        max_tokens=max_tokens,
        temperature=temperature,
        delay=delay,
    )


def _doc_tokens(doc: Optional[str]) -> int:
    """Estimate contract token count via the CHARS_PER_TOKEN heuristic."""
    return len(load_contract(doc)) // CHARS_PER_TOKEN


def _question_tokens(questions_file: Optional[str]) -> list[int]:
    """Return per-question token estimates via the CHARS_PER_TOKEN heuristic."""
    return [len(q) // CHARS_PER_TOKEN for q in load_questions(questions_file)]


def _turn_input_tokens(
    doc_toks: int, q_toks: list[int], turn_idx: int, max_tokens: int
) -> int:
    """Accumulated uncached input for one turn: doc + prior history + question."""
    avg_q = sum(q_toks) // N_TURNS
    history_toks = turn_idx * (avg_q + max_tokens)
    return doc_toks + history_toks + q_toks[turn_idx]


def _estimate_worst_case(cfg: RunConfig) -> float:
    """Sum price_uncached_turn over N_TURNS × 2 sides (fully-uncached upper bound)."""
    model = Model(cfg.model)
    doc_toks = _doc_tokens(cfg.doc)
    q_toks = _question_tokens(cfg.questions_file)
    total = 0.0
    for i in range(N_TURNS):
        inp = _turn_input_tokens(doc_toks, q_toks, i, cfg.max_tokens)
        turn_cost = price_uncached_turn(
            model=model, input_tokens=inp, output_tokens=cfg.max_tokens
        )
        total += 2 * turn_cost
    return total


def _confirm_spend(estimate: float) -> None:
    """Print the worst-case estimate and ask for confirmation (aborts on decline)."""
    typer.echo(f"Worst-case estimate: ${estimate:.4f}")
    typer.confirm("Proceed with real API calls?", abort=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    model: str = typer.Option(
        "sonnet-4.6", "--model", help="Model: sonnet-4.6"
    ),
    doc: Optional[str] = typer.Option(
        None, "--doc", help="Path to a custom contract document"
    ),
    questions_file: Optional[str] = typer.Option(
        None, "--questions-file", help="Path to a custom questions JSON file"
    ),
    ttl: _Ttl = typer.Option(_Ttl.min5, "--ttl", help="Cache TTL: 5m or 1h"),
    max_tokens: int = typer.Option(
        400, "--max-tokens", help="Max output tokens per turn"
    ),
    delay: float = typer.Option(0.0, "--delay", help="Pacing delay between turns (s)"),
) -> None:
    """Run the prompt-caching side-by-side live demo."""
    _load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo("Note: ANTHROPIC_API_KEY is not set — running in headless mode.")
    cfg = _build_config(
        model=model,
        doc=doc,
        questions_file=questions_file,
        ttl=ttl.value,
        max_tokens=max_tokens,
        delay=delay,
    )
    estimate = _estimate_worst_case(cfg)
    _confirm_spend(estimate)
    asyncio.run(run_tui(cfg))
