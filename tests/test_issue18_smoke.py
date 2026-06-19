"""
Integration smoke test for issue #18 — criterion 3.

Verifiable criterion:
  An integration smoke test runs `run_tui` / `resilient_session` for ~2 turns
  against the real API and asserts no crash and that cached cost < uncached cost;
  gated by the `integration` marker + skipif(not ANTHROPIC_API_KEY) (skips —
  never fails — when the key is absent).

Design:
  - Drive `run_tui` with a `Console(record=True)` and a limiting session_factory
    that wraps `resilient_session` and stops after 2 turns.
  - Assert on the returned `RenderState.cost_strip` totals.
  - Reuses the `@pytest.mark.integration` + `skipif` pattern from test_cycle6_runner.py.
"""

import os

import pytest

_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


@pytest.mark.integration
@pytest.mark.skipif(
    not _API_KEY,
    reason="ANTHROPIC_API_KEY not set — live API smoke test skipped (never fails)",
)
@pytest.mark.asyncio
async def test_when_run_tui_runs_two_turns_then_no_crash_and_cached_cost_is_less_than_uncached():
    """
    Full-stack two-turn smoke test driving run_tui → resilient_session → run_session.

    Asserts:
      1. run_tui completes without raising (no crash).
      2. Both cumulative costs are positive — real API calls were made.
      3. cost_strip.cached_total < cost_strip.uncached_total — caching saves money.
    """
    from io import StringIO

    from rich.console import Console

    from dejavu.resilience import resilient_session
    from dejavu.runner import RunConfig
    from dejavu.tui import run_tui

    cfg = RunConfig(max_tokens=100, delay=0.0)

    async def _two_turn_factory(cfg, *, on_token, on_note=None):
        """Wrap resilient_session and stop after 2 turns to keep live-API spend minimal."""
        count = 0
        async for turn in resilient_session(cfg, on_token=on_token, on_note=on_note):
            yield turn
            count += 1
            if count >= 2:
                return

    console = Console(file=StringIO(), width=120)
    state = await run_tui(cfg, console=console, session_factory=_two_turn_factory)

    assert state is not None, "run_tui must return the final RenderState"

    assert state.cost_strip.uncached_total > 0, (
        "uncached_total must be positive — uncached API calls were made"
    )
    assert state.cost_strip.cached_total > 0, (
        "cached_total must be positive — cached API calls were made"
    )
    assert state.cost_strip.cached_total < state.cost_strip.uncached_total, (
        "after 2 turns the rolling-cache panel must cost less than the uncached panel; "
        f"got cached={state.cost_strip.cached_total:.6f}, "
        f"uncached={state.cost_strip.uncached_total:.6f}"
    )
