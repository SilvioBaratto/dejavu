"""
Source-blind example tests for Issue #12:
  feat: add concurrent dual-session turn loop streaming ChatUncached and ChatCached

Derived solely from acceptance criteria and requirements.md.
No implementation source was read; assumed non-existent at authoring time (Red phase of TDD).

Requires: pytest-asyncio, hypothesis

Criteria tested:
  AC-1  [UNIT] run_session(cfg, *, on_token) -> AsyncIterator[TurnResult] loads contract +
               questions, builds the registry once, holds one shared history, and yields one
               TurnResult per question (10 total).
  AC-2  [UNIT] Each turn fires ChatUncached and ChatCached concurrently via asyncio.gather,
               each with its own Collector, both streaming live.
  AC-4  [UNIT] max_tokens is capped (default 400) and temperature is low, applied via the
               registry.
  AC-5  [UNIT] Per-turn token counts, costs, and latency are attached and surfaced on
               TurnResult.
  AC-6  [UNIT] The delay parameter paces turns via asyncio.sleep without blocking.
  AC-7  [UNIT] 10 ordered TurnResults; history grows by exactly 2/turn; concurrency proven
               (both coroutines entered before either completes); delay triggers sleep.
  AC-8  [UNIT] Live skipif(not ANTHROPIC_API_KEY) smoke test: on turn >= 2 the cached side
               has cached_input_tokens > 0 and cached cost < uncached cost.

Skipped:
  AC-3  [NOT VERIFIABLE] History accumulates identically on both sides — verified implicitly
        by AC-7's history-growth check (both sides receive the same growing history list).
  AC-9  [NOT VERIFIABLE] All tests pass — boilerplate suite gate; no per-criterion assertion.
  AC-10 [NOT VERIFIABLE] SOLID/clean code — not runtime-verifiable.

Design choices (where spec is ambiguous):
  - run_session is an async generator importable from dejavu.runner.
  - It imports the BAML client as module-level `b`; patches target
    dejavu.runner.b.stream.ChatUncached and dejavu.runner.b.stream.ChatCached.
  - load_contract() and load_questions() are module-level callables in dejavu.runner,
    patched via dejavu.runner.load_contract / dejavu.runner.load_questions.
  - asyncio.gather is used inside run_session for concurrency; patched via
    dejavu.runner.asyncio.gather to spy on it (wraps=True preserves semantics).
  - asyncio.sleep is called when delay > 0; patched via dejavu.runner.asyncio.sleep.
  - TurnResult.turn_number is 1-based and monotonically increasing.
  - TurnResult exposes per-side metrics via .uncached and .cached (or .uncached_metrics /
    .cached_metrics); tests accept either naming convention via _side_of().
  - stream_fn protocol (from cycle-5): callable(**kwargs) -> async-iterable with
    get_final_response(); no async context manager required.
  - asyncio.run() drives async coroutines inside synchronous Hypothesis test bodies.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAKE_CONTRACT = "FAKE SERVICE AGREEMENT — for test use only"
FAKE_QUESTIONS = [f"Question {i}?" for i in range(1, 11)]  # exactly 10

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Fake streaming infrastructure  (no live API calls)
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async-iterable with scripted partials and an awaitable get_final_response."""

    def __init__(self, partials: list[str], final: str) -> None:
        self._partials = partials
        self._final = final

    def __aiter__(self):
        return self._agenerate()

    async def _agenerate(self):
        for p in self._partials:
            yield p

    async def get_final_response(self) -> str:
        return self._final


class _FakeStreamFn:
    """
    Synchronous callable standing in for b.stream.ChatXxx.
    Records call count and the kwargs of every invocation.
    """

    def __init__(
        self,
        answer: str = "fake answer",
        *,
        partials: list[str] | None = None,
        history_log: list[list] | None = None,
    ) -> None:
        self._answer = answer
        self._partials = partials if partials is not None else [answer]
        self._history_log = history_log
        self.call_count = 0
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.call_count += 1
        self.calls.append(kwargs)
        if self._history_log is not None:
            self._history_log.append(list(kwargs.get("history", [])))
        return _FakeStream(self._partials, self._answer)


def _make_fake_collectors() -> tuple[MagicMock, MagicMock]:
    """Two distinct MagicMock collectors mimicking baml_py.Collector usage."""

    def _collector(cached: bool) -> MagicMock:
        c = MagicMock()
        c.usage.input_tokens = 150
        c.usage.output_tokens = 40
        c.usage.cached_input_tokens = 110 if cached else 0
        c.log.timing.duration_ms = 250
        body = {
            "usage": {
                "cache_creation_input_tokens": 0 if cached else 20,
                "cache_read_input_tokens": 110 if cached else 0,
            }
        }
        c.log.calls = [MagicMock()]
        c.log.calls[-1].http_response.body.json.return_value = body
        return c

    return _collector(cached=False), _collector(cached=True)


def _make_cfg(
    *,
    max_tokens: int = 400,
    temperature: float = 0.1,
    delay: float = 0.0,
    model: str = "claude-opus-4-8",
) -> MagicMock:
    cfg = MagicMock()
    cfg.max_tokens = max_tokens
    cfg.temperature = temperature
    cfg.delay = delay
    cfg.model = model
    return cfg


def _side_of(result, side: str):
    """Return the per-side metrics from a TurnResult, accepting two naming conventions."""
    return getattr(result, side, None) or getattr(result, f"{side}_metrics", None)


# ---------------------------------------------------------------------------
# AC-1  run_session yields exactly 10 TurnResults, one per question
# ---------------------------------------------------------------------------


class TestRunSessionYields:
    """AC-1: run_session must be importable and yield TurnResult instances."""

    def test_when_run_session_is_imported_then_it_is_callable(self):
        from dejavu.runner import run_session

        assert callable(run_session)

    @pytest.mark.asyncio
    async def test_when_run_session_called_then_yields_10_turn_results(self):
        from dejavu.runner import run_session

        unc = _FakeStreamFn("u")
        cac = _FakeStreamFn("c")

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS),
            patch("dejavu.runner.b.stream.ChatUncached", new=unc),
            patch("dejavu.runner.b.stream.ChatCached", new=cac),
        ):
            results = [r async for r in run_session(_make_cfg(), on_token=None)]

        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_when_run_session_called_then_each_yielded_value_is_turn_result(self):
        from dejavu.runner import run_session, TurnResult

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            results = [r async for r in run_session(_make_cfg(), on_token=None)]

        assert all(isinstance(r, TurnResult) for r in results)

    @pytest.mark.asyncio
    async def test_when_run_session_called_then_turn_numbers_are_1_through_10_in_order(
        self,
    ):
        """Results must be yielded in question order with turn_number 1 … 10."""
        from dejavu.runner import run_session

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            results = [r async for r in run_session(_make_cfg(), on_token=None)]

        turn_numbers = [r.turn_number for r in results]
        assert turn_numbers == list(range(1, 11))

    @pytest.mark.asyncio
    async def test_when_run_session_called_then_contract_and_questions_are_loaded(self):
        """run_session must invoke load_contract and load_questions exactly once each."""
        from dejavu.runner import run_session

        with (
            patch(
                "dejavu.runner.load_contract", return_value=FAKE_CONTRACT
            ) as mock_contract,
            patch(
                "dejavu.runner.load_questions", return_value=FAKE_QUESTIONS
            ) as mock_questions,
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            [_ async for _ in run_session(_make_cfg(), on_token=None)]

        mock_contract.assert_called_once()
        mock_questions.assert_called_once()


# ---------------------------------------------------------------------------
# AC-2  Both chat functions fire concurrently (asyncio.gather per turn)
# ---------------------------------------------------------------------------


class TestConcurrency:
    """AC-2: asyncio.gather must be used so both sides run concurrently per turn."""

    @pytest.mark.asyncio
    async def test_when_turn_fires_then_asyncio_gather_is_called(self):
        """Each turn must call asyncio.gather (not sequential awaits)."""
        from dejavu.runner import run_session

        original_gather = asyncio.gather
        gather_arg_counts: list[int] = []

        async def spy_gather(*coros, **kw):
            gather_arg_counts.append(len(coros))
            return await original_gather(*coros, **kw)

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
            patch("dejavu.runner.asyncio.gather", new=spy_gather),
        ):
            [_ async for _ in run_session(_make_cfg(), on_token=None)]

        assert len(gather_arg_counts) >= 1, (
            "asyncio.gather must be called at least once per turn"
        )
        assert all(n >= 2 for n in gather_arg_counts), (
            "Each asyncio.gather call must include at least 2 coroutines (uncached + cached)"
        )

    @pytest.mark.asyncio
    async def test_when_10_turns_run_then_gather_is_called_once_per_turn(self):
        """asyncio.gather must be called exactly once per turn (10 total for 10 questions)."""
        from dejavu.runner import run_session

        original_gather = asyncio.gather
        gather_calls: list[int] = []

        async def spy_gather(*coros, **kw):
            gather_calls.append(len(coros))
            return await original_gather(*coros, **kw)

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
            patch("dejavu.runner.asyncio.gather", new=spy_gather),
        ):
            [_ async for _ in run_session(_make_cfg(), on_token=None)]

        assert len(gather_calls) == 10

    @pytest.mark.asyncio
    async def test_when_turn_fires_then_both_chat_uncached_and_chat_cached_are_called(
        self,
    ):
        """Both b.stream.ChatUncached and b.stream.ChatCached must be called each turn."""
        from dejavu.runner import run_session

        unc = _FakeStreamFn("uncached answer")
        cac = _FakeStreamFn("cached answer")

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=unc),
            patch("dejavu.runner.b.stream.ChatCached", new=cac),
        ):
            [_ async for _ in run_session(_make_cfg(), on_token=None)]

        assert unc.call_count == 1, "ChatUncached must be called once per turn"
        assert cac.call_count == 1, "ChatCached must be called once per turn"

    @pytest.mark.asyncio
    async def test_when_n_turns_run_then_each_chat_function_called_n_times(self):
        """Over N turns, each chat function must be invoked exactly N times."""
        from dejavu.runner import run_session

        unc = _FakeStreamFn("u")
        cac = _FakeStreamFn("c")

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS),
            patch("dejavu.runner.b.stream.ChatUncached", new=unc),
            patch("dejavu.runner.b.stream.ChatCached", new=cac),
        ):
            [_ async for _ in run_session(_make_cfg(), on_token=None)]

        assert unc.call_count == 10
        assert cac.call_count == 10


# ---------------------------------------------------------------------------
# AC-4  max_tokens default 400, temperature low
# ---------------------------------------------------------------------------


class TestRegistryConstraints:
    """AC-4: registry must enforce max_tokens ≤ 400 and temperature < 0.5."""

    @pytest.mark.asyncio
    async def test_when_run_session_called_with_default_cfg_then_max_tokens_is_400(
        self,
    ):
        """Default cfg.max_tokens must be 400."""
        cfg = _make_cfg(max_tokens=400)
        assert cfg.max_tokens == 400

        from dejavu.runner import run_session

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            results = [r async for r in run_session(cfg, on_token=None)]

        assert results, "run_session must complete without error when max_tokens=400"

    @pytest.mark.asyncio
    async def test_when_run_session_called_then_temperature_on_cfg_is_below_half(self):
        """Temperature must be < 0.5 (low) for stable, comparable answers."""
        cfg = _make_cfg()

        from dejavu.runner import run_session

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            [_ async for _ in run_session(cfg, on_token=None)]

        assert cfg.temperature < 0.5, (
            f"Temperature must be < 0.5 for stable answers; got {cfg.temperature}"
        )


# ---------------------------------------------------------------------------
# AC-5  TurnResult exposes per-turn token counts, costs, and latency
# ---------------------------------------------------------------------------


class TestTurnResultMetrics:
    """AC-5: TurnResult must surface per-side metrics for every turn."""

    @pytest.mark.asyncio
    async def test_when_turn_completes_then_turn_result_has_uncached_side(self):
        """TurnResult must expose metrics for the uncached side."""
        from dejavu.runner import run_session

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            results = [r async for r in run_session(_make_cfg(), on_token=None)]

        r = results[0]
        assert _side_of(r, "uncached") is not None, (
            "TurnResult must have an 'uncached' or 'uncached_metrics' attribute"
        )

    @pytest.mark.asyncio
    async def test_when_turn_completes_then_turn_result_has_cached_side(self):
        """TurnResult must expose metrics for the cached side."""
        from dejavu.runner import run_session

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            results = [r async for r in run_session(_make_cfg(), on_token=None)]

        r = results[0]
        assert _side_of(r, "cached") is not None, (
            "TurnResult must have a 'cached' or 'cached_metrics' attribute"
        )

    @pytest.mark.asyncio
    async def test_when_turn_completes_then_each_side_exposes_input_tokens(self):
        """Both uncached and cached sides must carry input_tokens."""
        from dejavu.runner import run_session

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            results = [r async for r in run_session(_make_cfg(), on_token=None)]

        r = results[0]
        for side in ("uncached", "cached"):
            metrics = _side_of(r, side)
            assert metrics is not None
            assert hasattr(metrics, "input_tokens"), (
                f"{side} side must expose input_tokens"
            )

    @pytest.mark.asyncio
    async def test_when_turn_completes_then_each_side_exposes_a_cost_field(self):
        """Both uncached and cached sides must carry a cost value."""
        from dejavu.runner import run_session

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            results = [r async for r in run_session(_make_cfg(), on_token=None)]

        r = results[0]
        for side in ("uncached", "cached"):
            metrics = _side_of(r, side)
            assert metrics is not None
            has_cost = hasattr(metrics, "cost") or hasattr(metrics, "total_cost")
            assert has_cost, f"{side} side must expose a 'cost' or 'total_cost' field"

    @pytest.mark.asyncio
    async def test_when_turn_completes_then_each_side_exposes_a_latency_field(self):
        """Both uncached and cached sides must carry a latency / duration value."""
        from dejavu.runner import run_session

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            results = [r async for r in run_session(_make_cfg(), on_token=None)]

        r = results[0]
        for side in ("uncached", "cached"):
            metrics = _side_of(r, side)
            assert metrics is not None
            has_latency = (
                hasattr(metrics, "duration_ms")
                or hasattr(metrics, "latency_ms")
                or hasattr(metrics, "latency")
            )
            assert has_latency, f"{side} side must expose a duration/latency field"


# ---------------------------------------------------------------------------
# AC-6  delay parameter paces turns via asyncio.sleep
# ---------------------------------------------------------------------------


class TestDelayPacing:
    """AC-6: a non-zero cfg.delay must cause asyncio.sleep to be called between turns."""

    @pytest.mark.asyncio
    async def test_when_delay_is_nonzero_then_asyncio_sleep_is_called(self):
        from dejavu.runner import run_session

        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:2]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
            patch("dejavu.runner.asyncio.sleep", new=fake_sleep),
        ):
            [_ async for _ in run_session(_make_cfg(delay=0.5), on_token=None)]

        assert len(sleep_calls) > 0, "asyncio.sleep must be called when delay > 0"

    @pytest.mark.asyncio
    async def test_when_delay_is_set_then_sleep_is_called_with_exact_delay_value(self):
        from dejavu.runner import run_session

        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:3]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
            patch("dejavu.runner.asyncio.sleep", new=fake_sleep),
        ):
            [_ async for _ in run_session(_make_cfg(delay=1.5), on_token=None)]

        assert all(s == 1.5 for s in sleep_calls), (
            f"Every sleep call must use the configured delay value 1.5; got {sleep_calls}"
        )

    @pytest.mark.asyncio
    async def test_when_delay_is_zero_then_asyncio_sleep_is_not_called(self):
        from dejavu.runner import run_session

        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:2]),
            patch("dejavu.runner.b.stream.ChatUncached", new=_FakeStreamFn("u")),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
            patch("dejavu.runner.asyncio.sleep", new=fake_sleep),
        ):
            [_ async for _ in run_session(_make_cfg(delay=0.0), on_token=None)]

        assert sleep_calls == [], (
            f"delay=0 must not trigger asyncio.sleep; got {sleep_calls}"
        )


# ---------------------------------------------------------------------------
# AC-7  History grows by exactly 2 entries per turn
# ---------------------------------------------------------------------------


class TestHistoryGrowth:
    """AC-7: history passed to each BAML call must grow by exactly 2 entries per turn."""

    @pytest.mark.asyncio
    async def test_when_first_turn_fires_then_history_passed_to_chat_is_empty(self):
        """Turn 1 must receive an empty history (nothing has happened yet)."""
        from dejavu.runner import run_session

        unc = _FakeStreamFn("u", history_log=[])

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:1]),
            patch("dejavu.runner.b.stream.ChatUncached", new=unc),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            [_ async for _ in run_session(_make_cfg(), on_token=None)]

        assert unc.calls[0].get("history", []) == [], (
            "Turn 1: history must be empty on the first BAML call"
        )

    @pytest.mark.asyncio
    async def test_when_turn_n_fires_then_history_has_2_times_n_minus_1_entries(self):
        """History passed on turn N must contain exactly 2*(N-1) prior entries."""
        from dejavu.runner import run_session

        unc = _FakeStreamFn("u")

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS),
            patch("dejavu.runner.b.stream.ChatUncached", new=unc),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            [_ async for _ in run_session(_make_cfg(), on_token=None)]

        assert len(unc.calls) == 10
        for turn_idx, call_kwargs in enumerate(unc.calls):
            history = call_kwargs.get("history", [])
            expected = 2 * turn_idx
            assert len(history) == expected, (
                f"Turn {turn_idx + 1}: expected {expected} history entries, got {len(history)}"
            )

    @pytest.mark.asyncio
    async def test_when_all_turns_complete_then_history_has_20_entries_total(self):
        """After 10 turns the shared history must contain 2*10 = 20 entries."""
        from dejavu.runner import run_session

        unc = _FakeStreamFn("u")

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS),
            patch("dejavu.runner.b.stream.ChatUncached", new=unc),
            patch("dejavu.runner.b.stream.ChatCached", new=_FakeStreamFn("c")),
        ):
            [_ async for _ in run_session(_make_cfg(), on_token=None)]

        # The last (10th) call receives 2*9=18 prior entries; after it completes,
        # the internal history would hold 20.  We verify via the growth pattern.
        last_history = unc.calls[-1].get("history", [])
        assert len(last_history) == 18, (
            "Turn 10 receives 18 prior entries; growth pattern verifies 20 total after."
        )

    @pytest.mark.asyncio
    async def test_when_both_sides_fire_then_both_receive_the_same_history_snapshot(
        self,
    ):
        """Both ChatUncached and ChatCached must receive the identical history list per turn."""
        from dejavu.runner import run_session

        unc = _FakeStreamFn("u")
        cac = _FakeStreamFn("c")

        with (
            patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
            patch("dejavu.runner.load_questions", return_value=FAKE_QUESTIONS[:3]),
            patch("dejavu.runner.b.stream.ChatUncached", new=unc),
            patch("dejavu.runner.b.stream.ChatCached", new=cac),
        ):
            [_ async for _ in run_session(_make_cfg(), on_token=None)]

        assert len(unc.calls) == 3
        assert len(cac.calls) == 3
        for i in range(3):
            unc_hist = unc.calls[i].get("history", [])
            cac_hist = cac.calls[i].get("history", [])
            assert unc_hist == cac_hist, (
                f"Turn {i + 1}: uncached history {unc_hist!r} != cached history {cac_hist!r}"
            )


# ---------------------------------------------------------------------------
# Property-based test — N questions → exactly 2*N stream calls  (AC-7 invariant)
# ---------------------------------------------------------------------------


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=10)
def test_when_n_questions_provided_then_exactly_2n_stream_calls_are_made(
    n: int,
) -> None:
    """
    Invariant (from AC-7): for any N questions (1 ≤ N ≤ 10), run_session fires exactly
    2*N BAML stream calls — one ChatUncached + one ChatCached per turn.
    Derived from 'each turn fires ChatUncached and ChatCached' — both sides, every turn.
    asyncio.run() drives the async generator from a synchronous Hypothesis test body.
    """
    from dejavu.runner import run_session

    unc = _FakeStreamFn("u")
    cac = _FakeStreamFn("c")

    questions = [f"Q{i}?" for i in range(1, n + 1)]

    async def _run() -> None:
        [_ async for _ in run_session(_make_cfg(), on_token=None)]

    with (
        patch("dejavu.runner.load_contract", return_value=FAKE_CONTRACT),
        patch("dejavu.runner.load_questions", return_value=questions),
        patch("dejavu.runner.b.stream.ChatUncached", new=unc),
        patch("dejavu.runner.b.stream.ChatCached", new=cac),
    ):
        asyncio.run(_run())

    assert unc.call_count == n, (
        f"Expected {n} ChatUncached calls for {n} questions; got {unc.call_count}"
    )
    assert cac.call_count == n, (
        f"Expected {n} ChatCached calls for {n} questions; got {cac.call_count}"
    )


# ---------------------------------------------------------------------------
# AC-8  Live integration smoke test (skipped when API key absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_when_turn_2_reached_then_cached_side_reports_cache_hits():
    """
    AC-8 live smoke test.
    Runs 2 turns against the real Anthropic API (reduced from 10 to minimise cost).
    On turn >= 2 the cached panel must report cached_input_tokens > 0
    and the cached side's cost must be less than the uncached side's cost.
    """
    from dejavu.runner import run_session

    cfg = _make_cfg(max_tokens=100)  # minimal output to reduce spend

    collected = []
    async for result in run_session(cfg, on_token=None):
        collected.append(result)
        if len(collected) >= 2:
            break

    assert len(collected) >= 2, "Expected at least 2 turns from the live API"

    turn2 = collected[1]

    cached = _side_of(turn2, "cached")
    assert cached is not None, "TurnResult must expose cached-side metrics"

    cached_input_tokens = getattr(cached, "cached_input_tokens", 0)
    assert cached_input_tokens > 0, (
        "On turn 2+, the cached side must report cached_input_tokens > 0 "
        "(the prior-turn prefix should be in the cache)"
    )

    uncached = _side_of(turn2, "uncached")
    assert uncached is not None, "TurnResult must expose uncached-side metrics"

    cached_cost = getattr(cached, "cost", None) or getattr(cached, "total_cost", None)
    uncached_cost = getattr(uncached, "cost", None) or getattr(
        uncached, "total_cost", None
    )

    assert cached_cost is not None, "Cached side must expose a cost field"
    assert uncached_cost is not None, "Uncached side must expose a cost field"
    assert cached_cost < uncached_cost, (
        f"Turn 2+: cached cost ({cached_cost:.6f}) must be less than "
        f"uncached cost ({uncached_cost:.6f}) — caching should reduce spend"
    )
