"""
Source-blind example tests for Issue #11:
  feat: add single-session async streaming consumer with per-call Collector

Derived solely from acceptance criteria and requirements.md.
No implementation source was read; assumed non-existent at authoring time (Red phase of TDD).

Requires: pytest-asyncio

Criteria tested:
  AC-1  [UNIT] run_side(*, index, side, stream_fn, document, history, question,
                        registry, model, on_token, build_result) -> SideResult is implemented.
  AC-2  [UNIT] run_side creates its own baml_py.Collector and passes collector +
                client_registry through baml_options.
  AC-3  [UNIT] Each streamed partial emits exactly one TokenEvent(index, side, partial)
                to on_token.
  AC-4  [UNIT] _emit awaits on_token when it returns an awaitable, otherwise calls it.
  AC-5  [UNIT] Returns build_result(text, collector, model) carrying the final response text.
  AC-6  [UNIT] Tests use a fake stream_fn with __aiter__ and get_final_response() — no live API.

Skipped:
  AC-7  All tests pass — boilerplate suite gate; no per-criterion assertion
  AC-8  SOLID/clean code — not runtime-verifiable

Design choices (where spec is ambiguous):
  - stream_fn is called synchronously and returns an async-iterable object (matches BAML
    b.stream.* pattern from requirements.md §4).
  - baml_options is passed as a dict with keys "collector" and "client_registry" (idiomatic
    BAML Python; requirements.md §Additional Notes).
  - _emit is a module-level async function importable from dejavu.runner (private helper).
  - asyncio.run() is used inside property tests to drive async code from synchronous Hypothesis
    test bodies.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Fake stream infrastructure  (AC-6: no live API calls)
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
    """Synchronous callable; captures call kwargs and returns a _FakeStream."""

    def __init__(self, partials: list[str] = [], final: str = "final") -> None:
        self._partials = list(partials)
        self._final = final
        self.last_kwargs: dict = {}

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._partials, self._final)


# ---------------------------------------------------------------------------
# AC-1: run_side is importable with the full documented keyword signature
# ---------------------------------------------------------------------------


class TestRunSideSignature:
    """AC-1: run_side exists in dejavu.runner with all ten keyword parameters."""

    def test_when_run_side_is_imported_then_it_is_callable(self):
        from dejavu.runner import run_side

        assert callable(run_side)

    @pytest.mark.parametrize(
        "param",
        [
            "index",
            "side",
            "stream_fn",
            "document",
            "history",
            "question",
            "registry",
            "model",
            "on_token",
            "build_result",
        ],
    )
    def test_when_run_side_signature_is_inspected_then_param_exists(self, param: str):
        from dejavu.runner import run_side

        assert param in inspect.signature(run_side).parameters


# ---------------------------------------------------------------------------
# AC-2: run_side creates its own Collector and threads it through baml_options
# ---------------------------------------------------------------------------


class TestRunSideCollectorWiring:
    """AC-2: run_side passes a fresh baml_py.Collector and the supplied registry via baml_options."""

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_stream_fn_receives_baml_options_kwarg(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        stream_fn = _FakeStreamFn(partials=["tok"], final="tok")
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=stream_fn,
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: MagicMock(),
        )
        assert "baml_options" in stream_fn.last_kwargs

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_baml_options_contains_collector_key(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        stream_fn = _FakeStreamFn(partials=["tok"], final="tok")
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=stream_fn,
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: MagicMock(),
        )
        assert "collector" in stream_fn.last_kwargs["baml_options"]

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_baml_options_collector_is_baml_py_collector(
        self,
    ):
        """The collector must be an actual baml_py.Collector, not a mock or stub."""
        import baml_py
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        stream_fn = _FakeStreamFn(partials=["tok"], final="tok")
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=stream_fn,
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: MagicMock(),
        )
        collector = stream_fn.last_kwargs["baml_options"]["collector"]
        assert isinstance(collector, baml_py.Collector)

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_baml_options_contains_client_registry_key(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        stream_fn = _FakeStreamFn(partials=["tok"], final="tok")
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=stream_fn,
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: MagicMock(),
        )
        assert "client_registry" in stream_fn.last_kwargs["baml_options"]

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_baml_options_client_registry_is_the_passed_registry(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        sentinel_registry = MagicMock()
        stream_fn = _FakeStreamFn(partials=["tok"], final="tok")
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=stream_fn,
            document="d",
            history=[],
            question="q?",
            registry=sentinel_registry,
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: MagicMock(),
        )
        assert (
            stream_fn.last_kwargs["baml_options"]["client_registry"]
            is sentinel_registry
        )

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_twice_then_each_call_gets_its_own_collector_instance(
        self,
    ):
        """Per-call Collector: the instance must not be shared across invocations."""
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        collectors: list = []

        def capture(t, c, m):
            collectors.append(c)
            return MagicMock()

        for _ in range(2):
            await run_side(
                index=0,
                side=Side.UNCACHED,
                stream_fn=_FakeStreamFn(partials=["tok"], final="tok"),
                document="d",
                history=[],
                question="q?",
                registry=MagicMock(),
                model=Model.OPUS_48,
                on_token=lambda e: None,
                build_result=capture,
            )

        assert len(collectors) == 2
        assert collectors[0] is not collectors[1]


# ---------------------------------------------------------------------------
# AC-3: Each streamed partial emits exactly one TokenEvent
# ---------------------------------------------------------------------------


class TestTokenEventEmission:
    """AC-3: run_side emits a TokenEvent per partial, in stream order, with correct fields."""

    @pytest.mark.asyncio
    async def test_when_stream_yields_no_partials_then_on_token_is_never_called(self):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        calls: list = []
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=[], final=""),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: calls.append(e),
            build_result=lambda t, c, m: MagicMock(),
        )
        assert calls == []

    @pytest.mark.asyncio
    async def test_when_stream_yields_three_partials_then_on_token_is_called_exactly_three_times(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        calls: list = []
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=["a", "b", "c"], final="abc"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: calls.append(e),
            build_result=lambda t, c, m: MagicMock(),
        )
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_when_stream_yields_partial_then_event_carries_the_run_side_index(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        calls: list = []
        await run_side(
            index=5,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=["tok"], final="tok"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: calls.append(e),
            build_result=lambda t, c, m: MagicMock(),
        )
        assert calls[0].index == 5

    @pytest.mark.asyncio
    async def test_when_stream_yields_partial_then_event_carries_the_run_side_side(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        calls: list = []
        await run_side(
            index=0,
            side=Side.CACHED,
            stream_fn=_FakeStreamFn(partials=["tok"], final="tok"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: calls.append(e),
            build_result=lambda t, c, m: MagicMock(),
        )
        assert calls[0].side == Side.CACHED

    @pytest.mark.asyncio
    async def test_when_stream_yields_partial_then_event_partial_matches_the_streamed_token(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        calls: list = []
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=["hello world"], final="hello world"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: calls.append(e),
            build_result=lambda t, c, m: MagicMock(),
        )
        assert calls[0].partial == "hello world"

    @pytest.mark.asyncio
    async def test_when_stream_yields_multiple_partials_then_event_order_matches_stream_order(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        calls: list = []
        partials = ["alpha", "beta", "gamma"]
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=partials, final="".join(partials)),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: calls.append(e),
            build_result=lambda t, c, m: MagicMock(),
        )
        assert [e.partial for e in calls] == partials

    @pytest.mark.asyncio
    async def test_when_stream_yields_partial_then_the_emitted_event_is_a_token_event_instance(
        self,
    ):
        from dejavu.runner import run_side, Side, TokenEvent
        from dejavu.pricing import Model

        calls: list = []
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=["tok"], final="tok"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: calls.append(e),
            build_result=lambda t, c, m: MagicMock(),
        )
        assert isinstance(calls[0], TokenEvent)


# Invariant: for any list of N partials, exactly N TokenEvents are emitted.
@given(st.lists(st.text(), min_size=0, max_size=20))
@settings(max_examples=30)
def test_when_stream_yields_n_partials_then_exactly_n_token_events_are_emitted(
    partials: list[str],
) -> None:
    """
    AC-3 count invariant: #events == #partials for every non-negative list size.
    Derived from 'each streamed partial emits exactly one TokenEvent'.
    asyncio.run() drives the coroutine from a sync Hypothesis test body.
    """
    from dejavu.runner import run_side, Side
    from dejavu.pricing import Model

    calls: list = []

    async def _run():
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=partials, final="".join(partials)),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: calls.append(e),
            build_result=lambda t, c, m: None,
        )

    asyncio.run(_run())
    assert len(calls) == len(partials)


# ---------------------------------------------------------------------------
# AC-4: _emit dispatches to sync or async on_token correctly
# ---------------------------------------------------------------------------


class TestEmit:
    """AC-4: _emit calls on_token if sync, awaits it if async — both sinks are supported."""

    def test_when_emit_is_imported_then_it_is_callable(self):
        from dejavu.runner import _emit

        assert callable(_emit)

    def test_when_on_token_is_sync_then_emit_calls_it_and_event_is_delivered(self):
        from dejavu.runner import _emit, Side, TokenEvent

        calls: list = []
        event = TokenEvent(index=0, side=Side.UNCACHED, partial="tok")
        asyncio.run(_emit(lambda e: calls.append(e), event))
        assert calls == [event]

    def test_when_on_token_is_async_then_emit_awaits_it_and_event_is_delivered(self):
        from dejavu.runner import _emit, Side, TokenEvent

        calls: list = []

        async def async_sink(e):
            calls.append(e)

        event = TokenEvent(index=1, side=Side.CACHED, partial="async-tok")
        asyncio.run(_emit(async_sink, event))
        assert calls == [event]

    def test_when_on_token_is_sync_returning_none_then_emit_does_not_raise(self):
        from dejavu.runner import _emit, Side, TokenEvent

        event = TokenEvent(index=0, side=Side.UNCACHED, partial="x")
        asyncio.run(_emit(lambda e: None, event))

    def test_when_on_token_is_async_returning_none_then_emit_does_not_raise(self):
        from dejavu.runner import _emit, Side, TokenEvent

        async def noop(e):
            pass

        event = TokenEvent(index=0, side=Side.UNCACHED, partial="x")
        asyncio.run(_emit(noop, event))

    @pytest.mark.asyncio
    async def test_when_run_side_uses_sync_on_token_then_all_partials_are_delivered_in_order(
        self,
    ):
        """Integration: sync sink works end-to-end through run_side / _emit."""
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        received: list = []
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=["p1", "p2", "p3"], final="p1p2p3"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: received.append(e.partial),
            build_result=lambda t, c, m: MagicMock(),
        )
        assert received == ["p1", "p2", "p3"]

    @pytest.mark.asyncio
    async def test_when_run_side_uses_async_on_token_then_all_partials_are_delivered_in_order(
        self,
    ):
        """Integration: async sink works end-to-end through run_side / _emit."""
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        received: list = []

        async def async_sink(e):
            received.append(e.partial)

        await run_side(
            index=0,
            side=Side.CACHED,
            stream_fn=_FakeStreamFn(partials=["x", "y", "z"], final="xyz"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=async_sink,
            build_result=lambda t, c, m: MagicMock(),
        )
        assert received == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# AC-5: run_side returns build_result(text, collector, model)
# ---------------------------------------------------------------------------


class TestRunSideBuildResult:
    """AC-5: run_side passes final text, collector, and model to build_result and returns its value."""

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_return_value_is_build_result_return_value(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        sentinel = object()
        result = await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=["tok"], final="final answer"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: sentinel,
        )
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_build_result_receives_get_final_response_text(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        received_text: list = []
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(
                partials=["hello ", "world"], final="my final text"
            ),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: received_text.append(t),
        )
        assert received_text == ["my final text"]

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_build_result_receives_the_model(self):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        received_models: list = []
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=["tok"], final="tok"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.FABLE_5,
            on_token=lambda e: None,
            build_result=lambda t, c, m: received_models.append(m),
        )
        assert received_models == [Model.FABLE_5]

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_build_result_receives_a_baml_py_collector(
        self,
    ):
        import baml_py
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        received_collectors: list = []
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=["tok"], final="tok"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: received_collectors.append(c),
        )
        assert len(received_collectors) == 1
        assert isinstance(received_collectors[0], baml_py.Collector)

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_then_build_result_collector_is_same_instance_as_baml_options_collector(
        self,
    ):
        """
        The Collector threaded into baml_options (AC-2) must be the same Python object
        forwarded to build_result (AC-5) — verifies there is one Collector per call, used
        throughout the entire run_side execution.
        """
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        stream_fn = _FakeStreamFn(partials=["tok"], final="tok")
        build_collectors: list = []

        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=stream_fn,
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: build_collectors.append(c),
        )

        baml_options_collector = stream_fn.last_kwargs["baml_options"]["collector"]
        assert build_collectors[0] is baml_options_collector

    @pytest.mark.asyncio
    async def test_when_run_side_is_called_with_opus_model_then_build_result_receives_opus_model(
        self,
    ):
        from dejavu.runner import run_side, Side
        from dejavu.pricing import Model

        received_models: list = []
        await run_side(
            index=0,
            side=Side.UNCACHED,
            stream_fn=_FakeStreamFn(partials=["tok"], final="tok"),
            document="d",
            history=[],
            question="q?",
            registry=MagicMock(),
            model=Model.OPUS_48,
            on_token=lambda e: None,
            build_result=lambda t, c, m: received_models.append(m),
        )
        assert received_models == [Model.OPUS_48]
