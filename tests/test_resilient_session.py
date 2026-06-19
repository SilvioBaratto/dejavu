"""
Source-blind tests for Issue #15: resilient session wrapper retrying 429/5xx.

All tests derived exclusively from acceptance criteria and requirements.md.
No implementation source was consulted.

Oracle classification:
  [UNIT]          criterion 1 — async pass-through of TurnResult
  [UNIT]          criterion 2 — _is_transient status codes
  [UNIT]          criterion 3 — _backoff_delay monotone + capped
  [UNIT]          criterion 4 — on_note / sleep / restart / skip / clear
  [NOT VERIFIABLE] criterion 5 — non-transient errors propagate (skipped)
  [UNIT]          criterion 6 — injectable dependencies, no real API
  [NOT VERIFIABLE] criteria 7-8 — suite gate / SOLID prose (skipped)
"""

import pytest
from hypothesis import given, strategies as st


# ---------------------------------------------------------------------------
# Fixtures — built from acceptance-criteria text, not from implementation
# ---------------------------------------------------------------------------


class _FakeCfg:
    """Minimal stand-in for the cfg argument that resilient_session passes through."""


class _StatusExc(Exception):
    """
    Stub exception with a .status_code attribute.

    _is_transient is expected to duck-type on this attribute rather than
    checking isinstance — consistent with the injected-dependency criterion
    that demands full testability with no real Anthropic SDK objects.
    """

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeTurn:
    """
    Sentinel that stands in for TurnResult.

    Reading TurnResult's definition would require opening implementation source,
    which is forbidden here.  Tests verify that resilient_session passes through
    whatever the session_factory yields — structural identity is sufficient.
    """

    def __init__(self, index: int) -> None:
        self.index = index

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeTurn) and self.index == other.index

    def __repr__(self) -> str:
        return f"_FakeTurn({self.index})"


# ---------------------------------------------------------------------------
# Criterion 1 — async-generates every TurnResult the underlying session yields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_session_succeeds_without_errors_then_all_turns_are_yielded():
    """resilient_session is a transparent pass-through when the underlying session completes normally."""
    from dejavu.runner import resilient_session

    expected = [_FakeTurn(0), _FakeTurn(1), _FakeTurn(2)]

    async def good_session(cfg, *, on_token):
        for turn in expected:
            yield turn

    results = []
    async for turn in resilient_session(
        _FakeCfg(),
        on_token=lambda _t: None,
        session_factory=good_session,
        max_retries=3,
        base_delay=0.0,
    ):
        results.append(turn)

    assert results == expected


@pytest.mark.asyncio
async def test_when_underlying_session_is_empty_then_resilient_session_yields_nothing():
    """An empty underlying session must produce no output even through the wrapper."""
    from dejavu.runner import resilient_session

    async def empty_session(cfg, *, on_token):
        return
        yield  # makes this an async generator without emitting anything

    results = []
    async for turn in resilient_session(
        _FakeCfg(),
        on_token=lambda _t: None,
        session_factory=empty_session,
        max_retries=3,
        base_delay=0.0,
    ):
        results.append(turn)

    assert results == []


# ---------------------------------------------------------------------------
# Criterion 2 — _is_transient: True for 429/500/502/503/504, False for 400/401
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_when_exception_has_transient_status_then_is_transient_returns_true(
    status_code,
):
    from dejavu.runner import _is_transient

    assert _is_transient(_StatusExc(status_code)) is True


@pytest.mark.parametrize("status_code", [400, 401])
def test_when_exception_has_non_transient_status_then_is_transient_returns_false(
    status_code,
):
    from dejavu.runner import _is_transient

    assert _is_transient(_StatusExc(status_code)) is False


# ---------------------------------------------------------------------------
# Criterion 3 — _backoff_delay: monotonically increasing and capped
# ---------------------------------------------------------------------------


def test_when_attempt_number_increases_then_delay_is_nondecreasing():
    """Concrete check across the first five attempts."""
    from dejavu.runner import _backoff_delay

    delays = [_backoff_delay(i) for i in range(5)]
    for i in range(len(delays) - 1):
        assert delays[i] <= delays[i + 1], (
            f"delay decreased from attempt {i} to {i + 1}: "
            f"{delays[i]} > {delays[i + 1]}"
        )


def test_when_attempt_is_very_large_then_delay_plateaus():
    """The delay must reach a cap — attempts 90-100 must all return the same value."""
    from dejavu.runner import _backoff_delay

    tail = [_backoff_delay(i) for i in range(90, 101)]
    assert len(set(tail)) == 1, (
        f"Expected delay to plateau before attempt 90; got varying values: {tail}"
    )


@given(
    st.integers(min_value=0, max_value=50),
    st.integers(min_value=0, max_value=50),
)
def test_when_attempt_a_is_at_most_b_then_delay_a_is_at_most_delay_b(a, b):
    """Property: _backoff_delay is monotonically non-decreasing in its attempt argument."""
    from dejavu.runner import _backoff_delay

    if a <= b:
        assert _backoff_delay(a) <= _backoff_delay(b)


@given(st.integers(min_value=0, max_value=10_000))
def test_when_attempt_is_any_nonnegative_integer_then_delay_is_bounded_and_nonnegative(
    attempt,
):
    """Property: _backoff_delay returns a finite, non-negative value (≤ 3 600 s) for all inputs."""
    from dejavu.runner import _backoff_delay

    delay = _backoff_delay(attempt)
    assert 0 <= delay <= 3_600


# ---------------------------------------------------------------------------
# Criterion 4 — on_note / sleep / restart / skip already-yielded / clear on success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_transient_error_occurs_then_on_note_receives_retrying_message():
    """on_note must be called with the string 'retrying…' (Unicode ellipsis) on transient failure."""
    from dejavu.runner import resilient_session

    notes: list = []
    call_count = 0

    async def flaky_session(cfg, *, on_token):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _StatusExc(429)
        yield _FakeTurn(0)

    async def fake_sleep(_s: float) -> None:
        pass

    async for _ in resilient_session(
        _FakeCfg(),
        on_token=lambda _t: None,
        session_factory=flaky_session,
        on_note=notes.append,
        sleep=fake_sleep,
        max_retries=3,
        base_delay=0.0,
    ):
        pass

    assert "retrying…" in notes


@pytest.mark.asyncio
async def test_when_transient_error_occurs_then_injected_sleep_is_called():
    """The injected sleep callable — not asyncio.sleep — must be invoked on transient failure."""
    from dejavu.runner import resilient_session

    sleep_durations: list = []
    call_count = 0

    async def flaky_session(cfg, *, on_token):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _StatusExc(503)
        yield _FakeTurn(0)

    async def recording_sleep(seconds: float) -> None:
        sleep_durations.append(seconds)

    async for _ in resilient_session(
        _FakeCfg(),
        on_token=lambda _t: None,
        session_factory=flaky_session,
        sleep=recording_sleep,
        max_retries=3,
        base_delay=0.0,
    ):
        pass

    assert len(sleep_durations) >= 1


@pytest.mark.asyncio
async def test_when_retry_succeeds_then_on_note_is_cleared_with_none():
    """on_note(None) must be called after recovery to clear the 'retrying…' indicator."""
    from dejavu.runner import resilient_session

    notes: list = []
    call_count = 0

    async def flaky_session(cfg, *, on_token):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _StatusExc(502)
        yield _FakeTurn(0)

    async def fake_sleep(_s: float) -> None:
        pass

    async for _ in resilient_session(
        _FakeCfg(),
        on_token=lambda _t: None,
        session_factory=flaky_session,
        on_note=notes.append,
        sleep=fake_sleep,
        max_retries=3,
        base_delay=0.0,
    ):
        pass

    assert "retrying…" in notes
    retrying_idx = notes.index("retrying…")
    later_notes = notes[retrying_idx + 1 :]
    assert None in later_notes, (
        "on_note(None) must follow 'retrying…' once the retry succeeds"
    )


@pytest.mark.asyncio
async def test_when_failure_occurs_mid_stream_then_already_yielded_turns_are_not_repeated():
    """Turns yielded before a transient failure must not appear twice in the overall output."""
    from dejavu.runner import resilient_session

    call_count = 0

    async def mid_stream_failure(cfg, *, on_token):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _FakeTurn(0)
            yield _FakeTurn(1)
            raise _StatusExc(500)
        # Restart replays the full stream from the top
        yield _FakeTurn(0)
        yield _FakeTurn(1)
        yield _FakeTurn(2)
        yield _FakeTurn(3)

    async def fake_sleep(_s: float) -> None:
        pass

    results = []
    async for turn in resilient_session(
        _FakeCfg(),
        on_token=lambda _t: None,
        session_factory=mid_stream_failure,
        sleep=fake_sleep,
        max_retries=3,
        base_delay=0.0,
    ):
        results.append(turn)

    # Turns 0 and 1 were already surfaced; the retry must skip them and add only 2 and 3.
    assert results == [_FakeTurn(0), _FakeTurn(1), _FakeTurn(2), _FakeTurn(3)]


@pytest.mark.asyncio
async def test_when_transient_failure_occurs_then_session_factory_is_reinvoked():
    """On a transient failure resilient_session must restart by calling session_factory again."""
    from dejavu.runner import resilient_session

    call_count = 0

    async def flaky_session(cfg, *, on_token):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _StatusExc(504)
        yield _FakeTurn(0)

    async def fake_sleep(_s: float) -> None:
        pass

    async for _ in resilient_session(
        _FakeCfg(),
        on_token=lambda _t: None,
        session_factory=flaky_session,
        sleep=fake_sleep,
        max_retries=3,
        base_delay=0.0,
    ):
        pass

    assert call_count >= 2


@pytest.mark.asyncio
async def test_when_on_note_is_omitted_and_transient_failure_occurs_then_no_exception_is_raised():
    """Default on_note=None must be a no-op — no AttributeError or TypeError when emitting notes."""
    from dejavu.runner import resilient_session

    call_count = 0

    async def flaky_session(cfg, *, on_token):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _StatusExc(429)
        yield _FakeTurn(0)

    async def fake_sleep(_s: float) -> None:
        pass

    results = []
    async for turn in resilient_session(
        _FakeCfg(),
        on_token=lambda _t: None,
        session_factory=flaky_session,
        # on_note intentionally omitted — must default to None safely
        sleep=fake_sleep,
        max_retries=3,
        base_delay=0.0,
    ):
        results.append(turn)

    assert results == [_FakeTurn(0)]


# ---------------------------------------------------------------------------
# Criterion 6 — fully testable with injected dependencies and no real API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_all_dependencies_are_injected_then_session_runs_with_no_network_io():
    """End-to-end exercise using only in-process stubs — zero Anthropic API calls."""
    from dejavu.runner import resilient_session

    tokens_seen: list = []
    notes_seen: list = []
    sleep_calls: list = []

    async def stub_session(cfg, *, on_token):
        on_token("t1")
        yield _FakeTurn(0)
        on_token("t2")
        yield _FakeTurn(1)

    async def stub_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    results = []
    async for turn in resilient_session(
        _FakeCfg(),
        on_token=tokens_seen.append,
        session_factory=stub_session,
        on_note=notes_seen.append,
        sleep=stub_sleep,
        max_retries=3,
        base_delay=0.0,
    ):
        results.append(turn)

    assert results == [_FakeTurn(0), _FakeTurn(1)]
    assert "t1" in tokens_seen
    assert "t2" in tokens_seen
    assert sleep_calls == []  # no failures → sleep never invoked
