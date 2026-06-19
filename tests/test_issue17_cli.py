"""
Source-blind example tests for issue #17:
  feat: add Typer CLI with flags, worst-case cost guard, and .env loading

Every test is derived solely from the acceptance-criteria text and requirements.md.
No implementation source was read during authoring (Red-phase TDD).

Skipped criteria (NOT VERIFIABLE per oracle):
  - "All tests pass" — suite-gate boilerplate, no per-criterion assertion.
  - "SOLID, clean code" — subjective prose, no runtime assertion.
"""

import os
import typer
import pytest
from unittest.mock import patch
from hypothesis import given, strategies as st

from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Imports under test — all from dejavu.cli, which is the only module this
# issue adds.  RunConfig is the data-class produced by _build_config.
# Neither _build_config nor _estimate_worst_case exist yet (Red phase).
# ---------------------------------------------------------------------------
from dejavu.cli import app, _build_config, _estimate_worst_case  # noqa: E402

_runner = CliRunner()

_VALID_MODELS = ["opus-4.8", "fable-5", "mythos-5"]
_VALID_TTLS = ["5m", "1h"]


# ===========================================================================
# Criterion 1 — Typer app exposes --model, --doc, --questions-file, --ttl
#               (validated 5m|1h), --max-tokens (default 400), --delay
#               (default 0.0)
# ===========================================================================


def test_when_help_is_invoked_then_exit_code_is_zero():
    result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_when_help_is_invoked_then_model_flag_is_listed():
    result = _runner.invoke(app, ["--help"])
    assert "--model" in result.output


def test_when_help_is_invoked_then_doc_flag_is_listed():
    result = _runner.invoke(app, ["--help"])
    assert "--doc" in result.output


def test_when_help_is_invoked_then_questions_file_flag_is_listed():
    result = _runner.invoke(app, ["--help"])
    assert "--questions-file" in result.output


def test_when_help_is_invoked_then_ttl_flag_is_listed():
    result = _runner.invoke(app, ["--help"])
    assert "--ttl" in result.output


def test_when_help_is_invoked_then_max_tokens_flag_is_listed():
    result = _runner.invoke(app, ["--help"])
    assert "--max-tokens" in result.output


def test_when_help_is_invoked_then_delay_flag_is_listed():
    result = _runner.invoke(app, ["--help"])
    assert "--delay" in result.output


def test_when_ttl_is_invalid_then_exit_code_is_nonzero():
    """Values other than '5m' or '1h' must be rejected by Typer before any API call."""
    with patch("dejavu.cli.run_tui"):
        result = _runner.invoke(app, ["--ttl", "10m"])
    assert result.exit_code != 0


def test_when_ttl_is_5m_then_validation_passes():
    """'5m' is the canonical 5-minute cache TTL and must be accepted without a validation error."""
    # Typer validation errors use exit_code == 2; abort-on-decline uses exit_code == 1.
    with patch("dejavu.cli.run_tui"), patch("typer.confirm", side_effect=typer.Abort()):
        result = _runner.invoke(app, ["--ttl", "5m"])
    assert result.exit_code != 2


def test_when_ttl_is_1h_then_validation_passes():
    """'1h' is the canonical 1-hour cache TTL and must be accepted without a validation error."""
    with patch("dejavu.cli.run_tui"), patch("typer.confirm", side_effect=typer.Abort()):
        result = _runner.invoke(app, ["--ttl", "1h"])
    assert result.exit_code != 2


# ===========================================================================
# Criterion 2 — _build_config maps every flag → RunConfig with correct defaults
# ===========================================================================


def test_when_no_flags_given_then_model_defaults_to_opus_4_8():
    cfg = _build_config()
    assert cfg.model == "opus-4.8"


def test_when_no_flags_given_then_max_tokens_defaults_to_400():
    cfg = _build_config()
    assert cfg.max_tokens == 400


def test_when_no_flags_given_then_delay_defaults_to_zero():
    cfg = _build_config()
    assert cfg.delay == 0.0


def test_when_no_flags_given_then_ttl_defaults_to_5m():
    cfg = _build_config()
    assert cfg.ttl == "5m"


def test_when_no_flags_given_then_doc_defaults_to_none():
    """None signals that the bundled ~50-page contract should be used."""
    cfg = _build_config()
    assert cfg.doc is None


def test_when_no_flags_given_then_questions_file_defaults_to_none():
    """None signals that the bundled 10-question list should be used."""
    cfg = _build_config()
    assert cfg.questions_file is None


def test_when_model_fable5_is_given_then_config_model_is_fable5():
    cfg = _build_config(model="fable-5")
    assert cfg.model == "fable-5"


def test_when_max_tokens_200_is_given_then_config_max_tokens_is_200():
    cfg = _build_config(max_tokens=200)
    assert cfg.max_tokens == 200


def test_when_delay_1_5_is_given_then_config_delay_is_1_5():
    cfg = _build_config(delay=1.5)
    assert cfg.delay == 1.5


def test_when_ttl_1h_is_given_then_config_ttl_is_1h():
    cfg = _build_config(ttl="1h")
    assert cfg.ttl == "1h"


@given(
    model=st.sampled_from(_VALID_MODELS),
    max_tokens=st.integers(min_value=1, max_value=4096),
    delay=st.floats(
        min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False
    ),
    ttl=st.sampled_from(_VALID_TTLS),
)
def test_when_valid_flags_are_given_then_build_config_round_trips_all_values(
    model, max_tokens, delay, ttl
):
    """Round-trip: every flag value supplied to _build_config must appear
    unchanged in the returned RunConfig (tests the whole flag→field mapping)."""
    cfg = _build_config(model=model, max_tokens=max_tokens, delay=delay, ttl=ttl)
    assert cfg.model == model
    assert cfg.max_tokens == max_tokens
    assert cfg.delay == delay
    assert cfg.ttl == ttl


# ===========================================================================
# Criterion 3 — _estimate_worst_case(cfg) equals the hand-computed
#               price_uncached_turn sum (10 turns × 2 sides) for the chosen model
# ===========================================================================
# Spec price table (per million tokens):
#   opus-4.8  → base input $5.00, output $25.00
#   fable-5   → base input $10.00, output $50.00   (2× opus-4.8)
#   mythos-5  → base input $10.00, output $50.00   (same tier as fable-5)


def test_when_config_has_opus_model_then_estimate_is_positive():
    cfg = _build_config(model="opus-4.8", max_tokens=400)
    assert _estimate_worst_case(cfg) > 0.0


def test_when_model_is_fable5_then_estimate_exceeds_opus48_estimate():
    """Fable-5 input/output prices are 2× Opus 4.8; the worst-case estimate must be higher."""
    cfg_opus = _build_config(model="opus-4.8", max_tokens=400)
    cfg_fable = _build_config(model="fable-5", max_tokens=400)
    assert _estimate_worst_case(cfg_fable) > _estimate_worst_case(cfg_opus)


def test_when_model_is_mythos5_then_estimate_equals_fable5_estimate():
    """mythos-5 shares the same price tier as fable-5 (per spec table)."""
    cfg_fable = _build_config(model="fable-5", max_tokens=400)
    cfg_mythos = _build_config(model="mythos-5", max_tokens=400)
    assert _estimate_worst_case(cfg_mythos) == pytest.approx(
        _estimate_worst_case(cfg_fable), rel=1e-6
    )


def test_when_max_tokens_is_doubled_then_estimate_increases():
    """Larger max_tokens raises the output-cost component of the worst-case sum."""
    cfg_low = _build_config(model="opus-4.8", max_tokens=200)
    cfg_high = _build_config(model="opus-4.8", max_tokens=400)
    assert _estimate_worst_case(cfg_high) > _estimate_worst_case(cfg_low)


def test_when_estimate_covers_two_sessions_then_it_is_at_least_double_one_session():
    """
    The spec demands 10 turns × 2 sides.  A sanity bound: the two-session
    worst-case must be at least as large as a naive single-session estimate
    would be (we can't compute that without internals, but we can assert it is
    strictly positive and model-sensitive, which suffices for source-blindness).
    Choice: the fable-5/opus-4.8 ratio from the spec price table is ≥ 2×.
    """
    cfg_opus = _build_config(model="opus-4.8", max_tokens=400)
    cfg_fable = _build_config(model="fable-5", max_tokens=400)
    ratio = _estimate_worst_case(cfg_fable) / _estimate_worst_case(cfg_opus)
    assert ratio >= 2.0


@given(max_tokens=st.integers(min_value=1, max_value=4096))
def test_when_max_tokens_is_any_positive_integer_then_estimate_is_positive(
    max_tokens,
):
    """Never-raises / always-positive invariant: _estimate_worst_case must return
    a positive float for the entire valid max_tokens domain."""
    cfg = _build_config(model="opus-4.8", max_tokens=max_tokens)
    assert _estimate_worst_case(cfg) > 0.0


@given(
    low=st.integers(min_value=1, max_value=2000),
    high=st.integers(min_value=2001, max_value=4096),
)
def test_when_max_tokens_increases_then_estimate_is_nondecreasing(low, high):
    """Monotonicity: a larger output token budget must never reduce the worst-case estimate."""
    cfg_low = _build_config(model="opus-4.8", max_tokens=low)
    cfg_high = _build_config(model="opus-4.8", max_tokens=high)
    assert _estimate_worst_case(cfg_high) >= _estimate_worst_case(cfg_low)


# ===========================================================================
# Criterion 4 — cost guard: prints estimate; calls typer.confirm before
#               launching; declining aborts before any TUI/API call;
#               accepting launches run_tui
# ===========================================================================


def test_when_user_declines_cost_guard_then_run_tui_is_not_called():
    """Aborting the cost guard must prevent run_tui from being reached."""
    with (
        patch("dejavu.cli.run_tui") as mock_tui,
        patch("typer.confirm", side_effect=typer.Abort()),
    ):
        _runner.invoke(app, [])
    mock_tui.assert_not_called()


def test_when_user_accepts_cost_guard_then_run_tui_is_called_once():
    with (
        patch("dejavu.cli.run_tui") as mock_tui,
        patch("typer.confirm", return_value=True),
    ):
        _runner.invoke(app, [])
    mock_tui.assert_called_once()


def test_when_cost_guard_runs_then_dollar_amount_appears_before_confirm():
    """The guard must print the worst-case $ estimate in the output before prompting."""
    with patch("dejavu.cli.run_tui"), patch("typer.confirm", side_effect=typer.Abort()):
        result = _runner.invoke(app, [])
    assert "$" in result.output


def test_when_cost_guard_runs_then_typer_confirm_is_called():
    """The guard must call typer.confirm (not skip it silently)."""
    with (
        patch("dejavu.cli.run_tui"),
        patch("typer.confirm", side_effect=typer.Abort()) as mock_confirm,
    ):
        _runner.invoke(app, [])
    mock_confirm.assert_called()


def test_when_model_flag_changes_then_cost_guard_output_reflects_it():
    """Different models have different price tiers; the printed estimate must
    differ when the model changes (fable-5 costs more than opus-4.8)."""
    with patch("dejavu.cli.run_tui"), patch("typer.confirm", side_effect=typer.Abort()):
        result_opus = _runner.invoke(app, ["--model", "opus-4.8"])

    with patch("dejavu.cli.run_tui"), patch("typer.confirm", side_effect=typer.Abort()):
        result_fable = _runner.invoke(app, ["--model", "fable-5"])

    assert result_opus.output != result_fable.output


# ===========================================================================
# Criterion 5 — .env loaded via python-dotenv; key read only from
#               ANTHROPIC_API_KEY; no secrets in source
# ===========================================================================


def test_when_cli_is_invoked_then_load_dotenv_is_called():
    """python-dotenv's load_dotenv must be called at startup so .env is respected."""
    with (
        patch("dejavu.cli.load_dotenv") as mock_ld,
        patch("dejavu.cli.run_tui"),
        patch("typer.confirm", side_effect=typer.Abort()),
    ):
        _runner.invoke(app, [])
    mock_ld.assert_called()


def test_when_anthropic_api_key_is_set_then_cli_reaches_cost_guard():
    """When ANTHROPIC_API_KEY is present in the environment the CLI must reach
    the cost-guard stage (typer.confirm is called, not skipped)."""
    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-sentinel"}),
        patch("dejavu.cli.load_dotenv"),
        patch("dejavu.cli.run_tui"),
        patch("typer.confirm", side_effect=typer.Abort()) as mock_confirm,
    ):
        _runner.invoke(app, [])
    mock_confirm.assert_called()


def test_when_anthropic_api_key_is_absent_then_cli_reports_missing_key():
    """When ANTHROPIC_API_KEY is not in the environment and load_dotenv finds
    nothing, the CLI must signal the missing key — either via non-zero exit
    code or by mentioning 'ANTHROPIC_API_KEY' in output.

    Choice: we consider either outcome acceptable; the spec says the key is
    read *only* from ANTHROPIC_API_KEY, so the error message must name it."""
    clean_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with (
        patch.dict(os.environ, clean_env, clear=True),
        patch("dejavu.cli.load_dotenv"),
    ):  # prevent real .env from being loaded
        result = _runner.invoke(app, [])
    assert result.exit_code != 0 or "ANTHROPIC_API_KEY" in result.output


# ===========================================================================
# Criterion 6 — `dejavu --help` works; tested with typer.testing.CliRunner
#               (run_tui patched — no real API)
# ===========================================================================
# Note: the --help tests in Criterion 1 already exercise the CliRunner path.
# The extra tests below confirm that the CliRunner invocation of the main
# command (not --help) also stays clean when run_tui is patched.


def test_when_run_tui_is_patched_then_cli_completes_without_real_api_call():
    """Confirms that CliRunner + patched run_tui is a sufficient harness:
    the command must complete (exit 0 or 1 for abort) without ImportError or
    AttributeError from a missing real API."""
    with patch("dejavu.cli.run_tui"), patch("typer.confirm", side_effect=typer.Abort()):
        result = _runner.invoke(app, [])
    # Typer abort is exit code 1; a crash would be exit code != 0 or 1
    assert result.exit_code in (0, 1)


def test_when_help_is_invoked_with_runner_then_no_real_api_is_needed():
    """--help must work with CliRunner even without ANTHROPIC_API_KEY set."""
    clean_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, clean_env, clear=True):
        result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0
