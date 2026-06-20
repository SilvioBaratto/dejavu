"""
Source-blind example tests for issue #10:
  feat: add runtime capped/low-temperature ClientRegistry and model selection for the runner

Tests derived solely from acceptance criteria and requirements.md.
No implementation source was read.

Design choices (where spec is ambiguous):
- Model is imported from dejavu.runner (natural home per project structure in requirements.md).
- "claude-sonnet-4-6" is the exact ID for Sonnet 4.6 (the single model this project exercises).
- temperature range [0.0, 1.0] inferred from "low-temperature" in the issue title.
- set_primary test patches baml_py.ClientRegistry and forces a fresh import of dejavu.runner so
  the patch is active when the module binds its reference to the class, regardless of import style.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import baml_py
import pytest
from hypothesis import given, settings, strategies as st


# ─── _MODEL_IDS: maps the single Model member to its claude-* id ─────────────


def test_when_model_ids_inspected_then_every_model_member_is_mapped():
    from dejavu.runner import Model, _MODEL_IDS

    assert set(_MODEL_IDS.keys()) == set(Model)


def test_when_model_ids_inspected_then_all_values_start_with_claude():
    from dejavu.runner import _MODEL_IDS

    for model_id in _MODEL_IDS.values():
        assert model_id.startswith("claude-"), (
            f"Expected 'claude-' prefix, got {model_id!r}"
        )


def test_when_model_ids_inspected_then_claude_sonnet_4_6_is_among_values():
    """The single model this project exercises maps to claude-sonnet-4-6."""
    from dejavu.runner import _MODEL_IDS

    assert "claude-sonnet-4-6" in _MODEL_IDS.values()


# ─── _client_options: returns correct dict contents ──────────────────────────


def test_when_client_options_called_then_model_key_equals_model_ids_entry(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from dejavu.runner import Model, _MODEL_IDS, _client_options

    model = next(iter(Model))
    opts = _client_options(model, 400, 0.1)
    assert opts["model"] == _MODEL_IDS[model]


def test_when_client_options_called_then_api_key_equals_anthropic_api_key_env(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-sentinel-abc-123")
    from dejavu.runner import Model, _client_options

    model = next(iter(Model))
    opts = _client_options(model, 400, 0.1)
    assert opts["api_key"] == "sk-sentinel-abc-123"


def test_when_client_options_called_then_max_tokens_value_is_preserved(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from dejavu.runner import Model, _client_options

    model = next(iter(Model))
    opts = _client_options(model, 512, 0.1)
    assert opts["max_tokens"] == 512


def test_when_client_options_called_then_temperature_value_is_preserved(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from dejavu.runner import Model, _client_options

    model = next(iter(Model))
    opts = _client_options(model, 400, 0.3)
    assert opts["temperature"] == pytest.approx(0.3)


def test_when_client_options_called_then_allowed_role_metadata_is_cache_control_list(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from dejavu.runner import Model, _client_options

    model = next(iter(Model))
    opts = _client_options(model, 400, 0.1)
    assert opts["allowed_role_metadata"] == ["cache_control"]


def test_when_client_options_called_for_each_model_member_then_id_matches_model_ids(
    monkeypatch,
):
    """Each Model member maps to its _MODEL_IDS value in the returned dict."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from dejavu.runner import Model, _MODEL_IDS, _client_options

    for model in Model:
        opts = _client_options(model, 400, 0.1)
        assert opts["model"] == _MODEL_IDS[model], (
            f"Model {model!r}: expected {_MODEL_IDS[model]!r}, got {opts['model']!r}"
        )


# ─── API key is read from ANTHROPIC_API_KEY (no hardcoded secret) ────────────


def test_when_anthropic_api_key_env_set_then_api_key_option_reflects_it(monkeypatch):
    """The key must come from ANTHROPIC_API_KEY, not be baked into source."""
    secret = "sk-env-driven-key-unique-xyz"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    from dejavu.runner import Model, _client_options

    model = next(iter(Model))
    opts = _client_options(model, 400, 0.1)
    assert opts["api_key"] == secret


# ─── build_registry: returns ClientRegistry and calls set_primary ────────────


def test_when_build_registry_called_then_result_is_baml_client_registry_instance(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from dejavu.runner import Model, build_registry

    model = next(iter(Model))
    result = build_registry(model, max_tokens=400, temperature=0.0)
    assert isinstance(result, baml_py.ClientRegistry)


def test_when_build_registry_called_then_set_primary_is_invoked_on_registry(
    monkeypatch,
):
    """
    Verifies set_primary is called on the ClientRegistry instance.
    Forces a fresh import of dejavu.runner inside the patch context so the
    implementation's reference to baml_py.ClientRegistry is the mock,
    regardless of whether it uses `import baml_py` or `from baml_py import ...`.
    monkeypatch.delitem restores sys.modules to its pre-test state after the test.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delitem(sys.modules, "dejavu.runner", raising=False)

    mock_instance = MagicMock()
    with patch("baml_py.ClientRegistry", return_value=mock_instance):
        from dejavu.runner import Model, build_registry

        model = next(iter(Model))
        build_registry(model, max_tokens=400, temperature=0.0)

    mock_instance.set_primary.assert_called_once()


# ─── Properties: _client_options invariants over valid input domain ───────────

_REQUIRED_KEYS = {
    "model",
    "api_key",
    "max_tokens",
    "temperature",
    "allowed_role_metadata",
}


@given(
    max_tokens=st.integers(min_value=1, max_value=8192),
    temperature=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=30)
def test_when_client_options_called_with_any_valid_params_then_all_required_keys_are_present(
    max_tokens: int,
    temperature: float,
) -> None:
    """
    Invariant: for any (max_tokens, temperature) in the valid domain, _client_options
    returns a dict containing all five required keys.
    Derived from criterion: '_client_options returns model id, api_key, max_tokens,
    temperature, and allowed_role_metadata == ["cache_control"]'.
    """
    from dejavu.runner import Model, _client_options

    model = next(iter(Model))
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-hyp-test"}):
        opts = _client_options(model, max_tokens, temperature)
    assert _REQUIRED_KEYS.issubset(opts.keys())


@given(
    max_tokens=st.integers(min_value=1, max_value=8192),
    temperature=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=30)
def test_when_client_options_called_with_any_valid_params_then_allowed_role_metadata_is_always_cache_control(
    max_tokens: int,
    temperature: float,
) -> None:
    """
    Invariant: allowed_role_metadata == ["cache_control"] holds for every valid input.
    Required by BAML prompt-caching setup (requirements.md §Feature 1 + §Additional Notes).
    """
    from dejavu.runner import Model, _client_options

    model = next(iter(Model))
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-hyp-test"}):
        opts = _client_options(model, max_tokens, temperature)
    assert opts["allowed_role_metadata"] == ["cache_control"]
