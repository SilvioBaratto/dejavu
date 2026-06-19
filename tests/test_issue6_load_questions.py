"""
Source-blind example tests — Issue #6
feat: add fixed 10-question set with overridable load_questions hook

Every test is derived solely from the acceptance criteria and requirements.md.
No implementation source was read; no implementation file was opened.
"""

import json
import pathlib
import tempfile

import pytest
from hypothesis import given, strategies as st

from dejavu.conversation import load_questions

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_QUESTIONS_JSON = _REPO_ROOT / "dejavu" / "data" / "questions.json"
_CONVERSATION_PY = _REPO_ROOT / "dejavu" / "conversation.py"


def _tmp_json(
    directory: str, data: object, name: str = "questions.json"
) -> pathlib.Path:
    path = pathlib.Path(directory) / name
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Criterion 1 — bundled questions.json: exactly 10 distinct, non-empty items
# ---------------------------------------------------------------------------


def test_when_bundled_questions_json_is_loaded_it_is_a_list():
    data = json.loads(_QUESTIONS_JSON.read_text())
    assert isinstance(data, list)


def test_when_bundled_questions_json_is_loaded_it_has_exactly_10_items():
    data = json.loads(_QUESTIONS_JSON.read_text())
    assert len(data) == 10


def test_when_bundled_questions_json_is_loaded_all_items_are_non_empty_strings():
    data = json.loads(_QUESTIONS_JSON.read_text())
    for item in data:
        assert isinstance(item, str) and item.strip() != ""


def test_when_bundled_questions_json_is_loaded_all_items_are_distinct():
    data = json.loads(_QUESTIONS_JSON.read_text())
    assert len(set(data)) == len(data)


# ---------------------------------------------------------------------------
# Criterion 2 — load_questions() returns list[str] of 10, identical each call
# ---------------------------------------------------------------------------


def test_when_load_questions_is_called_it_returns_a_list():
    result = load_questions()
    assert isinstance(result, list)


def test_when_load_questions_is_called_it_returns_exactly_10_items():
    result = load_questions()
    assert len(result) == 10


def test_when_load_questions_is_called_all_items_are_str():
    result = load_questions()
    assert all(isinstance(q, str) for q in result)


def test_when_load_questions_is_called_twice_results_are_identical():
    """Determinism: repeated no-arg calls return the same ordered list."""
    assert load_questions() == load_questions()


# ---------------------------------------------------------------------------
# Criterion 3 — validation raises ValueError with a clear message on bad input
# ---------------------------------------------------------------------------


def test_when_override_file_has_9_items_then_value_error_is_raised():
    with tempfile.TemporaryDirectory() as d:
        path = _tmp_json(d, [f"Q{i}" for i in range(9)])
        with pytest.raises(ValueError):
            load_questions(path)


def test_when_override_file_has_11_items_then_value_error_is_raised():
    with tempfile.TemporaryDirectory() as d:
        path = _tmp_json(d, [f"Q{i}" for i in range(11)])
        with pytest.raises(ValueError):
            load_questions(path)


def test_when_override_file_has_zero_items_then_value_error_is_raised():
    with tempfile.TemporaryDirectory() as d:
        path = _tmp_json(d, [])
        with pytest.raises(ValueError):
            load_questions(path)


def test_when_override_file_contains_an_empty_string_then_value_error_is_raised():
    with tempfile.TemporaryDirectory() as d:
        questions = [f"Q{i}" for i in range(9)] + [""]
        path = _tmp_json(d, questions)
        with pytest.raises(ValueError):
            load_questions(path)


def test_when_override_file_contains_a_non_string_item_then_value_error_is_raised():
    with tempfile.TemporaryDirectory() as d:
        questions = [f"Q{i}" for i in range(9)] + [42]
        path = _tmp_json(d, questions)
        with pytest.raises(ValueError):
            load_questions(path)


def test_when_override_file_is_not_a_json_array_then_value_error_is_raised():
    with tempfile.TemporaryDirectory() as d:
        payload = {"questions": [f"Q{i}" for i in range(10)]}
        path = _tmp_json(d, payload)
        with pytest.raises(ValueError):
            load_questions(path)


def test_when_wrong_count_raises_value_error_the_message_is_non_empty():
    """The error message must be informative, not blank."""
    with tempfile.TemporaryDirectory() as d:
        path = _tmp_json(d, [f"Q{i}" for i in range(5)])
        with pytest.raises(ValueError) as exc_info:
            load_questions(path)
        assert str(exc_info.value).strip() != ""


def test_when_malformed_item_raises_value_error_the_message_is_non_empty():
    """The error message must be informative for type/content violations too."""
    with tempfile.TemporaryDirectory() as d:
        path = _tmp_json(d, [f"Q{i}" for i in range(9)] + [""])
        with pytest.raises(ValueError) as exc_info:
            load_questions(path)
        assert str(exc_info.value).strip() != ""


# ---------------------------------------------------------------------------
# Criterion 4 — load_questions(path) reads and returns the given JSON file's data
# ---------------------------------------------------------------------------


def test_when_valid_override_provided_load_questions_returns_exact_questions():
    questions = [f"What does clause {i + 1} require?" for i in range(10)]
    with tempfile.TemporaryDirectory() as d:
        path = _tmp_json(d, questions)
        result = load_questions(path)
    assert result == questions


def test_when_valid_override_file_provided_result_preserves_order():
    questions = [f"Question {i + 1}" for i in range(10)]
    with tempfile.TemporaryDirectory() as d:
        path = _tmp_json(d, questions)
        result = load_questions(path)
    assert result == questions  # order must match the file, not be sorted/shuffled


def test_when_valid_override_file_provided_result_is_list_of_str():
    questions = [f"Override Q{i + 1}?" for i in range(10)]
    with tempfile.TemporaryDirectory() as d:
        path = _tmp_json(d, questions)
        result = load_questions(path)
    assert isinstance(result, list)
    assert all(isinstance(q, str) for q in result)


# ---------------------------------------------------------------------------
# Criterion 5 — conversation.py stays well under 500 lines
# ---------------------------------------------------------------------------


def test_when_conversation_py_exists_it_is_under_500_lines():
    """Questions must live in the .json file, not inlined inside conversation.py."""
    assert _CONVERSATION_PY.exists(), "dejavu/conversation.py does not exist yet"
    lines = _CONVERSATION_PY.read_text().splitlines()
    assert len(lines) < 500, f"conversation.py is {len(lines)} lines; must be < 500"


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

_VALID_QUESTION = st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")
_VALID_10 = st.lists(_VALID_QUESTION, min_size=10, max_size=10)
_WRONG_COUNT = st.lists(st.text(min_size=1), min_size=0, max_size=30).filter(
    lambda lst: len(lst) != 10
)


@given(_VALID_10)
def test_when_any_10_non_empty_strings_are_written_then_loaded_original_is_returned(
    questions,
):
    """Round-trip: 10 non-empty strings to JSON then back returns originals."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "questions.json"
        path.write_text(json.dumps(questions))
        assert load_questions(path) == questions


@given(_WRONG_COUNT)
def test_when_question_count_is_not_10_value_error_is_always_raised(questions):
    """Wrong count must always raise ValueError, never silently succeed."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "questions.json"
        path.write_text(json.dumps(questions))
        with pytest.raises(ValueError):
            load_questions(path)
