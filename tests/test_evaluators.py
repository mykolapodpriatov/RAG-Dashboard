"""Tests for the offline heuristic evaluator.

Only pytest + pandas are required — we import ``evaluators`` directly and never
touch ``app.py`` (which would pull in streamlit / plotly).
"""

import pandas as pd
import pytest

from evaluators import evaluate_dataframe, heuristic_evaluate

_METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def test_identical_answer_and_context_gives_full_faithfulness():
    scores = heuristic_evaluate(
        question="what are the greek letters",
        answer="alpha beta gamma",
        contexts=["alpha beta gamma"],
    )
    assert scores["faithfulness"] == 1.0
    assert scores["context_precision"] == 1.0


def test_disjoint_answer_and_context_scores_zero():
    scores = heuristic_evaluate(
        question="q",
        answer="alpha beta",
        contexts=["completely different words here"],
    )
    assert scores["faithfulness"] == 0.0
    assert scores["context_precision"] == 0.0


def test_empty_contexts_do_not_divide_by_zero():
    scores = heuristic_evaluate(
        question="alpha",
        answer="alpha beta",
        contexts=[],
    )
    assert scores["faithfulness"] == 0.0
    assert scores["context_precision"] == 0.0


def test_empty_answer_does_not_divide_by_zero():
    scores = heuristic_evaluate(
        question="alpha beta",
        answer="",
        contexts=["alpha beta"],
    )
    assert scores["faithfulness"] == 0.0
    assert scores["answer_relevancy"] == 0.0
    assert scores["context_precision"] == 0.0


def test_missing_required_column_raises_value_error():
    df = pd.DataFrame({"question": ["q"], "answer": ["a"]})  # no `contexts`
    with pytest.raises(ValueError, match="contexts"):
        evaluate_dataframe(df)


def test_unknown_backend_raises_value_error():
    df = pd.DataFrame(
        {"question": ["q"], "answer": ["a"], "contexts": [["a"]]}
    )
    with pytest.raises(ValueError, match="backend"):
        evaluate_dataframe(df, backend="ragas")


@pytest.mark.parametrize("backend", ["heuristic", "mock"])
def test_scores_always_within_unit_interval(backend):
    df = pd.DataFrame(
        {
            "question": ["what is x", "who did y", ""],
            "answer": ["x is a thing", "y was done by z", "unrelated text"],
            "contexts": [["x is indeed a thing"], ["z did the work"], []],
        }
    )
    result = evaluate_dataframe(df, backend=backend)
    for col in _METRICS:
        assert result[col].between(0.0, 1.0).all()


def test_heuristic_is_deterministic():
    df = pd.DataFrame(
        {
            "question": ["what is x"],
            "answer": ["x is a thing"],
            "contexts": [["x is indeed a thing"]],
        }
    )
    first = evaluate_dataframe(df)
    second = evaluate_dataframe(df)
    pd.testing.assert_frame_equal(first, second)


def test_original_columns_are_preserved():
    df = pd.DataFrame(
        {
            "question": ["what is x"],
            "answer": ["x is a thing"],
            "contexts": [["x is indeed a thing"]],
        }
    )
    result = evaluate_dataframe(df)
    for col in ("question", "answer", "contexts", *_METRICS):
        assert col in result.columns
