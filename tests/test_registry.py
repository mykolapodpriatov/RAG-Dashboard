"""Tests for the evaluator registry.

The rules worth having a registry for are the ones asserted here: an unknown
name raises rather than silently falling back, and a backend that breaks the
frame contract is caught in the dispatch rather than reaching the charts.
"""

from __future__ import annotations

import pandas as pd
import pytest

import evaluators
from evaluators import (
    available_backends,
    evaluate_dataframe,
    get_evaluator,
    register_evaluator,
)


@pytest.fixture
def clean_registry():
    """Restore the registry after a test registers something."""
    saved = dict(evaluators._REGISTRY)
    yield
    evaluators._REGISTRY.clear()
    evaluators._REGISTRY.update(saved)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "answer": ["a1", "a2"],
            "contexts": [["c1"], ["c2"]],
        }
    )


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_the_shipped_backends_register_themselves() -> None:
    names = available_backends()

    assert "heuristic" in names
    assert "mock" in names
    assert names["heuristic"].is_real is False


def test_a_registered_backend_is_reachable_through_evaluate(clean_registry) -> None:
    def constant(df: pd.DataFrame) -> pd.DataFrame:
        for col in ("faithfulness", "answer_relevancy", "context_precision"):
            df[col] = 0.42
        return df

    register_evaluator("fake-real", constant, is_real=True, description="test")

    scored = evaluate_dataframe(_frame(), backend="fake-real")

    assert list(scored["faithfulness"]) == [0.42, 0.42]
    assert get_evaluator("fake-real").is_real is True


def test_is_real_travels_on_the_backend_not_the_process(clean_registry) -> None:
    # The UI warning has to describe the backend that actually ran.
    register_evaluator("real-ish", lambda df: df, is_real=True)

    assert get_evaluator("real-ish").is_real is True
    assert get_evaluator("heuristic").is_real is False


def test_a_duplicate_name_is_an_error_unless_replace_is_asked_for(clean_registry) -> None:
    # Two packages claiming one name should not resolve to whichever imported
    # last.
    register_evaluator("dup", lambda df: df, is_real=False)
    with pytest.raises(ValueError, match="already registered"):
        register_evaluator("dup", lambda df: df, is_real=False)

    register_evaluator("dup", lambda df: df, is_real=True, replace=True)
    assert get_evaluator("dup").is_real is True


def test_a_blank_name_is_rejected(clean_registry) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        register_evaluator("   ", lambda df: df, is_real=False)


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------


def test_an_unknown_backend_raises_and_names_the_registered_ones() -> None:
    # Never a silent fallback: showing proxy numbers while the caller believes
    # they asked for Ragas is the worst outcome available here.
    with pytest.raises(ValueError, match="heuristic") as exc:
        evaluate_dataframe(_frame(), backend="ragas")

    assert "ragas" in str(exc.value)


# ---------------------------------------------------------------------------
# the frame contract
# ---------------------------------------------------------------------------


def test_a_backend_that_omits_a_metric_is_rejected(clean_registry) -> None:
    def partial(df: pd.DataFrame) -> pd.DataFrame:
        df["faithfulness"] = 1.0
        return df

    register_evaluator("partial", partial, is_real=False)

    with pytest.raises(ValueError, match="answer_relevancy"):
        evaluate_dataframe(_frame(), backend="partial")


def test_a_backend_that_changes_the_row_count_is_rejected(clean_registry) -> None:
    # The comparison view groups two runs by metric; a backend that drops rows
    # would make it silently compare different things.
    def dropping(df: pd.DataFrame) -> pd.DataFrame:
        for col in ("faithfulness", "answer_relevancy", "context_precision"):
            df[col] = 1.0
        return df.head(1)

    register_evaluator("dropping", dropping, is_real=False)

    with pytest.raises(ValueError, match="row"):
        evaluate_dataframe(_frame(), backend="dropping")


def test_a_backend_that_returns_the_wrong_type_is_rejected(clean_registry) -> None:
    register_evaluator("nonsense", lambda df: "not a frame", is_real=False)

    with pytest.raises(ValueError, match="DataFrame"):
        evaluate_dataframe(_frame(), backend="nonsense")


# ---------------------------------------------------------------------------
# the seam is a refactor
# ---------------------------------------------------------------------------


def test_the_heuristic_backend_scores_exactly_as_before() -> None:
    """The registry must be provably a refactor, not a behaviour change."""
    df = pd.DataFrame(
        {
            "question": ["What is the notice period?", "Who owns the IP?"],
            "answer": ["Thirty days notice.", "The customer owns it."],
            "contexts": [["thirty days written notice"], ["intellectual property vests"]],
        }
    )

    scored = evaluate_dataframe(df, backend="heuristic")

    expected = [
        evaluators.heuristic_evaluate(row["question"], row["answer"], row["contexts"])
        for _, row in df.iterrows()
    ]
    for metric in ("faithfulness", "answer_relevancy", "context_precision"):
        assert list(scored[metric]) == [e[metric] for e in expected]


def test_the_mock_backend_stays_reproducible() -> None:
    first = evaluate_dataframe(_frame(), backend="mock")
    second = evaluate_dataframe(_frame(), backend="mock")

    assert list(first["faithfulness"]) == list(second["faithfulness"])


def test_ground_truths_still_add_answer_correctness_under_any_backend() -> None:
    df = _frame()
    df["ground_truths"] = [["a1"], ["nope"]]

    scored = evaluate_dataframe(df, backend="mock")

    assert "answer_correctness" in scored.columns
    assert scored["answer_correctness"].iloc[0] > scored["answer_correctness"].iloc[1]
