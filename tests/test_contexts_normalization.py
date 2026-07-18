"""Tests for `contexts` normalisation and CSV/JSON parity.

Only pytest + pandas are required; ``evaluators`` is imported directly.
"""

from pathlib import Path

import pandas as pd

from evaluators import _normalize_contexts, evaluate_dataframe

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


def test_list_passes_through():
    assert _normalize_contexts(["a", "b"]) == ["a", "b"]


def test_tuple_becomes_list():
    assert _normalize_contexts(("a", "b")) == ["a", "b"]


def test_list_literal_string_is_parsed():
    assert _normalize_contexts("['a', 'b']") == ["a", "b"]
    assert _normalize_contexts('["a", "b"]') == ["a", "b"]


def test_plain_string_becomes_single_context():
    assert _normalize_contexts("just a single sentence") == ["just a single sentence"]


def test_non_list_literal_is_kept_as_single_context():
    # A bare number parses via literal_eval but must not explode into digits.
    assert _normalize_contexts("42") == ["42"]


def test_nan_and_none_become_empty_list():
    assert _normalize_contexts(float("nan")) == []
    assert _normalize_contexts(None) == []


def test_empty_inputs_become_empty_list():
    assert _normalize_contexts([]) == []
    assert _normalize_contexts("") == []
    assert _normalize_contexts("   ") == []


def test_blank_csv_cell_normalises_to_empty_list():
    # Reproduce how pandas surfaces a blank cell: a scalar NaN.
    series = pd.Series([None], dtype="object")
    assert _normalize_contexts(series.iloc[0]) == []


def test_csv_and_json_examples_score_identically():
    csv_df = pd.read_csv(EXAMPLES / "sample_eval.csv")
    json_df = pd.read_json(EXAMPLES / "sample_eval.json")

    # The two uploads carry `contexts` as different Python types on the way in.
    assert isinstance(csv_df.loc[0, "contexts"], str)
    assert isinstance(json_df.loc[0, "contexts"], list)

    csv_scores = evaluate_dataframe(csv_df)[_METRICS].reset_index(drop=True)
    json_scores = evaluate_dataframe(json_df)[_METRICS].reset_index(drop=True)

    pd.testing.assert_frame_equal(csv_scores, json_scores)
    # Sanity: the fixture is not the degenerate all-zero case.
    assert csv_scores.to_numpy().sum() > 0
