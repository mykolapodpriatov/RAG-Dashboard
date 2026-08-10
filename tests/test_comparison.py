"""Tests for the run-comparison helpers used by the "Сравнение запусков" view.

Only pytest + pandas are required — we exercise ``combine_runs`` /
``run_metric_means`` directly and never touch ``app.py`` (streamlit / plotly).
"""

import pandas as pd
import pytest

from evaluators import combine_runs, evaluate_dataframe, run_metric_means

_METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def _run(rows: list[dict]) -> pd.DataFrame:
    return evaluate_dataframe(pd.DataFrame(rows))


def test_combine_runs_labels_rows_by_run():
    run_a = _run([{"question": "q", "answer": "alpha", "contexts": ["alpha"]}])
    run_b = _run([{"question": "q", "answer": "beta", "contexts": ["beta"]}])

    comparison = combine_runs(run_a, run_b, label_a="Run A", label_b="Run B")

    assert set(comparison["run"]) == {"Run A", "Run B"}
    assert set(comparison.columns) == {"run", "metric", "score"}


def test_combine_runs_covers_all_base_metrics_for_each_run():
    run_a = _run([{"question": "q", "answer": "alpha", "contexts": ["alpha"]}])
    run_b = _run([{"question": "q", "answer": "beta", "contexts": ["beta"]}])

    comparison = combine_runs(run_a, run_b)

    for run_label in ("Run A", "Run B"):
        metrics_seen = set(comparison.loc[comparison["run"] == run_label, "metric"])
        assert metrics_seen == set(_METRICS)


def test_combine_runs_handles_metric_present_in_only_one_run():
    # Run A supplies ground_truths (-> answer_correctness); Run B does not.
    run_a = _run(
        [
            {
                "question": "q",
                "answer": "alpha beta",
                "contexts": ["alpha beta"],
                "ground_truths": ["alpha beta"],
            }
        ]
    )
    run_b = _run([{"question": "q", "answer": "beta", "contexts": ["beta"]}])

    comparison = combine_runs(run_a, run_b, label_a="Run A", label_b="Run B")

    metrics_a = set(comparison.loc[comparison["run"] == "Run A", "metric"])
    metrics_b = set(comparison.loc[comparison["run"] == "Run B", "metric"])

    assert "answer_correctness" in metrics_a
    assert "answer_correctness" not in metrics_b
    # The base metrics are still present for both runs.
    assert set(_METRICS) <= metrics_a
    assert set(_METRICS) <= metrics_b


def test_combine_runs_handles_different_row_counts():
    run_a = _run(
        [
            {"question": "q1", "answer": "alpha", "contexts": ["alpha"]},
            {"question": "q2", "answer": "beta", "contexts": ["beta"]},
            {"question": "q3", "answer": "gamma", "contexts": ["gamma"]},
        ]
    )
    run_b = _run([{"question": "q1", "answer": "alpha", "contexts": ["alpha"]}])

    comparison = combine_runs(run_a, run_b, label_a="Run A", label_b="Run B")

    # 3 rows * 3 metrics for Run A, 1 row * 3 metrics for Run B.
    assert (comparison["run"] == "Run A").sum() == 9
    assert (comparison["run"] == "Run B").sum() == 3


def test_run_metric_means_aggregates_one_row_per_run_and_metric():
    run_a = _run(
        [
            {"question": "q", "answer": "alpha beta", "contexts": ["alpha beta"]},
            {"question": "q", "answer": "alpha", "contexts": ["alpha beta gamma"]},
        ]
    )
    run_b = _run([{"question": "q", "answer": "beta", "contexts": ["beta"]}])

    comparison = combine_runs(run_a, run_b, label_a="Run A", label_b="Run B")
    means = run_metric_means(comparison)

    assert set(means.columns) == {"run", "metric", "score"}
    # One aggregated row per (run, metric) combination.
    assert len(means) == len(set(zip(means["run"], means["metric"])))
    assert means["score"].between(0.0, 1.0).all()

    # Cross-check one aggregate against a manual mean.
    faithfulness_a = means.loc[
        (means["run"] == "Run A") & (means["metric"] == "faithfulness"), "score"
    ].item()
    assert faithfulness_a == pytest.approx(run_a["faithfulness"].mean())


def test_combine_runs_default_labels():
    run_a = _run([{"question": "q", "answer": "alpha", "contexts": ["alpha"]}])
    run_b = _run([{"question": "q", "answer": "beta", "contexts": ["beta"]}])

    comparison = combine_runs(run_a, run_b)

    assert set(comparison["run"]) == {"Run A", "Run B"}
