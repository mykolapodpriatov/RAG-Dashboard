"""Tests for the offline batch-evaluation CLI.

Only pytest + pandas are required — ``evaluate_cli`` imports ``evaluators``
directly and never touches ``app.py`` (streamlit / plotly).
"""

from pathlib import Path

import pandas as pd

from evaluate_cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


def test_cli_writes_scored_table_and_prints_means(tmp_path, capsys):
    out = tmp_path / "scored.csv"

    exit_code = main([str(EXAMPLES / "sample_eval.csv"), "--out", str(out)])

    assert exit_code == 0
    assert out.exists()

    scored = pd.read_csv(out)
    for col in _METRICS:
        assert col in scored.columns

    # The fixture is not the degenerate all-zero case.
    means = scored[_METRICS].mean()
    assert means.between(0.0, 1.0).all()
    assert means.sum() > 0

    stdout = capsys.readouterr().out
    for col in _METRICS:
        assert col in stdout


def test_cli_without_out_only_prints(tmp_path, capsys):
    exit_code = main([str(EXAMPLES / "sample_eval.json")])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    for col in _METRICS:
        assert col in stdout
    # No table was requested, so nothing is written to disk.
    assert list(tmp_path.iterdir()) == []
