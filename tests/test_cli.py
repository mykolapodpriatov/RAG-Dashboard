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


def _write_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_cli_compare_prints_means_and_correct_delta(tmp_path, capsys):
    # Identical answer/context → faithfulness 1.0; disjoint → 0.0. Delta is B − A.
    run_a = _write_csv(
        tmp_path / "run_a.csv",
        [{"question": "q", "answer": "alpha", "contexts": "['alpha']"}],
    )
    run_b = _write_csv(
        tmp_path / "run_b.csv",
        [{"question": "q", "answer": "beta", "contexts": "['gamma']"}],
    )

    exit_code = main([str(run_a), "--compare", str(run_b)])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Run A" in stdout
    assert "Run B" in stdout
    assert "delta" in stdout
    for col in _METRICS:
        assert col in stdout

    faith_line = next(line for line in stdout.splitlines() if line.startswith("faithfulness"))
    cells = faith_line.split()
    assert cells[1] == "1.0000"
    assert cells[2] == "0.0000"
    assert cells[3] == "-1.0000"


def test_cli_compare_writes_longform_out(tmp_path, capsys):
    run_a = _write_csv(
        tmp_path / "run_a.csv",
        [{"question": "q", "answer": "alpha", "contexts": "['alpha']"}],
    )
    run_b = _write_csv(
        tmp_path / "run_b.csv",
        [{"question": "q", "answer": "beta", "contexts": "['beta']"}],
    )
    out = tmp_path / "comparison.csv"

    exit_code = main([str(run_a), "--compare", str(run_b), "--out", str(out)])

    assert exit_code == 0
    assert out.exists()
    written = pd.read_csv(out)
    assert list(written.columns) == ["run", "metric", "score"]
    assert set(written["run"]) == {"Run A", "Run B"}
    assert set(_METRICS) <= set(written["metric"])
    stdout = capsys.readouterr().out
    assert "Wrote comparison table" in stdout


def test_cli_compare_asymmetric_answer_correctness(tmp_path, capsys):
    # Only Run A supplies ground_truths, so answer_correctness is A-only.
    run_a = _write_csv(
        tmp_path / "run_a.csv",
        [
            {
                "question": "q",
                "answer": "alpha",
                "contexts": "['alpha']",
                "ground_truths": "['alpha']",
            }
        ],
    )
    run_b = _write_csv(
        tmp_path / "run_b.csv",
        [{"question": "q", "answer": "beta", "contexts": "['beta']"}],
    )

    exit_code = main([str(run_a), "--compare", str(run_b)])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    ac_line = next(
        line for line in stdout.splitlines() if line.startswith("answer_correctness")
    )
    cells = ac_line.split()
    assert cells[1] == "1.0000"
    assert cells[2] == "-"
    assert cells[3] == "-"
    for col in _METRICS:
        assert col in stdout


def test_fail_under_high_threshold_exits_1(capsys):
    exit_code = main(
        [str(EXAMPLES / "sample_eval.csv"), "--fail-under", "faithfulness=1.1"]
    )

    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert "FAIL faithfulness" in stdout
    assert "< 1.10" in stdout


def test_fail_under_zero_threshold_exits_0(capsys):
    exit_code = main(
        [str(EXAMPLES / "sample_eval.csv"), "--fail-under", "faithfulness=0.0"]
    )

    assert exit_code == 0
    assert "FAIL" not in capsys.readouterr().out


def test_fail_under_unknown_metric_exits_2(capsys):
    exit_code = main(
        [str(EXAMPLES / "sample_eval.csv"), "--fail-under", "not_a_metric=0.5"]
    )

    assert exit_code == 2
    stdout = capsys.readouterr().out
    assert "Unknown metric" in stdout
    assert "not_a_metric" in stdout
    for col in _METRICS:
        assert col in stdout
    assert "answer_correctness" not in stdout.split("present:", 1)[-1]


def test_fail_under_still_writes_out(tmp_path, capsys):
    out = tmp_path / "scored.csv"

    exit_code = main(
        [
            str(EXAMPLES / "sample_eval.csv"),
            "--fail-under",
            "faithfulness=1.1",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 1
    assert out.exists()
    assert "Wrote scored table" in capsys.readouterr().out


def _question_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("  ")]


def test_worst_1_prints_one_question_per_metric(capsys):
    exit_code = main([str(EXAMPLES / "sample_eval.csv"), "--worst", "1"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    for col in _METRICS:
        assert f"worst 1 by {col}:" in stdout
    assert "worst 1 by answer_correctness:" not in stdout
    assert len(_question_lines(stdout)) == len(_METRICS)
    for line in _question_lines(stdout):
        assert "  " in line.strip() or line.strip()[0].isdigit()


def test_worst_0_is_silent_besides_means(capsys):
    exit_code = main([str(EXAMPLES / "sample_eval.csv"), "--worst", "0"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    for col in _METRICS:
        assert f"{col}:" in stdout
    assert "worst" not in stdout
    assert _question_lines(stdout) == []


def test_worst_prints_even_on_fail_under(capsys):
    exit_code = main(
        [
            str(EXAMPLES / "sample_eval.csv"),
            "--worst",
            "1",
            "--fail-under",
            "faithfulness=1.1",
        ]
    )

    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert "FAIL faithfulness" in stdout
    assert len(_question_lines(stdout)) == len(_METRICS)
