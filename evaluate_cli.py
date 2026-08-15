"""Offline batch-evaluation CLI for the RAG dashboard.

Scores a CSV/JSON dataset with the pure, dependency-light evaluator without
launching the Streamlit UI — a headless path suited to CI / regression checks.

Usage::

    python evaluate_cli.py INPUT[.csv|.json] [--backend heuristic] [--out scored.csv]
    python evaluate_cli.py BASE.csv --compare CANDIDATE.csv [--out comparison.csv]

The mean of every emitted metric is printed to stdout; the full scored table is
written to ``--out`` (CSV) when that flag is supplied. With ``--compare``, stdout
is a per-metric Run A / Run B / delta table and ``--out`` is the long-form
comparison (``run`` / ``metric`` / ``score``).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from evaluators import combine_runs, evaluate_dataframe, run_metric_means

# Columns whose per-row means we report. ``answer_correctness`` is optional and
# only present when the dataset carried a ``ground_truths`` column.
_METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "answer_correctness",
)

_LABEL_A = "Run A"
_LABEL_B = "Run B"
_MISSING = "-"


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load *path* into a DataFrame, dispatching on the file extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(
        f"Unsupported input extension {suffix!r}; expected .csv or .json."
    )


def _metric_mean(means: pd.DataFrame, run: str, metric: str) -> float | None:
    """Return the mean *metric* for *run*, or ``None`` if that pair is absent."""
    rows = means.loc[(means["run"] == run) & (means["metric"] == metric), "score"]
    if rows.empty:
        return None
    return float(rows.iloc[0])


def _fmt_score(value: float | None) -> str:
    return _MISSING if value is None else f"{value:.4f}"


def _print_comparison(means: pd.DataFrame) -> None:
    """Print per-metric Run A / Run B means and the B − A delta."""
    metrics: list[str] = []
    seen = set(means["metric"])
    for col in _METRIC_COLUMNS:
        if col in seen:
            metrics.append(col)
    for metric in means["metric"]:
        if metric not in metrics:
            metrics.append(metric)

    header = f"{'metric':<22} {_LABEL_A:>10} {_LABEL_B:>10} {'delta':>10}"
    print(header)
    for metric in metrics:
        mean_a = _metric_mean(means, _LABEL_A, metric)
        mean_b = _metric_mean(means, _LABEL_B, metric)
        if mean_a is None or mean_b is None:
            delta = _MISSING
        else:
            delta = f"{mean_b - mean_a:+.4f}"
        print(
            f"{metric:<22} {_fmt_score(mean_a):>10} {_fmt_score(mean_b):>10} "
            f"{delta:>10}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_cli.py",
        description="Offline batch evaluation of a RAG dataset (CSV or JSON).",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input dataset (.csv or .json).",
    )
    parser.add_argument(
        "--backend",
        default="heuristic",
        choices=("heuristic", "mock"),
        help="Evaluator backend (default: heuristic).",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional second dataset to compare against the positional input.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional path to write CSV. Single-run mode writes the scored "
            "table; --compare writes the long-form run/metric/score table."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batch evaluation. Returns a process exit code (``0`` on success)."""
    args = _build_parser().parse_args(argv)

    df = _load_dataframe(args.input)
    scored = evaluate_dataframe(df, backend=args.backend)

    if args.compare is not None:
        scored_b = evaluate_dataframe(
            _load_dataframe(args.compare), backend=args.backend
        )
        comparison = combine_runs(
            scored, scored_b, label_a=_LABEL_A, label_b=_LABEL_B
        )
        _print_comparison(run_metric_means(comparison))
        if args.out is not None:
            comparison.to_csv(args.out, index=False)
            print(f"Wrote comparison table to {args.out}")
        return 0

    metric_columns = [col for col in _METRIC_COLUMNS if col in scored.columns]
    means = scored[metric_columns].mean()
    for col in metric_columns:
        print(f"{col}: {means[col]:.4f}")

    if args.out is not None:
        scored.to_csv(args.out, index=False)
        print(f"Wrote scored table to {args.out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
