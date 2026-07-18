"""Offline batch-evaluation CLI for the RAG dashboard.

Scores a CSV/JSON dataset with the pure, dependency-light evaluator without
launching the Streamlit UI — a headless path suited to CI / regression checks.

Usage::

    python evaluate_cli.py INPUT[.csv|.json] [--backend heuristic] [--out scored.csv]

The mean of every emitted metric is printed to stdout; the full scored table is
written to ``--out`` (CSV) when that flag is supplied.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from evaluators import evaluate_dataframe

# Columns whose per-row means we report. ``answer_correctness`` is optional and
# only present when the dataset carried a ``ground_truths`` column.
_METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "answer_correctness",
)


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
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the scored table as CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batch evaluation. Returns a process exit code (``0`` on success)."""
    args = _build_parser().parse_args(argv)

    df = _load_dataframe(args.input)
    scored = evaluate_dataframe(df, backend=args.backend)

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
