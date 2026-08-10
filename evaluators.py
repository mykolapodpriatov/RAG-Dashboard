"""Offline, dependency-free evaluators for the RAG dashboard.

Two backends are available via ``evaluate_dataframe(df, backend=...)``:

* ``"heuristic"`` (default) — deterministic lexical proxy metrics computed with
  the standard library only. Not a substitute for Ragas / Open RAG Eval, but a
  reproducible, offline signal that actually reacts to the text.
* ``"mock"`` — the legacy reproducible-random placeholder, kept for demos.
"""

import ast
import random
import re

import pandas as pd

# True once a real Ragas / Open RAG Eval backend is wired in. While False, the
# UI warns that the displayed metrics are proxy/heuristic scores.
USING_REAL_EVALUATOR = False

# Fixed seed so the legacy mock scores stay reproducible across runs/demos.
_MOCK_SEED = 42

_REQUIRED_COLUMNS = ("question", "answer", "contexts")
_METRIC_COLUMNS = ("faithfulness", "answer_relevancy", "context_precision")

# Optional reference-based metric: computed per row only when the input carries a
# ``ground_truths`` column (see :func:`evaluate_dataframe`).
_GROUND_TRUTHS_COLUMN = "ground_truths"
_REFERENCE_METRIC_COLUMN = "answer_correctness"

# Unicode-aware word tokenizer: runs of word characters, case-folded.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: object) -> set[str]:
    """Return the set of case-folded word tokens in *text* (``set()`` if empty)."""
    if text is None:
        return set()
    return set(_TOKEN_RE.findall(str(text).casefold()))


def _normalize_contexts(value: object) -> list[str]:
    """Coerce a ``contexts`` cell into a ``list[str]`` regardless of its source.

    Uploads deliver the same logical data as different Python types, so we
    normalise them all to a list of strings:

    * ``list`` / ``tuple`` (typical ``pd.read_json`` result) — element-wise
      ``str`` coercion.
    * List-literal string (``pd.read_csv`` round-trip, e.g. ``"['a', 'b']"``) —
      parsed safely with :func:`ast.literal_eval`. A parse failure, or a literal
      that is not a list/tuple, falls back to treating the whole string as a
      single context.
    * ``NaN`` / ``None`` (blank cell) or a blank string — ``[]``.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return []
    # Scalar NaN (a blank CSV cell). ``pd.isna`` on non-scalars can raise or
    # return an array, hence the guard — lists were already handled above.
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return [text]


def heuristic_evaluate(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """Compute deterministic lexical proxy metrics, each bounded to ``[0, 1]``.

    * ``faithfulness`` — share of answer tokens that also occur in the contexts.
    * ``answer_relevancy`` — Jaccard overlap between question and answer tokens.
    * ``context_precision`` — share of contexts that share at least one token
      with the answer.

    Every division is guarded, so empty inputs yield ``0.0`` instead of raising
    ``ZeroDivisionError``.
    """
    answer_tokens = _tokenize(answer)
    question_tokens = _tokenize(question)
    per_context_tokens = [_tokenize(ctx) for ctx in contexts]
    context_tokens: set[str] = (
        set().union(*per_context_tokens) if per_context_tokens else set()
    )

    faithfulness = (
        len(answer_tokens & context_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )

    union = question_tokens | answer_tokens
    answer_relevancy = (
        len(question_tokens & answer_tokens) / len(union) if union else 0.0
    )

    context_precision = (
        sum(1 for tokens in per_context_tokens if tokens & answer_tokens)
        / len(per_context_tokens)
        if per_context_tokens
        else 0.0
    )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
    }


def answer_correctness(answer: object, ground_truths: object) -> float:
    """Token Jaccard overlap between *answer* and its reference *ground_truths*.

    A deterministic, offline, reference-based proxy for answer correctness. The
    ``ground_truths`` cell is coerced with :func:`_normalize_contexts` (so a
    native list, a list-literal string, a bare string or ``NaN`` are all
    accepted), its tokens are pooled, and the result is the Jaccard similarity
    of the answer tokens and the pooled reference tokens.

    The score is inherently bounded to ``[0, 1]``. Empty inputs — no answer
    tokens, no reference tokens, or both — yield ``0.0`` instead of raising
    ``ZeroDivisionError``.
    """
    answer_tokens = _tokenize(answer)
    reference_tokens: set[str] = set()
    for reference in _normalize_contexts(ground_truths):
        reference_tokens |= _tokenize(reference)

    union = answer_tokens | reference_tokens
    if not union:
        return 0.0
    return len(answer_tokens & reference_tokens) / len(union)


def _mock_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy reproducible-random placeholder, gated behind ``backend='mock'``."""
    rng = random.Random(_MOCK_SEED)
    for col in _METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = [rng.uniform(0.5, 1.0) for _ in range(len(df))]
    return df


def evaluate_dataframe(df: pd.DataFrame, backend: str = "heuristic") -> pd.DataFrame:
    """Evaluate a DataFrame with ``question``/``answer``/``contexts`` columns.

    Args:
        df: Input rows. A missing required column raises ``ValueError``.
        backend: ``"heuristic"`` (default, offline lexical proxy) or ``"mock"``
            (legacy reproducible-random placeholder).

    Returns:
        A copy of *df* with ``faithfulness``, ``answer_relevancy`` and
        ``context_precision`` columns appended. When the input carries a
        ``ground_truths`` column, a deterministic reference-based
        ``answer_correctness`` column is appended as well; otherwise the
        three-metric output is unchanged.
    """
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Отсутствует обязательная колонка: {col}. "
                f"Доступные колонки: {df.columns.tolist()}"
            )

    df = df.copy()
    # Normalise `contexts` up front so CSV (list-literal strings / NaN) and JSON
    # (native lists) uploads of the same data score identically downstream.
    df["contexts"] = df["contexts"].apply(_normalize_contexts)

    if backend == "mock":
        df = _mock_scores(df)
    elif backend == "heuristic":
        scores = [
            heuristic_evaluate(row["question"], row["answer"], row["contexts"])
            for _, row in df.iterrows()
        ]
        scores_df = pd.DataFrame(scores, index=df.index, columns=list(_METRIC_COLUMNS))
        for col in _METRIC_COLUMNS:
            df[col] = scores_df[col]
    else:
        raise ValueError(
            f"Неизвестный backend: {backend!r}. Ожидается 'heuristic' или 'mock'."
        )

    # Reference-based metric is deterministic and backend-independent; only
    # emitted when ground-truth references are supplied.
    if _GROUND_TRUTHS_COLUMN in df.columns:
        df[_REFERENCE_METRIC_COLUMN] = [
            answer_correctness(row["answer"], row[_GROUND_TRUTHS_COLUMN])
            for _, row in df.iterrows()
        ]
    return df


# All metric columns that may appear on an evaluated dataframe, in the order
# they should be compared. ``answer_correctness`` is optional (see above).
_COMPARISON_METRIC_COLUMNS = (*_METRIC_COLUMNS, _REFERENCE_METRIC_COLUMN)


def combine_runs(
    run_a: pd.DataFrame,
    run_b: pd.DataFrame,
    label_a: str = "Run A",
    label_b: str = "Run B",
) -> pd.DataFrame:
    """Combine two already-evaluated runs into one long-form table for comparison charts.

    Each input is expected to already carry the metric columns produced by
    :func:`evaluate_dataframe` (``faithfulness``, ``answer_relevancy``,
    ``context_precision`` and, optionally, ``answer_correctness``). The two
    runs are melted independently, so they need neither matching columns nor
    matching row counts:

    * A metric present in only one run (e.g. ``answer_correctness`` when only
      one upload supplied ``ground_truths``) simply contributes rows for that
      run alone — it is never dropped or padded with placeholder values.
    * Differing row counts never cause a shape mismatch, since each run
      contributes its own independent set of rows.

    Returns:
        A dataframe with columns ``run``, ``metric``, ``score`` — one row per
        (row, metric) pair, labelled by *label_a* / *label_b* — ready for
        ``plotly.express`` grouped bar/box charts via ``color="run"``.
    """
    frames = []
    for label, df in ((label_a, run_a), (label_b, run_b)):
        metric_cols = [col for col in _COMPARISON_METRIC_COLUMNS if col in df.columns]
        melted = df.melt(value_vars=metric_cols, var_name="metric", value_name="score")
        melted.insert(0, "run", label)
        frames.append(melted)
    return pd.concat(frames, ignore_index=True)


def run_metric_means(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a :func:`combine_runs` table into per-run, per-metric means.

    Args:
        comparison: The long-form ``run``/``metric``/``score`` output of
            :func:`combine_runs`.

    Returns:
        A dataframe with columns ``run``, ``metric``, ``score`` — one row per
        (run, metric) pair holding the mean score — suitable for a grouped bar
        chart of run averages.
    """
    return comparison.groupby(["run", "metric"], as_index=False)["score"].mean()
