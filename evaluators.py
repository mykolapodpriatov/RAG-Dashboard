"""Evaluator backends for the RAG dashboard.

Backends are **registered**, not branched on, so a team with its own evaluator
(or one wanting Ragas) can plug it in without editing this module. Two ship
here and register themselves at import:

* ``"heuristic"`` (default) — deterministic lexical proxy metrics computed with
  the standard library only. Not a substitute for Ragas / Open RAG Eval, but a
  reproducible, offline signal that actually reacts to the text.
* ``"mock"`` — the legacy reproducible-random placeholder, kept for demos.

Two rules the registry exists to enforce:

* **An unknown backend name raises**, naming what is registered. A dashboard
  quietly showing proxy numbers while the user believes they are looking at
  Ragas is the worst outcome available here.
* **Every backend returns the same frame contract**: the same metric columns,
  one row per input row. That is checked in the dispatch rather than trusted to
  each backend, because the comparison view groups two runs by metric and would
  otherwise silently compare different things.

Whether the numbers are proxies is a property of the backend that ran, not of
the process, so it travels on the registration rather than a module constant.
"""

from __future__ import annotations

import ast
import random
import re
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

#: Signature every backend implements: rows in (with ``contexts`` already
#: normalised), the same rows out plus the metric columns.
EvaluatorFn = Callable[[pd.DataFrame], pd.DataFrame]

# Fixed seed so the legacy mock scores stay reproducible across runs/demos.
_MOCK_SEED = 42

_REQUIRED_COLUMNS = ("question", "answer", "contexts")
_METRIC_COLUMNS = ("faithfulness", "answer_relevancy", "context_precision")


@dataclass(frozen=True)
class Evaluator:
    """A registered backend.

    Attributes:
        name: The name callers pass as ``backend=``.
        fn: Takes the prepared frame, returns it with the metric columns added.
        is_real: Whether these are real evaluator scores rather than a proxy.
            The UI warning follows this, so it must describe the backend that
            actually ran.
        description: One line, shown by :func:`available_backends`.
    """

    name: str
    fn: EvaluatorFn
    is_real: bool
    description: str = ""


_REGISTRY: dict[str, Evaluator] = {}


def register_evaluator(
    name: str,
    fn: EvaluatorFn,
    *,
    is_real: bool,
    description: str = "",
    replace: bool = False,
) -> None:
    """Register a backend under ``name``.

    Args:
        name: The name callers pass as ``backend=``.
        fn: The evaluator.
        is_real: Whether it produces real evaluator scores rather than proxies.
        description: One line for listings.
        replace: Allow overwriting an existing registration. Off by default so
            two packages claiming one name is an error rather than whichever
            imported last silently winning.

    Raises:
        ValueError: On a blank name, or a duplicate without ``replace``.
    """
    if not name.strip():
        raise ValueError("evaluator name must not be empty")
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"evaluator {name!r} is already registered; pass replace=True to override it"
        )
    _REGISTRY[name] = Evaluator(name=name, fn=fn, is_real=is_real, description=description)


def get_evaluator(name: str) -> Evaluator:
    """Look a backend up by name.

    Raises:
        ValueError: If nothing is registered under ``name``. Never falls back to
            the heuristic: showing proxy numbers while the caller believes they
            asked for something else is worse than failing.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(
            f"Неизвестный backend: {name!r}. Зарегистрированные: {known}."
        ) from None


def available_backends() -> dict[str, Evaluator]:
    """Every registered backend, keyed by name."""
    return dict(_REGISTRY)

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


def _heuristic_backend(df: pd.DataFrame) -> pd.DataFrame:
    """The default backend: deterministic lexical proxy metrics."""
    scores = [
        heuristic_evaluate(row["question"], row["answer"], row["contexts"])
        for _, row in df.iterrows()
    ]
    scores_df = pd.DataFrame(scores, index=df.index, columns=list(_METRIC_COLUMNS))
    for col in _METRIC_COLUMNS:
        df[col] = scores_df[col]
    return df


register_evaluator(
    "heuristic",
    _heuristic_backend,
    is_real=False,
    description="Deterministic lexical proxy, offline, no dependencies.",
)
register_evaluator(
    "mock",
    _mock_scores,
    is_real=False,
    description="Reproducible-random placeholder, for demos.",
)


def _check_contract(result: pd.DataFrame, source: pd.DataFrame, backend: str) -> pd.DataFrame:
    """Enforce the frame contract every backend owes its caller.

    Checked here rather than trusted to each backend: the comparison view
    groups two runs by metric, so a backend that omits a column or drops rows
    would make it silently compare different things.

    Raises:
        ValueError: On a missing metric column or a changed row count.
    """
    if not isinstance(result, pd.DataFrame):
        raise ValueError(f"backend {backend!r} returned {type(result).__name__}, not a DataFrame")
    if len(result) != len(source):
        raise ValueError(
            f"backend {backend!r} returned {len(result)} row(s) for {len(source)} input row(s)"
        )
    missing = [col for col in _METRIC_COLUMNS if col not in result.columns]
    if missing:
        raise ValueError(f"backend {backend!r} did not produce: {', '.join(missing)}")
    return result


def evaluate_dataframe(df: pd.DataFrame, backend: str = "heuristic") -> pd.DataFrame:
    """Evaluate a DataFrame with ``question``/``answer``/``contexts`` columns.

    Args:
        df: Input rows. A missing required column raises ``ValueError``.
        backend: A registered backend name; see :func:`available_backends`.
            ``"heuristic"`` (offline lexical proxy) is the default.

    Returns:
        A copy of *df* with ``faithfulness``, ``answer_relevancy`` and
        ``context_precision`` columns appended. When the input carries a
        ``ground_truths`` column, a deterministic reference-based
        ``answer_correctness`` column is appended as well; otherwise the
        three-metric output is unchanged.

    Raises:
        ValueError: On a missing required column, an unknown backend, or a
            backend that broke the frame contract.
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

    evaluator = get_evaluator(backend)
    df = _check_contract(evaluator.fn(df), df, backend)

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
