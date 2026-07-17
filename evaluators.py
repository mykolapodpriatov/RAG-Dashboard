"""Offline, dependency-free evaluators for the RAG dashboard.

Two backends are available via ``evaluate_dataframe(df, backend=...)``:

* ``"heuristic"`` (default) — deterministic lexical proxy metrics computed with
  the standard library only. Not a substitute for Ragas / Open RAG Eval, but a
  reproducible, offline signal that actually reacts to the text.
* ``"mock"`` — the legacy reproducible-random placeholder, kept for demos.
"""

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

# Unicode-aware word tokenizer: runs of word characters, case-folded.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: object) -> set[str]:
    """Return the set of case-folded word tokens in *text* (``set()`` if empty)."""
    if text is None:
        return set()
    return set(_TOKEN_RE.findall(str(text).casefold()))


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
        ``context_precision`` columns appended.
    """
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Отсутствует обязательная колонка: {col}. "
                f"Доступные колонки: {df.columns.tolist()}"
            )

    df = df.copy()

    if backend == "mock":
        return _mock_scores(df)
    if backend != "heuristic":
        raise ValueError(
            f"Неизвестный backend: {backend!r}. Ожидается 'heuristic' или 'mock'."
        )

    scores = [
        heuristic_evaluate(row["question"], row["answer"], row["contexts"])
        for _, row in df.iterrows()
    ]
    scores_df = pd.DataFrame(scores, index=df.index, columns=list(_METRIC_COLUMNS))
    for col in _METRIC_COLUMNS:
        df[col] = scores_df[col]
    return df
