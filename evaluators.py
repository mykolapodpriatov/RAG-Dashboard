import random

import pandas as pd

# True once a real Ragas / Open RAG Eval backend is wired in. While False,
# evaluate_dataframe() returns reproducible MOCK scores (the UI warns the user).
USING_REAL_EVALUATOR = False

# Fixed seed so the mock scores are reproducible across runs/demos instead of
# changing on every click.
_MOCK_SEED = 42


def evaluate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обертка для оценки DataFrame с колонками: question, answer, contexts.

    В реальном приложении здесь будет вызов Ragas или Open RAG Eval.
    Пока используем заглушку, генерирующую ВОСПРОИЗВОДИМЫЕ случайные оценки
    (с фиксированным seed), если соответствующие колонки ещё не заданы.
    """
    required_cols = ['question', 'answer', 'contexts']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Отсутствует обязательная колонка: {col}. Доступные колонки: {df.columns.tolist()}")

    df = df.copy()
    rng = random.Random(_MOCK_SEED)
    if 'faithfulness' not in df.columns:
        df['faithfulness'] = [rng.uniform(0.5, 1.0) for _ in range(len(df))]
    if 'answer_relevancy' not in df.columns:
        df['answer_relevancy'] = [rng.uniform(0.5, 1.0) for _ in range(len(df))]
    if 'context_precision' not in df.columns:
        df['context_precision'] = [rng.uniform(0.5, 1.0) for _ in range(len(df))]

    return df
