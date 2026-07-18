# RAG-Dashboard

RAG-Dashboard — визуальная панель для оценки и сравнения RAG-пайплайнов: загрузка результатов
экспериментов, расчёт метрик и их визуализация (Plotly).

> **Статус:** Рабочий UI с эвристическим (offline) оценщиком — метрики считаются
> детерминированно по лексическому пересечению токенов, без внешних сервисов.
> Интеграция реальных оценщиков (Ragas / Open RAG Eval) — в планах.

## Особенности
- Загрузка CSV/JSON с результатами экспериментов.
- Расчёт и визуализация метрик (Plotly), сравнение запусков.
- Контейнеризация (Docker, образ под non-root пользователем).

## В планах
- Замена эвристического оценщика на реальную интеграцию Ragas / Open RAG Eval.

## Формат данных

Загружаемый файл (CSV или JSON) должен содержать колонки:

| Колонка    | Тип          | Описание                                   |
|------------|--------------|--------------------------------------------|
| `question` | строка       | Вопрос пользователя.                       |
| `answer`   | строка       | Ответ RAG-системы.                         |
| `contexts` | список строк | Извлечённые фрагменты контекста.           |

Колонка `contexts` нормализуется автоматически, поэтому один и тот же набор
данных даёт **идентичные** метрики независимо от формата:

- в JSON это настоящий список — `["...", "..."]`;
- в CSV — строка со списком-литералом — `"['...', '...']"`;
- пустая ячейка (`NaN`) трактуется как отсутствие контекста — `[]`.

Готовые примеры: [`examples/sample_eval.csv`](examples/sample_eval.csv) и
[`examples/sample_eval.json`](examples/sample_eval.json) — оба дают одинаковые
оценки.

### Пример запуска оценки

```python
import pandas as pd
from evaluators import evaluate_dataframe

df = pd.read_csv("examples/sample_eval.csv")   # или pd.read_json(".../sample_eval.json")
scored = evaluate_dataframe(df)                # backend="heuristic" по умолчанию
print(scored[["faithfulness", "answer_relevancy", "context_precision"]])
```

Если во входных данных есть колонка `ground_truths`, дополнительно считается
референс-метрика `answer_correctness` (token Jaccard между ответом и эталоном).

### Оценка из командной строки (offline, без UI)

Для CI / регрессионных прогонов датасет можно оценить без запуска Streamlit —
`evaluate_cli.py` печатает среднее по каждой метрике и, при `--out`, сохраняет
оценённую таблицу в CSV:

```bash
python evaluate_cli.py examples/sample_eval.csv --out scored.csv
# опционально: --backend heuristic (по умолчанию) | mock
```

## Запуск локально
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Запуск в Docker
```bash
docker build -t rag-dashboard .
docker run -p 8501:8501 rag-dashboard
```
