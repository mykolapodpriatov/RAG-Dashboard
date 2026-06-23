# RAG-Dashboard

RAG-Dashboard – визуальная панель для оценки и сравнения RAG-потоков. 
Проект объединяет возможности существующих инструментов оценки (Open RAG Eval, Ragas) в удобный интерактивный веб-интерфейс.

## Особенности
- Загрузка CSV/JSON файлов с результатами экспериментов.
- Нативная интеграция метрик оценки качества генерации и поиска.
- Визуализация метрик (Plotly) и сравнение запусков.

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
