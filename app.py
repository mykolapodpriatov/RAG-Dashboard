import streamlit as st
import pandas as pd
import plotly.express as px
from evaluators import USING_REAL_EVALUATOR, combine_runs, evaluate_dataframe, run_metric_means

st.set_page_config(page_title="RAG-Dashboard", layout="wide")

st.title("RAG-Dashboard")
st.markdown("Панель для оценки и визуализации качества RAG-систем.")


def _read_uploaded(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded CSV/JSON file (Streamlit's ``UploadedFile``) into a DataFrame."""
    if uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_json(uploaded_file)


def _sync_upload_state(uploaded_file, id_key: str, df_key: str) -> None:
    """Invalidate the cached evaluation when a DIFFERENT file is uploaded under
    this slot, otherwise the previous file's scores/charts keep rendering
    against the new upload. file_id is a stable per-upload identifier provided
    by Streamlit's file_uploader.
    """
    upload_id = getattr(uploaded_file, 'file_id', None) or uploaded_file.name
    if st.session_state.get(id_key) != upload_id:
        st.session_state[id_key] = upload_id
        st.session_state.pop(df_key, None)


st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV или JSON с результатами", type=["csv", "json"])

st.sidebar.markdown("---")
compare_file = st.sidebar.file_uploader(
    "Сравнить с другим запуском (опционально)",
    type=["csv", "json"],
    key="compare_uploader",
)

if uploaded_file is not None:
    try:
        _sync_upload_state(uploaded_file, 'uploaded_id', 'df_evaluated')
        df = _read_uploaded(uploaded_file)

        st.subheader("Загруженные данные")
        st.dataframe(df.head())

        # The second run is optional and parsed independently, so a bad/absent
        # comparison file never breaks the primary single-run flow above.
        compare_df = None
        if compare_file is not None:
            try:
                _sync_upload_state(compare_file, 'compare_uploaded_id', 'compare_df_evaluated')
                compare_df = _read_uploaded(compare_file)
                st.subheader(f"Данные для сравнения: {compare_file.name}")
                st.dataframe(compare_df.head())
            except Exception as e:
                st.exception(e)
                compare_df = None

        st.markdown("---")
        st.subheader("Оценка (Live Evaluation)")

        if st.button("Запустить оценку Ragas / Open RAG Eval"):
            with st.spinner("Идет оценка..."):
                try:
                    # Persist the evaluation result so it survives the
                    # script re-runs that Streamlit triggers on every widget
                    # interaction (e.g. changing the metric selectbox below).
                    st.session_state['df_evaluated'] = evaluate_dataframe(df)
                    if compare_df is not None:
                        st.session_state['compare_df_evaluated'] = evaluate_dataframe(compare_df)
                except Exception as e:
                    st.exception(e)

        # Render results/visualization from session_state, OUTSIDE the button
        # block, so they remain visible when other widgets re-run the script.
        df_evaluated = st.session_state.get('df_evaluated')
        compare_df_evaluated = (
            st.session_state.get('compare_df_evaluated') if compare_df is not None else None
        )

        if df_evaluated is not None:
            st.success("Оценка завершена!")
            if not USING_REAL_EVALUATOR:
                st.warning(
                    "Внимание: показаны эвристические (offline) оценки на основе "
                    "лексического пересечения токенов — быстрый детерминированный "
                    "прокси, а не результат Ragas / Open RAG Eval. Реальная "
                    "интеграция оценщиков ещё не подключена."
                )
            st.dataframe(df_evaluated.head())

            st.markdown("---")
            st.subheader("Визуализация метрик")

            metrics = ['faithfulness', 'answer_relevancy', 'context_precision']
            available_metrics = [m for m in metrics if m in df_evaluated.columns]

            if available_metrics:
                # Boxplot for metrics distribution
                df_melted = df_evaluated.melt(value_vars=available_metrics, var_name='Metric', value_name='Score')
                fig = px.box(df_melted, x='Metric', y='Score', title="Распределение метрик")
                st.plotly_chart(fig, use_container_width=True)

                # Histogram for specific metric
                selected_metric = st.selectbox("Выберите метрику для детализации", available_metrics)
                fig2 = px.histogram(df_evaluated, x=selected_metric, nbins=20, title=f"Гистограмма: {selected_metric}")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Метрики для визуализации не найдены.")

            if compare_df_evaluated is not None:
                st.markdown("---")
                st.subheader("Сравнение запусков")

                label_a = f"Run A ({uploaded_file.name})"
                label_b = f"Run B ({compare_file.name})"
                comparison = combine_runs(df_evaluated, compare_df_evaluated, label_a, label_b)

                # Grouped bar chart: mean per metric, side by side per run.
                means = run_metric_means(comparison)
                fig3 = px.bar(
                    means,
                    x='metric',
                    y='score',
                    color='run',
                    barmode='group',
                    title="Среднее по метрикам: сравнение запусков",
                )
                st.plotly_chart(fig3, use_container_width=True)

                # Box plot of the full distribution, split/colored by run.
                fig4 = px.box(
                    comparison,
                    x='metric',
                    y='score',
                    color='run',
                    title="Распределение метрик по запускам",
                )
                st.plotly_chart(fig4, use_container_width=True)

                # Mismatched columns between the two runs (e.g. ground_truths
                # supplied in only one upload) are common and expected — flag
                # it instead of silently dropping or erroring.
                metrics_by_run = comparison.groupby('run')['metric'].apply(set)
                if len(metrics_by_run) == 2 and metrics_by_run.iloc[0] != metrics_by_run.iloc[1]:
                    only_a = metrics_by_run.iloc[0] - metrics_by_run.iloc[1]
                    only_b = metrics_by_run.iloc[1] - metrics_by_run.iloc[0]
                    notes = []
                    if only_a:
                        notes.append(f"только в «{metrics_by_run.index[0]}»: {', '.join(sorted(only_a))}")
                    if only_b:
                        notes.append(f"только в «{metrics_by_run.index[1]}»: {', '.join(sorted(only_b))}")
                    st.info(
                        "Наборы метрик у запусков не совпадают полностью — "
                        + "; ".join(notes)
                        + ". Это ожидаемо, если, например, колонка `ground_truths` "
                        "присутствует не во всех загруженных данных."
                    )

    except Exception as e:
        st.exception(e)
else:
    st.info("Пожалуйста, загрузите файл с данными для начала работы.")
    st.markdown("""
    **Ожидаемый формат файла:**
    Таблица (CSV/JSON) с колонками, необходимыми для оценки (например, `question`, `answer`, `contexts`, `ground_truths`).

    Дополнительно, во второй загрузчик в сайдбаре можно загрузить второй запуск
    (тот же формат) — после оценки появится раздел сравнения запусков.
    """)
