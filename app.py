import streamlit as st
import pandas as pd
import plotly.express as px
from evaluators import USING_REAL_EVALUATOR, evaluate_dataframe

st.set_page_config(page_title="RAG-Dashboard", layout="wide")

st.title("RAG-Dashboard")
st.markdown("Панель для оценки и визуализации качества RAG-систем.")

st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV или JSON с результатами", type=["csv", "json"])

if uploaded_file is not None:
    try:
        # Invalidate any cached evaluation when a DIFFERENT file is uploaded,
        # otherwise the previous file's scores/charts keep rendering against
        # the new upload. file_id is a stable per-upload identifier provided
        # by Streamlit's file_uploader.
        upload_id = getattr(uploaded_file, 'file_id', None) or uploaded_file.name
        if st.session_state.get('uploaded_id') != upload_id:
            st.session_state['uploaded_id'] = upload_id
            st.session_state.pop('df_evaluated', None)

        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        st.subheader("Загруженные данные")
        st.dataframe(df.head())

        st.markdown("---")
        st.subheader("Оценка (Live Evaluation)")

        if st.button("Запустить оценку Ragas / Open RAG Eval"):
            with st.spinner("Идет оценка..."):
                try:
                    # Persist the evaluation result so it survives the
                    # script re-runs that Streamlit triggers on every widget
                    # interaction (e.g. changing the metric selectbox below).
                    st.session_state['df_evaluated'] = evaluate_dataframe(df)
                except Exception as e:
                    st.exception(e)

        # Render results/visualization from session_state, OUTSIDE the button
        # block, so they remain visible when other widgets re-run the script.
        df_evaluated = st.session_state.get('df_evaluated')
        if df_evaluated is not None:
            st.success("Оценка завершена!")
            if not USING_REAL_EVALUATOR:
                st.warning(
                    "Внимание: показаны ДЕМОНСТРАЦИОННЫЕ (mock) оценки. "
                    "Реальная интеграция Ragas / Open RAG Eval ещё не подключена."
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

    except Exception as e:
        st.exception(e)
else:
    st.info("Пожалуйста, загрузите файл с данными для начала работы.")
    st.markdown("""
    **Ожидаемый формат файла:**
    Таблица (CSV/JSON) с колонками, необходимыми для оценки (например, `question`, `answer`, `contexts`, `ground_truths`).
    """)
