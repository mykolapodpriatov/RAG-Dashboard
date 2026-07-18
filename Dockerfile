FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user (least privilege) instead of root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Readiness probe via Streamlit's health endpoint, using only the stdlib
# (no curl / extra packages). A non-200 status or any exception exits non-zero,
# which Docker records as an unhealthy container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"]

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
