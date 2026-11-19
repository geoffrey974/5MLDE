FROM python:3.13.2-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .
COPY configs configs
COPY src src

RUN useradd --create-home --uid 1000 app \
    && mkdir -p data logs /home/app/.prefect \
    && chown -R app:app /app /home/app/.prefect
USER app

EXPOSE 8000
CMD ["uvicorn", "bike_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
