FROM python:3.11-slim

# LightGBM needs libgomp at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
COPY pipelines ./pipelines
COPY scripts ./scripts

RUN pip install --no-cache-dir .

# Alembic config for the migration job. The trained model is not baked into the
# image; mount it (compose: ./models, k8s: the shared data PVC) and point
# RANKER_MODEL_PATH at it.
COPY alembic.ini ./
COPY alembic ./alembic

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
