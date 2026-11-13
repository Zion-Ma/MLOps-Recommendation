# News Recommendation MLOps Platform

A production-style news recommendation system covering the full ML lifecycle: daily
ingestion, hybrid retrieval, learning-to-rank serving, feedback logging, scheduled
retraining, model registry, Kubernetes deployment, and monitoring with drift detection.

## Highlights

| | |
| --- | --- |
| Ranking quality | **0.750 NDCG@10**, 0.749 MRR, 0.973 Recall@50 on a held-out split (2.6x a random baseline) |
| Training data | 11.2M MIND events, 50K users, 51.5K articles; ranker trained on 50K impressions |
| Serving latency | cold path p50 **206 ms** / p95 **246 ms**; Redis-cached p50 3 ms |
| Orchestration | 3 Airflow DAGs, every step an isolated `KubernetesPodOperator` pod |
| Quality gates | 122 tests (115 app + 7 DAG), `mypy --strict`, ruff, 41 Kubernetes manifest checks, CI-validated DAGs |

Full methodology, baselines and reproduction commands: [docs/RESULTS.md](docs/RESULTS.md).
A catch worth noting: the first training run scored 0.962 NDCG@10 because of label
leakage in the popularity feature; it was found by auditing feature distributions and
fixed before reporting (see [TROUBLESHOOTING #15](docs/TROUBLESHOOTING.md)).

## Data sources

- **MIND** (Microsoft News dataset): historical impressions and clicks for offline
  training and evaluation.
- **Hacker News Algolia API**: continuously changing content for daily ingestion, so
  retrieval indexes and retraining see fresh data.

## Architecture

```text
Daily Article Ingestion (Hacker News API)
        ↓
PostgreSQL Article Store
        ↓
OpenSearch BM25 Overretrieval (high-recall candidate pool)
        ↓
SentenceTransformers Dense Reranking (cosine similarity)
        ↓
LightGBM Learning-to-Rank (NDCG-optimized)
        ↓
FastAPI Recommendation Service  ←→  Redis response cache
        ↓
User Feedback Logging (impression / click / dwell / skip)
        ↓
Airflow Retraining DAG (KubernetesPodOperator)
        ↓
MLflow Model Registry (evaluation gate → production alias)
        ↓
Prometheus + Grafana + Evidently Monitoring
```

**Stack:** FastAPI · SQLAlchemy 2 · PostgreSQL · Redis · OpenSearch · Qdrant ·
SentenceTransformers · LightGBM · MLflow · Apache Airflow · Docker · Kubernetes ·
Prometheus · Grafana · Evidently AI

## Repository layout

```text
app/
  api/            recommendation + feedback event endpoints, Redis cache
  retrieval/      BM25 (OpenSearch), dense (Qdrant), dense reranking
  ranking/        LightGBM ranker training/eval/serialization
  features/       ranking feature engineering
  monitoring/     Prometheus metrics + middleware
  db/             SQLAlchemy models, repositories, sessions
pipelines/        CLI pipelines (ingestion, indexing, training, evaluation,
                  promotion, CTR/latency/drift monitoring, alerting)
airflow/dags/     daily_ingestion, weekly_retraining, monitoring DAGs
k8s/              Kubernetes manifests (API + full infra + RBAC + jobs)
monitoring/       Prometheus scrape config, Grafana provisioning + dashboard
alembic/          database migrations
tests/            unit, API, pipeline, DAG, and manifest tests
```

## Quickstart (local)

Prerequisites: Python 3.11, Docker (with ~4 GB free memory for OpenSearch), and
optionally `kubectl` + a cluster for the Kubernetes parts.

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Start infrastructure (Postgres, Redis, OpenSearch, Qdrant, MLflow,
#    Prometheus, Grafana, and the API)
docker compose up -d

# 3. Migrate the database
alembic upgrade head

# 4. Ingest some articles and build the indexes
python -m pipelines.ingest_hn --days-back 2 --limit 200
python -m pipelines.index_bm25 --limit 1000
python -m pipelines.index_dense --limit 1000

# 5. Load MIND training data (downloads, ingests, and builds features)
python scripts/setup_mind_data.py --size small --split train

# 6. Train and register the ranker
python -m pipelines.train_ranker \
  --input-path data/ranking_features.parquet \
  --output-path models/ranker.txt \
  --mlflow-tracking-uri http://localhost:5001 \
  --register-model

# 7. Serve recommendations
uvicorn app.main:app --reload
```

Local UIs: API docs http://localhost:8000/docs · MLflow http://localhost:5001 ·
Prometheus http://localhost:9090 · Grafana http://localhost:3000 (admin/admin) ·
Airflow http://localhost:8080 (credentials printed in the `airflow` container logs).

The compose `airflow` service runs `airflow standalone` for the UI and DAG parsing.
DAG tasks are `KubernetesPodOperator`s, so actually executing them needs a reachable
cluster.

### Example requests

```bash
# Ranked recommendations (user UUID or external id)
curl "http://localhost:8000/recommendations/U1234?top_k=5"
```

```json
{
  "user_id": "0b1c…",
  "model_version": "news-lightgbm-ranker/production",
  "strategy": "ranked",
  "items": [
    {"article_id": "7f3a…", "title": "…", "url": "https://…", "source": "hn",
     "category": "tech", "score": 0.83, "rank": 1}
  ]
}
```

```bash
# Log feedback (user is created if unknown)
curl -X POST http://localhost:8000/events/click \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "U1234", "article_id": "7f3a…", "position": 0,
       "model_version": "news-lightgbm-ranker/production"}'
```

## API

| Endpoint | Description |
| --- | --- |
| `GET /recommendations/{user_id}` | Ranked recommendations (`?top_k=`, `?candidate_pool_size=`). Accepts a user UUID or external id. Falls back to recency ranking for cold-start users. |
| `POST /events/impression` `/click` `/dwell` `/skip` | Feedback logging with position, model version, and free-form metadata. |
| `GET /health` | Liveness/readiness probe. |
| `GET /metrics` | Prometheus metrics. |

Serving pipeline per request: user click history → query construction → BM25
overretrieval (pool of 200–1000) → dense cosine reranking → per-candidate feature
generation (bm25/dense/rerank scores, article age, popularity, category affinity) →
LightGBM scoring → top-k, persisted for offline analysis and cached in Redis.

## Pipelines

| Pipeline | Purpose |
| --- | --- |
| `ingest_hn` | Fetch, normalize, and deduplicate Hacker News stories |
| `ingest_mind` | Load MIND articles, users, impressions, and clicks |
| `index_bm25` / `index_dense` | Refresh OpenSearch and Qdrant indexes |
| `build_ranking_dataset` | Build labeled ranking feature rows (Parquet) |
| `train_ranker` | Train LightGBM lambdarank, log params/metrics/artifacts to MLflow |
| `evaluate_model` | NDCG@10 / MRR / Recall@50 with a promotion gate (`--min-ndcg`) |
| `promote_model` | Set the MLflow `production` alias, record in `model_versions` |
| `compute_ctr` / `compute_latency` | CTR from events; p95 latency + error rate from Prometheus |
| `generate_drift_report` | Evidently HTML report + PSI-based drift summary |
| `alert_if_degraded` | Threshold checks; non-zero exit fails the Airflow task |

## Orchestration (Airflow)

Three DAGs in `airflow/dags/`, all running steps as `KubernetesPodOperator` pods
built from the project image:

- **daily_ingestion** — ingest HN → update OpenSearch → update Qdrant
- **weekly_retraining** — build dataset → train + register → evaluate (NDCG gate) → promote
- **monitoring** — CTR + latency + drift report → alert_if_degraded

Airflow lives in its own environment (it pins SQLAlchemy 1.4; the app uses 2.x):

```bash
python -m venv .venv-airflow
.venv-airflow/bin/pip install -r airflow/requirements.txt
.venv-airflow/bin/pytest tests/test_dags.py
```

## Deployment (Kubernetes)

Manifests in `k8s/` cover the API (Deployment + Service + HPA + rolling updates),
Postgres/OpenSearch/Qdrant StatefulSets, Redis, MLflow, Airflow (scheduler +
webserver + RBAC for pod launching), Prometheus (pod discovery scraping), Grafana,
and one-shot Jobs for migrations and ad-hoc training.

```bash
kubectl apply -f k8s/namespace.yaml
cp k8s/secrets.example.yaml k8s/secrets.yaml   # fill in real values; never commit
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml -f k8s/storage.yaml
kubectl apply -f k8s/postgres.yaml -f k8s/redis.yaml -f k8s/opensearch.yaml \
  -f k8s/qdrant.yaml -f k8s/mlflow.yaml
kubectl apply -f k8s/jobs/migrate-job.yaml
kubectl apply -f k8s/api.yaml
kubectl -n news-rec create configmap airflow-dags --from-file=airflow/dags/
kubectl apply -f k8s/airflow.yaml -f k8s/prometheus.yaml -f k8s/grafana.yaml
```

## Monitoring

- **System:** request rate, p95 latency, error rate, retrieval latency by stage
  (BM25 / rerank), LightGBM scoring latency — Prometheus histograms surfaced in the
  provisioned Grafana dashboard (`monitoring/grafana/dashboards/`).
- **Model:** recommendations served by model version and strategy, feedback events,
  observed CTR, offline NDCG@10/MRR/Recall@50 per training run in MLflow.
- **Drift:** daily Evidently report over ranking features plus a PSI summary that
  feeds `alert_if_degraded`, which fails the monitoring DAG when CTR, latency,
  error rate, or drift share cross thresholds.

## Configuration

All services are configured through environment variables (compose sets them for
local use; Kubernetes injects them from `k8s/configmap.yaml` and the secret).

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection (from the secret in Kubernetes) |
| `OPENSEARCH_URL` / `QDRANT_URL` / `REDIS_URL` | Retrieval indexes and cache |
| `MLFLOW_TRACKING_URI` | Experiment tracking and model registry |
| `PROMETHEUS_URL` | Used by `compute_latency` to read p95 latency and error rate |
| `RANKER_MODEL_PATH` | LightGBM model file served by the API |
| `MODEL_VERSION` | Label attached to served recommendations and events |
| `RECOMMENDATION_CACHE_TTL` | Redis response cache TTL in seconds (default 300) |

## Design decisions

- **BM25 overretrieval + dense reranking.** BM25 gives a cheap, high-recall pool; dense
  similarity reorders it semantically before the learned ranker. Candidate embeddings
  come precomputed from Qdrant, which cut cold-path p95 latency 7.7x versus
  re-encoding candidates per request.
- **Leak-free features.** Popularity is built from global click counts excluding the
  user's own clicks, so training matches what serving can observe.
- **Evaluation-gated promotion.** `evaluate_model --min-ndcg` fails the retraining DAG
  before `promote_model` can move the MLflow `production` alias.
- **Airflow orchestrates, pods work.** Every pipeline step runs in an isolated pod from
  the project image, so Airflow stays thin and training can scale independently.
- **Separate Airflow environment.** Airflow pins SQLAlchemy 1.4 while the app needs
  2.x; keeping them apart mirrors production and avoids dependency breakage.

## Known limitations

- **The retraining loop is not fully closed.** The weekly DAG rebuilds its dataset from
  historical MIND impressions; live impressions logged by the API are not yet joined
  to subsequent clicks into labeled impression groups.
- **MIND data has no retrieval scores**, so bm25/dense/rerank features are 0 at training
  time and the model cannot use them; serving ties fall back to dense rerank order.
  Both limitations resolve once served impressions are logged with their scores.
- **The Qdrant index is the full corpus** (MIND + Hacker News in one collection).
- **Kubernetes manifests are validated structurally by tests**, not deployed to a
  live cluster in CI.

## Testing & CI

```bash
ruff check app pipelines scripts tests airflow
mypy                      # strict, app + pipelines
pytest --ignore=tests/test_dags.py   # needs local Postgres: docker compose up -d postgres && alembic upgrade head
.venv-airflow/bin/pytest tests/test_dags.py   # DAG tests, in the Airflow environment
```

GitHub Actions runs lint + tests against a Postgres service, validates the Airflow
DAGs in a dedicated environment, and on merge to `main` builds and pushes the API
image to GHCR (used by both Kubernetes and the Airflow pods).

Hit an error? Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — every
issue encountered during development is logged there with root cause and fix
(SQLAlchemy/Airflow conflict, macOS LightGBM segfault, port 5000 vs AirPlay, …).
