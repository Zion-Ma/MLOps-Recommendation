# Quantified Results

Measured 2026-08-25 on the local stack (Docker Compose, Apple Silicon).
Reproduction commands at the bottom.

## Data scale

| Quantity | Value |
| --- | --- |
| User events ingested | 11,187,051 |
| Impressions | 5,843,448 |
| Clicks | 5,343,598 |
| Users | 50,058 |
| Articles | 51,555 |
| Ranking dataset | 49,995 rows / 1,376 impression groups |

## Offline ranking quality

Held-out 20% group split (9,840 rows, 275 impression groups). Model:
LightGBM lambdarank, registered in MLflow as `news-lightgbm-ranker` v1,
promoted to the `production` alias.

| Metric | LightGBM ranker | Random baseline | Popularity baseline |
| --- | --- | --- | --- |
| NDCG@10 | **0.750** | 0.285 | 0.294 |
| MRR | **0.749** | 0.241 | 0.255 |
| Recall@50 | **0.973** | 0.921 | 0.917 |

NDCG@10 is 2.6x the random baseline (+163% relative).

**Leakage audit.** The first training run scored 0.962 NDCG@10. A feature
audit showed `article_popularity` was derived from the user's own click list
and therefore contained the label (popularity > 0 for 99.7% of positives vs
0.7% of negatives). After rebuilding the feature from global click counts
(minus the user's own clicks) and dropping the candidate from the click
history used for category affinity, the honest score is the 0.750 above.
Details in TROUBLESHOOTING.md #15.

## Serving latency

30 distinct users with click history, full pipeline per request (BM25 pool of
200 -> dense rerank -> feature generation -> LightGBM -> persist), measured
against the containerized API.

**Before optimization** — every request re-encoded ~200 candidate texts with
SentenceTransformers on CPU:

| Path | p50 | p95 | mean |
| --- | --- | --- | --- |
| Cold (re-encode candidates) | 1,462 ms | 1,902 ms | 1,422 ms |
| Redis-cached | 3 ms | 5 ms | 4 ms |

**After optimization** — candidate embeddings fetched from Qdrant (51,578
precomputed vectors); only the query is embedded per request, with on-the-fly
encoding as fallback for unindexed articles. 60 cold requests:

| Path | p50 | p95 | mean | max |
| --- | --- | --- | --- | --- |
| Cold (precomputed vectors) | **206 ms** | **246 ms** | 218 ms | 508 ms |
| Redis-cached | 3 ms | 4 ms | 3 ms | — |

Cold-path p95 improved 7.7x (1,902 ms -> 246 ms) and p50 7.1x. The remaining
tail (~500 ms) correlates with users with very long click histories. Redis
caching is a separate layer on top for repeat requests; no organic-traffic
hit rate exists yet, so no blended latency is claimed.

## Engineering

- 115 tests passing (unit, API, pipeline, DAG, k8s manifest); mypy `--strict`
  clean; 0 ruff findings
- 41 automated structural checks over the Kubernetes manifests (probes,
  resources, selector/label consistency)
- 3 Airflow DAGs (daily ingestion, weekly retraining with NDCG promotion
  gate, monitoring), each step an isolated KubernetesPodOperator pod
- 9-service stack (FastAPI, PostgreSQL, Redis, OpenSearch, Qdrant, MLflow,
  Airflow, Prometheus, Grafana); CI validates DAGs in a dedicated Airflow
  environment and ships the API image to GHCR
- 15 development/production issues documented with root cause and fix in
  TROUBLESHOOTING.md

## Known limitations (be precise about these in interviews)

- **The retraining loop is not closed yet.** The weekly DAG currently
  rebuilds its dataset from historical MIND impression labels; live
  impressions logged through the API are skipped because nothing yet joins a
  served recommendation to a subsequent click into a labeled impression
  group. The events table is real and written by the serving path, but its
  only live rows are manual test events.
- **Historical MIND data carries no retrieval scores** (bm25/dense/rerank
  features are 0), so the trained model cannot use them and serving-time
  candidates can tie on score; tied ordering falls back to the dense rerank
  stage. Both limitations resolve together once served impressions are
  logged with their retrieval scores and group ids.
- **The Qdrant index (51,578 vectors) is the full article corpus** — MIND
  plus Hacker News plus test rows in one collection. Do not label it as
  MIND-only or HN-only.

## Reproduction

```bash
python -m pipelines.build_ranking_dataset --output-path data/ranking_features_v3.parquet --limit 50000
python -m pipelines.train_ranker --input-path data/ranking_features_v3.parquet \
  --output-path models/ranker_v3.txt --mlflow-tracking-uri http://localhost:5001 --register-model
python -m pipelines.evaluate_model --model-path models/ranker_v3.txt \
  --dataset-path data/ranking_features_v3.parquet --output-path reports/eval_metrics.json
```

Latency: time repeated `GET /recommendations/{user_id}` calls for 30 distinct
users, first hit vs second hit (Redis cache TTL 300s).

---

# Resume project description (final, high-value set)

**News Recommendation System with MLOps Pipeline**

- Trained a LightGBM learning-to-rank model on 50K impressions sampled from
  5.8M historical Microsoft MIND interactions, reaching 0.75 NDCG@10 and
  0.97 Recall@50 on a held-out split (2.6x a random-order baseline), with
  MLflow-tracked runs and evaluation-gated promotion to a production alias
- Caught label leakage that had inflated NDCG@10 to 0.96 by auditing feature
  distributions against labels, then rebuilt the popularity feature from
  global click counts so training matched what serving observes
- Profiled the FastAPI serving pipeline (OpenSearch BM25 overretrieval of
  200+ candidates, SentenceTransformers dense reranking, LightGBM scoring),
  identified per-request candidate re-encoding as the bottleneck, and cut
  cold-path p95 latency 7.7x (1.9s to 250ms) by serving precomputed Qdrant
  embeddings, with Redis caching layered on top for repeat requests
- Isolated Airflow into its own environment and CI job after diagnosing that
  its SQLAlchemy 1.4 pin silently broke the application's SQLAlchemy 2.x
  data layer, mirroring the production separation of orchestration and
  serving
- Orchestrated the full lifecycle with three Airflow DAGs (daily Hacker News
  ingestion feeding the live retrieval indexes, weekly retraining with an
  NDCG promotion gate, drift monitoring) running each step as an isolated
  KubernetesPodOperator pod, on a CI-validated 9-service Kubernetes stack
  with Prometheus and Grafana dashboards and PSI-based drift alerts that
  fail the monitoring DAG

## Alternate bullets (swap in per posting)

- Ranking/search roles: expand bullet 3 into separate retrieval and latency
  bullets; the two-stage retrieval design deserves its own line
- Platform/infra roles: expand bullet 5 back into separate Airflow and
  Kubernetes bullets (HPAs, PVCs, RBAC, rolling updates, 41 automated
  manifest checks)
- AI-forward companies (optional): "Directed AI coding agents through a
  test-driven, phase-based build, reviewing implementation plans and
  revising architectural decisions"
