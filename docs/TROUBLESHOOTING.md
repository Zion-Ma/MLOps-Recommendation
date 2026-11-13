# Troubleshooting Log

Errors encountered while building and running this project, with root causes and
fixes. Newest issues at the bottom.

---

## 1. Installing Airflow broke the app environment

**Symptom**

```text
ImportError: cannot import name 'mapped_column' from 'sqlalchemy.orm'
```

Every app/DB test failed at collection right after `pip install apache-airflow`.

**Root cause** — Airflow 2.x pins SQLAlchemy `<2.0`. Installing it into the app
environment silently downgraded SQLAlchemy from 2.x to 1.4, and the app's models
use the SQLAlchemy 2.0 API (`Mapped`, `mapped_column`).

**Fix** — Airflow lives in its own environment, never the app's:
`airflow/requirements.txt` + a dedicated `dag-tests` CI job. If this happens
again: `pip uninstall apache-airflow...` then `pip install "sqlalchemy>=2.0"`.

---

## 2. Segmentation fault in LightGBM during tests (macOS)

**Symptom**

```text
Fatal Python error: Segmentation fault
  File ".../lightgbm/basic.py", line 2301 in __init_from_np2d
```

The full pytest suite crashed at `test_evaluate_model`, but every test file
passed when run alone.

**Root cause** — an OpenMP runtime clash on macOS. It requires all three of:
torch loaded first (via `sentence-transformers` in retrieval tests), pyarrow
used (parquet I/O), and LightGBM training. Each pair works; the triple
segfaults. `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` do NOT help.

**Fix** — import order: loading LightGBM before torch keeps one OpenMP runtime
in charge. `tests/conftest.py` does `import lightgbm` at the top so it always
loads before any test module pulls in torch. (App code is unaffected because
`app/api/recommendations.py` happens to import lightgbm before
sentence-transformers.)

---

## 3. Airflow DagBag tests hit a nonexistent metadata database

**Symptom**

```text
sqlite3.OperationalError: no such table: dag
```

DAG import tests passed, but any test calling `dag_bag.get_dag("...")` failed.

**Root cause** — `DagBag.get_dag()` queries the Airflow metadata DB (for paused
state etc.), which doesn't exist in a unit-test environment.

**Fix** — read parsed DAGs from the in-memory dict instead:
`dag_bag.dags["weekly_retraining"]`.

---

## 4. kubectl dry-run validation needs a live cluster

**Symptom**

```text
kubectl apply --dry-run=client -f k8s/ -R
... dial tcp [::1]:8080: connect: connection refused
```

**Root cause** — even `--dry-run=client` contacts the API server for group/
version discovery; there is no local cluster on this machine.

**Fix** — manifest validation moved into the test suite
(`tests/test_k8s_manifests.py`): PyYAML parsing plus structural checks (probes,
resources, selector↔label consistency, expected components). Runs in CI with no
cluster required.

---

## 5. Ruff B008 on FastAPI `Depends` defaults

**Symptom**

```text
B008 Do not perform function call `Depends` in argument defaults
```

**Root cause** — the classic `db: Session = Depends(get_db)` pattern triggers
flake8-bugbear's function-call-in-default rule.

**Fix** — the modern FastAPI style: `db: Annotated[Session, Depends(get_db)]`,
with dependency parameters ordered before defaulted query parameters. Do NOT
work around it with `= None` defaults — FastAPI may treat that as a real
default.

---

## 6. Kubernetes ConfigMap values do not expand `$(VAR)`

**Symptom (caught in review, would have failed at deploy time)** — the Airflow
DB connection string in a ConfigMap contained `$(POSTGRES_PASSWORD)`, which
Kubernetes never expands inside ConfigMap data.

**Root cause** — `$(VAR)` substitution only works in a container's `env:` value
fields, referencing variables defined *earlier in the same env list*.

**Fix** — `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` moved into each container's
`env:` list, after the `POSTGRES_PASSWORD` secret ref (see `k8s/airflow.yaml`).

---

## 7. Docker build would fail in CI: `COPY models` on a gitignored directory

**Symptom (caught in review)** — the Dockerfile copied `models/` into the
image, but `models/` is in `.gitignore`, so a CI checkout has no such directory
and `COPY` fails the build.

**Fix** — the model is not baked into the image. Compose mounts `./models` into
the api container (`RANKER_MODEL_PATH=/app/models/ranker.txt`); Kubernetes
serves it from the shared data PVC.

---

## 8. `docker compose up` fails: port 5000 already in use

**Symptom**

```text
Error response from daemon: Ports are not available:
listen tcp 0.0.0.0:5000: bind: address already in use
```

**Root cause** — macOS AirPlay Receiver (the `ControlCenter` process) listens
on port 5000. Verify with `lsof -nP -iTCP:5000 -sTCP:LISTEN`.

**Fix** — MLflow's host port is mapped to **5001** in docker-compose.yml
(container port stays 5000, so in-network URLs like `http://mlflow:5000` are
unchanged). Local MLflow UI: http://localhost:5001. Alternative: disable
AirPlay Receiver in System Settings → General → AirDrop & Handoff.

---

## 9. API container served only `/health` (stale image)

**Symptom** — the compose stack was healthy, but `/recommendations`,
`/events/*`, and `/metrics` returned `{"detail":"Not Found"}`; the OpenAPI spec
listed only `/health`.

**Root cause** — `docker compose up` reuses an existing image; the api image
had been built back when the project only had a health endpoint, and compose
never rebuilds automatically after code changes.

**Fix**

```bash
docker compose up -d --build api
```

---

## 10. `GET /recommendations/{user_id}` returns 404 for a new user

**Symptom**

```text
404 {"detail": "User 'zion' not found"}
```

**Root cause** — by design, not a bug. The recommendations endpoint never
creates users; only the feedback endpoints (`POST /events/*`) auto-create
unknown external ids.

**Fix / usage pattern** — log an event first, then request recommendations:

```bash
curl -X POST localhost:8000/events/click -H 'Content-Type: application/json' \
  -d '{"user_id": "zion", "article_id": "<existing-article-uuid>"}'
curl localhost:8000/recommendations/zion
```

Or use an existing MIND user id (e.g. `U13740`).

---

## 11. RESOLVED (2026-08-25): ranker predicted constant 0.0 scores

**Symptom** — every recommendation has `"score": 0.0`; ordering effectively
comes from the dense rerank stage, not LightGBM.

**Root cause** — the bundled `models/ranker.txt` was trained on a dataset where
`bm25_score`, `dense_score`, and `rerank_score` were hardcoded to 0.0
(`pipelines/build_ranking_dataset.py` sets them to 0 for MIND impressions
because no retrieval scores were logged for historical data). The remaining
features produced no useful splits, so the model outputs a constant.

**Fix** — retrain once real retrieval scores exist in the training data. Either
(a) log live impressions through the API (which will carry real scores once
the recommendations→events loop populates them), or (b) backfill retrieval
scores when building the dataset. Then:

```bash
python -m pipelines.build_ranking_dataset --output-path data/ranking_features.parquet
python -m pipelines.train_ranker --input-path data/ranking_features.parquet \
  --output-path models/ranker.txt --mlflow-tracking-uri http://localhost:5001 \
  --register-model
docker compose restart api
```

**Resolution** — retrained on a rebuilt 50k-row dataset (see #12–#15 below).
Validation NDCG@10 0.750 / MRR 0.749 / Recall@50 0.973 vs a 0.285 NDCG random
baseline. Remaining limitation: within one serving candidate set, scores can
still tie because the retrieval-score features (bm25/dense/rerank) were 0 in
historical training data, so the model never learned to use them — ordering of
tied candidates falls back to the dense rerank stage. Fixes itself once live
impressions logged through the API (which carry real retrieval scores) feed
retraining.

---

## 12. Dataset build crashed: `KeyError: 'mind_impression_id'`

**Symptom** — `pipelines.build_ranking_dataset` crashed once live impressions
existed alongside MIND ones.

**Root cause** — impressions logged through `POST /events/impression` carry
arbitrary metadata; the builder assumed every labeled impression had MIND's
`mind_impression_id` group key.

**Fix** — impressions without a group key are skipped (they cannot form
ranking groups). See `build_rows()` in `pipelines/build_ranking_dataset.py`.

---

## 13. MLflow client/server version mismatch

**Symptom**

```text
API request to endpoint /api/2.0/mlflow/logged-models failed with error code 404
```

**Root cause** — the app environment uses the MLflow 3 client, which calls the
logged-models API; the compose server image was v2.16, which doesn't have it.

**Fix** — compose now runs `ghcr.io/mlflow/mlflow:v3.14.0`. Keep client and
server on the same major version.

---

## 14. MLflow refused to serialize the sklearn LightGBM wrapper

**Symptom**

```text
MlflowException: The saved sklearn model references untrusted types ...
['lightgbm.sklearn.LGBMRanker']
```

**Root cause** — MLflow 3 serializes sklearn-flavor models with skops, which
rejects `LGBMRanker` as an untrusted type.

**Fix** — `train_ranker` logs the native booster (`model.booster_`) via the
lightgbm flavor instead of the sklearn wrapper.

---

## 15. Label leakage: NDCG@10 of 0.96 was bogus

**Symptom** — first real training run reported validation NDCG@10 0.962, which
is implausibly high for two usable features.

**Root cause** — `article_popularity` was computed from the user's own click
list, which contained the very click being predicted: 99.7% of positive rows
had popularity > 0 vs 0.7% of negatives. The model learned "was clicked →
predict clicked". It was also train/serve skew: at serving time the user's
clicked articles are filtered out, so the feature was always 0 in production.

**Fix** — popularity is now the global click count across all users
(`UserEventRepository.count_clicks_by_article`), minus the user's own clicks
on the candidate; the user's clicks on the candidate are also dropped from the
history used for category affinity (drop-one). Honest metrics after the fix:
NDCG@10 0.750, MRR 0.749, Recall@50 0.973 (held-out 20% group split, 275
validation impression groups) vs 0.285/0.241/0.921 random baseline.

**Lesson** — when a ranking metric looks too good, check whether any feature
can see the label. Compare feature distributions grouped by label
(`df.groupby("label")[features].mean()`) before trusting the number.
