# News Recommendation MLOps Platform — Codex Implementation Plan

## Overview

Build a production-style news recommendation system focused on:

- Retrieval engineering
- Ranking systems
- MLOps
- Apache Airflow orchestration
- Kubernetes deployment
- Monitoring and retraining

This project is optimized for:

- ML Engineering roles
- MLOps roles
- AI Infrastructure roles
- Backend + AI systems roles
- Search / Recommendation engineering roles

---

# Project Goal

Build a recommendation platform that:

- Ingests new articles daily
- Performs BM25 retrieval
- Performs dense vector retrieval
- Uses hybrid retrieval
- Reranks candidates with LightGBM
- Serves recommendations through FastAPI
- Logs user interactions
- Retrains models weekly
- Tracks models in MLflow
- Uses Airflow for orchestration
- Deploys services on Kubernetes
- Monitors system and model health

---

# Core Architecture

```text
Daily Article Ingestion
        ↓
PostgreSQL Article Store
        ↓
OpenSearch BM25 Index
        ↓
Qdrant Vector Index
        ↓
Hybrid Candidate Retrieval
        ↓
LightGBM Ranker
        ↓
FastAPI Recommendation Service
        ↓
User Feedback Logging
        ↓
Airflow Retraining DAG
        ↓
MLflow Model Registry
        ↓
Kubernetes Deployment
        ↓
Prometheus + Grafana Monitoring
```

---

# Technology Stack

## Backend

- Python
- FastAPI
- SQLAlchemy
- Pydantic

## Retrieval

- OpenSearch
- Qdrant
- SentenceTransformers

## Ranking

- LightGBM Ranker

## Databases

- PostgreSQL
- Redis

## MLOps

- Apache Airflow
- MLflow
- Docker
- Kubernetes

## Monitoring

- Prometheus
- Grafana
- Evidently AI

---

# Data Sources

## Historical Recommendation Data

### MIND Dataset

Use for:

- Offline training
- Ranking evaluation
- Feature engineering
- User click simulation

Dataset:

https://msnews.github.io/

---

## Daily Dynamic Data

### Hacker News Algolia API

Use for:

- Daily article ingestion
- Continuously changing content
- Weekly retraining
- Retrieval updates

API:

https://hn.algolia.com/api

---

# Repository Structure

```text
news-rec-mlops/
│
├── app/
│   ├── api/
│   ├── retrieval/
│   ├── ranking/
│   ├── features/
│   ├── monitoring/
│   └── db/
│
├── airflow/
│   ├── dags/
│   └── plugins/
│
├── pipelines/
│   ├── ingest_hn.py
│   ├── ingest_mind.py
│   ├── train_ranker.py
│   ├── evaluate_model.py
│   └── generate_drift_report.py
│
├── scripts/
├── tests/
├── k8s/
├── monitoring/
├── docker/
├── notebooks/
├── data/
├── reports/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# Development Phases

# Phase 1 — Repository Scaffold

## Goal

Create clean project foundation.

## Tasks

- Setup Python project
- Setup FastAPI
- Setup pytest
- Setup Docker
- Setup docker-compose
- Setup linting and formatting
- Add /health endpoint

## Codex Prompt

```text
Create a production-style Python project for a recommendation system.

Set up:
- FastAPI
- pytest
- ruff
- mypy
- Docker
- docker-compose
- health endpoint

Use clean project structure and typed Python code.
```

---

# Phase 2 — PostgreSQL Schema

## Goal

Create persistent storage.

## Tables

```text
articles
users
user_events
recommendations
model_versions
```

## Important Fields

### articles

```text
id
title
url
text
source
published_at
category
embedding_id
created_at
```

### user_events

```text
user_id
article_id
event_type
position
model_version
timestamp
```

## Codex Prompt

```text
Add SQLAlchemy models and Alembic migrations for:
- articles
- users
- user_events
- recommendations
- model_versions

Add repository classes and tests.
```

---

# Phase 3 — Daily Article Ingestion

## Goal

Continuously collect new content.

## Source

Hacker News Algolia API.

## Tasks

- Fetch daily stories
- Normalize article schema
- Deduplicate URLs
- Store articles

## Codex Prompt

```text
Implement Hacker News ingestion pipeline.

Requirements:
- fetch recent stories
- normalize data
- deduplicate by URL
- store in PostgreSQL
- support CLI args:
  --days-back
  --limit
```

---

# Phase 4 — MIND Dataset Loader

## Goal

Load historical recommendation data.

## Tasks

- Parse news.tsv
- Parse behaviors.tsv
- Create training events
- Create impression logs

## Codex Prompt

```text
Implement MIND dataset ingestion.

Load:
- articles
- users
- impressions
- clicks

Store into PostgreSQL.
```

---

# Phase 5 — BM25 Retrieval

## Goal

Implement lexical retrieval.

## Stack

- OpenSearch

## Tasks

- Create index mappings
- Index articles
- Implement BM25 retrieval

## Codex Prompt

```text
Add OpenSearch BM25 retrieval.

Implement:
- index_articles()
- search_bm25(query, top_k)

Add integration tests.
```

---

# Phase 6 — Dense Retrieval

## Goal

Implement vector search.

## Stack

- SentenceTransformers
- Qdrant

## Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Tasks

- Generate embeddings
- Create Qdrant collection
- Vector search

## Codex Prompt

```text
Add dense retrieval using Qdrant.

Requirements:
- embedding generation
- vector indexing
- ANN retrieval
- configurable embedding model
```

---

# Phase 7 — Hybrid Retrieval

## Goal

Combine BM25 and dense retrieval.

## Formula

```text
hybrid_score =
  0.5 * normalized_bm25 +
  0.5 * normalized_dense
```

## Tasks

- Score normalization
- Deduplication
- Candidate merging

## Codex Prompt

```text
Implement hybrid retrieval combining BM25 and dense retrieval.

Requirements:
- score normalization
- deduplication
- configurable weighting
```

---

# Phase 8 — Ranking Features

## Goal

Build ranking dataset.

## Features

```text
bm25_score
dense_score
hybrid_score
article_age_hours
article_popularity
user_category_affinity
topic_similarity
recent_click_similarity
```

## Codex Prompt

```text
Implement ranking feature generation for:
- user_id
- candidate articles

Return model-ready feature rows.
```

---

# Phase 9 — Train LightGBM Ranker

## Goal

Train reranking model.

## Metrics

```text
NDCG@10
MRR
Recall@50
```

## Tasks

- Create training dataset
- Train ranker
- Evaluate metrics
- Register model in MLflow

## Codex Prompt

```text
Train LightGBM learning-to-rank model.

Track:
- metrics
- parameters
- artifacts

Log everything to MLflow.
```

---

# Phase 10 — Recommendation API

## Goal

Serve ranked recommendations.

## Endpoint

### GET /recommendations/{user_id}

## Pipeline

```text
user history
    ↓
candidate retrieval
    ↓
feature generation
    ↓
ranking model
    ↓
top recommendations
```

## Codex Prompt

```text
Implement recommendation serving API.

Flow:
- retrieve candidates
- generate features
- score candidates
- return ranked articles
```

---

# Phase 11 — Feedback Logging

## Goal

Track recommendation quality.

## Events

```text
impression
click
dwell
skip
```

## Endpoints

```text
POST /events/impression
POST /events/click
POST /events/dwell
```

## Codex Prompt

```text
Implement feedback event logging endpoints.

Store:
- user_id
- article_id
- model_version
- timestamp
- event metadata
```

---

# Phase 12 — Airflow Orchestration

# DAG 1 — Daily Ingestion DAG

## Tasks

```text
fetch_hn_articles
store_articles
update_opensearch
generate_embeddings
update_qdrant
```

---

# DAG 2 — Weekly Retraining DAG

## Tasks

```text
extract_training_events
build_features
train_ranker
evaluate_model
register_model
promote_model
```

---

# DAG 3 — Monitoring DAG

## Tasks

```text
compute_ctr
compute_latency
generate_drift_report
alert_if_degraded
```

---

## Important Requirement

Use:

```text
KubernetesPodOperator
```

for:

- Training jobs
- Evaluation jobs
- Drift jobs

## Codex Prompt

```text
Implement Airflow DAGs for:
- daily ingestion
- weekly retraining
- monitoring

Use KubernetesPodOperator for training jobs.

Add DAG import tests.
```

---

# Phase 13 — Kubernetes Deployment

## Goal

Deploy production-style infrastructure.

## Deploy

```text
FastAPI
Airflow
MLflow
PostgreSQL
Redis
OpenSearch
Qdrant
Prometheus
Grafana
```

## Kubernetes Features

### Include

- Deployments
- Services
- ConfigMaps
- Secrets
- PVCs
- HorizontalPodAutoscaler

### Important

Add:
- Training Jobs
- Rolling updates
- Autoscaling

## Codex Prompt

```text
Create Kubernetes manifests for:
- FastAPI
- Airflow
- MLflow
- PostgreSQL
- Redis
- OpenSearch
- Qdrant
- Prometheus
- Grafana

Include:
- HPA
- PVCs
- ConfigMaps
- Secrets templates
```

---

# Phase 14 — Monitoring & Drift

## Monitor

### System Metrics

```text
request latency
throughput
error rate
retrieval latency
ranking latency
```

### Model Metrics

```text
CTR
NDCG@10
prediction distribution
embedding drift
feature drift
```

## Monitoring Stack

- Prometheus
- Grafana
- Evidently AI

## Codex Prompt

```text
Add Prometheus metrics and Evidently drift reports.

Expose:
GET /metrics

Generate daily drift reports.
```

---

# Local Development Stack

Use Docker Compose.

## Services

```text
api
postgres
redis
opensearch
qdrant
mlflow
airflow
prometheus
grafana
```

---

# Final Resume Value

This project demonstrates:

- Retrieval engineering
- Ranking systems
- Recommendation infrastructure
- MLOps
- Airflow orchestration
- Kubernetes deployment
- Monitoring
- Model retraining
- Production ML systems
- Backend engineering
- Distributed systems thinking

---

# Strong Resume Bullets

> Built a production-style recommendation platform using BM25, dense retrieval, hybrid search, and LightGBM reranking with FastAPI-based serving.

> Designed Airflow-orchestrated retraining pipelines using KubernetesPodOperator for distributed model training, evaluation, and automated MLflow model promotion.

> Deployed recommendation infrastructure on Kubernetes with autoscaling, monitoring, drift detection, Prometheus metrics, and Grafana dashboards.

---

# Recommended Build Order

```text
1. Repo scaffold
2. PostgreSQL schema
3. HN ingestion
4. MIND ingestion
5. BM25 retrieval
6. Dense retrieval
7. Hybrid retrieval
8. Feature generation
9. LightGBM training
10. Recommendation API
11. Feedback logging
12. Airflow DAGs
13. Kubernetes deployment
14. Monitoring & drift
15. README polish
```

---

# Codex Best Practices

Use:
- AGENTS.md
- incremental tasks
- small PR-sized prompts
- clear acceptance criteria
- test-driven prompts

Useful references:

- https://developers.openai.com/codex/learn/best-practices
- https://developers.openai.com/cookbook/articles/codex_exec_plans

