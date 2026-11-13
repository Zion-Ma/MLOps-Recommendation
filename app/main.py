# creates FastAPI app
from fastapi import FastAPI

from app.api.events import router as events_router
from app.api.recommendations import router as recommendations_router
from app.monitoring.metrics import metrics_endpoint, prometheus_middleware

app = FastAPI(title="News Recommendation API")

app.middleware("http")(prometheus_middleware)

app.include_router(recommendations_router)
app.include_router(events_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> object:
    return metrics_endpoint()
