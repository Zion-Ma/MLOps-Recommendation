import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)

RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds",
    "Candidate retrieval latency in seconds",
    ["stage"],
)

RANKING_LATENCY = Histogram(
    "ranking_latency_seconds",
    "LightGBM scoring latency in seconds",
)

RECOMMENDATIONS_SERVED = Counter(
    "recommendations_served_total",
    "Recommendation responses served",
    ["model_version", "strategy"],
)

FEEDBACK_EVENTS = Counter(
    "feedback_events_total",
    "Feedback events recorded",
    ["event_type"],
)


def _route_template(request: Request) -> str:
    # Use the route template (e.g. /recommendations/{user_id}) instead of the raw
    # path to keep Prometheus label cardinality bounded.
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str):
        return path
    return "unmatched"


async def prometheus_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    path = _route_template(request)
    REQUEST_COUNT.labels(
        method=request.method,
        path=path,
        status=str(response.status_code),
    ).inc()
    REQUEST_LATENCY.labels(method=request.method, path=path).observe(elapsed)

    return response


def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
