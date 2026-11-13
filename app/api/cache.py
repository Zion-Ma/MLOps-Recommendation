import os
from functools import lru_cache
from typing import Any

DEFAULT_CACHE_TTL_SECONDS = 300


def cache_ttl_seconds() -> int:
    return int(os.getenv("RECOMMENDATION_CACHE_TTL", str(DEFAULT_CACHE_TTL_SECONDS)))


@lru_cache(maxsize=1)
def get_cache_client() -> Any | None:
    """Return a Redis client, or None when Redis is unreachable.

    The API must keep serving recommendations when Redis is down, so cache
    failures degrade to uncached responses instead of errors.
    """
    try:
        import redis

        client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        client.ping()
        return client
    except Exception:
        return None


def cache_get(client: Any | None, key: str) -> str | None:
    if client is None:
        return None
    try:
        value = client.get(key)
    except Exception:
        return None
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def cache_set(client: Any | None, key: str, value: str, ttl_seconds: int) -> None:
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, value)
    except Exception:
        return
