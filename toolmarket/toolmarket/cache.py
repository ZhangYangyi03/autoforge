"""Caching, with Redis when there is one and correctly without when there isn't.

The rule this module exists to enforce: **a cache that is absent must not be a
cache that lies.** Every read path here returns either a hit or "no opinion",
and nothing above it may treat "no opinion" as an error. That is why there is a
`NullCache` implementing the same surface rather than `Optional[Cache]` threaded
through callers — the latter breeds `if cache is not None:` at every call site,
and the first one somebody forgets is a 500 on a deployment that simply did not
attach Redis.

Two details that are easy to get wrong and are therefore deliberate:

  * Invalidation happens on **write**, in `ResourceRegistry.save`, and not at
    every call site that might have mutated a record. There is exactly one place
    a record changes; putting the invalidation anywhere else means eventually
    there are two, and the second one is the bug.
  * Keys carry a **schema prefix** (`tm:v1:...`). A deploy that changes the
    record shape must not silently serve records serialised by the previous
    release out of a warm Redis. Bumping `CACHE_NAMESPACE` is the whole
    migration story, and it is a one-line change on purpose.

Redis is reached through `redis-py`; the import is deferred so the base install
never needs it.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

CACHE_NAMESPACE = "tm:v1"
DEFAULT_TTL = 30.0

# The process-wide instance, built on first use from `REDIS_URL`.
_CACHE: Optional[Any] = None
_CACHE_LOCK = threading.Lock()


def resource_key(resource_id: str) -> str:
    return f"{CACHE_NAMESPACE}:resource:{resource_id}"


class NullCache:
    """The cache you get when there is no Redis. Every read misses, on purpose.

    Not a stub: it is the *correct* behaviour for a single-process deployment,
    where the registry's in-memory projection is already the fastest copy of the
    data and a cache in front of it would only add a way to serve stale records.
    """

    enabled = False
    backend = "none"

    def get(self, key: str) -> Optional[Any]:
        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        return None

    def delete(self, *keys: str) -> int:
        return 0

    def delete_prefix(self, prefix: str) -> int:
        return 0

    def incr(self, key: str, ttl: Optional[float] = None) -> int:
        """Counters must still work when Redis does not.

        Returns 1 every time: with no shared store there is no window to count
        within, and pretending otherwise would make a limiter that silently
        allowed everything while reporting limits it was not enforcing.
        """
        return 1

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None


class RedisCache:
    """A JSON cache over Redis. `decode_responses=True` throughout, so values
    arrive as `str` and the JSON round-trip is the only serialisation."""

    enabled = True
    backend = "redis"

    def __init__(self, url: str, *, prefix: str = CACHE_NAMESPACE,
                 socket_timeout: float = 2.0) -> None:
        try:
            import redis  # noqa: PLC0415 - deferred so the base install stays light
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "RedisCache needs redis-py, which failed to import: "
                f"{exc!r}. Install the extra: pip install 'tool-market[redis]'"
            ) from exc
        self.url = url
        self.prefix = prefix
        self._client = redis.Redis.from_url(
            url, decode_responses=True, socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
        )
        # Namespacing is applied here, once, so no caller can forget it and
        # quietly write into another application's keyspace.
        self._ns = f"{prefix}:"

    @property
    def client(self) -> Any:
        return self._client

    def _k(self, key: str) -> str:
        return key if key.startswith(self._ns) else f"{self._ns}{key}"

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(self._k(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            # A corrupt entry is a miss, not a crash: an unrelated writer or a
            # truncated value should never take the API down.
            self._client.delete(self._k(key))
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        payload = json.dumps(value, default=str)
        self._client.set(self._k(key), payload, ex=int(ttl or DEFAULT_TTL))

    def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return int(self._client.delete(*[self._k(k) for k in keys]))

    def delete_prefix(self, prefix: str) -> int:
        """SCAN + DEL, never KEYS.

        KEYS blocks the server for the duration of a full keyspace walk, which on
        a shared Redis is a production outage triggered by your own cache
        invalidation. SCAN is O(1) per call and safe to run against live traffic.
        """
        pattern = f"{self._k(prefix)}*"
        removed = 0
        for chunk in self._client.scan_iter(match=pattern, count=200):
            removed += int(self._client.delete(chunk))
        return removed

    def incr(self, key: str, ttl: Optional[float] = None) -> int:
        k = self._k(key)
        pipe = self._client.pipeline()
        pipe.incr(k)
        if ttl:
            # NX: only the first increment in a window sets the expiry, so a
            # counter never has its window extended by the traffic it is meant
            # to be measuring. Without NX, a steady stream of requests keeps
            # pushing the deadline out and the limit never resets.
            pipe.expire(k, int(ttl), nx=True)
        count, _ = pipe.execute()
        return int(count)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


class MemoryCache:
    """A bounded in-process TTL cache.

    This is what the substrate runs on when a single process serves everything:
    a limiter still counts, and a hot resource is not re-serialised per request.
    It is *not* a Redis substitute behind more than one worker — two processes
    have two counters, so a configured limit is per-process. The deployment doc
    says so out loud rather than leaving the reader to discover it under load.
    """

    enabled = True
    backend = "memory"

    def __init__(self, *, max_entries: int = 2048, prefix: str = CACHE_NAMESPACE) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._max = max_entries
        self.prefix = prefix

    def _live(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires and expires < time.time():
            self._data.pop(key, None)
            return None
        return value

    def get(self, key: str) -> Optional[Any]:
        return self._live(key)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if len(self._data) >= self._max:
            # Evict the soonest-to-expire entry. Cheap, and close enough to LRU
            # for a cache whose entries all share a TTL.
            oldest = min(self._data, key=lambda k: self._data[k][0] or float("inf"))
            self._data.pop(oldest, None)
        self._data[key] = (time.time() + float(ttl or DEFAULT_TTL), value)

    def delete(self, *keys: str) -> int:
        return sum(1 for k in keys if self._data.pop(k, None) is not None)

    def delete_prefix(self, prefix: str) -> int:
        doomed = [k for k in self._data if k.startswith(prefix)]
        for k in doomed:
            self._data.pop(k, None)
        return len(doomed)

    def incr(self, key: str, ttl: Optional[float] = None) -> int:
        current = self._live(key)
        nxt = int(current or 0) + 1
        if current is None:
            self._data[key] = (time.time() + float(ttl or DEFAULT_TTL), nxt)
        else:
            expires, _ = self._data[key]
            self._data[key] = (expires, nxt)
        return nxt

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        self._data.clear()


def make_cache(url: Optional[str] = None) -> Any:
    """`REDIS_URL` (or `url`) -> a cache. Unset -> `MemoryCache`; `"none"` ->
    `NullCache`.

    Memory is the default rather than `NullCache` because the single-process
    deployment is the common case and a counting limiter is worth having there.
    `"none"` is how you say "I want the substrate to have no cache at all",
    which is what the tests that assert on store call counts need.
    """
    raw = (url or os.environ.get("REDIS_URL") or "").strip()
    if raw.lower() in ("none", "off", "disabled"):
        return NullCache()
    if not raw:
        return MemoryCache()
    return RedisCache(raw)


def reset_cache_singleton() -> None:
    """Drop the process-wide cache. Tests call this between cases; the API calls
    it once at shutdown so a Redis connection is not leaked on reload."""
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None:
            _CACHE.close()
        _CACHE = None


def get_cache() -> Any:
    """The process-wide cache, built once from the environment.

    A singleton rather than one per caller because the *point* of the cache is
    that two components — the read path and the write path that invalidates it —
    are looking at the same store. Two instances of `MemoryCache` would mean a
    write invalidating nothing the reader can see.
    """
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = make_cache()
        return _CACHE
