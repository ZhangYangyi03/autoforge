"""The cache boundary: three backends, one contract, and the cases that differ.

The property under test is not "Redis works" — it is that *absent* and *disabled*
are different answers, and that neither one silently becomes "always a miss, so
always correct". A cache that degrades into lying about freshness is the failure
mode this module exists to prevent, so the tests pin the degradation path down.
"""
from __future__ import annotations

import time

import pytest

from toolmarket.cache import (
    CACHE_NAMESPACE,
    MemoryCache,
    NullCache,
    RedisCache,
    get_cache,
    make_cache,
    reset_cache_singleton,
    resource_key,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_cache_singleton()
    yield
    reset_cache_singleton()


# ------------------------------------------------------------------ memory
def test_memory_cache_round_trip():
    c = MemoryCache()
    assert c.enabled and c.backend == "memory"
    assert c.get("missing") is None
    c.set("k", {"a": 1})
    assert c.get("k") == {"a": 1}
    c.delete("k")
    assert c.get("k") is None


def test_memory_cache_expires_on_ttl():
    """A TTL that does not expire is a cache that serves yesterday's resource.

    The sleep is deliberately short and the assertion is on the *expired* state,
    because the interesting bug is not `set` failing to store, it is `get`
    returning a value past its deadline.
    """
    c = MemoryCache()
    c.set("k", "v", ttl=0.05)
    assert c.get("k") == "v"
    time.sleep(0.06)
    assert c.get("k") is None


def test_memory_cache_zero_ttl_is_treated_as_no_expiry():
    # `ttl=0` reaching here would otherwise mean "expire immediately", which
    # would turn `CACHE_TTL=0` (documented as *disable the cache*) into a cache
    # that always misses while still paying the write. The API guards this too;
    # the test is here because the guard is a one-line truthiness check somebody
    # will eventually "clean up".
    c = MemoryCache()
    c.set("k", "v", ttl=0)
    assert c.get("k") == "v"


def test_memory_cache_incr_counts_within_a_window():
    c = MemoryCache()
    assert c.incr("n", ttl=60) == 1
    assert c.incr("n", ttl=60) == 2
    assert c.incr("n", ttl=60) == 3


def test_memory_cache_delete_prefix():
    c = MemoryCache()
    c.set(f"{CACHE_NAMESPACE}:resource:a", 1)
    c.set(f"{CACHE_NAMESPACE}:resource:b", 2)
    c.set(f"{CACHE_NAMESPACE}:other", 3)
    removed = c.delete_prefix(f"{CACHE_NAMESPACE}:resource:")
    assert removed == 2
    assert c.get(f"{CACHE_NAMESPACE}:resource:a") is None
    assert c.get(f"{CACHE_NAMESPACE}:other") == 3


def test_memory_cache_ping_and_close():
    c = MemoryCache()
    assert c.ping() is True
    c.close()  # must not raise


# -------------------------------------------------------------------- null
def test_null_cache_misses_everything_and_says_so():
    c = NullCache()
    c.set("k", "v")
    assert c.get("k") is None
    assert c.enabled is False
    assert c.backend == "none"
    # `ping` True is the deliberate part: NullCache is a *chosen* configuration,
    # not a broken dependency, so a readiness probe must not call the process
    # unready because somebody turned the cache off.
    assert c.ping() is True


def test_null_cache_incr_never_pretends_to_enforce_a_window():
    """`incr` returns 1 forever — and that is the honest answer.

    With no shared store there is no window to count within. Returning 1 makes a
    counting limiter behave as if every request were the first, which is visible
    in the 429 rate; returning anything else would be a limiter inventing a
    budget it cannot actually enforce.
    """
    c = NullCache()
    assert [c.incr("n") for _ in range(5)] == [1, 1, 1, 1, 1]


# ---------------------------------------------------------------- dispatch
def test_make_cache_dispatch(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(make_cache(), MemoryCache)
    monkeypatch.setenv("REDIS_URL", "none")
    assert isinstance(make_cache(), NullCache)
    monkeypatch.setenv("REDIS_URL", "off")
    assert isinstance(make_cache(), NullCache)
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    try:
        import redis  # noqa: F401
        have_redis = True
    except Exception:  # noqa: BLE001
        have_redis = False
    if not have_redis:
        # Without the extra, the failure must name it rather than being an
        # ImportError from a module the caller never mentioned.
        with pytest.raises(RuntimeError) as excinfo:
            make_cache()
        assert "redis" in str(excinfo.value)
        return
    # Constructing a Redis backend must not connect: the API builds the cache at
    # import time, and connect-on-construct would turn a down Redis into an
    # import-time crash instead of a readiness failure.
    c = make_cache()
    assert isinstance(c, RedisCache)
    # An unreachable Redis is a *usable* object whose calls fail — ping reports
    # False rather than raising past the caller's try/except.
    assert c.ping() is False


def test_get_cache_is_a_singleton_and_resets(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "none")
    first = get_cache()
    assert get_cache() is first
    reset_cache_singleton()
    assert get_cache() is not first


def test_resource_key_shape():
    assert resource_key("tool:slugify") == f"{CACHE_NAMESPACE}:resource:tool:slugify"
