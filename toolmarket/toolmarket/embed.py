"""embed.py -- embedding client for the toolmarket retrieval index.

Stdlib only (urllib). The toolmarket service runs on an interpreter without
numpy, so embeddings can never be computed in-process; they come from an HTTP
endpoint and are stored as float32 blobs by vectorstore.py.

Configuration, all through the environment, so no secret is ever written here:

    EMBED_BASE_URL   default https://aiping.cn/api/v1
    EMBED_API_KEY    when unset, the key is read read-only from EMBED_CONFIG
    EMBED_CONFIG     default C:/Users/china/.autoforge/config.json
    EMBED_MODEL      default Qwen3-Embedding-0.6B  (multilingual, 1024 dims)
    EMBED_DIM        default 1024
    EMBED_TIMEOUT    seconds per request, default 20
    EMBED_BATCH      inputs per request, default 32

Failure is explicit: EmbedUnavailable, never a silently empty vector. A caller
that cannot embed must fall back to lexical search, and say that it did.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_CONFIG = "C:/Users/china/.autoforge/config.json"


class EmbedUnavailable(RuntimeError):
    """Raised when an embedding could not be produced, for any reason."""


def _read_config_key(path):
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return ""
    if not isinstance(cfg, dict):
        return ""
    return str(cfg.get("api_key") or "")


def settings():
    base = (os.environ.get("EMBED_BASE_URL") or "https://aiping.cn/api/v1").rstrip("/")
    key = os.environ.get("EMBED_API_KEY") or _read_config_key(
        os.environ.get("EMBED_CONFIG") or DEFAULT_CONFIG)
    try:
        dim = int(os.environ.get("EMBED_DIM") or "1024")
    except Exception:
        dim = 1024
    try:
        timeout = float(os.environ.get("EMBED_TIMEOUT") or "20")
    except Exception:
        timeout = 20.0
    try:
        batch = int(os.environ.get("EMBED_BATCH") or "32")
    except Exception:
        batch = 32
    return {"base_url": base, "api_key": key,
            "model": os.environ.get("EMBED_MODEL") or "Qwen3-Embedding-0.6B",
            "dim": dim, "timeout": timeout, "batch": batch}


def describe(s=None):
    """Settings with the key masked -- safe to log or return from a route."""
    s = s or settings()
    key = s["api_key"] or ""
    masked = (key[:5] + "..." + key[-3:]) if len(key) > 10 else ("<set>" if key else "<unset>")
    return {"base_url": s["base_url"], "model": s["model"], "dim": s["dim"],
            "api_key": masked, "batch": s["batch"], "configured": bool(key)}


def _post(base, path, key, payload, timeout):
    req = urllib.request.Request(
        base + path,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def embed_texts(texts, s=None, batch=None, retries=2):
    """Embed a list of strings; one vector per string, in the given order."""
    s = s or settings()
    if not s["api_key"]:
        raise EmbedUnavailable("no embedding api key: set EMBED_API_KEY or EMBED_CONFIG")
    texts = list(texts)
    if not texts:
        return []
    batch = batch or s["batch"]
    out = []
    for start in range(0, len(texts), batch):
        chunk = [t if isinstance(t, str) and t.strip() else " "
                 for t in texts[start:start + batch]]
        payload = {"model": s["model"], "input": chunk, "encoding_format": "float"}
        last = None
        for attempt in range(retries + 1):
            try:
                data = _post(s["base_url"], "/embeddings", s["api_key"], payload, s["timeout"])
                items = sorted((data.get("data") or []), key=lambda d: d.get("index", 0))
                vecs = [d.get("embedding") for d in items]
                if len(vecs) != len(chunk) or any(not isinstance(v, list) or not v for v in vecs):
                    raise EmbedUnavailable(
                        "endpoint returned %d vectors for %d inputs" % (len(vecs), len(chunk)))
                if s["dim"] and len(vecs[0]) != int(s["dim"]):
                    raise EmbedUnavailable(
                        "model %s returned dim=%d, EMBED_DIM says %d"
                        % (s["model"], len(vecs[0]), s["dim"]))
                out.extend(vecs)
                last = None
                break
            except EmbedUnavailable:
                raise
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read(200).decode("utf-8", "replace")
                except Exception:
                    pass
                last = EmbedUnavailable("HTTP %s from %s: %s" % (exc.code, s["base_url"], body))
            except Exception as exc:  # noqa: BLE001 - any transport failure
                last = EmbedUnavailable("%s: %s" % (type(exc).__name__, exc))
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
        if last is not None:
            raise last
    return out



_RERANK_STATE = {"last": 0.0}


def rerank_texts(query, docs, model=None, s=None, retries=3):
    """Score (query, document) pairs with a cross-encoder; best first.

    Returns [(index, relevance_score)] sorted by score, or raises
    EmbedUnavailable so the caller can keep the fusion order and say that it did.

    A hosted reranker rate-limits a tight loop -- 112 back-to-back calls earned a
    429 during measurement. Calls are spaced by RERANK_INTERVAL, and 429/503 are
    retried with a widening backoff instead of surfacing as an outage on the
    first rejection.
    """
    s = s or settings()
    if not s["api_key"]:
        raise EmbedUnavailable("no api key for rerank: set EMBED_API_KEY or EMBED_CONFIG")
    docs = [d if isinstance(d, str) and d.strip() else " " for d in (docs or [])]
    if not docs:
        return []
    model = model or os.environ.get("RERANK_MODEL") or "bge-reranker-v2-m3"
    try:
        timeout = float(os.environ.get("RERANK_TIMEOUT") or s["timeout"])
    except Exception:
        timeout = 20.0
    try:
        gap = float(os.environ.get("RERANK_INTERVAL") or "0.35")
    except Exception:
        gap = 0.35

    payload = {"model": model, "query": query, "documents": docs}
    last = None
    for attempt in range(retries + 1):
        wait = gap - (time.time() - _RERANK_STATE["last"])
        if wait > 0:
            time.sleep(wait)
        _RERANK_STATE["last"] = time.time()
        try:
            data = _post(s["base_url"], "/rerank", s["api_key"], payload, timeout)
            out = [(int(it.get("index", 0)), float(it.get("relevance_score") or 0.0))
                   for it in (data.get("results") or [])]
            out.sort(key=lambda x: (-x[1], x[0]))
            return out
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read(200).decode("utf-8", "replace")
            except Exception:
                pass
            if exc.code in (429, 503) and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            last = EmbedUnavailable("HTTP %s from rerank: %s" % (exc.code, body))
        except Exception as exc:
            last = EmbedUnavailable("%s: %s" % (type(exc).__name__, exc))
        if attempt < retries:
            time.sleep(0.8 * (attempt + 1))
    raise last

def embed_query(text, s=None):
    vecs = embed_texts([text], s=s, batch=1)
    if not vecs:
        raise EmbedUnavailable("empty embedding for query")
    return vecs[0]


def main():
    s = settings()
    print("settings:", json.dumps(describe(s), ensure_ascii=False))
    try:
        t0 = time.time()
        vecs = embed_texts(["read a file from disk",
                            "\u5217\u51fa\u672c\u673a\u6b63\u5728\u8fd0\u884c\u7684 python \u8fdb\u7a0b"], s=s)
        print("shapes:", [(len(v), round(sum(x * x for x in v) ** 0.5, 4)) for v in vecs])
        print("elapsed_ms:", int((time.time() - t0) * 1000))
    except EmbedUnavailable as exc:
        print("UNAVAILABLE:", exc)
    print("OK")


if __name__ == "__main__":
    main()
