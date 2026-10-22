"""vectorstore.py -- stdlib-only lexical + vector index for toolmarket resources.

No third-party dependencies: sqlite3, math, json, re, struct only. BM25 works
today. Dense vectors are stored, normalized and brute-forced; they are worth
filling only once an embedding backend is wired up outside the toolmarket
process (that interpreter has no numpy).

Design:
  - one sqlite file per deployment, WAL, one writer (the toolmarket service).
  - content_hash gates incremental reindex: unchanged text is never re-tokenized.
  - vectors are normalized on insert, so cosine is a plain dot product.
  - an empty collection returns [], it never raises.
  - collections are pinned to (dim, model_id): mixing embedding models is refused.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import time

_WORD = re.compile("[a-z0-9_]+")
_CJK = re.compile("[" + chr(0x4E00) + "-" + chr(0x9FFF) + "]")

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections(
  name TEXT PRIMARY KEY,
  dim INTEGER,
  metric TEXT NOT NULL DEFAULT 'cosine',
  model_id TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS vectors(
  resource_id TEXT PRIMARY KEY,
  collection TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL,
  norm REAL NOT NULL,
  meta TEXT,
  content_hash TEXT,
  model_id TEXT,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_collection ON vectors(collection);
CREATE TABLE IF NOT EXISTS lexical(
  resource_id TEXT PRIMARY KEY,
  collection TEXT NOT NULL,
  tf TEXT NOT NULL,
  doclen INTEGER NOT NULL,
  meta TEXT,
  content_hash TEXT,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lexical_collection ON lexical(collection);
"""


def tokenize(text):
    """Lowercase word tokens plus single CJK characters.

    An identifier is also emitted as its parts. `mcp_x_pdf_merge` is one word to
    `_WORD`, so it tokenized to a single opaque term and a query for `pdf` shared
    nothing with it -- BM25 scored 0 and, on a shelf reached without the embedder
    (a second process, a rebuilt index), search returned nothing for a document
    that was sitting right there. Measured 2026-09-17: resource
    `mcp_x_pdf_merge`, query `pdf merge`, `bm25_search` -> [].

    The stem is kept as well as the parts, so an exact-name query (`mcp_x_pdf_merge`
    typed in full) still ranks above the generic `merge`. Duplicates are dropped
    because token frequency is what the scorer reads.
    """
    if not text:
        return []
    low = str(text).lower()
    out = _WORD.findall(low)
    for word in list(out):
        if "_" in word:
            parts = [p for p in word.split("_") if p]
            if len(parts) > 1:
                out.extend(parts)
    out.extend(_CJK.findall(low))
    return out


# Bump when `tokenize` changes what a document becomes. The stored row holds a
# *pre-computed term frequency*, not the text, so a tokenizer fix does not reach
# documents already indexed: they keep the old tokens and stay wrong, silently,
# while every hash comparison says they are current. Versioning the hash is what
# turns that into a reindex.
TOKENIZER_VERSION = 2


def hash_text(text):
    return hashlib.sha256(
        ("%d:%s" % (TOKENIZER_VERSION, text or "")).encode("utf-8")).hexdigest()[:16]


def pack_vector(vec):
    return struct.pack("<%df" % len(vec), *[float(x) for x in vec])


def unpack_vector(blob):
    n = len(blob) // 4
    return list(struct.unpack("<%df" % n, blob))


def build_resource_text(rec):
    """Weighted text for one toolmarket resource: name x3, tags x2, then prose."""
    if not isinstance(rec, dict):
        return ""
    parts = []
    name = rec.get("name") or ""
    if name:
        parts.append((name + " ") * 3)
    meta = rec.get("metadata") or {}
    tags = meta.get("tags") or []
    if tags:
        parts.append((" ".join(str(t) for t in tags) + " ") * 2)
    desc = rec.get("description") or ""
    if desc:
        parts.append(desc)
    params = ((rec.get("contract") or {}).get("parameters") or {}).get("properties") or {}
    for pname, pspec in params.items():
        parts.append(str(pname))
        if isinstance(pspec, dict) and pspec.get("description"):
            parts.append(str(pspec["description"]))
    return " ".join(parts)


def rrf_fuse(rankings, k=60, limit=None):
    """Reciprocal rank fusion. No weights to tune without a labelled set."""
    agg = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            rid = item[0]
            agg[rid] = agg.get(rid, 0.0) + 1.0 / (k + rank)
    out = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:limit] if limit else out


class Store:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        d = os.path.dirname(self.path)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        # check_same_thread=False: the index is shared by a FastAPI app whose
        # synchronous routes run in a worker thread, so the connection is used from
        # a different thread than the one that made it. Without this, every route
        # that touches the index raises ProgrammingError and returns a 500 -- which
        # is exactly what /search did. Serialization is left to SQLite itself: WAL
        # plus busy_timeout below is what makes that safe here.
        self.conn = sqlite3.connect(self.path, timeout=15.0, isolation_level=None,
                                    check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=15000")
        self.conn.executescript(SCHEMA)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def create_collection(self, name, dim=None, metric="cosine", model_id=None):
        row = self.conn.execute(
            "SELECT dim, metric, model_id FROM collections WHERE name=?", (name,)
        ).fetchone()
        if row is not None:
            # A collection that never declared a dimension adopts the first one it
            # is handed, and likewise for the model. After that both are pinned, so
            # a second model can never be mixed into one vector space unnoticed.
            if dim is not None and row[0] is None:
                self.conn.execute("UPDATE collections SET dim=? WHERE name=?",
                                  (int(dim), name))
                row = (int(dim), row[1], row[2])
            if dim is not None and row[0] is not None and int(dim) != int(row[0]):
                raise ValueError(
                    "collection %r is dim=%s, refusing dim=%s" % (name, row[0], dim)
                )
            if model_id is not None and row[2] is None:
                self.conn.execute("UPDATE collections SET model_id=? WHERE name=?",
                                  (model_id, name))
                row = (row[0], row[1], model_id)
            if model_id is not None and row[2] not in (None, model_id):
                raise ValueError(
                    "collection %r is pinned to model_id=%r, refusing %r"
                    % (name, row[2], model_id)
                )
            return False
        self.conn.execute(
            "INSERT INTO collections(name, dim, metric, model_id, created_at) VALUES(?,?,?,?,?)",
            (name, dim, metric, model_id, time.time()),
        )
        return True
    def list_collections(self):
        cur = self.conn.execute(
            "SELECT name, dim, metric, model_id, created_at FROM collections ORDER BY name"
        )
        return [
            {"name": r[0], "dim": r[1], "metric": r[2], "model_id": r[3], "created_at": r[4]}
            for r in cur.fetchall()
        ]

    def drop_collection(self, name):
        n = self.conn.execute(
            "SELECT COUNT(*) FROM lexical WHERE collection=?", (name,)
        ).fetchone()[0]
        m = self.conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE collection=?", (name,)
        ).fetchone()[0]
        self.conn.execute("DELETE FROM lexical WHERE collection=?", (name,))
        self.conn.execute("DELETE FROM vectors WHERE collection=?", (name,))
        self.conn.execute("DELETE FROM collections WHERE name=?", (name,))
        return {"lexical": n, "vectors": m}

    def stats(self, collection=None):
        if collection:
            lx = self.conn.execute(
                "SELECT COUNT(*) FROM lexical WHERE collection=?", (collection,)
            ).fetchone()[0]
            dv = self.conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection=?", (collection,)
            ).fetchone()[0]
        else:
            lx = self.conn.execute("SELECT COUNT(*) FROM lexical").fetchone()[0]
            dv = self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        return {"lexical": lx, "vectors": dv}

    def count(self, collection=None):
        return self.stats(collection)["lexical"]

    def needs_reindex(self, collection, resource_id, chash):
        row = self.conn.execute(
            "SELECT content_hash FROM lexical WHERE resource_id=? AND collection=?",
            (resource_id, collection),
        ).fetchone()
        return row is None or row[0] != chash

    def index_text(self, collection, resource_id, text, meta=None, chash=None):
        """Tokenize and store one document. Skips the write when unchanged."""
        chash = chash or hash_text(text)
        if not self.needs_reindex(collection, resource_id, chash):
            return False
        toks = tokenize(text)
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        self.conn.execute(
            "INSERT OR REPLACE INTO lexical(resource_id, collection, tf, doclen, meta,"
            " content_hash, updated_at) VALUES(?,?,?,?,?,?,?)",
            (resource_id, collection, json.dumps(tf, ensure_ascii=False), len(toks),
             json.dumps(meta or {}, ensure_ascii=False), chash, time.time()),
        )
        return True

    def index_texts(self, collection, rows, skip_unchanged=True):
        """rows: iterable of (resource_id, text) or (resource_id, text, meta)."""
        written = 0
        self.conn.execute("BEGIN")
        try:
            for row in rows:
                rid, text = row[0], row[1]
                meta = row[2] if len(row) > 2 else None
                chash = hash_text(text)
                if skip_unchanged and not self.needs_reindex(collection, rid, chash):
                    continue
                toks = tokenize(text)
                tf = {}
                for t in toks:
                    tf[t] = tf.get(t, 0) + 1
                self.conn.execute(
                    "INSERT OR REPLACE INTO lexical(resource_id, collection, tf, doclen, meta,"
                    " content_hash, updated_at) VALUES(?,?,?,?,?,?,?)",
                    (rid, collection, json.dumps(tf, ensure_ascii=False), len(toks),
                     json.dumps(meta or {}, ensure_ascii=False), chash, time.time()),
                )
                written += 1
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return written

    def _lexical_rows(self, collection):
        return self.conn.execute(
            "SELECT resource_id, tf, doclen, meta FROM lexical WHERE collection=?"
            " ORDER BY resource_id",
            (collection,),
        ).fetchall()

    def bm25_search(self, collection, query, k=5, k1=1.5, b=0.75):
        rows = self._lexical_rows(collection)
        if not rows:
            return []
        qterms = tokenize(query)
        if not qterms:
            return []
        doc_tf = [(r[0], json.loads(r[1]), r[2], r[3]) for r in rows]
        n_docs = len(doc_tf)
        avgdl = (sum(d[2] for d in doc_tf) / float(n_docs)) or 1.0
        df = {}
        for term in set(qterms):
            df[term] = sum(1 for d in doc_tf if term in d[1])
        scored = []
        for rid, tf, dl, meta in doc_tf:
            score = 0.0
            for term in qterms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl))
            if score > 0.0:
                scored.append((rid, score, json.loads(meta) if meta else None))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def index_vector(self, collection, resource_id, vector, meta=None, chash=None, model_id=None):
        vec = [float(x) for x in vector]
        dim = len(vec)
        if dim == 0:
            raise ValueError("empty vector for %r" % resource_id)
        col = self.conn.execute(
            "SELECT dim, metric, model_id FROM collections WHERE name=?", (collection,)
        ).fetchone()
        if col is None:
            raise ValueError("no such collection: %r" % collection)
        if col[0] is not None and int(col[0]) != dim:
            raise ValueError(
                "collection %r expects dim=%s, got dim=%d for %r"
                % (collection, col[0], dim, resource_id)
            )
        if model_id is not None and col[2] not in (None, model_id):
            raise ValueError("collection %r is pinned to model_id=%r" % (collection, col[2]))
        norm = math.sqrt(sum(x * x for x in vec))
        unit = [x / norm for x in vec] if norm > 0 else vec
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors(resource_id, collection, dim, vec, norm, meta,"
            " content_hash, model_id, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (resource_id, collection, dim, pack_vector(unit), norm,
             json.dumps(meta or {}, ensure_ascii=False), chash, model_id, time.time()),
        )
        return True

    def dense_search(self, collection, vector, k=5):
        col = self.conn.execute(
            "SELECT dim, metric FROM collections WHERE name=?", (collection,)
        ).fetchone()
        metric = (col[1] if col else "cosine") or "cosine"
        q = [float(x) for x in vector]
        rows = self.conn.execute(
            "SELECT resource_id, vec, meta FROM vectors WHERE collection=?", (collection,)
        ).fetchall()
        if not rows:
            return []
        if col and col[0] is not None and len(q) != int(col[0]):
            raise ValueError("query dim=%d, collection %r dim=%s" % (len(q), collection, col[0]))
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        qu = [x / qn for x in q]
        scored = []
        for rid, blob, meta in rows:
            v = unpack_vector(blob)
            if len(v) != len(qu):
                continue
            dot = 0.0
            for a, c in zip(qu, v):
                dot += a * c
            score = dot if metric != "l2" else -math.sqrt(max(0.0, 2.0 - 2.0 * dot))
            scored.append((rid, score, json.loads(meta) if meta else None))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def search(self, collection, query=None, vector=None, k=5, mode="bm25", rrf_k=60):
        if mode == "bm25":
            return self.bm25_search(collection, query, k=k) if query else []
        if mode == "dense":
            return self.dense_search(collection, vector, k=k) if vector else []
        if mode == "hybrid":
            rankings = []
            if query:
                rankings.append(self.bm25_search(collection, query, k=max(k * 5, 50)))
            if vector:
                rankings.append(self.dense_search(collection, vector, k=max(k * 5, 50)))
            if not rankings:
                return []
            if len(rankings) == 1:
                return rankings[0][:k]
            fused = rrf_fuse(rankings, k=rrf_k, limit=k)
            meta = {}
            for ranking in rankings:
                for rid, _s, m in ranking:
                    meta.setdefault(rid, m)
            return [(rid, score, meta.get(rid)) for rid, score in fused]
        raise ValueError("unknown mode: %r" % mode)

    def get(self, resource_id, collection=None):
        if collection:
            row = self.conn.execute(
                "SELECT collection, doclen, meta, content_hash, updated_at FROM lexical"
                " WHERE resource_id=? AND collection=?",
                (resource_id, collection),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT collection, doclen, meta, content_hash, updated_at FROM lexical"
                " WHERE resource_id=?",
                (resource_id,),
            ).fetchone()
        if row is None:
            return None
        return {"resource_id": resource_id, "collection": row[0], "doclen": row[1],
                "meta": json.loads(row[2]) if row[2] else None, "content_hash": row[3],
                "updated_at": row[4]}

    def delete(self, resource_id, collection=None):
        if collection:
            a = self.conn.execute(
                "DELETE FROM lexical WHERE resource_id=? AND collection=?", (resource_id, collection)
            ).rowcount
            c = self.conn.execute(
                "DELETE FROM vectors WHERE resource_id=? AND collection=?", (resource_id, collection)
            ).rowcount
        else:
            a = self.conn.execute(
                "DELETE FROM lexical WHERE resource_id=?", (resource_id,)
            ).rowcount
            c = self.conn.execute(
                "DELETE FROM vectors WHERE resource_id=?", (resource_id,)
            ).rowcount
        return {"lexical": a, "vectors": c}


    def health(self, collection=None, records=None, expected_model=None):
        """Cheap self-check for silent corruption.

        Everything here is derived from what is on disk rather than from what was
        supposed to be written: a count that merely echoes the write path would
        pass even when the write path is the thing that is broken.
        """
        problems = []
        warnings = []
        counts = {}
        for col in self.list_collections():
            name = col["name"]
            if collection and name != collection:
                continue
            dim = col["dim"]
            n_lex = self.conn.execute(
                "SELECT COUNT(*) FROM lexical WHERE collection=?", (name,)).fetchone()[0]
            n_vec = self.conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection=?", (name,)).fetchone()[0]
            info = {"lexical": n_lex, "vectors": n_vec, "dim": dim, "model_id": col["model_id"]}
            if dim is None and n_vec:
                problems.append("%s: %d vectors stored but the collection never declared a dim"
                                % (name, n_vec))
            if dim is not None and expected_model and col["model_id"] not in (None, expected_model):
                problems.append("%s: vectors are from model %r, expected %r"
                                % (name, col["model_id"], expected_model))
            bad_len = off_norm = zero = 0
            for _rid, blob in self.conn.execute(
                    "SELECT resource_id, vec FROM vectors WHERE collection=?", (name,)):
                if dim is not None and len(blob) != int(dim) * 4:
                    bad_len += 1
                    continue
                v = unpack_vector(blob)
                s = math.sqrt(sum(x * x for x in v))
                if not v or s == 0.0:
                    zero += 1
                elif abs(s - 1.0) > 0.02:
                    off_norm += 1
            if bad_len:
                problems.append("%s: %d vectors have the wrong byte length for dim=%s"
                                % (name, bad_len, dim))
            if off_norm:
                problems.append("%s: %d vectors are not unit-normalized" % (name, off_norm))
            if zero:
                problems.append("%s: %d vectors are zero-length" % (name, zero))
            orphan_v = self.conn.execute(
                "SELECT COUNT(*) FROM vectors v LEFT JOIN lexical l"
                " ON v.resource_id = l.resource_id WHERE l.resource_id IS NULL").fetchone()[0]
            orphan_l = self.conn.execute(
                "SELECT COUNT(*) FROM lexical l LEFT JOIN vectors v"
                " ON v.resource_id = l.resource_id"
                " WHERE v.resource_id IS NULL AND l.collection = ?", (name,)).fetchone()[0]
            info["vectors_without_lexical"] = orphan_v
            info["lexical_without_vector"] = orphan_l
            if orphan_v:
                warnings.append("%s: %d vectors have no lexical row" % (name, orphan_v))
            if orphan_l and dim is not None:
                warnings.append("%s: %d lexical rows have no vector; dense search misses them"
                                % (name, orphan_l))
            if orphan_l and dim is None and n_lex >= 10:
                # Every document indexed, no vector space declared: the collection
                # was created without a dimension, so it can never accept a vector
                # and the dense leg is dead permanently while search still answers.
                # This is a problem, not a hint -- it is the exact state that made
                # a whole shelf score as if retrieval were simply weak.
                n_vec_docs = n_lex - orphan_l
                if n_vec_docs <= 0:
                    problems.append(
                        "%s: %d documents indexed but no vector space (dim is unset); "
                        "search is lexical-only until the collection is recreated with a dim"
                        % (name, n_lex))
            if records is not None:
                by_id = {}
                for r in records:
                    rid = r.get("id") or r.get("name")
                    if rid:
                        by_id[rid] = hash_text(build_resource_text(r))
                stale = not_on_shelf = 0
                for rid, chash in self.conn.execute(
                        "SELECT resource_id, content_hash FROM lexical WHERE collection=?", (name,)):
                    cur = by_id.get(rid)
                    if cur is None:
                        not_on_shelf += 1
                    elif chash != cur:
                        stale += 1
                info["resources_not_on_shelf"] = not_on_shelf
                info["stale_documents"] = stale
                if stale:
                    problems.append("%s: %d documents changed since indexing; sync is behind"
                                    % (name, stale))
                if not_on_shelf:
                    warnings.append("%s: %d indexed resources are no longer on the shelf"
                                    % (name, not_on_shelf))
            counts[name] = info
        return {"ok": not problems, "problems": problems, "warnings": warnings, "counts": counts}


    def sync_from_records(self, collection, records, prune=True):
        """Incremental index of toolmarket resource dicts. Returns a report."""
        self.create_collection(collection)
        seen = set()
        rows = []
        for rec in records:
            rid = rec.get("id") or rec.get("name")
            if not rid:
                continue
            seen.add(rid)
            rows.append((rid, build_resource_text(rec),
                         {"name": rec.get("name"), "state": rec.get("state")}))
        written = self.index_texts(collection, rows)
        pruned = 0
        if prune:
            for r in self._lexical_rows(collection):
                if r[0] not in seen:
                    self.delete(r[0], collection)
                    pruned += 1
        return {"seen": len(seen), "written": written, "skipped": len(rows) - written,
                "pruned": pruned}


if __name__ == "__main__":
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(prefix="vsdemo_"), "index.sqlite")
    docs = [
        ("tool:read_file", "read file contents from disk and print them to stdout"),
        ("tool:run_python", "execute a python source string on the host with a timeout"),
        ("tool:agent_bus_append", "append one json message line to an agent bus board file"),
        ("tool:git_push", "push local commits to a remote git repository"),
        ("tool:query_bus", "read messages addressed to an agent from a shared board"),
    ]
    with Store(tmp) as st:
        st.create_collection("demo")
        print("first pass:", st.index_texts("demo", docs))
        print("collections:", st.list_collections())
        print("stats:", st.stats("demo"))
        for q in ["read a file from disk", "send a message to another agent"]:
            print("Q:", q)
            for rid, score, meta in st.search("demo", query=q, k=3):
                print("   %-26s %.4f" % (rid, score))
        print("empty collection query:", st.search("nope", query="x", k=3))
        print("second pass (unchanged):", st.index_texts("demo", docs))
        print("stats after reindex:", st.stats("demo"))
    print("OK")
