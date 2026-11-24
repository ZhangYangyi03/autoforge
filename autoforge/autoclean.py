"""Hand a crawl result to entity-graph, if it is worth cleaning, without being asked.

The operator's rule (2026-11-24): do not wait to be told to clean -- judge whether the
crawl produced something worth keeping, and clean it when it did.

So this is the hook on the crawl path. It is deliberately small, and deliberately
forgiving in three specific ways:

  it never re-fetches     it takes the result already in hand. A hook that fetched again
                          would double the traffic and, on a live crawl, recurse.
  it never raises         any failure is a one-line note in the ledger and the crawl
                          result is returned unchanged. A cleaning bug must not be able
                          to break fetching.
  it never decides alone  the value judgement lives in entitygraph.auto.judge, next to
                          the cleaner it is guarding, and the reason is recorded either
                          way -- including for the crawls it chose not to keep.

Set AUTOFORGE_AUTOCLEAN=0 to turn it off; ENTITYGRAPH_AUTO_STORE overrides where the
rolling graph is written.
"""
from __future__ import annotations

import os

DEFAULT_ENTITYGRAPH = r"D:\Users\china\Desktop\项目_开发\entity-graph"
_LEDGER = "_autoclean"


def enabled() -> bool:
    return (os.environ.get("AUTOFORGE_AUTOCLEAN", "1") or "1").strip() not in ("0", "false", "no")


def _import_entitygraph():
    import sys
    root = os.environ.get("ENTITYGRAPH_SRC") or DEFAULT_ENTITYGRAPH
    if root and root not in sys.path:
        sys.path.insert(0, root)
    from entitygraph import auto  # type: ignore
    return auto


def after_crawl(result, *, engine: str = ""):
    """Judge and maybe ingest. Returns the verdict, or None when disabled.

    Never raises: the caller is a live fetch and its return value is the crawl.
    """
    if not enabled() or not isinstance(result, dict):
        return None
    try:
        auto = _import_entitygraph()
        verdict = auto.maybe_clean(result, engine=engine)
    except Exception as exc:
        return {"worth": False, "cleaned": False,
                "reason": "autoclean unavailable: %s" % str(exc)[:200]}
    try:
        if isinstance(result, dict):
            result[_LEDGER] = {k: verdict.get(k) for k in
                               ("worth", "cleaned", "reason", "store",
                                "store_nodes_after", "dropped_oldest", "caveats")}
    except Exception:
        pass
    return verdict


def report() -> str:
    """One line about whether the hook is on and where it writes."""
    state = "on" if enabled() else "off (AUTOFORGE_AUTOCLEAN=0)"
    try:
        auto = _import_entitygraph()
        store = os.environ.get("ENTITYGRAPH_AUTO_STORE") or auto.DEFAULT_STORE
        exists = os.path.exists(store)
        return "autoclean %s; store %s (%s)" % (
            state, store, "present" if exists else "not created yet")
    except Exception as exc:
        return "autoclean %s; entitygraph not importable: %s" % (state, str(exc)[:120])


def one_line(verdict) -> str:
    """The one line appended to a crawl report, so the caller sees what was kept."""
    if not verdict:
        return "autoclean: skipped"
    if verdict.get("cleaned"):
        return ("autoclean: kept -- %s | store %d node(s), %d dropped"
                % (verdict.get("reason", ""), verdict.get("store_nodes_after", 0),
                   verdict.get("dropped_oldest", 0)))
    return "autoclean: not kept -- %s" % verdict.get("reason", "no reason recorded")
