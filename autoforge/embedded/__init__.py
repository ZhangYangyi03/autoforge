"""toolmarket, shipped inside autoforge.

Why it moved in
---------------
The market was a sibling checkout. Everything autoforge does with it -- pushing
a freshly forged tool, asking the shelf whether a need is already served, the
`market_sync` field on the ledger -- then depended on a directory that existed
on one machine. On a fresh clone the market was simply absent, and absence read
as "nothing published yet" rather than as "the package is not installed".

So the source lives at the repo root (`toolmarket/`, joined by git subtree so
its 39 commits came with it) and this module is the one place that knows how to
reach it. Two consequences, both easy to undo by accident later:

  * The path is *computed*, not written down. The old launcher hard coded
    D:\\Users\\china\\Desktop\\... which works for one person until they move a
    folder.
  * The market is importable *only* inside `importable()`. Putting that directory
    on sys.path at import time would shadow a pip-installed `toolmarket` for the
    whole process, including code that never asked for this one.

Nothing here imports fastapi or uvicorn at module level: those are the `market`
extra, and a user who never runs `auto market` should not need them.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Iterator

__all__ = ["source_root", "available", "missing_reason", "importable",
           "serve", "app", "store_url", "publish"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def source_root() -> Path:
    """The directory that *contains* the `toolmarket` package."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "toolmarket",   # <repo>/autoforge/embedded -> <repo>/toolmarket
        here.parents[1],                  # a checkout that put toolmarket beside autoforge/
        Path.cwd() / "toolmarket",
    ]
    for candidate in candidates:
        if (candidate / "toolmarket" / "__init__.py").is_file():
            return candidate
    return candidates[0]


def installed_root() -> Path | None:
    """Where `toolmarket` already lives, if it is importable at all.

    Two ways this module succeeds and they are different situations: the source
    tree is in the repo (a checkout), or the package was installed -- either
    from this wheel, which now carries it, or from PyPI through the `market`
    extra. A wheel install has no sibling directory to walk up into, so without
    this branch `auto market` would report a missing tree on a machine where the
    market is one import away.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("toolmarket")
    except Exception:                                     # noqa: BLE001
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent.parent


def available() -> bool:
    if (source_root() / "toolmarket" / "__init__.py").is_file():
        return True
    return installed_root() is not None


def missing_reason() -> str:
    if available():
        return ""
    return ("toolmarket is neither beside this autoforge (%s) nor installed. "
            "pip install 'autoforge-agent[market]', or run from a checkout."
            % source_root())


@contextlib.contextmanager
def importable() -> Iterator[Path]:
    """Put the embedded tree first on sys.path for the duration of the block.

    First, and restored afterwards: a pip-installed toolmarket has to keep
    winning outside the block, or importing autoforge would swap the market out
    from under a process that installed its own.
    """
    if not available():
        raise ImportError(missing_reason())
    if not (source_root() / "toolmarket" / "__init__.py").is_file():
        # Importable from site-packages: nothing to add to sys.path, and adding
        # a directory that does not hold the package would shadow the copy that
        # works with one that does not exist.
        yield installed_root() or source_root()
        return
    root = str(source_root())
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        yield Path(root)
    finally:
        if added:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def store_url(explicit: str | None = None) -> str:
    """The store the console opens with.

    `:memory:` is the right default for a test and the wrong one for a shelf
    nobody restarts twice: the second start lists zero tools, and that reads as
    "the other agents never published anything" rather than as "the database was
    in RAM". So the default is a file beside the ledger; TOOLMARKET_STORE and
    DATABASE_URL still override it.
    """
    if explicit:
        return explicit
    env = os.environ.get("TOOLMARKET_STORE") or os.environ.get("DATABASE_URL")
    if env:
        return env
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    path = Path(base) / "autoforge" / "toolmarket.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.as_posix()


def app(*, store: str | None = None, seed_demo: bool = False) -> Any:
    """The market's FastAPI app, built from the embedded tree."""
    with importable():
        from toolmarket.registry import ResourceRegistry
        from toolmarket.store import make_store
        from toolmarket.web import build_app

        registry = ResourceRegistry(make_store(store_url(store)))
        return build_app(registry, seed_demo=seed_demo)


def serve(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
          store: str | None = None, open_browser: bool = False,
          seed_demo: bool = False) -> None:
    """Run the market console in this process. Blocks."""
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("the market console needs the extra:\n"
                         "    pip install 'autoforge-agent[market]'")
    application = app(store=store, seed_demo=seed_demo)
    url = f"http://{host}:{port}/"
    print(f"\n  autoforge market  ->  {url}")
    print(f"  store : {store_url(store)}")
    print("  ctrl-c to stop\n")
    if open_browser:
        import threading
        import time
        import webbrowser

        def _open() -> None:
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(application, host=host, port=port, log_level="warning")


def publish(spec: Any, *, url: str | None = None) -> dict[str, Any]:
    """Put one already-forged tool on the shelf, idempotently.

    This is the repair path, not the forge path: `_sync_to_market` runs during a
    forge and is allowed to fail, which leaves tools that work and are invisible.
    `auto market push` walks the ledger and calls this, so a night of failed
    syncs is fixed by one command instead of by re-forging everything.

    Look before writing, for the same reason the forge does: a duplicate POST
    returns 409, and a 409 read as an error would make the repair look broken on
    exactly the tools that are already fine.
    """
    import json as _json
    import urllib.error
    import urllib.request

    base = (url or os.environ.get("TOOLMARKET_URL") or "http://127.0.0.1:8000").rstrip("/")
    payload = {
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.parameters,
        "code": spec.code,
        "source": spec.source or "generated",
        "generator": spec.generator or "",
        "invariances": list(getattr(spec, "invariances", None) or []),
        "effect_signature": getattr(spec, "effect_signature", "") or "",
        "tags": list(getattr(spec, "tags", None) or []),
    }
    try:
        with urllib.request.urlopen(f"{base}/resources/tool:{spec.name}", timeout=8) as resp:
            if 200 <= resp.status < 300:
                return {"ok": True, "note": "already there"}
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return {"ok": False, "error": "lookup HTTP %s" % exc.code}
    except Exception as exc:                              # noqa: BLE001
        return {"ok": False, "error": ("%s: %s" % (type(exc).__name__, exc))[:160]}
    try:
        req = urllib.request.Request(
            base + "/resources", data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            return {"ok": 200 <= resp.status < 300, "status": resp.status}
    except Exception as exc:                              # noqa: BLE001
        if "409" in str(exc):
            return {"ok": True, "note": "already there (raced the lookup)"}
        return {"ok": False, "error": ("%s: %s" % (type(exc).__name__, exc))[:160]}
