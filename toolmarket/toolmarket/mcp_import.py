'''Import MCP servers off the public registry onto the shelf.

Why this exists
---------------
`mcp_facade.py` answers the first half of the question -- other agents can now
*read* our shelf. This is the other direction: our shelf can read theirs. The
official MCP registry (`registry.modelcontextprotocol.io`) is a cursor-paged
JSON list of servers, one entry per (name, version), and every entry is
something an agent might one day need.

Three things this module is careful about
----------------------------------------
1. It does not lie about what it imported. A registry entry is a *pointer*: a
   remote URL, or a package to launch (npx/uvx/docker). Only the first kind can
   be reached from here without executing somebody else's code, so only the
   first kind gets a contract with code in it. A package entry is imported with
   an empty contract and `effect_signature="unbound"` -- visible, searchable,
   and honestly marked not-yet-runnable. Importing 10k entries where 9k pretend
   to be callable would poison `/search` for everything else on the shelf.

2. It is resumable and idempotent. The registry's cursor is a version stamp, not
   an offset, so a half-finished import must be able to continue where it
   stopped and a re-run over an already-imported page must be a no-op. Both are
   `--cursor` plus a per-id existence check.

3. It never promotes anything. Everything lands in DRAFT. "A stranger's server
   exists" and "this agent may run it" are different claims; the second one is
   the trust policy's, and an importer that wrote ACTIVE would be bypassing the
   one mechanism this codebase has for exactly this danger.

Retries are not decoration. Measured 2026-09-17: ~10s per 100-entry page,
frequent read timeouts under load, and ~39% of the first page was additional
versions of a name already seen -- which is why dedupe is per name, not per row.
'''
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from toolmarket.protocol.resources import (
    ResourceRecord,
    ResourceType,
    ToolContract,
)

REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"
USER_AGENT = "autoforge-toolmarket-importer/0.1"
PAGE_LIMIT = 100
FETCH_TIMEOUT = 25.0
FETCH_ATTEMPTS = 3

# The remote call a bound resource performs. Kept as source text because that is
# what a contract carries: the shelf stores code, not a closure, so a tool
# imported today still runs tomorrow in a process that never saw the import.
# Placeholders are substituted with str.replace rather than str.format, because
# the body is JSON-RPC and its braces are data.
REMOTE_CALL_TEMPLATE = '''def __NAME__(tool="", arguments=None):
    """Call a tool on the MCP server at __URL__, or list its tools.

    One resource per *server*, not per remote tool: the registry lists servers,
    and a server's tool names are only knowable by asking it. An empty `tool`
    returns that list, so the first call is discovery and the second is use.
    The docstring cannot contain a double quote -- it is machine-generated
    source, and a stray quote here produces a SyntaxError in a stored contract
    that nothing would notice until the tool is first called.
    """
    import json, urllib.request
    SEP = chr(10)
    url = "__URL__"
    arguments = arguments or {}
    method = "tools/list" if not tool else "tools/call"
    params = {} if not tool else {"name": tool, "arguments": arguments}
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                       "params": params}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream",
                 "User-Agent": "autoforge-toolmarket/0.2"})
    # The User-Agent is not decoration: several of these servers sit behind
    # Cloudflare and answer the bare "Python-urllib/3.x" with 403 Forbidden --
    # measured 2026-09-17, identical request, only the header differed.
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", "replace")
    try:
        payload = json.loads(raw)
    except ValueError:
        return raw[:4000]
    if "error" in payload:
        return "MCP error: " + json.dumps(payload["error"], ensure_ascii=False)
    result = payload.get("result") or {}
    if method == "tools/list":
        return json.dumps([t.get("name") for t in (result.get("tools") or [])],
                          ensure_ascii=False)
    parts = [c.get("text", "") for c in (result.get("content") or [])
             if isinstance(c, dict)]
    return SEP.join(parts) if parts else json.dumps(result, ensure_ascii=False)
'''

BOUND_SIGNATURE = "network+foreign-code"
UNBOUND_SIGNATURE = "unbound"


def resource_id_for(server_name: str) -> str:
    '''A registry name -> a resource id that is stable and path-safe.

    The registry uses `io.github.owner/repo` and `ac.vendor/thing`. Slashes,
    dots and dashes survive a round trip through the registry but not through a
    shell or a path segment, so they are folded to underscores. The original
    name is kept in provenance; this is only the handle.
    '''
    safe = (server_name or "").strip().lower()
    for ch in ("/", ".", "-", " "):
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return "mcp_" + safe.strip("_")


def remote_urls(entry: dict[str, Any]) -> list[str]:
    out = []
    for r in entry.get("remotes") or []:
        url = (r or {}).get("url")
        if isinstance(url, str) and url.startswith("http"):
            out.append(url)
    return out


def package_commands(entry: dict[str, Any]) -> list[str]:
    '''How a package entry *would* be launched. Reported, never executed.

    Running a stranger's package is a decision for a human or a policy engine,
    not a side effect of an import, so this exists to make the record honest
    about what it is missing rather than to be run.
    '''
    cmds = []
    for p in entry.get("packages") or []:
        reg = (p or {}).get("registryType") or (p or {}).get("registry_name") or "?"
        ident = (p or {}).get("identifier") or "?"
        cmds.append(f"{reg}:{ident}")
    return cmds


def to_record(entry: dict[str, Any]) -> Optional[ResourceRecord]:
    '''One registry entry as a DRAFT resource, or None if it has no name.

    Versioned duplicates collapse here: the id comes from the name alone, so the
    tenth version of a server *is* the first resource, and `import_servers` skips
    it. A remote is bound with code; a package-only entry is imported unbound.
    '''
    name = (entry or {}).get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    rid = resource_id_for(name)
    urls = remote_urls(entry)
    pkgs = package_commands(entry)
    repo = entry.get("repository")

    if urls:
        code = (REMOTE_CALL_TEMPLATE
                .replace("__NAME__", rid)
                .replace("__URL__", urls[0]))
        params = {
            "type": "object",
            "properties": {
                "tool": {"type": "string",
                         "description": "Remote tool name; empty lists them."},
                "arguments": {"type": "object",
                              "description": "Arguments for the remote tool."},
            },
            "required": [],
            "additionalProperties": False,
        }
        effect = BOUND_SIGNATURE
    else:
        code = ""
        params = {"type": "object", "properties": {},
                  "additionalProperties": False}
        effect = UNBOUND_SIGNATURE

    homepage = entry.get("websiteUrl")
    if not homepage and isinstance(repo, dict):
        homepage = repo.get("url")

    return ResourceRecord(
        id=ResourceRecord.make_id(rid),
        type=ResourceType.TOOL,
        name=rid,
        description=(entry.get("description") or entry.get("title") or "")[:500],
        contract=ToolContract(parameters=params, code=code,
                              effect_signature=effect),
        provenance={
            "source": "mcp-registry",
            "generator": "mcp-registry",
            "origin_name": name,
            "version": entry.get("version"),
            "title": entry.get("title"),
            "remotes": urls,
            "packages": pkgs,
            "homepage": homepage,
            "imported_at": time.time(),
        },
        metadata={"schema_source": "declared",
                  "callable": bool(urls),
                  "requires_launch": bool(pkgs) and not urls},
    )


def fetch_page(cursor: Optional[str] = None, limit: int = PAGE_LIMIT,
               url: str = REGISTRY_URL, timeout: float = FETCH_TIMEOUT,
               attempts: int = FETCH_ATTEMPTS, search: Optional[str] = None,
               sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    '''One page, with backoff retries. A dict with `_error` when all failed.

    `search` is server-side: measured 2026-09-17, `?search=pdf` returned pdf
    servers while `?q=` and `?query=` were ignored. It is what makes looking
    something up on demand affordable -- one page instead of a ten-thousand
    entry walk, which is the difference between "ask the registry when you need
    something" and "mirror the registry every night".
    '''
    last: Optional[Exception] = None
    for attempt in range(attempts):
        target = f"{url}?limit={int(limit)}"
        if search:
            target += "&search=" + urllib.parse.quote(str(search))
        if cursor:
            target += "&cursor=" + urllib.parse.quote(cursor)
        try:
            req = urllib.request.Request(
                target, headers={"User-Agent": USER_AGENT,
                                 "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
            if attempt + 1 < attempts:
                sleep(1.5 * (attempt + 1))
    return {"_error": f"{type(last).__name__}: {last}" if last
            else "unknown fetch failure"}


def import_servers(reg: Any, *, limit: int = 0, cursor: Optional[str] = None,
                   pages: int = 0,
                   fetch: Callable[..., dict[str, Any]] = fetch_page,
                   log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    '''Walk the registry and register what is not already on the shelf.

    Stops on the first of: `limit` insertions, `pages` pages fetched, the cursor
    running out, or a page that failed after its retries. The report carries the
    cursor to resume from, which is the only thing a caller needs to continue.
    '''
    inserted = skipped = seen = bound = unbound = 0
    fetched_pages = 0
    errors: list[str] = []
    partial_page = False

    while True:
        if limit and inserted >= limit:
            break
        if pages and fetched_pages >= pages:
            break
        # The cursor we are about to read *from*. Kept because the registry's
        # cursor is a forward-only version stamp: there is no way to resume in
        # the middle of a page, so if `limit` stops us mid-page the only
        # lossless resume point is the start of the page in hand. Re-reading it
        # costs one page and the per-id check makes the repeat a no-op.
        page_cursor_in = cursor
        page = fetch(cursor)
        if not page:
            break
        if page.get("_error"):
            errors.append(str(page["_error"]))
            break
        batch = page.get("servers") or []
        fetched_pages += 1
        for item in batch:
            if limit and inserted >= limit:
                partial_page = True
                cursor = page_cursor_in
                break
            entry = (item or {}).get("server") or {}
            rec = to_record(entry)
            if rec is None:
                continue
            seen += 1
            if reg.get(rec.id) is not None:
                skipped += 1
                continue
            reg.save(rec)
            inserted += 1
            if rec.contract.effect_signature == UNBOUND_SIGNATURE:
                unbound += 1
            else:
                bound += 1
        if partial_page:
            break
        cursor = (page.get("metadata") or {}).get("nextCursor")
        log(f"page {fetched_pages}: seen {seen}, inserted {inserted}, "
            f"skipped {skipped}")
        if not batch or not cursor:
            break

    return {"inserted": inserted, "skipped": skipped, "seen": seen,
            "pages": fetched_pages, "bound": bound, "unbound": unbound,
            "cursor": cursor, "exhausted": cursor is None and not partial_page,
            "partial_page": partial_page, "errors": errors}


def find_servers(reg: Any, query: str, *, limit: int = 25,
                 fetch: Callable[..., dict[str, Any]] = fetch_page,
                 log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    '''Search the registry by keyword and register the hits. Returns the ids.

    This is the on-demand half. A full import is a mirror: it only knows what
    existed the day it ran. A search is a lookup, and it is the reason an agent
    can be told "when you need a tool, go and look" without that being a promise
    to re-read the internet every time.

    It reports `found` and `inserted` separately, because a query whose hits are
    all already on the shelf is a success and must not read like a failure.
    '''
    page = fetch(limit=limit, search=query)
    if page.get("_error"):
        return {"query": query, "found": 0, "entries": 0, "inserted": 0,
                "ids": [], "error": page["_error"]}
    entries = inserted = 0
    ids: list[str] = []
    for item in page.get("servers") or []:
        rec = to_record((item or {}).get("server") or {})
        if rec is None:
            continue
        # The registry returns one row per version, so the same server arrives
        # several times. `entries` is the raw row count; `found` is distinct
        # resources, which is the number a caller can act on -- "found 25" next
        # to a list of 16 ids is the kind of arithmetic a caller has to redo.
        entries += 1
        if rec.id in ids:
            continue
        ids.append(rec.id)
        if reg.get(rec.id) is None:
            reg.save(rec)
            inserted += 1
    found = len(ids)
    log(f"search {query!r}: {entries} rows, {found} distinct, inserted {inserted}")
    return {"query": query, "found": found, "entries": entries,
            "inserted": inserted, "ids": ids, "error": None}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="toolmarket.mcp-import",
        description="Import MCP registry servers onto the shelf as DRAFT "
                    "resources. Resumable: pass back the cursor it prints.")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after this many insertions (0 = no limit)")
    p.add_argument("--pages", type=int, default=0,
                   help="stop after this many registry pages (0 = no limit)")
    p.add_argument("--cursor", default=None, help="resume from this cursor")
    p.add_argument("--search", default=None,
                   help="look up a keyword instead of walking the registry, "
                        "and register whatever it returns")
    p.add_argument("--store", default=None,
                   help="store URL; defaults to TOOLMARKET_STORE")
    args = p.parse_args(argv)

    from toolmarket.registry import ResourceRegistry
    from toolmarket.store import make_store

    reg = ResourceRegistry(make_store(args.store) if args.store else make_store())
    if args.search:
        report = find_servers(reg, args.search, limit=args.limit or 25,
                              log=lambda m: print(m, flush=True))
    else:
        report = import_servers(reg, limit=args.limit, cursor=args.cursor,
                                pages=args.pages,
                                log=lambda m: print(m, flush=True))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
