"""Importing the MCP registry: the three properties that make it safe to run.

Tested offline against a fake `fetch`, because the properties worth testing are
about behaviour under a *hostile* registry (duplicates, missing names, an entry
with neither remote nor package, a page that fails after retries), not about the
official one's current contents. The network is exercised by the live run
recorded in the commit message, not here.
"""
from __future__ import annotations

import json

import pytest

from toolmarket import mcp_import as imp
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore


def _entry(name, version="1.0.0", url=None, pkg=None, desc="A server."):
    e = {"name": name, "description": desc, "version": version, "title": name}
    if url:
        e["remotes"] = [{"type": "streamable-http", "url": url}]
    if pkg:
        e["packages"] = [{"registryType": pkg[0], "identifier": pkg[1]}]
    return {"server": e, "_meta": {}}


def _pager(pages):
    """A fake fetch that walks a list of pre-built registry pages."""
    calls = {"n": 0}

    def fetch(cursor=None, **kw):
        idx = calls["n"]
        calls["n"] += 1
        if idx >= len(pages):
            return {"servers": [], "metadata": {}}
        body = pages[idx]
        cursor_out = body.get("cursor")
        return {"servers": body.get("servers", []),
                "metadata": ({"nextCursor": cursor_out} if cursor_out else {}),
                **({"_error": body["_error"]} if body.get("_error") else {})}

    return fetch, calls


@pytest.fixture()
def reg():
    return ResourceRegistry(ResourceStore(":memory:"))


# -- ids --------------------------------------------------------------------
def test_id_is_stable_and_folded_from_the_registry_name():
    assert imp.resource_id_for("io.github.owner/repo") == "mcp_io_github_owner_repo"
    assert imp.resource_id_for("ac.inference.sh/mcp") == "mcp_ac_inference_sh_mcp"
    # Two spellings that only differ by case must not become two resources.
    assert imp.resource_id_for("Foo.Bar/Baz") == imp.resource_id_for("foo.bar/baz")


def test_same_name_at_two_versions_is_one_resource(reg):
    fetch, _ = _pager([
        {"servers": [_entry("a/one"), _entry("a/one", version="2.0.0")],
         "cursor": "c1"},
        {"servers": [_entry("a/two")]},
    ])
    rep = imp.import_servers(reg, fetch=fetch)
    assert rep["inserted"] == 2
    assert rep["skipped"] == 1
    assert len(reg.list()) == 2


# -- honesty ----------------------------------------------------------------
def test_remote_entry_is_bound_and_carries_runnable_code(reg):
    fetch, _ = _pager([{"servers": [_entry("x/remote", url="https://example.com/mcp")]}])
    imp.import_servers(reg, fetch=fetch)
    rec = reg.get("tool:mcp_x_remote")
    assert rec.contract.effect_signature == "network+foreign-code"
    assert rec.metadata["callable"] is True
    # The stored code must be real, importable Python that names the url.
    ns = {}
    exec(compile(rec.contract.code, "<contract>", "exec"), ns)
    assert "https://example.com/mcp" in rec.contract.code
    assert callable(ns[rec.name])


def test_package_only_entry_is_unbound_and_says_so(reg):
    fetch, _ = _pager([{"servers": [_entry("x/pkg", pkg=("npm", "@x/pkg"))]}])
    imp.import_servers(reg, fetch=fetch)
    rec = reg.get("tool:mcp_x_pkg")
    assert rec.contract.effect_signature == "unbound"
    assert rec.contract.code == ""
    assert rec.metadata["callable"] is False
    assert rec.metadata["requires_launch"] is True
    assert rec.provenance["packages"] == ["npm:@x/pkg"]


def test_nothing_is_imported_as_active(reg):
    fetch, _ = _pager([{"servers": [_entry("x/a", url="https://e.com/mcp"),
                                    _entry("x/b", pkg=("npm", "b"))]}])
    imp.import_servers(reg, fetch=fetch)
    assert {r.state.value for r in reg.list()} == {"draft"}


def test_entry_without_a_name_is_skipped_not_fatal(reg):
    fetch, _ = _pager([{"servers": [{"server": {"description": "no name"}},
                                    _entry("x/ok", url="https://e.com/mcp")]}])
    rep = imp.import_servers(reg, fetch=fetch)
    assert rep["inserted"] == 1
    assert len(reg.list()) == 1


# -- resumability -----------------------------------------------------------
def test_report_carries_the_cursor_to_resume_from(reg):
    fetch, _ = _pager([{"servers": [_entry("x/a", url="https://e.com/mcp")],
                        "cursor": "c42"},
                       {"servers": [_entry("x/b", url="https://e.com/mcp")]}])
    rep = imp.import_servers(reg, fetch=fetch, pages=1)
    assert rep["pages"] == 1 and rep["cursor"] == "c42" and not rep["exhausted"]
    fetch2, _ = _pager([{"servers": [_entry("x/b", url="https://e.com/mcp")]}])
    rep2 = imp.import_servers(reg, fetch=fetch2, cursor=rep["cursor"])
    assert rep2["exhausted"] is True
    assert len(reg.list()) == 2


def test_rerunning_the_same_page_inserts_nothing(reg):
    page = {"servers": [_entry("x/a", url="https://e.com/mcp")]}
    fetch, _ = _pager([page])
    first = imp.import_servers(reg, fetch=fetch)
    fetch2, _ = _pager([page])
    second = imp.import_servers(reg, fetch=fetch2)
    assert first["inserted"] == 1 and second["inserted"] == 0
    assert second["skipped"] == 1


def test_limit_stops_early_and_resumes_losslessly(reg):
    """The cursor must point at a page start, never past a resource we skipped.

    Stopping mid-page and handing back that page's *outgoing* cursor would lose
    every entry after the one that tripped the limit -- silently, because the
    next run would start cleanly one page later.
    """
    page1 = {"servers": [_entry("x/a", url="https://e.com/mcp"),
                         _entry("x/b", url="https://e.com/mcp")],
             "cursor": "c1"}
    page2 = {"servers": [_entry("x/c", url="https://e.com/mcp")]}
    fetch, _ = _pager([page1, page2])
    rep = imp.import_servers(reg, fetch=fetch, limit=1)
    assert rep["inserted"] == 1
    assert rep["partial_page"] is True
    assert rep["exhausted"] is False, "a partial stop is not the end of the shelf"
    assert rep["cursor"] is None, "resume must re-read the page we stopped inside"

    # Resuming from there picks up b and c, and does not duplicate a.
    fetch2, _ = _pager([page1, page2])
    rep2 = imp.import_servers(reg, fetch=fetch2, cursor=rep["cursor"])
    assert rep2["inserted"] == 2 and rep2["skipped"] == 1
    assert sorted(r.name for r in reg.list()) == ["mcp_x_a", "mcp_x_b", "mcp_x_c"]


# -- failure -----------------------------------------------------------------
def test_a_failed_page_is_reported_and_the_shelf_keeps_what_it_got(reg):
    fetch, _ = _pager([{"servers": [_entry("x/a", url="https://e.com/mcp")],
                        "cursor": "c1"},
                       {"_error": "TimeoutError: read timed out"}])
    rep = imp.import_servers(reg, fetch=fetch)
    assert rep["inserted"] == 1
    assert rep["errors"] and "timed out" in rep["errors"][0]
    assert rep["cursor"] == "c1", "a failed page must not advance the cursor"


def test_fetch_page_retries_then_returns_error():
    calls = {"n": 0}

    def boom(url, timeout=None, **kw):
        calls["n"] += 1
        raise TimeoutError("read timed out")

    out = imp.fetch_page(attempts=3, sleep=lambda _s: None)
    # fetch_page is exercised through urllib; here we prove the retry count by
    # monkeypatching urlopen instead.
    import urllib.request as ur
    real = ur.urlopen

    def fake(req, timeout=None):
        return boom(getattr(req, "full_url", req), timeout=timeout)

    ur.urlopen = fake
    try:
        out = imp.fetch_page(attempts=3, sleep=lambda _s: None)
    finally:
        ur.urlopen = real
    assert calls["n"] == 3
    assert "_error" in out and "timed out" in out["_error"]


def test_empty_registry_is_not_an_error(reg):
    fetch, _ = _pager([{"servers": []}])
    rep = imp.import_servers(reg, fetch=fetch)
    assert rep == {**rep, "inserted": 0, "seen": 0, "exhausted": True,
                   "errors": []}


# -- on-demand lookup -------------------------------------------------------
def test_search_registers_hits_and_reports_already_present_ones(reg):
    """A query whose hits are all already on the shelf is a success.

    `found` and `inserted` are separate numbers for this reason: an agent told
    "0 inserted" would conclude we have nothing, when the truth is we have it
    and it was cheap to confirm.
    """
    hits = [_entry("s/pdf", url="https://p.example/mcp", desc="pdf tools"),
            _entry("s/other", url="https://o.example/mcp")]
    fetch, calls = _pager([{"servers": hits}])
    rep = imp.find_servers(reg, "pdf", fetch=fetch)
    assert rep["found"] == 2 and rep["inserted"] == 2 and rep["error"] is None
    assert rep["entries"] == 2
    assert calls["n"] == 1, "a lookup is one page, not a walk"
    again_fetch, _ = _pager([{"servers": hits}])
    rep2 = imp.find_servers(reg, "pdf", fetch=again_fetch)
    assert rep2["found"] == 2 and rep2["inserted"] == 0
    assert len(reg.list()) == 2


def test_search_passes_the_term_to_the_registry_not_a_local_filter():
    seen = {}

    def fetch(cursor=None, **kw):
        seen.update(kw)
        return {"servers": [], "metadata": {}}

    imp.find_servers(None, "pdf", fetch=fetch)
    assert seen.get("search") == "pdf"


def test_fetch_page_puts_search_in_the_query_string():
    import urllib.request as ur
    captured = {}
    real = ur.urlopen

    class _Resp:
        def read(self):
            return b'{"servers": [], "metadata": {}}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        captured["url"] = getattr(req, "full_url", req)
        return _Resp()

    ur.urlopen = fake
    try:
        imp.fetch_page(search="pdf and tools", sleep=lambda _s: None)
    finally:
        ur.urlopen = real
    assert "search=pdf+and+tools" in captured["url"] or \
           "search=pdf%20and%20tools" in captured["url"]


def test_search_failure_is_reported_not_raised(reg):
    fetch, _ = _pager([{"_error": "TimeoutError: read timed out"}])
    rep = imp.find_servers(reg, "pdf", fetch=fetch)
    assert rep["found"] == 0 and "timed out" in rep["error"]
    assert len(reg.list()) == 0


def test_search_counts_distinct_resources_not_version_rows(reg):
    """`found` must be actionable: the id list and the count must agree."""
    rows = [_entry("s/pdf", version="1.0.0", url="https://p.example/mcp"),
            _entry("s/pdf", version="2.0.0", url="https://p.example/mcp"),
            _entry("s/pdf", version="3.0.0", url="https://p.example/mcp"),
            _entry("s/other", url="https://o.example/mcp")]
    fetch, _ = _pager([{"servers": rows}])
    rep = imp.find_servers(reg, "pdf", fetch=fetch)
    assert rep["entries"] == 4
    assert rep["found"] == 2 and len(rep["ids"]) == 2
    assert rep["inserted"] == 2
