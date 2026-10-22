"""`POST /discover`: the shelf's door out to the public MCP registry.

Tested with the upstream call stubbed, because what matters here is the
*contract of the route* -- what it reports, what it refuses, how it behaves when
upstream is off or absent -- not the registry's current contents (which move).
The live path is proven separately: a real run on 2026-09-17 registered 8 pdf
servers from the registry and `/search` ranked them afterwards.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from toolmarket import mcp_import
from toolmarket.api.main import create_app
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore


@pytest.fixture()
def client(tmp_path):
    reg = ResourceRegistry(ResourceStore(str(tmp_path / "discover.db")))
    return TestClient(create_app(reg)), reg


def _fake_find(monkeypatch, recs, report=None):
    """Stand in for the registry walk; returns the ids like the real one does."""
    def fake(reg, query, **kw):
        ids = []
        for rec in recs:
            if reg.get(rec.id) is None:
                reg.save(rec)
            ids.append(rec.id)
        base = {"query": query, "found": len(ids), "entries": len(ids),
                "inserted": len(ids), "ids": ids, "error": None}
        base.update(report or {})
        return base

    monkeypatch.setattr(mcp_import, "find_servers", fake)


def _remote(name):
    return mcp_import.to_record({
        "name": name, "description": "does a thing",
        "version": "1.0.0", "remotes": [{"url": "https://e.example/mcp"}]})


def _package(name):
    return mcp_import.to_record({
        "name": name, "description": "a package", "version": "1.0.0",
        "packages": [{"registryType": "npm", "identifier": "@x/y"}]})


def test_discover_is_a_post_because_it_writes(client):
    """/discover registers resources, so a GET would mutate on every retry."""
    c, _ = client
    assert c.get("/discover").status_code == 405


def test_empty_query_is_refused(client):
    c, _ = client
    assert c.post("/discover", json={"q": "   "}).status_code == 422


def test_discover_reports_found_and_inserted_separately(monkeypatch, client):
    c, reg = client
    recs = [_remote("x/one"), _remote("x/two")]
    _fake_find(monkeypatch, recs, {"inserted": 0})
    body = c.post("/discover", json={"q": "widget", "limit": 5}).json()
    # found > 0 with inserted == 0 must read as "we already hold these".
    assert body["found"] == 2 and body["inserted"] == 0
    assert body["note"] == "hit the shelf already held"
    assert body["count"] == 2


def test_discover_marks_which_hits_are_actually_callable(monkeypatch, client):
    c, _ = client
    _fake_find(monkeypatch, [_remote("x/remote"), _package("x/pkg")])
    results = {r["name"]: r for r in
               c.post("/discover", json={"q": "widget"}).json()["results"]}
    assert results["mcp_x_remote"]["callable"] is True
    assert results["mcp_x_remote"]["requires_launch"] is False
    assert results["mcp_x_pkg"]["callable"] is False
    assert results["mcp_x_pkg"]["requires_launch"] is True


def test_discover_keeps_the_origin_name_and_remotes(monkeypatch, client):
    """The handle is folded; the origin is what a human and the registry share."""
    c, _ = client
    _fake_find(monkeypatch, [_remote("io.github.owner/repo")])
    r = c.post("/discover", json={"q": "widget"}).json()["results"][0]
    assert r["name"] == "mcp_io_github_owner_repo"
    assert r["origin"] == "io.github.owner/repo"
    assert r["remotes"] == ["https://e.example/mcp"]


def test_nothing_is_promoted_by_discovery(monkeypatch, client):
    c, reg = client
    _fake_find(monkeypatch, [_remote("x/remote")])
    c.post("/discover", json={"q": "widget"})
    assert [r.state.value for r in reg.list()] == ["draft"]


def test_discovery_can_be_switched_off_without_touching_the_network(
        monkeypatch, client):
    c, _ = client
    called = {"n": 0}

    def boom(*a, **kw):
        called["n"] += 1
        raise AssertionError("upstream must not be reached")

    monkeypatch.setattr(mcp_import, "find_servers", boom)
    monkeypatch.setenv("TOOLMARKET_DISCOVER", "off")
    body = c.post("/discover", json={"q": "widget"}).json()
    assert body["enabled"] is False and body["found"] == 0
    assert called["n"] == 0


def test_upstream_failure_is_reported_not_raised(monkeypatch, client):
    c, reg = client
    _fake_find(monkeypatch, [], {"found": 0, "inserted": 0,
                                 "error": "TimeoutError: read timed out"})
    body = c.post("/discover", json={"q": "widget"}).json()
    assert body["count"] == 0 and "timed out" in body["error"]
    assert reg.list() == []


def test_what_discovery_finds_is_visible_to_search(monkeypatch, client):
    """The measured bug this guards: 8 found, 8 inserted, /search still 0.

    A discovery path whose results the next lookup cannot see is a path that
    found nothing -- the caller's very next move is `/search`, and it has to
    work.
    """
    c, _ = client
    _fake_find(monkeypatch, [_remote("x/PDF-merge")])
    c.post("/discover", json={"q": "pdf merge"})
    hits = c.get("/search", params={"q": "pdf merge"}).json()
    assert hits["count"] >= 1
    assert any(r["name"] == "mcp_x_pdf_merge" for r in hits["results"])
