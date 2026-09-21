"""The forge must look on the shelf BEFORE it builds, not only after.

`_sync_to_market` has always checked for an existing entry -- but it runs
after the tool already exists. The forge path itself never looked, so the
principle "check the market before forging" lived in the prompt for a day
with zero executions: 158 forges, not one pre-forge lookup. A principle is
not a gate.

These tests pin the gate, not the prose: the lookup happens on the path the
forge takes, it runs before `pipeline.forge`, and -- the subtle half -- a
market that cannot answer is NOT read as "the shelf does not have it".

The gate has two halves, because one of them cannot see the case that
matters. `/resources` answers "is this name already on the shelf"; the
market's hybrid `/search` answers "is this job already on the shelf under
other words". Both are exercised below, along with the noise each half was
measured to produce before it was tightened: description-word coincidences
(`host_artifact_scan` for a question about processes) and prefix matches
(`report` for `repo`).
"""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import urllib.request

from autoforge.agent import ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import MockLLMClient


def _agent() -> ForgeAgent:
    return ForgeAgent(MockLLMClient(), policy=FULL_FREEDOM)


class _Resp:
    def __init__(self, payload):
        raw = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode("utf-8")
        self._b = io.BytesIO(raw)

    def read(self):
        return self._b.read()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _open(payload):
    def _u(req, timeout=None):
        return _Resp(payload)
    return _u


def _router(resources=None, search=None, resources_raise=False, search_raise=False):
    """Answer `/resources` and `/search` separately, as the real market does."""
    def _u(req, timeout=None):
        url = getattr(req, "full_url", None) or str(req)
        if "/search" in url:
            if search_raise:
                raise OSError("connection refused")
            return _Resp(search if search is not None else {"results": []})
        if resources_raise:
            raise OSError("connection refused")
        return _Resp(resources if resources is not None else [])
    return _u


def _hits(pairs):
    return {"results": [{"name": n, "score": s} for n, s in pairs]}


class TestPreLookupRunsAndIsHonest:
    def test_unreachable_market_returns_no_answer_not_no_entry(self, monkeypatch):
        """A dead market must not read as an empty one."""
        def _boom(req, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        out = _agent()._prelookup_market("read the last lines of a board file")
        assert out == "", (
            "an unreachable shelf must not be read as permission to forge: " + repr(out))

    def test_empty_shelf_says_so(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _open([]))
        out = _agent()._prelookup_market("read the last lines of a board file")
        assert "no" in out.lower()
        assert "duplicate" in out.lower()

    def test_overlapping_entry_is_named(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _open([
            {"id": "tool:read_bus_ndjson", "name": "read_bus_ndjson",
             "description": "Read the last N lines of an agent-bus ndjson board file"},
        ]))
        out = _agent()._prelookup_market("read bus ndjson board lines")
        assert "read_bus_ndjson" in out
        assert "Do not re-forge" in out

    def test_lookup_lands_before_the_forge(self, monkeypatch):
        """The point of the whole change: it is on the forge path."""
        monkeypatch.setattr(urllib.request, "urlopen", _open([]))
        a = _agent()
        seen = {}

        def _fake_forge(need, context=None, should_abort=None, **kw):
            seen["context"] = context or ""
            seen["need"] = need
            return SimpleNamespace(ok=False, aborted=False, rounds=1, spec=None)

        a.pipeline = SimpleNamespace(forge=_fake_forge)
        a.registry.get("forge_tool").fn("read the last lines of a board file")
        assert "Tool-market pre-lookup" in seen.get("context", ""), (
            "the shelf was never consulted on the forge path: " + repr(seen))

    def test_a_down_market_does_not_block_the_forge(self, monkeypatch):
        """Best-effort by construction: the gate reports, it never refuses."""
        def _boom(req, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        a = _agent()
        seen = {}

        def _fake_forge(need, context=None, should_abort=None, **kw):
            seen["called"] = True
            seen["context"] = context or ""
            return SimpleNamespace(ok=False, aborted=False, rounds=1, spec=None)

        a.pipeline = SimpleNamespace(forge=_fake_forge)
        out = a.registry.get("forge_tool").fn("read the last lines of a board file")
        assert seen.get("called"), "a dead market stopped the forge"
        assert "pre-lookup" not in seen["context"].lower()


class TestSameNameHalf:
    def test_a_description_coincidence_is_not_a_name_match(self, monkeypatch):
        """`host_artifact_scan` was named for a need about processes.

        The words `running` and `processes` were in its description. That is a
        coincidence of English, not a duplicate, and naming it teaches the agent
        to ignore the gate.
        """
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[{"name": "host_artifact_scan",
                        "description": "Scan the host for running processes, ports and temp files"}],
            search={"results": []}))
        out = _agent()._prelookup_market("count the running processes on this windows machine")
        assert "host_artifact_scan" not in out
        assert "no entry" in out.lower() or "not a duplicate" in out.lower()

    def test_stopwords_do_not_count_toward_the_threshold(self, monkeypatch):
        """`and`/`the`/`with` are how two unrelated names reach two words."""
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[{"name": "inspect_repo_the_and_gitignore_dir",
                        "description": "x"}],
            search={"results": []}))
        out = _agent()._prelookup_market("transcode the video and report the bitrate")
        assert "inspect_repo" not in out

    def test_a_prefix_is_not_a_word(self, monkeypatch):
        """`repo` must not match `report`, which is how it named a gitignore tool."""
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[{"name": "inspect_repo_gitignore_and_systemdrive_dir",
                        "description": "Report on a repository's ignore file"}],
            search={"results": []}))
        out = _agent()._prelookup_market("transcode a video with ffmpeg and report the bitrate")
        assert "inspect_repo" not in out

    def test_two_shared_words_in_the_name_are_a_match(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[{"name": "list_agent_processes",
                        "description": "List the agent processes on this host"}],
            search={"results": []}))
        out = _agent()._prelookup_market("list the agent processes running here")
        assert "list_agent_processes" in out
        assert "(name)" in out

    def test_an_abbreviation_is_left_to_the_semantic_half(self, monkeypatch):
        """`win` is not `window`. Pinned as a known, covered-elsewhere gap.

        The name half answers "is this name the need's"; it is not a
        spell-checker, and stretching it into one is how `repo` matched
        `report`. The semantic half is what catches this, and asserting the gap
        here means the day someone widens the rule they see what it was
        protecting.
        """
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[{"name": "list_agent_processes_win",
                        "description": "List agent processes on Windows"}],
            search={"results": []}))
        out = _agent()._prelookup_market("count the running processes on this windows machine")
        assert "list_agent_processes_win" not in out

    def test_the_whole_name_verbatim_always_matches(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[{"name": "read_bus_ndjson", "description": "x"}],
            search={"results": []}))
        out = _agent()._prelookup_market("read_bus_ndjson")
        assert "read_bus_ndjson" in out


class TestSameJobHalf:
    """The half `/resources` cannot see: the same job under different words."""

    def test_a_synonym_with_no_shared_name_words_is_named(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[{"name": "list_agent_processes_win", "description": "unrelated text"}],
            search=_hits([("list_agent_processes_win", 0.99), ("tail_board_file", 0.97)])))
        out = _agent()._prelookup_market("count the running processes on this windows machine")
        assert "list_agent_processes_win" in out
        assert "synonym" in out

    def test_a_low_score_also_ran_is_not_reported_as_a_synonym(self, monkeypatch):
        """Measured on the live shelf: correct answer 0.10, runner-up 0.038."""
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[], search=_hits([("list_agent_processes_win", 0.1003),
                                        ("powershell_run", 0.0154)])))
        out = _agent()._prelookup_market("count the running processes on this windows machine")
        assert "list_agent_processes_win" in out
        assert "powershell_run" not in out

    def test_the_markets_own_confidence_verdict_is_obeyed(self, monkeypatch):
        """It says `confident: false`; taking it at its word is the honest read."""
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[],
            search={"results": [{"name": "scan_skills_library", "score": 0.9}],
                    "confidence": {"confident": False, "threshold": 0.01}}))
        out = _agent()._prelookup_market("transcode a video with ffmpeg")
        assert "scan_skills_library" not in out

    def test_search_down_is_a_partial_answer_not_silence(self, monkeypatch):
        """`/resources` answered. That is an answer, and a partial one."""
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[], search_raise=True))
        out = _agent()._prelookup_market("transcode a video with ffmpeg and report the bitrate")
        assert out != "", "a partial answer must not collapse to no answer"
        assert "/search" in out
        assert "cannot be ruled out" in out

    def test_both_halves_down_is_still_no_answer(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources_raise=True, search_raise=True))
        assert _agent()._prelookup_market("anything at all") == ""

    def test_a_malformed_search_payload_does_not_raise(self, monkeypatch):
        """The market is another process; its failures are not ours to crash on."""
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[], search={"results": "not a list"}))
        out = _agent()._prelookup_market("read the last lines of a board file")
        assert isinstance(out, str) and out

    def test_the_ledger_records_both_lookups(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _router(
            resources=[], search=_hits([("tail_board_file", 0.9)])))
        a = _agent()
        a._prelookup_market("read the last lines of a board file")
        kinds = {e.get("kind") for e in a.trace}
        assert "market_prelookup" in kinds
        assert "market_semantic_lookup" in kinds


class TestUpstreamHalf:
    """The third leg: the shelf is 153 resources, the world is not.

    `/resources` finds the same name, `/search` finds the same job under other
    words -- both only over what *we* hold. On 2026-09-17 that made "not on the
    shelf" and "does not exist" the same sentence, which is how an agent forges
    what it could have used. This leg asks the market to look in the public MCP
    registry (`POST /discover`) and reports what came back.
    """

    @staticmethod
    def _router(resources, search, discover):
        def _u(req, timeout=None):
            url = getattr(req, "full_url", None) or str(req)
            if "/discover" in url:
                return _Resp(discover)
            if "/search" in url:
                return _Resp(search)
            return _Resp(resources)
        return _u

    def test_it_is_asked_only_when_the_shelf_has_nothing(self, monkeypatch):
        """A shelf that already answered yes does not need the internet."""
        calls = {"discover": 0}

        def _u(req, timeout=None):
            url = getattr(req, "full_url", None) or str(req)
            if "/discover" in url:
                calls["discover"] += 1
                return _Resp({"results": []})
            if "/search" in url:
                return _Resp(_hits([("tail_board_file", 0.9)]))
            return _Resp([{"name": "tail_board_file", "description": "x"}])

        monkeypatch.setattr(urllib.request, "urlopen", _u)
        _agent()._prelookup_market("read the last lines of a board file")
        assert calls["discover"] == 0

    def test_upstream_hits_are_reported_as_draft_not_as_callable_tools(self, monkeypatch):
        discover = {"found": 2, "inserted": 2, "results": [
            {"name": "mcp_x_pdf", "state": "draft", "callable": True,
             "requires_launch": False, "description": "merge pdfs",
             "origin": "x/pdf"},
            {"name": "mcp_x_pkg", "state": "draft", "callable": False,
             "requires_launch": True, "description": "a package",
             "origin": "x/pkg"}]}
        monkeypatch.setattr(urllib.request, "urlopen",
                            self._router([], {"results": []}, discover))
        out = _agent()._prelookup_market("merge two pdf files into one")
        assert "mcp_x_pdf" in out and "mcp_x_pkg" in out
        assert "public MCP registry" in out
        # The two kinds must not read the same: one is callable, one is a pointer.
        assert "callable over the network" in out
        assert "needs its package launched" in out
        assert "DRAFT" in out
        assert "Do not re-forge" in out

    def test_a_need_is_reduced_to_one_content_word(self, monkeypatch):
        """The registry's search is conjunctive -- measured 2026-09-17:
        "pdf merge" and "github issues" returned 0, "pdf" and "github" returned
        8 each. A sentence has to become a word before it is worth sending."""
        seen = []

        def _u(req, timeout=None):
            url = getattr(req, "full_url", None) or str(req)
            if "/discover" in url:
                seen.append(json.loads(req.data.decode("utf-8"))["q"])
                return _Resp({"results": []})
            if "/search" in url:
                return _Resp({"results": []})
            return _Resp([])

        monkeypatch.setattr(urllib.request, "urlopen", _u)
        _agent()._prelookup_market("count the running processes on this windows box")
        assert seen, "the registry was never asked"
        assert seen[0] == "processes", (
            "the longest content word is the informative one: " + repr(seen))
        assert all(" " not in q for q in seen)

    def test_it_sends_at_most_two_queries(self, monkeypatch):
        seen = []

        def _u(req, timeout=None):
            url = getattr(req, "full_url", None) or str(req)
            if "/discover" in url:
                seen.append(json.loads(req.data.decode("utf-8"))["q"])
                return _Resp({"results": []})
            if "/search" in url:
                return _Resp({"results": []})
            return _Resp([])

        monkeypatch.setattr(urllib.request, "urlopen", _u)
        _agent()._prelookup_market(
            "transcode an enormous video file with hardware acceleration")
        assert 1 <= len(seen) <= 2, seen

    def test_a_need_with_no_content_word_does_not_query(self, monkeypatch):
        calls = {"n": 0}

        def _u(req, timeout=None):
            url = getattr(req, "full_url", None) or str(req)
            if "/discover" in url:
                calls["n"] += 1
            if "/search" in url:
                return _Resp({"results": []})
            return _Resp([])

        monkeypatch.setattr(urllib.request, "urlopen", _u)
        out = _agent()._prelookup_market("do it")
        assert calls["n"] == 0
        assert "no entry" in out.lower() or "not a duplicate" in out.lower()

    def test_a_market_without_the_endpoint_is_not_recorded_as_found_nothing(self, monkeypatch):
        """A 404 body parses cleanly. Reading it as an answer would turn
        "this market is too old" into "the world has nothing"."""
        monkeypatch.setattr(urllib.request, "urlopen", self._router(
            [], {"results": []}, {"detail": "Not Found"}))
        a = _agent()
        assert a._discover_upstream("merge two pdf files into one") == []
        events = [e for e in a.trace if e.get("kind") == "market_discover"]
        assert events and events[0].get("ok") is False
        assert "malformed" in str(events[0].get("error", "")) or \
               "results" in str(events[0].get("error", ""))

    def test_upstream_failure_leaves_the_no_entry_answer_intact(self, monkeypatch):
        def _u(req, timeout=None):
            url = getattr(req, "full_url", None) or str(req)
            if "/discover" in url:
                raise OSError("connection refused")
            if "/search" in url:
                return _Resp({"results": []})
            return _Resp([])

        monkeypatch.setattr(urllib.request, "urlopen", _u)
        out = _agent()._prelookup_market("merge two pdf files into one")
        assert out != ""
        assert "no entry" in out.lower() or "not a duplicate" in out.lower()

    def test_the_ledger_records_what_upstream_answered(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", self._router(
            [], {"results": []},
            {"found": 1, "inserted": 1, "results": [
                {"name": "mcp_x_pdf", "state": "draft", "callable": True,
                 "description": "merge pdfs", "origin": "x/pdf"}]}))
        a = _agent()
        a._discover_upstream("merge two pdf files into one")
        events = [e for e in a.trace if e.get("kind") == "market_discover"]
        assert events and events[0]["ok"] is True
        assert events[0]["found"] == 1 and events[0]["inserted"] == 1
        assert events[0]["hits"] == ["mcp_x_pdf"]

# ---------------------------------------------------------------------------
# A shelf on this host is one node of the market, not the market.
# ---------------------------------------------------------------------------

def _peer_router(markets: dict[str, dict]):
    """Answer each peer URL with its own canned shelf.

    Keyed by host:port so one address can be live and another refused in the
    same test -- which is the whole point: "my shelf is down" and "the peer is
    down" are different situations and must not print the same.
    """
    def _u(req, timeout=None):
        url = getattr(req, "full_url", None) or str(req)
        for key, canned in markets.items():
            if key in url:
                if canned is None:
                    raise OSError("connection refused")
                return _Resp(canned)
        raise OSError("connection refused")
    return _u


class TestPeerShelves:
    def test_peer_hit_is_reported_as_a_peer_with_its_name(self, monkeypatch):
        """A hit on another machine must name the machine, not 404 quietly.

        The lookup answered "is this already made" about one host out of two
        until a sibling on the same LAN turned out to carry the tool. A peer hit
        is not the same fact as a local hit -- the tool has to be called *there*
        -- so the verdict carries the peer label through to the model.
        """
        agent = _agent()
        monkeypatch.setenv("TOOLMARKET_URL", "http://127.0.0.1:8000")
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "kos=http://10.9.9.9:8000")
        monkeypatch.setattr(urllib.request, "urlopen", _peer_router({
            "127.0.0.1:8000": [],                                  # local shelf: empty
            "10.9.9.9:8000": _hits([("seconds_to_hms", 4.69), ("hms_many", 4.2)]),
        }))
        out = agent._prelookup_market("convert seconds into HH:MM:SS")
        assert "peer machine's shelf" in out
        assert "seconds_to_hms" in out and "kos" in out
        assert "Do not re-forge" in out

    def test_a_dead_local_shelf_does_not_skip_a_live_peer(self, monkeypatch):
        """Found by testing this against a dead local shelf.

        The peer lookup was written *after* the `if neither lookup came back:
        return ""` line, so a host whose own shelf was down returned "no answer"
        without ever asking the peer that was up. Same class of bug as the one
        this whole file exists for: two different states collapsed into one
        sentence because of where a line happened to sit.
        """
        agent = _agent()
        monkeypatch.setenv("TOOLMARKET_URL", "http://127.0.0.1:8001")
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "kos=http://10.9.9.9:8000")
        monkeypatch.setattr(urllib.request, "urlopen", _peer_router({
            "127.0.0.1:8001": None,                                # local: refused
            "10.9.9.9:8000": _hits([("seconds_to_hms", 4.69)]),
        }))
        out = agent._prelookup_market("convert seconds into HH:MM:SS")
        assert out, "a live peer must be enough to answer, even with the local shelf down"
        assert "seconds_to_hms" in out

    def test_peer_answered_but_carries_nothing_licenses_the_forge_and_says_which_half_is_missing(self, monkeypatch):
        """"Searched and empty" is an answer; "unreachable" is not.

        And when only the peer answered, the verdict must say the local half is
        missing rather than let one machine's answer read as the whole market's.
        """
        agent = _agent()
        monkeypatch.setenv("TOOLMARKET_URL", "http://127.0.0.1:8001")
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "kos=http://10.9.9.9:8000")
        monkeypatch.setattr(urllib.request, "urlopen", _peer_router({
            "127.0.0.1:8001": None,
            "10.9.9.9:8000": {"results": [], "confidence": {"confident": False}},
        }))
        out = agent._prelookup_market("parse a parquet file into a dataframe")
        assert out and "not a duplicate" in out
        assert "local half" in out, "must not present a one-machine answer as a whole-market one"

    def test_both_shelves_unreachable_is_still_no_answer(self, monkeypatch):
        """The rule the file was built on, extended to peers: two dead machines
        are not an empty world."""
        agent = _agent()
        monkeypatch.setenv("TOOLMARKET_URL", "http://127.0.0.1:8001")
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "kos=http://10.9.9.9:8000")
        monkeypatch.setattr(urllib.request, "urlopen", _peer_router({
            "127.0.0.1:8001": None, "10.9.9.9:8000": None}))
        assert agent._prelookup_market("convert seconds into HH:MM:SS") == ""

    def test_no_peers_configured_means_no_peer_requests(self, monkeypatch):
        """Silence, not a LAN sweep. An unconfigured forge must make exactly the
        requests it made before peers existed."""
        agent = _agent()
        monkeypatch.setenv("TOOLMARKET_URL", "http://127.0.0.1:8000")
        monkeypatch.delenv("AUTOFORGE_PEER_MARKETS", raising=False)
        seen = []

        def _u(req, timeout=None):
            seen.append(getattr(req, "full_url", str(req)))
            return _Resp([])

        monkeypatch.setattr(urllib.request, "urlopen", _u)
        agent._prelookup_market("convert seconds into HH:MM:SS")
        assert all("127.0.0.1:8000" in u for u in seen), seen

    def test_peer_token_is_read_from_a_file_and_sent_as_a_bearer(self, monkeypatch, tmp_path):
        """The `|FILE:` form exists so a token need not live in an env var.

        A peer reached through the node's proxy needs the node's bearer token,
        and env vars are visible to every process. The file form is what makes
        that path usable without widening who can read the secret.
        """
        token_file = tmp_path / "node.token"
        token_file.write_text("s3cret-token\n", encoding="utf-8")
        agent = _agent()
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS",
                           "win=http://10.9.9.9:8077/market|FILE:%s" % token_file)
        sent = {}

        def _u(req, timeout=None):
            sent["auth"] = req.get_header("Authorization")
            sent["url"] = req.full_url
            return _Resp({"results": []})

        monkeypatch.setattr(urllib.request, "urlopen", _u)
        agent._peer_lookup("anything")
        assert sent["auth"] == "Bearer s3cret-token"
        assert "/market/search" in sent["url"]

    def test_peer_config_parses_labels_urls_and_bare_urls(self, monkeypatch):
        agent = _agent()
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS",
                           "kos=http://192.168.1.108:8000, http://10.0.0.5:8000 ,")
        assert agent._peer_market_urls() == [
            ("kos", "http://192.168.1.108:8000"),
            ("http://10.0.0.5:8000", "http://10.0.0.5:8000"),
        ]
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "")
        assert agent._peer_market_urls() == []

    def test_a_slow_peer_is_not_an_absent_peer(self, monkeypatch):
        """The bug a real peer found: a cold hybrid search takes 22 seconds.

        Measured on the live shelf: /search with the cross-encoder rerank is
        22.4s cold and 1.7s warm. A five-second timeout did not fail to answer
        the question -- it failed to *ask* it, and then reported the silence as
        "no answer", which is the one conflation this whole lookup exists to
        prevent. The fix is a timeout long enough to ask, and a breaker that
        treats a timeout (there, busy) differently from a refusal (not there).
        """
        import time as _t

        agent = _agent()
        monkeypatch.setenv("TOOLMARKET_URL", "http://127.0.0.1:8000")

        def _slow(req, timeout=None):
            assert timeout is not None and timeout >= 20, (
                "a peer search must be allowed to be slow: got timeout=%r" % timeout)
            _t.sleep(0.05)
            return _Resp({"results": [{"name": "seconds_to_hms", "score": 4.69}],
                          "confidence": {"confident": True}})

        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "kos=http://10.9.9.9:8000")
        monkeypatch.setattr(urllib.request, "urlopen", _slow)
        hits, answered = agent._peer_lookup("convert seconds into HH:MM:SS")
        assert answered and hits and hits[0]["name"] == "seconds_to_hms"

    def test_a_peer_that_comes_back_is_asked_again(self, monkeypatch):
        """Recovery, not just backoff -- the half that only shows up over time.

        The rest is the point of the backoff, but a rest that never expires is a
        peer deleted by a temporary failure. A machine that went to sleep is the
        same machine that comes back, and the lookup has to hear it when it does.
        """
        import json as _json
        import time as _t

        agent = _agent()
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "kos=http://10.9.9.7:8000")
        calls = {"n": 0}

        def _u(req, timeout=None):
            calls["n"] += 1
            # Both legs, /search *and* the /resources fallback: a peer that is
            # switched off answers neither, and a stub that answered the second
            # request would be testing a peer that is up.
            if calls["n"] <= 2:
                raise OSError("connection refused")
            class _R:
                status = 200
                def read(self_inner):
                    return _json.dumps({"results": [
                        {"name": "seconds_to_hms", "score": 1.0}],
                        "confidence": {"confident": True}}).encode()
                def __enter__(self_inner): return self_inner
                def __exit__(self_inner, *a): return False
            return _R()

        monkeypatch.setattr(urllib.request, "urlopen", _u)
        assert agent._peer_lookup("seconds") == ([], False), "down = no answer"
        base = "http://10.9.9.7:8000"
        assert base in agent._peer_dead_until, "a failed peer must be rested"
        # Inside the window it must not be asked at all.
        assert agent._peer_lookup("seconds") == ([], False)
        assert calls["n"] == 2, "the resting peer was asked again"
        # Let the rest expire the way time would, and it must be heard.
        agent._peer_dead_until[base] = _t.time() - 1
        hits, answered = agent._peer_lookup("seconds")
        assert calls["n"] == 3, "the peer was never retried"
        assert answered is True and [h["name"] for h in hits] == ["seconds_to_hms"], (hits, answered)

    def test_a_refused_peer_rests_shorter_than_a_timed_out_one(self, monkeypatch):
        """Not-there and there-but-busy deserve different retry intervals."""
        import time as _t

        agent = _agent()
        monkeypatch.setenv("TOOLMARKET_URL", "http://127.0.0.1:8000")
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS",
                           "ref=http://10.9.9.8:8000,slow=http://10.9.9.9:8000")

        def _u(req, timeout=None):
            url = getattr(req, "full_url", str(req))
            if "10.9.9.8" in url:
                raise OSError("connection refused")
            raise TimeoutError("timed out")

        monkeypatch.setattr(urllib.request, "urlopen", _u)
        agent._peer_lookup("anything")
        now = _t.time()
        refused_until = agent._peer_dead_until["http://10.9.9.8:8000"] - now
        slow_until = agent._peer_dead_until["http://10.9.9.9:8000"] - now
        assert refused_until < slow_until, (refused_until, slow_until)

    def test_peer_spec_survives_a_process_that_already_started(self, tmp_path, monkeypatch):
        """setx writes the registry; a RUNNING process never sees it.

        Measured 2026-09-21: the federation check reported
        AUTOFORGE_PEER_MARKETS as unset *after* setx had exported it. A setting
        that only applies at the next launch is silently not in effect -- the
        same shape of failure the lookup exists to prevent. So the spec is also
        read from a file, which a live process can read.
        """
        import json

        monkeypatch.delenv("AUTOFORGE_PEER_MARKETS", raising=False)
        monkeypatch.setattr("autoforge.agent.ForgeAgent._peer_market_from_registry",
                            staticmethod(lambda: ""), raising=False)
        monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
        (tmp_path / "peers.json").write_text(
            json.dumps({"peers": {"kos": "http://192.168.1.108:8000"}}), encoding="utf-8")

        agent = _agent()
        assert agent._peer_market_urls() == [("kos", "http://192.168.1.108:8000")]

    def test_the_environment_wins_the_clash_without_discarding_the_file(
            self, tmp_path, monkeypatch):
        """Precedence yes, silent deletion no.

        This asserted that the environment *replaced* the file, and it was the
        behaviour that lost a real machine: on 2026-09-21 the operator's other
        host was one `peers.json` edit away from being visible, the edit was
        made, and the peer stayed invisible -- because the environment value
        wins and nothing said the file was being ignored. A configured peer that
        is dropped without a word is the same failure the lookup exists to
        prevent, one level down: not-in-effect looks exactly like not-set.

        So the env entry still wins the label it names, and the file's *other*
        peers survive instead of vanishing.
        """
        import json

        monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
        monkeypatch.setattr("autoforge.agent.ForgeAgent._peer_market_from_registry",
                            staticmethod(lambda: ""), raising=False)
        (tmp_path / "peers.json").write_text(
            json.dumps({"peers": {"kos": "http://from-file:8000",
                                  "lab": "http://from-file-lab:8000"}}), encoding="utf-8")
        monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "kos=http://from-env:8000")
        got = dict(_agent()._peer_market_urls())
        assert got["kos"] == "http://from-env:8000", "the environment must win the clash"
        assert got["lab"] == "http://from-file-lab:8000", (
            "a peer configured in the file was dropped silently")

    def test_no_peers_configured_means_no_peer_requests(self, tmp_path, monkeypatch):
        """The gate: with nothing configured the lookup must not call out at all."""
        monkeypatch.delenv("AUTOFORGE_PEER_MARKETS", raising=False)
        monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
        monkeypatch.setattr("autoforge.agent.ForgeAgent._peer_market_from_registry",
                            staticmethod(lambda: ""), raising=False)

        def _boom(*a, **k):
            raise AssertionError("no peer is configured; nothing may be requested")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        agent = _agent()
        assert agent._peer_market_urls() == []
        assert agent._peer_lookup("anything") == ([], False)
