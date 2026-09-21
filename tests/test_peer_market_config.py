"""Where the peer list comes from, and what happens when two places disagree.

Measured 2026-09-21: the operator's other machine was one `peers.json` edit away
from being visible, the edit was made, and nothing changed -- because the
environment value wins and a value in the environment is invisible to anyone
looking at the file. Three configuration sources with a silent precedence order
is the same failure this whole lookup exists to prevent (a setting that is not
in effect looks exactly like a setting that is), one level down.

The tests pin the property that matters: a configured peer is never discarded
without saying so.
"""
from __future__ import annotations

import json

import pytest

from autoforge.agent import ForgeAgent


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.delenv("AUTOFORGE_PEER_MARKETS", raising=False)
    monkeypatch.setattr(ForgeAgent, "_peer_market_from_registry",
                        staticmethod(lambda: ""), raising=False)
    return tmp_path


def _file(home, peers):
    (home / "peers.json").write_text(json.dumps({"peers": peers}), encoding="utf-8")


def test_the_file_is_used_when_nothing_is_in_the_environment(home):
    _file(home, {"kos": "http://192.168.1.108:8000"})
    assert ForgeAgent._peer_market_urls() == [("kos", "http://192.168.1.108:8000")]
    assert ForgeAgent._peer_market_spec_with_source()[1] == "file"


def test_a_peer_named_only_by_the_file_is_not_discarded(home, monkeypatch):
    """The regression: the environment used to *replace* the file silently."""
    _file(home, {"kos": "http://192.168.1.108:8000", "lab": "http://10.0.0.5:8000"})
    monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "win=http://10.1.1.1:8000")
    got = dict(ForgeAgent._peer_market_urls())
    assert got["win"] == "http://10.1.1.1:8000"
    assert got["kos"] == "http://192.168.1.108:8000", "the file's peer was silently dropped"
    assert got["lab"] == "http://10.0.0.5:8000"


def test_the_environment_wins_on_a_label_clash(home, monkeypatch):
    _file(home, {"kos": "http://192.168.1.108:8000"})
    monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "kos=http://10.9.9.9:8000")
    assert ForgeAgent._peer_market_urls() == [("kos", "http://10.9.9.9:8000")]


def test_an_explicitly_empty_environment_means_none(home, monkeypatch):
    """Not "unset": empty is how a test, or a machine that wants no peers, says so.

    Falling through to the file here would make whether the suite passes depend
    on whose machine it ran on -- the leak tests/conftest.py exists to close.
    """
    _file(home, {"kos": "http://192.168.1.108:8000"})
    monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "")
    assert ForgeAgent._peer_market_urls() == []


def test_no_configuration_anywhere_is_not_an_error(home):
    assert ForgeAgent._peer_market_urls() == []


def test_the_default_market_is_not_its_own_peer(home, monkeypatch):
    """Double-counting one shelf as both "the market" and "a peer".

    With TOOLMARKET_URL set to another machine -- which is exactly how a node
    joins a shared shelf -- the same address gets named in peers.json too. The
    lookup would then ask that one shelf twice, report its tool once as on the
    shelf and once as on a peer, and pay the per-peer timeout twice.
    """
    _file(home, {"kos": "http://192.168.1.108:8000", "lab": "http://10.0.0.5:8000"})
    monkeypatch.setenv("TOOLMARKET_URL", "http://192.168.1.108:8000")
    got = ForgeAgent._peer_market_urls()
    assert [lab for lab, _ in got] == ["lab"], got


def test_a_peer_behind_a_token_file_is_still_deduplicated(home, monkeypatch):
    _file(home, {"kos": "http://192.168.1.108:8077/market|FILE:C:/t.token"})
    monkeypatch.setenv("TOOLMARKET_URL", "http://192.168.1.108:8077/market")
    assert ForgeAgent._peer_market_urls() == []
