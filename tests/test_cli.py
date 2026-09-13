"""CLI surface tests: the `auto` entry point must be safe to call anywhere.

The rule that matters: bare `auto` summons the agent on a terminal, but a piped
or cron invocation gets help instead of a hung REPL waiting on stdin.
"""
from __future__ import annotations

import argparse
import io
import json
import threading

import pytest

from autoforge import cli


def _args(**kw) -> argparse.Namespace:
    base = dict(model=None, base_url=None, api_key=None, fast=False,
                max_tokens=None, no_proxy=False, proxy=None)
    base.update(kw)
    return argparse.Namespace(**base)


# -- config resolution --------------------------------------------------
def test_config_reads_env(monkeypatch):
    monkeypatch.delenv("AIPING_API_KEY", raising=False)
    monkeypatch.delenv("AUTOFORGE_API_KEY", raising=False)
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("AUTOFORGE_MODEL", "qwen2.5:7b")
    cfg = cli._config(_args())
    assert cfg["model"] == "qwen2.5:7b" and cfg["proxy"] is False
    assert cfg["key"] == "ollama"          # local endpoints need no key


def test_config_flags_beat_env(monkeypatch):
    monkeypatch.setenv("AUTOFORGE_MODEL", "from-env")
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "http://127.0.0.1:11434/v1")
    cfg = cli._config(_args(model="from-flag"))
    assert cfg["model"] == "from-flag"


def test_missing_key_hint_names_a_variable_that_works(monkeypatch):
    """The advice must not send the reader down a closed path.

    Naming AIPING_API_KEY on a non-aiping endpoint is worse than saying nothing:
    the reader sets it, gets 401, and has no reason to suspect the variable.
    """
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://api.deepseek.com/v1")
    with pytest.raises(SystemExit) as e:
        cli._config(_args())
    assert "AUTOFORGE_API_KEY" in str(e.value) and "AIPING_API_KEY" not in str(e.value)

    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://aiping.cn/api/v1")
    with pytest.raises(SystemExit) as e:
        cli._config(_args())
    assert "AIPING_API_KEY" in str(e.value)


def test_remote_without_key_exits_with_advice(monkeypatch):
    monkeypatch.delenv("AIPING_API_KEY", raising=False)
    monkeypatch.delenv("AUTOFORGE_API_KEY", raising=False)
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://example.invalid/v1")
    with pytest.raises(SystemExit, match="no API key"):
        cli._config(_args())


def test_aiping_key_only_stands_in_for_aiping(monkeypatch):
    """`AIPING_API_KEY` is provider-specific, not a generic fallback.

    A DeepSeek run inheriting it read "Authorization Required" out of a 401 and
    nothing in the output named the credential as the cause — the base_url had
    moved but the key had not. Scope the fallback to the endpoint it belongs to.
    """
    monkeypatch.setenv("AIPING_API_KEY", "QC-aiping-only")

    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://api.deepseek.com/v1")
    with pytest.raises(SystemExit, match="no API key"):
        cli._config(_args())

    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://aiping.cn/api/v1")
    assert cli._config(_args())["key"] == "QC-aiping-only"


def test_autoforge_key_is_provider_agnostic(monkeypatch):
    """`AUTOFORGE_API_KEY` is ours, so it stands in for any endpoint."""
    monkeypatch.delenv("AIPING_API_KEY", raising=False)
    monkeypatch.setenv("AUTOFORGE_API_KEY", "sk-any-provider")
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://api.deepseek.com/v1")
    assert cli._config(_args())["key"] == "sk-any-provider"


def test_fast_flag_or_env_enables_fast(monkeypatch):
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("AUTOFORGE_FAST", "1")
    assert cli._config(_args())["fast"] is True
    monkeypatch.delenv("AUTOFORGE_FAST")
    assert cli._config(_args(fast=True))["fast"] is True


# -- summon semantics --------------------------------------------------
def test_bare_auto_without_tty_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli.main([]) == 0
    assert "usage: autoforge" in capsys.readouterr().out


def test_bare_auto_with_tty_enters_chat(monkeypatch, capsys):
    """On a terminal, `auto` must dispatch to chat, not help."""
    called: list[str] = []
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "cmd_chat", lambda a: called.append("chat") or 0)
    assert cli.main([]) == 0
    assert called == ["chat"]


def test_parser_exposes_every_subcommand():
    names = cli.build_parser()._subparsers._group_actions[0].choices
    assert set(names) == {"chat", "forge", "list", "setup", "config",
                          "web", "run", "modes"}


# -- trace rendering ---------------------------------------------------
def test_trace_renders_forge_lifecycle(capsys):
    agent = argparse.Namespace(trace=[
        {"kind": "call", "tool": "forge_tool"},
        {"kind": "forge_attempt", "need": "normalise ISBNs", "round": 1,
         "accepted": False, "error": "no ISBN found"},
        {"kind": "forge_done", "ok": True, "name": "isbn_normalizer"},
        {"kind": "auto_quarantine", "name": "bad_tool"},
    ])
    cli._show_trace(agent, 0)
    out = capsys.readouterr().out
    assert "forge_tool" in out
    assert "round 1" in out and "no ISBN found" in out
    assert "sealed" in out and "isbn_normalizer" in out
    assert "quarantined" in out and "bad_tool" in out


def test_a_successful_attempt_is_not_announced_twice(capsys):
    """A round that passed says nothing of its own — forge_done reports it."""
    agent = argparse.Namespace(trace=[
        {"kind": "forge_attempt", "need": "x", "round": 1, "accepted": True},
        {"kind": "forge_done", "ok": True, "name": "thing"},
    ])
    cli._show_trace(agent, 0)
    out = capsys.readouterr().out
    assert out.count("thing") == 1
    assert "round" not in out


def test_trace_from_index_skips_earlier_events(capsys):
    agent = argparse.Namespace(trace=[{"kind": "call", "tool": "old"},
                                      {"kind": "call", "tool": "new"}])
    cli._show_trace(agent, 1)
    out = capsys.readouterr().out
    assert "new" in out and "old" not in out


# -- list --------------------------------------------------------------
def test_list_prints_artifact(tmp_path, capsys):
    art = tmp_path / "tool.json"
    art.write_text(json.dumps({"name": "t", "code": "def t(): pass"}), encoding="utf-8")
    assert cli.cmd_list(argparse.Namespace(path=str(art))) == 0
    assert '"name": "t"' in capsys.readouterr().out


def test_list_reports_empty_directory(tmp_path, capsys):
    cli.cmd_list(argparse.Namespace(path=str(tmp_path)))
    assert "no artifacts" in capsys.readouterr().out


def test_list_enumerates_a_directory(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps({"name": "alpha"}), encoding="utf-8")
    cli.cmd_list(argparse.Namespace(path=str(tmp_path)))
    assert "alpha" in capsys.readouterr().out


# -- the confirmation gate, on the terminal ------------------------------
#
# The gate in `ToolRegistry.call` asks whoever is attached to the registry. On
# the command line that is this object. It is the difference between a policy
# field that refuses tools and one that only says it does, so its three
# answers get their own tests: yes, no, and nobody-here.
class _FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _cfg(**over) -> dict:
    cfg = dict(model="m", base="http://127.0.0.1:1/v1", key="k", max_tokens=16,
               proxy=False, fast=True, policy="full")
    cfg.update(over)
    return cfg


def test_the_cli_attaches_a_confirmer_to_the_agent(monkeypatch):
    # Wiring that is not checked is wiring that quietly stops existing, and
    # then "off" means nothing again on the one surface the user actually uses.
    monkeypatch.setenv("AUTOFORGE_HOME", ".")
    assert isinstance(cli._build(_cfg()).confirmer, cli._TerminalConfirmer)
    assert isinstance(cli._build_mode(_cfg(policy="supervised"), "minimal").confirmer,
                      cli._TerminalConfirmer)


def test_the_confirmer_is_silent_when_stdin_is_not_a_terminal(monkeypatch):
    # A piped or cron run has nobody at the other end. Answering "yes" there
    # would turn the gate into a formality.
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    assert cli._TerminalConfirmer()("forge_tool", {}, ["may_access_network"]) is None


def test_the_confirmer_does_not_prompt_from_a_worker_thread(monkeypatch, capsys):
    # The web harness runs agents in threads with a browser on the other end.
    # Prompting there would block a request on input nobody can see.
    monkeypatch.setattr("sys.stdin", _FakeTty("y\n"))
    seen: list = []
    t = threading.Thread(
        target=lambda: seen.append(
            cli._TerminalConfirmer()("forge_tool", {}, ["may_access_network"])))
    t.start()
    t.join()
    assert seen == [None]
    assert "confirmation gate" not in capsys.readouterr().out


@pytest.mark.parametrize("typed,expected", [
    ("y\n", True), ("yes\n", True), ("Y\n", True),
    ("\n", False), ("n\n", False), ("maybe\n", False),
])
def test_the_confirmer_defaults_to_no(monkeypatch, typed, expected):
    # Anything that is not an explicit yes is a no. An empty line is the
    # commonest thing a hurried operator types.
    monkeypatch.setattr("sys.stdin", _FakeTty(typed))
    assert cli._TerminalConfirmer()("forge_tool", {}, ["may_access_network"]) is expected


def test_the_confirmer_treats_eof_and_ctrl_c_as_nobody_answering(monkeypatch):
    monkeypatch.setattr("sys.stdin", _FakeTty(""))
    assert cli._TerminalConfirmer()("forge_tool", {}, ["may_access_network"]) is None
    monkeypatch.setattr("sys.stdin", _FakeTty("y\n"))

    def interrupted(*_a, **_kw):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)
    assert cli._TerminalConfirmer()("forge_tool", {}, ["may_access_network"]) is None


def test_the_prompt_names_the_switch_the_tool_and_the_arguments(monkeypatch, capsys):
    # An approval prompt that does not say what it is approving trains its
    # reader to say yes without looking, which is worse than no prompt.
    monkeypatch.setattr("sys.stdin", _FakeTty("n\n"))
    cli._TerminalConfirmer()("forge_tool", {"need": "a csv parser"},
                             ["may_access_network"])
    out = capsys.readouterr().out
    assert "forge_tool" in out
    assert "may_access_network" in out
    assert "a csv parser" in out
