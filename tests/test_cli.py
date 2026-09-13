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
                          "web", "run", "modes", "tick"}


def test_every_offered_subcommand_has_a_handler():
    """Regression: `tick` was offered by the parser and absent from dispatch.

    `schedule.wake_command` hands the OS `python -m autoforge tick --quiet`, so
    the mismatch would have been a registered task failing every interval,
    forever, with a KeyError nobody was awake to read.
    """
    names = set(cli.build_parser()._subparsers._group_actions[0].choices)
    table = cli.command_table()
    assert names == set(table), \
        f"offered but unrouted: {names - set(table)}; " \
        f"routed but unoffered: {set(table) - names}"
    for name, handler in table.items():
        assert callable(handler), f"{name} is wired to {handler!r}"


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


# -- the unattended entry point -----------------------------------------
# `python -m autoforge tick` is what gets registered with the OS scheduler, so
# these tests exercise it as the scheduler would: as a process that must
# terminate, and whose silence means "nothing to do".
def test_tick_on_an_empty_schedule_says_nothing_when_quiet(monkeypatch, capsys):
    """Quiet means quiet: a task firing every 30 minutes must not log 48
    lines a day to say it had nothing to do."""
    # A configured provider: `_config` exits before reaching the seam
    # otherwise, and what is under test here is the tick, not the setup.
    monkeypatch.setenv("AUTOFORGE_API_KEY", "test-key")
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AUTOFORGE_HOME", "/tmp")
    assert cli.main(["tick", "--quiet", "--file", "/tmp/af-empty-sched.jsonl"]) == 0
    assert capsys.readouterr().out == ""


def test_tick_reports_the_next_due_time_when_not_quiet(monkeypatch, capsys, tmp_path):
    from autoforge.schedule import Schedule
    path = tmp_path / "s.jsonl"
    Schedule(path).add("water the plants", "1d")
    # A configured provider: `_config` exits before reaching the seam
    # otherwise, and what is under test here is the tick, not the setup.
    monkeypatch.setenv("AUTOFORGE_API_KEY", "test-key")
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AUTOFORGE_HOME", "/tmp")
    assert cli.main(["tick", "--file", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Nothing is due" in out
    assert "water the plants" in out


def test_tick_does_not_build_an_agent_when_nothing_is_due(monkeypatch, tmp_path):
    """The common case must not pay for a model client.

    A tick that constructs an agent to discover it has nothing to do would make
    the unattended path depend on config and credentials it does not need --
    which is exactly how a scheduled task fails on a machine where the API key
    has expired.
    """
    built: list[str] = []
    monkeypatch.setattr(cli, "_build_mode", lambda *a, **k: built.append("agent"))
    # A configured provider: `_config` exits before reaching the seam
    # otherwise, and what is under test here is the tick, not the setup.
    monkeypatch.setenv("AUTOFORGE_API_KEY", "test-key")
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AUTOFORGE_HOME", "/tmp")
    assert cli.main(["tick", "--quiet", "--file", str(tmp_path / "none.jsonl")]) == 0
    assert built == [], "a no-op tick built an agent"


def test_tick_runs_the_due_task_through_the_agent(monkeypatch, tmp_path, capsys):
    from autoforge.schedule import Schedule
    path = tmp_path / "s.jsonl"
    Schedule(path).add("say hello", "0s")

    seen: dict = {}

    class _FakeAgent:
        def __init__(self, cfg=None):
            seen["built"] = True

        def run(self, prompt, **kw):
            seen["prompt"] = prompt
            return argparse.Namespace(content="done: hello said")

    monkeypatch.setattr(cli, "_build_mode", lambda cfg, mode: _FakeAgent(cfg))
    # A configured provider: `_config` exits before reaching the seam
    # otherwise, and what is under test here is the tick, not the setup.
    monkeypatch.setenv("AUTOFORGE_API_KEY", "test-key")
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AUTOFORGE_HOME", "/tmp")
    rc = cli.main(["tick", "--file", str(path)])
    assert rc == 0
    assert seen.get("built"), "a due task did not build an agent"
    assert "say hello" in seen["prompt"], \
        "the due task's text never reached the model"
    assert "hello said" in capsys.readouterr().out


def test_tick_closes_the_task_it_ran(monkeypatch, tmp_path):
    """A due task attended to must not still be due on the next tick."""
    from autoforge.schedule import Schedule
    path = tmp_path / "s.jsonl"
    task = Schedule(path).add("one shot", "0s")

    class _FakeAgent:
        def __init__(self, cfg=None):
            pass

        def run(self, prompt, **kw):
            return argparse.Namespace(content="ok")

    monkeypatch.setattr(cli, "_build_mode", lambda cfg, mode: _FakeAgent(cfg))
    # A configured provider: `_config` exits before reaching the seam
    # otherwise, and what is under test here is the tick, not the setup.
    monkeypatch.setenv("AUTOFORGE_API_KEY", "test-key")
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AUTOFORGE_HOME", "/tmp")
    assert cli.main(["tick", "--quiet", "--file", str(path)]) == 0
    assert Schedule(path).due() == []
    assert Schedule(path).get(task.id).runs == 1


def test_tick_closes_the_task_even_when_the_agent_raises(monkeypatch, tmp_path):
    """A run that crashed is a run that happened.

    Leaving it open means every subsequent tick re-runs a task that is already
    failing, and the agenda fills with the same broken entry forever.
    """
    from autoforge.schedule import Schedule
    path = tmp_path / "s.jsonl"
    Schedule(path).add("doomed", "0s")

    class _ExplodingAgent:
        def __init__(self, cfg=None):
            pass

        def run(self, prompt, **kw):
            raise RuntimeError("model unreachable")

    monkeypatch.setattr(cli, "_build_mode", lambda cfg, mode: _ExplodingAgent(cfg))
    # A configured provider: `_config` exits before reaching the seam
    # otherwise, and what is under test here is the tick, not the setup.
    monkeypatch.setenv("AUTOFORGE_API_KEY", "test-key")
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AUTOFORGE_HOME", "/tmp")
    cli.main(["tick", "--file", str(path)])
    closed = Schedule(path).get(Schedule(path).all()[0].id)
    assert closed.runs == 1 and closed.failures == 1
    assert "model unreachable" in closed.notes[-1]


def test_tick_with_a_missing_schedule_file_is_a_clean_no_op(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path / "nothing-here"))
    assert cli.main(["tick", "--quiet"]) == 0
