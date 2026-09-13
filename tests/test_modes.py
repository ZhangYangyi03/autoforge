"""Runtime modes: the two built-in tools and the minimal agent loop."""
from __future__ import annotations

import os

import pytest

from autoforge.core.llm import LLMClient, LLMResponse
from autoforge.core.message import ToolCall
from autoforge.modes import MinimalAgent, _bash, _editor, _minimal_tools

# --------------------------------------------------------------------------
# bash
# --------------------------------------------------------------------------


def test_bash_reports_output_and_exit_code():
    out = _bash("echo out && echo err >&2 && exit 7")
    assert "exit=7" in out and "out" in out and "err" in out


def test_bash_captures_a_failing_command_without_raising():
    out = _bash("this-command-does-not-exist-anywhere")
    assert out.startswith("exit=")
    assert "exit=0" not in out.split("\n")[0]


def test_bash_times_out_instead_of_hanging():
    out = _bash("sleep 5", timeout=1)
    assert "timeout" in out


# --------------------------------------------------------------------------
# str_replace_editor
# --------------------------------------------------------------------------


def test_editor_create_then_view(tmp_path):
    p = str(tmp_path / "a.txt")
    assert "created" in _editor("create", p, file_text="one\ntwo\n")
    view = _editor("view", p)
    assert "one" in view and "two" in view
    assert view.splitlines()[0].startswith("1")


def test_editor_view_range_limits_the_window(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("\n".join(f"line{i}" for i in range(1, 21)), encoding="utf-8")
    view = _editor("view", str(p), view_range=[5, 7])
    assert "line5" in view and "line7" in view
    assert "line8" not in view


def test_editor_str_replace_requires_a_unique_match(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("dup\ndup\n", encoding="utf-8")
    out = _editor("str_replace", str(p), old_str="dup", new_str="x")
    assert "appears 2 times" in out
    assert p.read_text(encoding="utf-8") == "dup\ndup\n"       # untouched

    p.write_text("keep\nswap\n", encoding="utf-8")
    assert "edited" in _editor("str_replace", str(p), old_str="swap", new_str="done")
    assert p.read_text(encoding="utf-8") == "keep\ndone\n"


def test_editor_refuses_a_missing_match_or_file(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("abc\n", encoding="utf-8")
    assert "not found" in _editor("str_replace", str(p), old_str="zzz", new_str="y")
    assert "no such file" in _editor("view", str(tmp_path / "nope.txt"))


def test_editor_insert_places_lines(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("a\nb\n", encoding="utf-8")
    assert "inserted" in _editor("insert", str(p), insert_line=1, new_str="X")
    assert p.read_text(encoding="utf-8") == "a\nX\nb\n"


def test_editor_rejects_an_unknown_command(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("a\n", encoding="utf-8")
    assert "unknown command" in _editor("delete", str(p))


# --------------------------------------------------------------------------
# the mode
# --------------------------------------------------------------------------


class Scripted(LLMClient):
    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):        # noqa: ANN001, ANN003
        self.calls += 1
        assert tools, "the agent must be handed tool schemas"
        return self.script.pop(0) if self.script else LLMResponse(content="done")


def test_minimal_exposes_exactly_two_tools():
    assert sorted(_minimal_tools()) == ["bash", "str_replace_editor"]
    assert all(s.state.value == "active" for s in _minimal_tools().values())


def test_minimal_agent_cannot_forge():
    agent = MinimalAgent(llm=Scripted([]))
    assert agent.pipeline is None
    assert agent.store is None


def test_minimal_agent_runs_a_tool_then_self_terminates(tmp_path):
    llm = Scripted([
        LLMResponse(content="", tool_calls=[ToolCall("c1", "bash", {"command": "echo hi"})]),
        LLMResponse(content="", tool_calls=[ToolCall("c2", "terminate",
                                                    {"summary": "all done"})]),
    ])
    agent = MinimalAgent(llm=llm, cwd=str(tmp_path))
    result = agent.run("echo hi")

    assert result.self_terminated is True
    assert result.termination_reason == ""
    assert "all done" in result.content
    assert result.tool_calls == ["bash", "terminate"]

    # bash is recorded as call+result; terminate is itself a tool call, so the
    # trajectory ends call -> finish.
    kinds = [e["kind"] for e in agent.trace]
    assert kinds == ["call", "result", "call", "finish"]
    tool_call = agent.trace[0]
    assert tool_call["tool"] == "bash" and tool_call["args"]["command"] == "echo hi"
    assert agent.trace[1]["tool"] == "bash" and agent.trace[1]["ok"] is True


def test_minimal_agent_pins_bash_to_the_mode_workspace(tmp_path):
    """A file written through the editor in the mode's cwd is visible to bash."""
    agent = MinimalAgent(llm=Scripted([]), cwd=str(tmp_path))
    agent.registry.call("str_replace_editor",
                        {"command": "create", "path": "note.txt", "file_text": "hi"})
    out = agent.registry.call("bash", {"command": "cat note.txt"})
    assert out.ok and "hi" in out.output
    assert (tmp_path / "note.txt").exists()


def test_minimal_report_shape_matches_what_the_ui_reads():
    rep = MinimalAgent(llm=Scripted([])).report()
    assert rep["mode"] == "minimal"
    assert {t["name"] for t in rep["tools"]["tools"]} == {"bash", "str_replace_editor"}


# --------------------------------------------------------------------------
# the CLI seam
# --------------------------------------------------------------------------


def test_build_mode_returns_the_minimal_agent_without_touching_the_network():
    from autoforge.cli import _build_mode

    cfg = {"model": "m", "base": "http://127.0.0.1:1/v1", "key": "k",
           "proxy": False, "fast": True, "max_tokens": 64}
    agent = _build_mode(cfg, "minimal")
    assert isinstance(agent, MinimalAgent)
    assert agent.cwd == os.getcwd()


def test_unknown_mode_is_refused():
    from autoforge.cli import _build_mode

    with pytest.raises(SystemExit) as err:
        _build_mode({}, "telepathy")
    assert "telepathy" in str(err.value)


def test_modes_are_declared_once():
    from autoforge.cli import MODES

    assert MODES == ("standard", "minimal")
