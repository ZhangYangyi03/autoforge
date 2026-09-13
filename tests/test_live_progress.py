"""A slow model must read as "waiting", never as "hung".

The observed failure: `run` rendered the trace only after the whole task
finished, so every model call — 5s or 60s — produced identical output: nothing.
There was no way for a user watching the terminal to tell a slow request from a
wedged process, and no way to tell which tool the agent was on.

Two things fix that, and both are pinned here:

  * the loop fires a hook *before* the request goes out (on_request), so the
    "asking the model" line is not delayed until the answer arrives
  * ForgeAgent.run forwards every trace record to a `progress` observer as it
    happens, so forging and self-amendment stream live instead of being
    rendered once, at the end, as a pile

These assert on the callback sequence, not on printed text, because the
sequence is the contract — the CLI's formatting is free to change.
"""
from __future__ import annotations

import io

from autoforge.agent import ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.cli import _LiveRun
from autoforge.core.agent import Agent
from autoforge.core.llm import LLMResponse, MockLLMClient, tool_call
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec


def _echo_spec() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="echo back",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        fn=lambda x="": x,
    )


# ----------------------------------------------------------------------
# the hook order is the contract
# ----------------------------------------------------------------------
def test_request_fires_before_the_model_is_called():
    """`on_request` must land *before* `chat`, or the waiting line is a lie."""
    seen: list[str] = []
    llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="done"))

    def on_request(turn: int) -> None:
        seen.append(f"request:{turn}:calls={len(llm.calls)}")

    def on_turn(turn, msg) -> None:
        seen.append(f"turn:{turn}:calls={len(llm.calls)}")

    Agent(llm, ToolRegistry(), on_request=on_request, on_turn=on_turn).run("hi")

    # at request time the client had made 0 calls; by turn time, 1.
    assert seen == ["request:1:calls=0", "turn:1:calls=1"]


def test_request_fires_once_per_turn_including_tool_rounds():
    llm = MockLLMClient(script=[
        LLMResponse(content="", tool_calls=[tool_call("echo", {"x": "a"})]),
        LLMResponse(content="finished"),
    ])
    reg = ToolRegistry()
    reg.register(_echo_spec())
    requests: list[int] = []
    Agent(llm, reg, on_request=requests.append).run("go")
    assert requests == [1, 2]


def test_a_raising_observer_does_not_kill_the_run():
    """Progress reporting is decoration; it must never be load-bearing."""
    llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="ok"))

    def explode(turn: int) -> None:
        raise RuntimeError("observer bug")

    result = Agent(llm, ToolRegistry(), on_request=explode).run("hi")
    assert result.content == "ok"


# ----------------------------------------------------------------------
# ForgeAgent.streams its trace
# ----------------------------------------------------------------------
def test_run_forwards_every_record_to_progress():
    llm = MockLLMClient(script=[
        LLMResponse(content="", tool_calls=[tool_call("echo", {"x": "a"})]),
        LLMResponse(content="finished"),
    ])
    agent = ForgeAgent(llm, policy=FULL_FREEDOM)
    agent.registry.register(_echo_spec())

    events: list[tuple[str, dict]] = []
    agent.run("go", progress=lambda kind, payload: events.append((kind, payload)))

    kinds = [k for k, _ in events]
    assert "request" in kinds, "no waiting signal reached the observer"
    assert "call" in kinds and "result" in kinds, "tool activity never streamed"
    assert [p["tool"] for k, p in events if k == "call"] == ["echo"]


def test_progress_is_detached_after_the_run():
    """A finished run must not keep printing into a stale observer."""
    llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="ok"))
    agent = ForgeAgent(llm, policy=FULL_FREEDOM)
    seen: list[str] = []
    agent.run("go", progress=lambda kind, payload: seen.append(kind))
    before = len(seen)
    agent.registry.register(_echo_spec())          # a later, unrelated change
    assert len(seen) == before, "observer leaked past the end of the run"
    assert agent._progress is None


def test_run_without_a_progress_observer_still_works():
    """The headless/library path must not require an observer."""
    llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="ok"))
    result = ForgeAgent(llm, policy=FULL_FREEDOM).run("go")
    assert result.content == "ok"


# ----------------------------------------------------------------------
# forging announces itself before the long silence
# ----------------------------------------------------------------------
def test_forge_start_is_emitted_before_any_round():
    """Round 1 begins with a model call; the user must hear about it first."""
    llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="x = 1"))
    agent = ForgeAgent(llm, policy=FULL_FREEDOM)

    events: list[tuple[str, dict]] = []
    agent.pipeline.on_event = lambda kind, payload: events.append((kind, payload))
    agent.pipeline.forge("a tool that returns one")

    kinds = [k for k, _ in events]
    assert kinds[0] == "forge_start", f"first event was {kinds[0]!r}, not the start"
    assert "forge_done" in kinds
    assert events[0][1]["need"] == "a tool that returns one"


def test_chat_streams_too_not_only_run():
    """The complaint was about the chat loop: it had no progress at all.

    Every entry point that runs a task must hand the same observer down, or the
    silence just moves to whichever command was missed.
    """
    import inspect

    from autoforge import cli

    src = inspect.getsource(cli.cmd_chat)
    assert "progress=live" in src, "chat runs the agent without a live observer"
    assert "_LiveRun" in src


def test_cmd_forge_attaches_the_observer_to_the_pipeline():
    """`forge` calls the pipeline directly, so it must wire the observer itself."""
    import inspect

    from autoforge import cli

    src = inspect.getsource(cli.cmd_forge)
    assert "_LiveRun" in src
    assert "_progress" in src, "the pipeline would record into the void"


def test_a_failed_round_is_not_reported_twice():
    """`forge_error` narrates the round; the attempt line must then stay quiet."""
    live = _LiveRun(stream=io.StringIO())
    live(("forge_error"), {"round": 2, "error": "TypeError: unhashable type"})
    live(("forge_attempt"), {"round": 2, "accepted": False,
                             "error": "TypeError: unhashable type"})
    out = live.stream.getvalue()
    assert out.count("unhashable") == 1, f"the same failure was said twice:\n{out}"


def test_a_distinct_failed_round_is_reported():
    """Suppression is per round — round 3 is not silenced by round 2's error."""
    live = _LiveRun(stream=io.StringIO())
    live(("forge_error"), {"round": 2, "error": "boom"})
    live(("forge_attempt"), {"round": 3, "accepted": False, "error": "a different wall"})
    out = live.stream.getvalue()
    assert "a different wall" in out


def test_forge_done_carries_the_tool_name():
    """A finished forge must name what it made — "sealed ?" says nothing.

    A *failed* forge has no spec and so no name; it is named by its `need`
    instead. Either way the reader gets an identifier, never a bare "?".
    """
    llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="x = 1"))
    agent = ForgeAgent(llm, policy=FULL_FREEDOM)

    done: list[dict] = []
    agent.pipeline.on_event = lambda kind, payload: (
        done.append(payload) if kind == "forge_done" else None)
    result = agent.pipeline.forge("a tool that returns one")

    assert done, "forge_done never fired"
    if result.ok:
        assert done[0]["name"] == result.spec.name, "a sealed forge must be named"
    else:
        # No spec was produced, so there is no name to report — but the CLI
        # still has `need` to fall back on, and that is what it renders.
        assert done[0]["name"] is None
        assert done[0]["need"], "a failed forge still has to be identifiable"


# ----------------------------------------------------------------------
# the CLI renderer
# ----------------------------------------------------------------------
def test_live_run_writes_plain_lines_when_not_a_tty():
    """Piped output must stay readable — no carriage-return ticking."""
    import io

    from autoforge.cli import _LiveRun

    buf = io.StringIO()
    live = _LiveRun(stream=buf)
    assert live.live is False, "StringIO was mistaken for a terminal"
    live.start()
    live("request", {"turn": 1})
    live("call", {"tool": "echo"})
    live.stop()

    out = buf.getvalue()
    assert "\r" not in out, "a non-tty stream got carriage returns"
    assert "asking the model" in out and "-> echo" in out


def test_live_run_done_line_counts_calls_not_lists():
    """`AgentResult.tool_calls` is a list; the summary must not print it raw."""
    import io
    from types import SimpleNamespace

    from autoforge.cli import _LiveRun

    buf = io.StringIO()
    live = _LiveRun(stream=buf)
    live.done(SimpleNamespace(turns=3, tool_calls=["forge_tool", "is_prime"]))
    out = buf.getvalue()
    assert "3 turn(s), 2 tool call(s)" in out
    assert "['forge_tool'" not in out
