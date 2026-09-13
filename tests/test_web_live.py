"""The console must stream while the run is happening, not after it ends.

The CLI had this bug and was fixed; the browser had the same one. `Harness.send`
called `agent.run(...)` with no observer and only pumped the trace once the run
returned, so the page showed "user said X" and then nothing for as long as the
model took — a slow answer and a wedged process looked identical.

These tests hold a run open mid-flight and assert the trajectory is already on
the wire. They also pin the other half of the fix: because the live observer and
the end-of-run backstop now share one cursor, nothing is delivered twice.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from autoforge.web import server as web


class _Agent:
    """A fake agent that pauses in the middle of a run, on purpose."""

    def __init__(self, gate: threading.Event) -> None:
        self.trace: list[dict] = []
        self.store = None
        self._gate = gate
        self.pipeline = None

    def run(self, text, history=None, progress=None):
        self.trace.append({"kind": "request", "turn": 1})
        if progress:
            progress("request", {"turn": 1})
        # Everything above must already have reached subscribers: the run is
        # still open, and the test asserts on that.
        self._gate.wait(10)
        self.trace.append({"kind": "call", "tool": "echo", "args": {"x": 1}})
        if progress:
            progress("call", {"tool": "echo"})
        self.trace.append({"kind": "finish", "turns": 1, "tools": 1,
                           "self_terminated": False})
        return SimpleNamespace(content="done", messages=[], self_terminated=False)


def _harness(gate: threading.Event) -> web.Harness:
    return web.Harness({}, lambda cfg, mode: _Agent(gate), default_mode="standard")


def _drain(q, seconds: float = 3.0) -> list[dict]:
    out: list[dict] = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            out.append(q.get(timeout=0.1))
        except Exception:                                          # noqa: BLE001
            continue
    return out


def _wait_idle(session, seconds: float = 10.0) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline and session.busy:
        time.sleep(0.02)


def test_events_arrive_while_the_run_is_still_open():
    gate = threading.Event()
    h = _harness(gate)
    s = h.create("standard")
    q = s.subscribe()

    h.send(s, "hi")
    # Collect without releasing the gate: the run is mid-flight by construction.
    seen = _drain(q, seconds=2.0)
    assert s.busy, "the run finished before the assertion — the gate did not hold"

    sources = [e["source"] for e in seen]
    assert "request" in sources, (
        "nothing reached the console while the run was open — "
        f"got {sources} (the trace is still being batched at the end)"
    )

    gate.set()
    _wait_idle(s)


def test_live_delivery_does_not_duplicate_the_backstop():
    gate = threading.Event()
    gate.set()                      # let it run straight through
    h = _harness(gate)
    s = h.create("standard")
    h.send(s, "hi")
    _wait_idle(s)

    requests = [e for e in s.events if e["source"] == "request"]
    calls = [e for e in s.events if e["source"] == "tool_call"]
    assert len(requests) == 1, f"the request event was sent {len(requests)} times"
    assert len(calls) == 1, f"the tool call was sent {len(calls)} times"


def test_a_run_without_a_progress_parameter_still_works():
    """MinimalAgent predates the observer; the harness must not assume it.

    Passing `progress=` unconditionally turned a working turn into a TypeError
    the first time this was wired up. The trajectory must still arrive — batched
    is acceptable, broken is not.
    """
    class _NoProgressAgent:
        def __init__(self) -> None:
            self.trace = []
            self.store = None
            self.pipeline = None

        def run(self, text, history=None):          # no progress kwarg, at all
            # The call belongs here, where a run puts it. Seeding it in
            # __init__ made it look like history the harness had already
            # delivered, so the end-of-run backstop correctly skipped it and the
            # test failed against its own fixture.
            self.trace.append({"kind": "call", "tool": "echo", "args": {}})
            self.trace.append({"kind": "finish", "turns": 1, "tools": 1,
                               "self_terminated": False})
            return SimpleNamespace(content="ok", messages=[], self_terminated=False)

    h = web.Harness({}, lambda cfg, mode: _NoProgressAgent(), default_mode="standard")
    s = h.create("standard")
    h.send(s, "hi")
    _wait_idle(s)

    assert s.error is None, f"an agent without a progress kwarg was broken: {s.error}"
    assert any(e["source"] == "tool_call" for e in s.events), \
        "the trajectory never arrived"


def test_events_keep_their_order_and_numbering():
    gate = threading.Event()
    gate.set()
    h = _harness(gate)
    s = h.create("standard")
    h.send(s, "hi")
    _wait_idle(s)

    seqs = [e["seq"] for e in s.events]
    assert seqs == list(range(1, len(seqs) + 1)), f"sequence has gaps: {seqs}"
    sources = [e["source"] for e in s.events]
    assert sources.index("user") < sources.index("request")
    assert sources[-1] in ("assistant", "system")


def test_forge_rounds_stream_too():
    """`forge` records through the agent, so the observer must be attached there."""
    gate = threading.Event()
    gate.set()

    class _Pipeline:
        def __init__(self, agent: _Agent) -> None:
            self._agent = agent

        def forge(self, need: str):
            self._agent.trace.append({"kind": "forge_start", "need": need})
            # The pipeline records via the agent; that forwarding is what has to
            # make these land live rather than in a lump at the end.
            if self._agent._progress:
                self._agent._progress("forge_start", {"need": need})
            self._agent.trace.append({"kind": "forge_done", "ok": True,
                                      "rounds": 1, "name": "thing"})
            return SimpleNamespace(ok=True, spec=None, rounds=1, attempts=[])

    class _ForgeAgent(_Agent):
        def __init__(self, gate: threading.Event) -> None:
            super().__init__(gate)
            self._progress = None
            self.pipeline = _Pipeline(self)

    h = web.Harness({}, lambda cfg, mode: _ForgeAgent(gate), default_mode="standard")
    s = h.create("standard")
    h.forge(s, "a tool")
    _wait_idle(s)

    assert s.error is None
    kinds = [e.get("raw", {}).get("kind") for e in s.events]
    assert "forge_start" in kinds, f"forge start never reached the console: {kinds}"
    assert kinds.count("forge_done") == 1, f"forge_done duplicated or lost: {kinds}"
