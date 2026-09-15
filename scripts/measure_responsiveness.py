"""Measure the two promises the operator is owed, end to end.

The unit tests assert the plumbing; this prints the numbers. Both defects it
measures were the same defect seen twice -- a long step that could not be
interrupted and would not speak:

  * an interjection reached the agent only at the boundary *after* the step that
    was already running, so the operator's correction landed minutes later, and
    the acknowledgement they got said so ("will reach the agent at the next
    step") -- which is precisely what being ignored sounds like;
  * a step in progress said nothing while it ran, and the only progress line
    was rewritten in place every second, so a run that was working looked
    identical to one that had died.

Run: python scripts/measure_responsiveness.py
"""
from __future__ import annotations

import io
import threading
import time

from autoforge.cli import _LiveRun
from autoforge.core.agent import Agent
from autoforge.core.llm import LLMAborted, LLMResponse, MockLLMClient
from autoforge.core.steering import Steering
from autoforge.tools.registry import ToolRegistry

# What the real client's read timeout is, and therefore the block the operator
# used to be stuck behind.
MODEL_STALL = 30.0
SPEAK_AFTER = 2.0


def measure_interjection(stall: float = MODEL_STALL,
                         speak_after: float = SPEAK_AFTER) -> dict:
    """Seconds from the operator's line to the run yielding to it."""
    sent: list[float] = []
    aborted_at: list[float] = []
    prompts: list[str] = []

    def handler(messages, tools, **kw):
        sent.append(time.monotonic())
        prompts.append(
            [m.content for m in messages if m.role == "user"][-1]
        )
        should_abort = kw.get("should_abort")
        while time.monotonic() - sent[-1] < stall:
            time.sleep(0.02)
            # Exactly the contract the real client offers a slow endpoint.
            if should_abort is not None and should_abort():
                aborted_at.append(time.monotonic())
                raise LLMAborted("the operator spoke")
        return LLMResponse(content="answered without interruption")

    said: list[tuple[float, str]] = []
    steer = Steering(printer=lambda t: said.append((time.monotonic(), t)))
    llm = MockLLMClient(handler=handler)
    agent = Agent(llm, ToolRegistry(), steer=steer, allow_self_terminate=False)

    box: dict = {}

    def run() -> None:
        box["result"] = agent.run("a task that takes a while")

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    time.sleep(speak_after)

    spoke = time.monotonic()
    steer.submit("actually, do it the other way round")
    acked = time.monotonic()

    worker.join(timeout=stall + 10.0)
    return {
        "ack_seconds": (said[0][0] - spoke) if said else None,
        "ack_text": said[0][1] if said else None,
        "yield_seconds": (aborted_at[0] - spoke) if aborted_at else None,
        "requestioned_seconds": (sent[1] - spoke) if len(sent) > 1 else None,
        "the_correction_was_in_the_next_prompt": (
            len(prompts) > 1 and "the other way round" in (prompts[1] or "")
        ),
        "steps_abandoned": len(aborted_at),
        "still_running_after_join": worker.is_alive(),
        "result": box.get("result"),
    }


def measure_reports(seconds: float = 3.3, every: float = 1.0) -> dict:
    """Whether a silent run keeps producing lines the operator can scroll to."""
    buf = io.StringIO()
    live = _LiveRun(stream=buf)          # not a tty: the durable-log case
    live.REPORT_EVERY = every
    live.start()
    live("request", {"turn": 1})         # a run is now in progress
    time.sleep(seconds)
    live._running = False
    live.stop()

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    reports = [ln for ln in lines if "still" in ln]
    stamps = [float(ln[ln.index("+") + 1: ln.index("s", ln.index("+"))])
              for ln in reports]
    gaps = [b - a for a, b in zip([0.0] + stamps, stamps)]
    return {
        "seconds": seconds,
        "report_interval": every,
        "reports": len(reports),
        "first": reports[0] if reports else None,
        "longest_gap_measured": max(gaps) if gaps else None,
        "longest_silence_allowed_by_config": _LiveRun.REPORT_EVERY,
        "shipped_cadence": _LiveRun.REPORT_EVERY,
        "tick_lines_in_a_log": len([ln for ln in lines
                                    if ln.strip().startswith("…")]),
    }


if __name__ == "__main__":
    print("=" * 72)
    print("1. the operator speaks during a 30s model call")
    print("=" * 72)
    m = measure_interjection()
    for key in ("ack_seconds", "ack_text", "yield_seconds", "requestioned_seconds",
                "the_correction_was_in_the_next_prompt", "steps_abandoned",
                "still_running_after_join"):
        print(f"  {key:42} {m[key]}")
    print()

    print("=" * 72)
    print("2. a run that says nothing else")
    print("=" * 72)
    r = measure_reports()
    for key, val in r.items():
        print(f"  {key:42} {val}")
