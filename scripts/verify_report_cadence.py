"""Evidence for one claim: no run goes quiet for longer than the cadence.

The operator asked for a line every 30 seconds -- "one step at a time, like
you do" -- and tolerated 100. What they watched instead was a 41s forge round
and a 58s model call pass in silence, because the ceiling was set at 60 and
sat *above* the pauses a real run produces.

So this is not a test of the constant; it is a run of the thing. A real
`Agent`, a real model call that stalls, the real `_LiveRun` observer, and the
real durable-line path (`editor.write`, which is what a terminal at a keyboard
gets). It prints every line the operator would have seen, then the longest gap
between two of them.

    python scripts/verify_report_cadence.py [stall_seconds]

Exit code is 1 if any gap exceeded `REPORT_EVERY + BEAT_SECONDS`.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoforge.cli import _LiveRun                                # noqa: E402
from autoforge.core.agent import Agent                            # noqa: E402
from autoforge.core.llm import LLMResponse, MockLLMClient         # noqa: E402
from autoforge.core.steering import Steering                      # noqa: E402
from autoforge.tools.registry import ToolRegistry                 # noqa: E402

#: Two cadences and a bit, so the run must report twice -- one report proves
#: the thread fired, two prove it keeps firing.
DEFAULT_STALL = 2 * _LiveRun.REPORT_EVERY + 8.0


class _Terminal:
    """The line editor, from `_LiveRun`'s point of view.

    Enough of one to take the same path a real session takes: durable lines go
    through `write`, the in-place tick through `tick`. Without it the run takes
    the piped-log path instead, and the operator's case goes untested.
    """

    available = True

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.ticks: list[str] = []

    def write(self, text: str = "") -> None:
        self.lines.extend(ln for ln in text.split("\n") if ln.strip())

    def tick(self, text: str) -> None:
        self.ticks.append(text)

    def elapsed_stamps(self) -> list[float]:
        out = []
        for ln in self.lines:
            if "+" not in ln or "s" not in ln:
                continue
            at = ln.index("+")
            try:
                out.append(float(ln[at + 1: ln.index("s", at)]))
            except ValueError:
                continue
        return out


def main(argv: list[str]) -> int:
    stall = float(argv[1]) if len(argv) > 1 else DEFAULT_STALL
    terminal = _Terminal()

    def handler(messages, tools, **kw):
        # Blocks like an endpoint that is thinking, honouring `should_abort` so
        # the shape matches a real call rather than a `sleep`.
        start = time.monotonic()
        while time.monotonic() - start < stall:
            time.sleep(0.05)
            if (kw.get("should_abort") or (lambda: False))():
                raise RuntimeError("interrupted")
        return LLMResponse(content="the answer, %ds late" % int(stall))

    live = _LiveRun(editor=terminal).start()

    def on_request(turn: int) -> None:
        # The same event the CLI subscribes to: emitted *before* the request
        # goes out, which is what makes the wait visible while it happens.
        live("request", {"turn": turn})

    def on_turn(turn: int, msg) -> None:
        live("turn", {"turn": turn})

    agent = Agent(MockLLMClient(handler=handler), ToolRegistry(),
                  steer=Steering(printer=lambda t: None),
                  allow_self_terminate=False,
                  on_request=on_request, on_turn=on_turn)

    box: dict = {}
    worker = threading.Thread(
        target=lambda: box.update(r=agent.run("a task with one slow call")),
        daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(timeout=stall + 30.0)
    live.stop()
    live.done(box["r"])

    print("=" * 72)
    print(f"a {stall:.0f}s model call, cadence {_LiveRun.REPORT_EVERY:.0f}s "
          f"+{_LiveRun.BEAT_SECONDS:.0f}s beat")
    print("=" * 72)
    for ln in terminal.lines:
        print(ln)
    note = ("the tick the operator sees every second, rewritten in place"
            if terminal.ticks else "no ticks (piped path)")
    print(f"  ({note} -- {len(terminal.ticks)} of them)")

    stamps = terminal.elapsed_stamps()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    bound = _LiveRun.REPORT_EVERY + _LiveRun.BEAT_SECONDS
    worst = max(gaps) if gaps else float("inf")
    print("-" * 72)
    print(f"  lines                       {len(terminal.lines)}")
    print(f"  reports (\"still ...\")        "
          f"{len([ln for ln in terminal.lines if 'still' in ln])}")
    print(f"  longest gap between lines   {worst:.1f}s")
    print(f"  cadence + beat              {bound:.1f}s")
    print(f"  wall clock                  {time.monotonic() - started:.1f}s")
    print(f"  VERDICT                     "
          f"{'within the cadence' if worst <= bound + 1.0 else 'TOO QUIET'}")
    return 0 if worst <= bound + 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
