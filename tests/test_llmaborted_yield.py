"""An interruption is a yield, never a red error and never a spin.

Two defects met on 2026-11-09, when typing mid-answer printed
`error: LLMAborted: abandoned a model call in flight after 3.6s because the
operator said something` and the run looked dead:

1. `Agent` decided whether an abort was its own by re-reading the steering
   queue *after* the exception unwound. A sibling loop, a burst fold or
   `_answer_the_operator` could drain that queue in between, so a line meant
   for this loop read as "somebody else's" and was re-raised to the CLI.
   Ownership is now recorded by `_yield_if_mine` at the instant the predicate
   fires.
2. The CLI and the web harness painted that re-raised yield as `error:`.
   Both now treat `LLMAborted` as "heard you", not as a failure.
"""
from __future__ import annotations

import io

from autoforge.cli import _LiveRun
from autoforge.core.agent import Agent
from autoforge.core.llm import LLMAborted, LLMResponse, MockLLMClient
from autoforge.core.steering import Steering
from autoforge.tools.registry import ToolRegistry
from autoforge.web.server import Harness, _event  # noqa: F401  (import smoke)


class _DrainingSteering(Steering):
    """A channel whose queue is emptied while the abort is unwinding.

    Stands in for `_answer_the_operator`, a nested loop or a burst fold --
    the race that made the old `if not self._operator_should_yield(): raise`
    guess wrong.
    """

    def __init__(self):
        super().__init__(printer=lambda _t: None)
        self.armed = True

    def has_pending(self) -> bool:
        return self.armed

    def stop_requested(self) -> bool:
        return False

    def take_supplements(self) -> list[str]:
        return []


def test_a_yield_is_not_re_raised_even_if_the_queue_drains_mid_abort():
    steer = _DrainingSteering()
    calls = {"n": 0}
    answers = []

    def handler(messages, tools, **kw):
        calls["n"] += 1
        pred = kw.get("should_abort")
        if calls["n"] == 1:
            # The predicate fires (recording ownership), then the line is
            # consumed elsewhere before the exception is caught.
            assert pred is not None and pred() is True
            steer.armed = False
            raise LLMAborted("abandoned a model call in flight after 3.6s "
                             "because the operator said something")
        answers.append("answered the operator")
        return LLMResponse(content="answered the operator")

    agent = Agent(MockLLMClient(handler=handler), ToolRegistry(),
                  steer=steer, allow_self_terminate=False, max_turns=6)
    result = agent.run("a task")

    assert "error" not in result.content.lower()
    assert answers == ["answered the operator"], (
        f"the yield was re-raised / spun instead of re-asking: {calls}"
    )
    assert calls["n"] == 2


def test_an_inherited_abort_still_goes_back_to_the_owner():
    """A nested loop with no channel must not swallow somebody else's line."""
    calls = {"n": 0}

    def handler(messages, tools, **kw):
        calls["n"] += 1
        raise LLMAborted("the parent's operator spoke")

    agent = Agent(MockLLMClient(handler=handler), ToolRegistry(),
                  steer=None, allow_self_terminate=False)
    try:
        agent.run("a task")
    except LLMAborted:
        pass
    else:
        raise AssertionError("a nested loop swallowed an abort it cannot answer")
    assert calls["n"] == 1, "the nested loop spun on an inherited predicate"


def test_the_cli_does_not_print_an_error_for_a_yield(monkeypatch):
    """The REPL's handler is reached through its own module, not by running it."""
    import autoforge.cli as cli

    assert issubclass(LLMAborted, InterruptedError)
    # The catch is ordered before `except Exception`; prove it by source order
    # so a future edit cannot silently move it back behind the red printer.
    import inspect
    src = inspect.getsource(cli.cmd_talk) if hasattr(cli, "cmd_talk") else None
    if src is None:
        # the REPL lives in whichever function owns the `you>` prompt
        src = "".join(
            inspect.getsource(getattr(cli, n))
            for n in dir(cli)
            if n.startswith("cmd_") and callable(getattr(cli, n))
            and "error:" in inspect.getsource(getattr(cli, n))
        )
    assert src, "could not find the REPL error handler"
    assert src.index("except LLMAborted") < src.index('error:'), (
        "a yield would be caught by `except Exception` again"
    )


def test_the_web_harness_does_not_record_a_yield_as_an_error():
    from autoforge.web.server import Session

    h = Harness(cfg={}, factory=lambda cfg, mode: None)
    s = Session("s1", {}, "standard", lambda cfg, mode: None)
    seen = []

    def work():
        raise LLMAborted("the operator spoke")

    h._run(s, work)
    assert s.error is None, f"a yield became a session error: {s.error}"
    assert s.busy is False
    assert not any(e.get("source") == "error" for e in s.events), (
        "the browser was sent a red error for a yield"
    )
