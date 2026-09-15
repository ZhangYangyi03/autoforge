"""The agent loop.

Deliberately thin in mechanism, deliberately free in policy.

The three things that make this loop different from every other agent loop:

1. **No hidden turn cap.** `max_turns=None` (the default) means the agent runs
   until it produces a text answer or calls `terminate`. A cap exists only if
   the caller imposes one — and when one is imposed, it is logged, not silent.
   The agent decides when a task is done, not an arbitrary number baked into
   the framework.

2. **Self-termination is a first-class tool.** The agent can end its own turn
   loop, returning a summary and a reason. Combined with `unlimited_turns`,
   this means the stopping condition is *semantic* (the agent judges it's done)
   rather than *mechanical* (the counter hit N).

Everything else — tool dispatch, message threading — is ordinary, because that
part isn't what's broken in other frameworks.

3. **The context is compacted, never silently truncated.** An unbounded loop
   meets a bounded context window eventually, and the failure there is a
   provider error at the far end of the task. `compaction.py` replaces the
   middle of the history with a summary at a provably safe cut point. The loop
   itself only calls it; the policy — when, how much to keep, and what to do if
   summarising fails — lives there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .llm import LLMAborted, LLMClient
from .message import Message
from .compaction import OUTPUT_MAX_CHARS, bound_output

DEFAULT_SYSTEM = (
    "You are a capable agent. Call a tool when it genuinely helps; answer "
    "directly when none is needed. Do not call a tool you do not need.\n"
    "When you have fully completed the task, either answer in plain text or "
    "call terminate with a summary of what you did."
)


@dataclass
class AgentResult:
    content: str
    messages: list[Message]
    turns: int
    tool_calls: list[str] = field(default_factory=list)
    self_terminated: bool = False
    termination_reason: str = ""
    # Kept apart from self_terminated on purpose: "the agent judged itself
    # done" and "a person stopped it" are different facts, and a report that
    # blurs them credits the agent with a decision it did not make.
    stopped_by_operator: bool = False

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


class _TerminateSignal(BaseException):
    """Raised by the terminate tool to end the loop cleanly.

    Inherits BaseException (not Exception) so that the registry's
    `except Exception` tool-boundary handler does NOT swallow it — a
    self-termination must propagate all the way back to the loop.
    """

    def __init__(self, summary: str, reason: str = "") -> None:
        super().__init__(summary)
        self.summary = summary
        self.reason = reason


def _notify(hook: Callable[..., None] | None, *args: Any) -> None:
    """Call an observer hook, swallowing whatever it throws.

    Every hook here is a *reporter* — progress lines, trace records, UI. None of
    them decide anything, so a broken reporter must not take the run down with
    it. Before this, a CLI writing to a closed stream killed the task mid-loop.
    """
    if hook is None:
        return
    try:
        hook(*args)
    except Exception:  # noqa: BLE001 — reporting is never load-bearing
        pass


def _mark_run(steer: Any, live: bool) -> None:
    """Tell the channel whether a run is live, when it tracks that at all.

    The channel's reply to the operator promises that "the step in progress
    will yield to it". Only this loop can make that a fact rather than a hope,
    so the loop holds the flag -- and the guard is for the duck-typed channels
    (tests, embedders) that are usable without it: a missing marker must never
    be the thing that kills a run, exactly as a broken channel reads as "no"
    in `_operator_wants_the_floor`.
    """
    fn = getattr(steer, "begin_run" if live else "end_run", None)
    if fn is None:
        return
    try:
        fn()
    except Exception:  # noqa: BLE001 — see docstring
        pass


class Agent:
    """A tool-using loop over an LLMClient and a ToolRegistry.

    Parameters
    ----------
    max_turns:
        `None` (default) means unbounded — the agent stops when it answers or
        self-terminates. An integer imposes a cap, which is recorded on the
        result so it is visible, never silent.
    allow_self_terminate:
        Registers a `terminate` tool the model can call to end the loop with
        a summary. Requires a registry that accepts registration.
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: Any,
        *,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        allow_self_terminate: bool = True,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, Any], None] | None = None,
        on_turn: Callable[[int, Message], None] | None = None,
        on_request: Callable[[int], None] | None = None,
        on_steer: Callable[[str], None] | None = None,
        steer: Any = None,
        compactor: Any = None,
        on_compact: Callable[[Any], None] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        self.max_turns = max_turns
        self.allow_self_terminate = allow_self_terminate
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_turn = on_turn
        self.on_request = on_request
        self.on_steer = on_steer
        # Anything with `take_supplements()` / `stop_requested()`: the channel a
        # person uses to talk to this run while it is happening. Optional, so a
        # run with nobody watching is exactly what it was before.
        self.steer = steer
        # Anything with `maybe_compact(msgs, turn) -> event | None`. None means
        # the context is never compacted, which is correct for a short run and
        # is why the framework itself turns this off by default rather than
        # deciding a window size for the caller.
        self.compactor = compactor
        self.on_compact = on_compact
        self._terminated: _TerminateSignal | None = None
        if allow_self_terminate:
            self._register_terminate_tool()

    def _bounded(self, output: str, tool: str) -> str:
        """`output` cut down to the cap before it enters the context.

        Ported from Hermes' per-message truncation limits (`_CONTENT_MAX` with a
        head and a tail). Hermes applies them when it hands messages to the
        summarizer; this loop applies them at *ingest*, because the damage is
        already done by the time a summary runs: an unbounded `dir /s` or a whole
        file read is one message, and one 60k-token message can walk a transcript
        from under the threshold to three times over it in a single turn — past
        anything a cut of the *middle* can fix.

        The cap comes from the compactor's policy when a compactor is present, so
        whoever sized the window sized this too; with no compactor there is no
        threshold to protect and the module default stands.
        """
        cap = getattr(getattr(self.compactor, "policy", None),
                      "max_output_chars", OUTPUT_MAX_CHARS)
        return bound_output(output, max_chars=cap or 0, name=tool)

    def _register_terminate_tool(self) -> None:
        from ..tools.spec import ToolSpec, ToolState

        registry = self.registry
        if not hasattr(registry, "register") or "terminate" in getattr(registry, "_tools", {}):
            return

        def terminate(summary: str = "", reason: str = "") -> str:
            raise _TerminateSignal(summary or "task complete", reason)

        registry.register(
            ToolSpec(
                name="terminate",
                description=(
                    "End the task early and return a summary. Use this when the "
                    "task is fully complete or cannot proceed further."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "what was accomplished"},
                        "reason": {"type": "string", "description": "why stopping now"},
                    },
                },
                fn=terminate,
                source="builtin",
                tags=["meta"],
                state=ToolState.ACTIVE,
            )
        )

    def _absorb(self, msgs: list[Message]) -> bool:
        """Take whatever the operator said since the last check.

        Supplements land as user messages at the next safe point — before a
        model call, or right after a tool result — which is the latest moment
        they can still change the run they belong to. Returns True if the
        operator also asked to stop: the current step finishes, then the run
        ends. A stop is never applied mid-tool, because the alternative is
        killing a subprocess or a write halfway through.

        No `try/except` on purpose. The whole point is that a line the operator
        typed is never dropped, so a bug in this channel must be loud rather
        than swallowed into a run that quietly ignored them.
        """
        if self.steer is None:
            return False
        for text in self.steer.take_supplements():
            msgs.append(Message.user(text))
            _notify(self.on_steer, text)
        return bool(self.steer.stop_requested())

    @staticmethod
    def _stopped(msgs: list[Message], turns: int, used: list[str]) -> AgentResult:
        return AgentResult(
            "(stopped by the operator)", msgs, turns, used,
            stopped_by_operator=True, termination_reason="stopped by the operator",
        )

    def _operator_wants_the_floor(self) -> bool:
        """Whether the operator has said something the loop has not consumed.

        Asked by *long steps* about themselves -- an in-flight model call, a
        running sandbox child -- so their line lands inside the step rather than
        after it. `_absorb` at the loop boundaries is the same question asked at
        the cheap moments; this is the one asked at the expensive ones, and the
        difference is minutes.

        A stop counts as well as a note. A `/stop` that can only be honoured at
        the next boundary is not a stop, it is a note that the current step is
        still going -- and the step it is ignored by is the longest one.

        A failure here reads as "no". This is polled from inside a request; if
        the steering channel is somehow unusable, the right outcome is a run
        that finishes, not a run that dies investigating its own intercom.
        """
        if self.steer is None:
            return False
        try:
            return bool(self.steer.has_pending()) or bool(self.steer.stop_requested())
        except Exception:                     # noqa: BLE001 - see docstring
            return False

    def run(self, task: str, history: Sequence[Message] | None = None) -> AgentResult:
        """Run to a result, telling the channel that a run is live throughout.

        The channel's reply to the operator promises that the step in progress
        will yield to them. This is the loop that has a step in progress, so
        this is the place that makes the sentence true. The `finally` carries
        as much of it as the entry: a run that raises, or that self-terminates
        through the terminate tool, must not leave the channel promising a
        yield from a loop that is no longer there.
        """
        _mark_run(self.steer, True)
        try:
            return self._run(task, history)
        finally:
            _mark_run(self.steer, False)

    def _run(self, task: str, history: Sequence[Message] | None = None) -> AgentResult:
        msgs = list(history or [])
        if not msgs or msgs[0].role != "system":
            msgs.insert(0, Message.system(self.system_prompt))
        msgs.append(Message.user(task))

        used: list[str] = []
        turn = 0
        self._terminated = None

        while True:
            turn += 1
            if self.max_turns is not None and turn > self.max_turns:
                return AgentResult(
                    f"(turn cap of {self.max_turns} reached before completion)",
                    msgs, turn - 1, used,
                )
            # Before the model is asked: the cheapest place to find out the
            # task just changed.
            if self._absorb(msgs):
                return self._stopped(msgs, turn - 1, used)

            # After `_absorb`, so anything the operator just typed is part of
            # the list when the cut point is chosen — and therefore protected by
            # the rule that their last line is never summarized. Compacting
            # first would let a brand-new instruction land in the dropped range.
            #
            # It also has to be here rather than mid-tool: this is the only
            # point in the loop where the list is guaranteed to hold whole
            # groups (an assistant message and all the results answering it),
            # which is what a cut needs to be safe.
            if self.compactor is not None:
                event = self.compactor.maybe_compact(msgs, turn)
                # Only when something happened. A no-op is most turns, and a
                # reporter that fires with `None` trains its reader to ignore it.
                if event is not None:
                    _notify(self.on_compact, event)

            _notify(self.on_request, turn)
            try:
                resp = self.llm.chat(
                    msgs, tools=self.registry.schemas(),
                    should_abort=self._operator_wants_the_floor,
                )
            except LLMAborted:
                # They spoke while the answer was in flight. Nothing has been
                # decided yet -- no tool ran, no message was appended -- so the
                # only correct move is to go round again, where `_absorb` folds
                # their line into the list and the same question is asked, now
                # informed. The turn is given back: a turn that produced no
                # answer is not a turn spent, and counting it would let an
                # operator's own corrections race the turn cap.
                # Unless the abort was not this loop's to answer. A nested loop
                # -- a verification probe agent, an adversary's attacker -- runs
                # on the same client as the run that spawned it, so it inherits
                # that run's abort predicate and can be woken by a line
                # addressed to somebody else. Going round again here would spin:
                # this loop's own channel is empty, `_absorb` has nothing to
                # fold in, and the inherited predicate is still true, so the
                # next call aborts immediately and forever. Only a line this
                # loop can actually absorb is worth another turn; anything else
                # belongs to the caller, which is where the operator's message
                # is waiting.
                if not self._operator_wants_the_floor():
                    raise
                turn -= 1
                continue
            msgs.append(Message.assistant(resp.content, resp.tool_calls))
            _notify(self.on_turn, turn, msgs[-1])

            if not resp.tool_calls:
                return AgentResult(resp.content, msgs, turn, used)

            for tc in resp.tool_calls:
                used.append(tc.name)
                _notify(self.on_tool_call, tc.name, tc.arguments)
                try:
                    result = self.registry.call(tc.name, tc.arguments)
                except _TerminateSignal as sig:
                    self._terminated = sig
                    return AgentResult(
                        sig.summary, msgs, turn, used,
                        self_terminated=True, termination_reason=sig.reason,
                    )
                _notify(self.on_tool_result, tc.name, result)
                msgs.append(Message.tool(self._bounded(result.output, tc.name),
                                         tc.id, tc.name))
                # A note typed while that tool ran lands here, before the model
                # is asked again — a forge can take minutes, and waiting until
                # the task ended would make the correction useless.
                if self._absorb(msgs):
                    return self._stopped(msgs, turn, used)


__all__ = ["Agent", "AgentResult", "DEFAULT_SYSTEM", "_TerminateSignal"]