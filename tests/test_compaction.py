"""Context compaction: a long run must not forget why it started.

The interesting half of this is not "does it shrink the list". It is the four
things that make shrinking it *safe*, each of which has an obvious wrong
version that only shows up much later:

* the cut never orphans a tool result (the provider answers that with a 400,
  mid-run, naming a `tool_call_id` the caller never wrote);
* the operator's own lines survive — verbatim, not paraphrased;
* a summarizer that fails leaves the history alone instead of leaving nothing;
* and the loop actually calls it, since a compactor nobody invokes is the same
  fiction as a policy field nobody reads.

Tested against real objects: a real `Agent`, a real `ToolRegistry`, a real
socket-free `MockLLMClient`.
"""
from __future__ import annotations

import pytest

from autoforge.core.agent import Agent
from autoforge.core.compaction import (CompactionEvent, CompactionPolicy,
                                      Compactor, DeterministicSummarizer,
                                      LLMSummarizer, estimate_text_tokens,
                                      estimate_tokens, group_starts,
                                      is_compaction_note, plan_cut)
from autoforge.core.llm import LLMResponse, MockLLMClient, ToolCall
from autoforge.core.message import Message
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def api_problems(msgs) -> list[str]:
    """Reasons a real OpenAI-compatible endpoint would reject this list.

    The rule, in one place so every test checks the same thing: a `tool` message
    is only legal while a tool call it answers is still outstanding, and every
    outstanding call must be answered before the next non-tool message.
    """
    problems: list[str] = []
    pending: set[str] = set()
    for m in msgs:
        if m.role == "assistant" and m.tool_calls:
            if pending:
                problems.append(f"{len(pending)} call(s) unanswered before a new assistant msg")
            pending = {tc.id for tc in m.tool_calls}
        elif m.role == "tool":
            if m.tool_call_id not in pending:
                problems.append(f"tool result {m.tool_call_id!r} answers nothing")
            else:
                pending.discard(m.tool_call_id)
        else:
            if pending:
                problems.append(f"{len(pending)} call(s) unanswered before a {m.role} msg")
            pending = set()
    if pending:
        problems.append(f"{len(pending)} call(s) left unanswered at the end")
    return problems


def long_history(turns: int = 30, *, task: str = "do the thing",
                 body: int = 400, n_results: int = 1) -> list[Message]:
    """A plausible transcript: system, task, then tool turns."""
    msgs = [Message.system("you are a capable agent"), Message.user(task)]
    for i in range(turns):
        calls = [ToolCall(id=f"c{i}_{k}", name="read_file",
                          arguments={"path": f"/work/file{i}_{k}.txt"})
                 for k in range(n_results)]
        msgs.append(Message.assistant(f"step {i}", calls))
        for k in range(n_results):
            msgs.append(Message.tool("x" * body, f"c{i}_{k}", "read_file"))
    return msgs


def tiny_policy(**over) -> CompactionPolicy:
    """A window small enough for a test to cross it, and no log file."""
    kw = dict(max_context_tokens=1500, keep_recent_groups=3, persist=False)
    kw.update(over)
    return CompactionPolicy(**kw)


def compactor(**over) -> Compactor:
    return Compactor(tiny_policy(**over),
                     summarizer=None, fallback=DeterministicSummarizer())


# ---------------------------------------------------------------------------
# estimating
# ---------------------------------------------------------------------------
def test_empty_costs_nothing():
    assert estimate_text_tokens("") == 0
    assert estimate_tokens([]) == 0


def test_cjk_is_estimated_far_heavier_than_latin():
    """A char-per-token script must not be estimated at four chars per token.

    Underestimating here does not make the summary worse — it makes the run
    reach the provider over its window, which kills it at the far end.
    """
    cjk = estimate_text_tokens("这是一个测试" * 10)
    latin = estimate_text_tokens("a" * 60)
    # (docstring said 40; the string is 60 chars -> ~16 tokens)
    assert cjk > latin, (cjk, latin)
    assert cjk >= 60, cjk           # at least one token per CJK char


def test_estimate_counts_tool_arguments_not_just_content():
    """Tool arguments are part of the request and can dwarf the content."""
    bare = [Message.assistant("", [ToolCall("c1", "write_file", {})])]
    fat = [Message.assistant("", [ToolCall("c1", "write_file",
                                           {"body": "y" * 4000})])]
    assert estimate_tokens(fat) > estimate_tokens(bare) + 500


# ---------------------------------------------------------------------------
# where it is safe to cut
# ---------------------------------------------------------------------------
def test_groups_keep_a_multi_result_turn_together():
    msgs = long_history(turns=3, n_results=4)
    # [system, user] then one group per turn: an assistant message plus the four
    # results answering it. A cut is only offered at 0, 1, 2, 7 and 12.
    assert group_starts(msgs) == [0, 1, 2, 7, 12]


def test_plan_cut_never_lands_on_a_tool_result():
    msgs = long_history(turns=30)
    for keep in range(1, 12):
        cut = plan_cut(msgs, keep_head=2, keep_recent_groups=keep)
        if cut is not None:
            assert msgs[cut].role != "tool", (keep, cut)


def test_plan_cut_keeps_the_operators_last_line_out_of_the_summary():
    """Their newest instruction is an instruction, not history."""
    msgs = long_history(turns=20)
    msgs.insert(8, Message.user("actually, use the other file"))
    cut = plan_cut(msgs, keep_head=2, keep_recent_groups=3)
    assert cut is not None and cut <= 8


def test_a_compaction_note_does_not_count_as_the_operators_last_word():
    """A note is a `user` message because that is the role providers accept for
    injected context — not because a person wrote it. Counting it as the
    operator's last instruction would clamp the cut to the note's own position
    and stop the run compacting ever again."""
    msgs = long_history(turns=40)
    plain = plan_cut(long_history(turns=40), keep_head=2, keep_recent_groups=3)
    msgs[2:2] = [Message.user("[context compacted at turn 1]\n\nearlier stuff")]
    cut = plan_cut(msgs, keep_head=2, keep_recent_groups=3)
    # One extra message in, one extra message dropped — the note travelled into
    # the compacted range instead of pinning the cut to index 2.
    assert cut is not None and cut == plain + 1
    assert cut > 3


def test_plan_cut_declines_when_there_is_nothing_worth_doing():
    msgs = long_history(turns=2)
    assert plan_cut(msgs, keep_head=2, keep_recent_groups=6) is None


# ---------------------------------------------------------------------------
# the summary itself
# ---------------------------------------------------------------------------
def test_deterministic_summary_is_structural_and_says_so():
    msgs = long_history(turns=6)
    text = DeterministicSummarizer().summarize(msgs)
    assert "6 assistant turn(s)" in text
    assert "read_file×6" in text
    assert "/work/file0_0.txt" in text


def steered(*lines: str, before: int = 40, after: int = 8) -> list[Message]:
    """A history in which every `line` but the last lands in the dropped range.

    `plan_cut` clamps the cut to the operator's *last* message, so pushing an
    earlier operator line into the compacted range takes a later one after it —
    which is exactly the shape a steered long run has, and the only shape in
    which "did the operator's words survive" is a real question.
    """
    msgs = long_history(turns=before)
    for line in lines:
        msgs.append(Message.user(line))
        msgs.extend(long_history(turns=after)[2:])
    return msgs


def context_of(msgs) -> str:
    return "\n".join(m.content for m in msgs)


def test_operator_words_survive_in_the_compacted_context_not_in_the_summary():
    """The deterministic summarizer reports structure; it does not carry the
    operator's words. The Compactor does, structurally — so the guarantee holds
    even when the summarizer is a model that paraphrases, or fails entirely.

    Testing it on the summarizer would be testing the wrong object: a promise
    that has to survive a summarizer cannot be delegated to one.
    """
    msgs = steered("stop using tabs", "carry on")
    c = Compactor(tiny_policy(), summarizer=DeterministicSummarizer(),
                  fallback=None)
    ev = c.maybe_compact(msgs, 1)
    assert ev is not None and ev.acted, ev and ev.error
    # Absent from the summary text...
    assert "stop using tabs" not in ev.summary
    # ...present in the context the model will actually be sent.
    assert "stop using tabs" in context_of(msgs)


def test_operator_words_survive_a_summarizer_that_lies():
    """A summarizer that invents plausible prose must not be able to erase an
    instruction, because it never held the instruction in the first place."""
    class Gaslighter:
        name = "gaslighter"

        def summarize(self, msgs):
            return "The operator asked for nothing of note."

    msgs = steered("NEVER touch prod", "carry on")
    c = Compactor(tiny_policy(), summarizer=Gaslighter(),
                  fallback=DeterministicSummarizer())
    ev = c.maybe_compact(msgs, 1)
    assert ev is not None and ev.acted, ev and ev.error
    assert "NEVER touch prod" in context_of(msgs)


def test_operator_words_survive_a_second_compaction():
    """The generation that kills a naive implementation: the first note is a
    `user` message, so a second pass that re-derives operator lines from the
    dropped range finds none — the first generation's instruction is inside a
    note, and notes are skipped. The accumulator is what prevents that."""
    c = Compactor(tiny_policy(), summarizer=DeterministicSummarizer(),
                  fallback=None)
    msgs = steered("use the OTHER branch", "carry on")
    first = c.maybe_compact(msgs, 1)
    assert first is not None and first.acted, first and first.error

    # More work, then a second cut — with the first note now inside it.
    msgs.extend(long_history(turns=20)[2:])
    msgs.append(Message.user("still going"))
    msgs.extend(long_history(turns=4)[2:])
    ev = c.maybe_compact(msgs, 2)
    assert ev is not None and ev.acted, ev and ev.error
    assert "use the OTHER branch" in context_of(msgs)


def test_the_operator_block_is_capped_loudly_rather_than_silently():
    """A cap is unavoidable; a silent one is amnesia in a smaller font."""
    c = Compactor(tiny_policy(), summarizer=DeterministicSummarizer(),
                  fallback=None, max_operator_chars=40)
    msgs = steered("a" * 30, "b" * 30, "carry on")
    ev = c.maybe_compact(msgs, 1)
    assert ev is not None and ev.acted, ev and ev.error
    text = context_of(msgs)
    assert "did not fit" in text
    assert "1 further operator message(s)" in text


def test_llm_summarizer_sends_no_tools_and_no_history():
    """A summarizer that could call tools could change the world it is
    compacting the record of."""
    llm = MockLLMClient(handler=lambda *a, **k: LLMResponse(content="GOAL: x"))
    text = LLMSummarizer(llm).summarize(long_history(turns=2))
    assert text == "GOAL: x"
    messages, tools = llm.calls[0]
    assert tools is None
    assert len(messages) == 2                 # system + one transcript
    assert "GOAL" in messages[0].content      # the instruction names the headings


def test_llm_summarizer_refuses_an_empty_reply():
    """An empty reply is the reasoning-budget failure, and it must raise so the
    fallback runs rather than compacting to a blank note."""
    llm = MockLLMClient(handler=lambda *a, **k: LLMResponse(content="   "))
    with pytest.raises(Exception):
        LLMSummarizer(llm).summarize(long_history(turns=2))


def test_transcript_elision_is_announced():
    llm = MockLLMClient(handler=lambda *a, **k: LLMResponse(content="ok"))
    s = LLMSummarizer(llm, max_transcript_chars=400)
    s.summarize(long_history(turns=40))
    sent = llm.calls[0][0][1].content
    assert "chars skipped" in sent


# ---------------------------------------------------------------------------
# compacting
# ---------------------------------------------------------------------------
def test_below_threshold_is_a_no_op():
    msgs = long_history(turns=2)
    before = list(msgs)
    c = compactor(max_context_tokens=10_000_000)
    assert c.maybe_compact(msgs, 1) is None
    assert msgs == before


def test_compaction_shrinks_the_list_and_stays_valid():
    msgs = long_history(turns=30)
    assert api_problems(msgs) == []          # the fixture itself is legal
    c = compactor()
    ev = c.maybe_compact(msgs, 7)
    assert ev is not None and ev.acted
    assert ev.after_tokens < ev.before_tokens
    assert api_problems(msgs) == [], api_problems(msgs)
    assert msgs[0].role == "system"
    assert msgs[1].content == "do the thing"       # task kept as written


def test_the_task_survives_even_when_history_is_prepended():
    """`run(task, history)` puts the task last, not at index 1."""
    history = long_history(turns=20)
    msgs = list(history) + [Message.user("THE REAL TASK")]
    c = compactor()
    ev = c.maybe_compact(msgs, 3)
    assert ev is not None and ev.acted
    assert msgs[-1].content == "THE REAL TASK"
    assert "THE REAL TASK" not in "".join(m.content for m in msgs[:-1])


def test_the_note_declares_its_own_fidelity():
    msgs = long_history(turns=30)
    compactor().maybe_compact(msgs, 4)
    note = next(m for m in msgs if m.role == "user" and "context compacted" in m.content)
    assert "history, not a new request" in note.content
    assert "produced without a model" in note.content
    assert "NOT preserved" in note.content


def test_an_llm_summary_declares_a_different_fidelity():
    msgs = long_history(turns=30)
    llm = MockLLMClient(handler=lambda *a, **k: LLMResponse(content="GOAL: it"))
    c = Compactor(tiny_policy(), summarizer=LLMSummarizer(llm))
    ev = c.maybe_compact(msgs, 2)
    assert ev is not None and ev.summarizer == "llm"
    note = next(m for m in msgs if m.role == "user" and "context compacted" in m.content)
    assert "paraphrased" in note.content


def test_a_failing_summarizer_falls_back_rather_than_dropping():
    msgs = long_history(turns=30)
    n_before = len(msgs)

    def boom(*a, **k):
        raise RuntimeError("gateway 503")

    c = Compactor(tiny_policy(), summarizer=LLMSummarizer(MockLLMClient(handler=boom)))
    ev = c.maybe_compact(msgs, 1)
    assert ev is not None and ev.acted
    assert ev.summarizer == "deterministic"     # the fallback did it
    assert len(msgs) < n_before


def test_no_summary_at_all_leaves_the_history_alone():
    """The whole point: dropping the range with nothing to replace it is the
    silent amnesia this module exists to prevent."""
    msgs = long_history(turns=30)
    before = list(msgs)
    c = Compactor(tiny_policy(), summarizer=None, fallback=None)
    ev = c.maybe_compact(msgs, 1)
    assert ev is not None and not ev.acted
    assert msgs == before
    assert "no summarizer" in ev.error
    assert "skipped" in ev.describe()


def test_the_note_reports_the_real_number_of_dropped_messages():
    """The note's whole job is to tell the model what it is looking at. A frame
    that says "0 earlier messages were replaced" while sitting where forty used
    to be is worse than no frame: it is a lie with a citation.

    This shipped broken once — `event.dropped` was assigned after `_frame` ran,
    so every note claimed zero.
    """
    import re
    msgs = long_history(turns=40)
    c = Compactor(tiny_policy(), summarizer=DeterministicSummarizer(),
                  fallback=None, log_path="")
    ev = c.maybe_compact(msgs, 1)
    assert ev is not None and ev.acted
    note = next(m.content for m in msgs if is_compaction_note(m))
    claimed = int(re.search(r"(\d+) earlier message\(s\)", note).group(1))
    assert claimed == ev.dropped > 0


def test_ineffective_compaction_backs_off_after_two_strikes():
    """A cut that cannot shrink the context is retried once, then dropped.

    Ported from Hermes' anti-thrash rule (`_MAX_INEFFECTIVE_COMPRESSIONS`).
    One ineffective pass is not a verdict: the summarizer can be transiently
    bad, and the flag this replaces was latched on the first failure — which
    meant one oversized message disabled compaction for the whole run while the
    transcript kept growing. Two consecutive passes with no reduction is where
    retrying is provably pointless, because the next turn asks the same
    question of the same list.
    """
    class Huge:
        name = "huge"

        def summarize(self, msgs):
            return "y" * 200_000

    msgs = long_history(turns=20)
    c = Compactor(tiny_policy(), summarizer=Huge(), fallback=None)
    ev = c.maybe_compact(msgs, 1)
    assert ev is not None and not ev.acted
    assert "did not reduce" in ev.error
    assert "Strike 1 of 2" in ev.error
    assert c.stalled is False                # one pass is not yet a verdict
    assert c.report()["stalled"] is False

    c.maybe_compact(msgs, 2)
    assert c.stalled is True                 # the second one makes it one
    assert c.report()["stalled"] is True
    assert c.report()["ineffective"] == 2
    assert c.maybe_compact(msgs, 3) is None  # and now it stops paying


def test_strikes_clear_when_the_context_is_under_the_line_again():
    """The ledger tracks a condition, not a history of failures.

    A strike earned while the context was over the threshold must not go on
    suppressing compaction after the context is back under it: that is the bug
    the latch had, and the reason Hermes resets the count on any reading that
    clears the line.
    """
    msgs = long_history(turns=20)
    c = Compactor(tiny_policy(max_context_tokens=10_000_000))
    c.ineffective = 2
    assert c.stalled is True                 # blocked, as if from two bad cuts
    assert c.maybe_compact(msgs, 1) is None
    assert c.ineffective == 0 and c.stalled is False


def test_a_grown_transcript_earns_one_probe():
    """Blocked is not sealed: growth past the blocked size buys one more look.

    Hermes allows its probe after 300 seconds of continuous block
    (`_ANTI_THRASH_RECOVERY_SECONDS`) on the reasoning that the transcript has
    changed underneath the decision. Growth is the same signal without a clock,
    and exactly one strike is granted, so a transcript that still cannot be
    compacted re-trips on the next pass instead of looping.
    """
    class Huge:
        name = "huge"

        def summarize(self, msgs):
            return "y" * 200_000

    msgs = long_history(turns=20)
    c = Compactor(tiny_policy(), summarizer=Huge(), fallback=None)
    c.maybe_compact(msgs, 1)
    c.maybe_compact(msgs, 2)
    assert c.stalled is True
    blocked_at = c.blocked_at_tokens

    # Barely grown is not grown enough: still blocked, no summarizer call.
    assert c.maybe_compact(msgs, 3) is None

    msgs.extend(long_history(turns=60)[2:])
    assert estimate_tokens(msgs) > blocked_at * 1.5
    ev = c.maybe_compact(msgs, 4)            # the probe is allowed through
    assert ev is not None and not ev.acted
    assert c.stalled is True                 # it was ineffective, so it re-trips


def test_an_uncuttable_pass_moves_the_counter():
    """No cut is a strike, same as no reduction.

    A transcript that is over the threshold with nothing safe to cut answers
    every future turn identically. Counting that as ineffective is what stops
    the loop from paying a fresh planning pass on every turn — Hermes counts its
    too-few-messages case for the same reason.
    """
    msgs = [Message.system("sys"), Message.user("task"),
            Message.tool("z" * 400_000, "c0", "read_file")]
    c = Compactor(tiny_policy())
    ev = c.maybe_compact(msgs, 1)
    assert ev is not None and not ev.acted
    assert "too short to cut" in ev.error
    assert c.ineffective == 1


def test_persistence_appends_and_reports_the_failure_instead_of_raising(tmp_path):
    msgs = long_history(turns=30)
    log = tmp_path / "sub" / "compaction.log.md"
    c = Compactor(tiny_policy(persist=True), log_path=str(log))
    ev = c.maybe_compact(msgs, 1)
    assert ev.persisted == str(log) and log.exists()
    assert "context compacted" in log.read_text(encoding="utf-8")

    # An unwritable path is a lost file, not a lost run.
    msgs2 = long_history(turns=30)
    bad = Compactor(tiny_policy(persist=True),
                    log_path=str(tmp_path / "nope" / "x" / "y"))
    (tmp_path / "nope").write_text("i am a file, not a directory")
    ev2 = bad.maybe_compact(msgs2, 2)
    assert ev2.acted and ev2.persisted == ""
    assert "not persisted" in ev2.error


def test_disabled_policy_does_nothing():
    msgs = long_history(turns=30)
    before = list(msgs)
    c = compactor(enabled=False)
    assert c.maybe_compact(msgs, 1) is None
    assert msgs == before


def test_keep_recent_groups_is_clamped_rather_than_crashing_the_run():
    assert CompactionPolicy(keep_recent_groups=0).keep_recent_groups == 1
    assert CompactionPolicy(keep_head=0).keep_head == 1


def test_report_counts_only_what_actually_happened():
    msgs = long_history(turns=30)
    c = compactor()
    c.maybe_compact(msgs, 1)
    rep = c.report()
    assert rep["compactions"] == 1
    assert rep["messages_dropped"] > 0
    assert rep["tokens_saved"] > 0


# ---------------------------------------------------------------------------
# the loop actually calls it
# ---------------------------------------------------------------------------
def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    def read_file(path: str = "") -> str:
        return "x" * 400

    reg.register(ToolSpec(name="read_file", description="read",
                          parameters={"type": "object",
                                      "properties": {"path": {"type": "string"}}},
                          fn=read_file, source="builtin",
                          effect_signature="read_only"))
    return reg


def test_the_loop_compacts_so_a_late_request_sees_a_smaller_context():
    seen: list[list[Message]] = []

    def handler(messages, tools, **kw):
        seen.append(list(messages))
        return LLMResponse(content="", tool_calls=[
            ToolCall(id=f"t{len(seen)}", name="read_file",
                     arguments={"path": f"/work/f{len(seen)}.txt"})])

    events = []
    agent = Agent(MockLLMClient(handler=handler), _registry(),
                  max_turns=40, compactor=compactor(),
                  on_compact=events.append)
    agent.run("do the thing")

    assert events and any(e.acted for e in events)
    sizes = [len(s) for s in seen]
    assert min(sizes) < max(sizes), "the context never shrank"
    # Every request the model saw was legal, including the compacted ones.
    for snap in seen:
        assert api_problems(snap) == [], api_problems(snap)
    # And the compaction note reached the model.
    assert any(any("context compacted" in m.content for m in s) for s in seen)


def test_a_steered_line_typed_mid_run_is_never_lost():
    """The steering channel promises an operator's line is never dropped. It
    would be a strange thing to build that and let compaction eat the same line
    an hour later."""

    class Drip:
        def __init__(self) -> None:
            self.q = ["please use the OTHER branch"]

        def take_supplements(self):
            return [self.q.pop(0)] if self.q else []

        def stop_requested(self) -> bool:
            return False

    def handler(messages, tools, **kw):
        return LLMResponse(content="", tool_calls=[
            ToolCall(id=f"t{len(messages)}", name="read_file",
                     arguments={"path": "/work/f.txt"})])

    agent = Agent(MockLLMClient(handler=handler), _registry(),
                  max_turns=40, compactor=compactor(), steer=Drip())
    result = agent.run("do the thing")

    text = "\n".join(m.content for m in result.messages)
    assert "please use the OTHER branch" in text


def test_an_invoked_compactor_is_reported_not_silent():
    """A compaction the operator never sees is a run that quietly stops
    knowing things, which is the failure this whole module is about."""
    def handler(messages, tools, **kw):
        return LLMResponse(content="", tool_calls=[
            ToolCall(id=f"t{len(messages)}", name="read_file", arguments={"path": "/w/f"})])

    events = []
    Agent(MockLLMClient(handler=handler), _registry(), max_turns=30,
          compactor=compactor(), on_compact=events.append).run("task")
    assert events, "the loop compacted without telling anyone"
    assert events[0].describe().startswith("compacted")


def test_a_broken_observer_does_not_take_the_run_down():
    def handler(messages, tools, **kw):
        return LLMResponse(content="", tool_calls=[
            ToolCall(id=f"t{len(messages)}", name="read_file", arguments={"path": "/w/f"})])

    def exploding(event):
        raise ValueError("my printer is on fire")

    result = Agent(MockLLMClient(handler=handler), _registry(), max_turns=30,
                   compactor=compactor(), on_compact=exploding).run("task")
    assert result.turns > 0
