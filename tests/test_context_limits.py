"""One message cannot own the window, and a wedged run leaves evidence.

Three failures, one file, because they are one failure seen from three
distances:

* `bound_output` — a tool result enters the context already cut to a head and a
  tail, so one `find` over a repo, or one build log, cannot put a transcript
  past every cut that compaction could make. Ported from Hermes' per-message
  truncation limits (`_CONTENT_MAX` / head / tail).
* `dump_transcript` — when a compaction declines, or fails to reduce, the
  transcript is written to disk with its heaviest messages named, so "why is
  this run stuck?" has an answer that is not a token estimate and a shrug.
* `plan_cut(tail_tokens=...)` — the protected tail is *measured* in tokens
  instead of counted in groups: Hermes' `tail_token_budget`.

Plus the wiring: a cap that the loop does not apply protects nothing, so the
last test drives a real agent through a tool that returns two million
characters and reads what the model would have been sent.
"""
from __future__ import annotations

import json
import os

from autoforge.agent import ForgeAgent
from autoforge.core import compaction as C
from autoforge.core.compaction import (
    OUTPUT_HEAD_CHARS,
    OUTPUT_MAX_CHARS,
    OUTPUT_TAIL_CHARS,
    Message,
    bound_output,
    dump_transcript,
    estimate_text_tokens,
    estimate_tokens,
    plan_cut,
    spill_dir,
)
from autoforge.core.llm import LLMResponse, MockLLMClient, ToolCall
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec


# ---------------------------------------------------------------------------
# a head, a tail, and a way back to the whole thing
# ---------------------------------------------------------------------------
def test_a_result_under_the_cap_is_untouched():
    """The common case pays a length check and nothing else.

    No footer, no marker, no reformatting: a bounded pass that rewrote every
    small result would be a bug wearing a fix's clothes.
    """
    text = "line one\nline two\n"
    assert bound_output(text) == text
    assert bound_output("x" * OUTPUT_MAX_CHARS) == "x" * OUTPUT_MAX_CHARS


def test_a_long_result_keeps_its_head_and_its_tail():
    text = "head" * 100_000
    out = bound_output(text, name="dump")

    assert "[TRUNCATED]" in out
    assert out.startswith(text[:OUTPUT_HEAD_CHARS])
    assert out.endswith(text[-OUTPUT_TAIL_CHARS:])
    # The arithmetic in the footer has to be the arithmetic it did.
    omitted = len(text) - OUTPUT_HEAD_CHARS - OUTPUT_TAIL_CHARS
    assert f"{omitted:,} chars omitted" in out


def test_the_whole_output_is_recoverable_from_the_spill_file():
    """A cut the model cannot go back from is a result thrown away."""
    text = "abcdefghij" * 50_000
    out = bound_output(text, name="build log")
    path = _spilled(out)

    assert path.startswith(spill_dir())
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == text


def _spilled(rendered: str) -> str:
    """The path the footer names, so the test reads what the model was told."""
    for line in rendered.splitlines():
        if "The complete text is at " in line:
            return line.split("The complete text is at ", 1)[1].split(" — ")[0]
    raise AssertionError(f"no spill path in {rendered[:400]!r}")


def test_the_cut_lands_on_a_line_boundary():
    """An excerpt that starts mid-token invites the model to guess.

    Same rule the web fetcher's truncation follows: back up to the last newline
    when one is close, forward to the next when one is close, and take the
    character boundary only when the line is enormous.
    """
    body = "\n".join(f"row {i} " + "y" * 60 for i in range(2_000))
    out = bound_output(body, max_chars=4_000, head_chars=2_000, tail_chars=1_000)

    head = out.split("\n\n")[0]                 # rows are single-spaced, so the
    tail = out.rsplit("\n\n", 1)[-1]            # first blank line ends each part
    assert body.startswith(head) and body[len(head)] == "\n"
    assert body.endswith(tail) and body[-len(tail) - 1] == "\n"


def test_a_zero_cap_turns_the_bound_off():
    text = "z" * (OUTPUT_MAX_CHARS * 3)
    assert bound_output(text, max_chars=0) == text


def test_cjk_is_cut_by_characters_not_bytes():
    text = "汉字测试" * 20_000          # 80k chars, 240k bytes in UTF-8
    out = bound_output(text)
    assert len(out) < len(text)
    assert out.startswith("汉字测试" * 100)


# ---------------------------------------------------------------------------
# the transcript on disk
# ---------------------------------------------------------------------------
def test_a_transcript_names_the_heaviest_messages_first(tmp_path):
    """Which message is carrying the tokens — the question a wedged run asks."""
    msgs = [
        Message.system("sys"),
        Message.user("task"),
        Message.assistant("small", [ToolCall("c0", "read_file", {})]),
        Message.tool("x" * 40_000, "c0", "read_file"),
        Message.tool("tiny", "c1", "read_file"),
    ]
    path = dump_transcript(msgs, "compaction-1", turn=7, directory=str(tmp_path))

    lines = open(path, encoding="utf-8").read().splitlines()
    header = json.loads(lines[0])
    assert header["_transcript"] and header["turn"] == 7
    assert header["messages"] == len(msgs) == len(lines) - 1
    assert header["sizes"][0]["role"] == "tool"
    assert header["sizes"][0]["index"] == 3
    assert header["sizes"][0]["tokens"] == estimate_text_tokens(msgs[3].content)
    assert header["sizes"][1]["tokens"] <= header["sizes"][0]["tokens"]
    # Every line is a message the provider would accept, so the file can be
    # replayed rather than only read.
    assert all(json.loads(line)["role"] for line in lines[1:])


def test_an_unwritable_transcript_is_not_a_lost_run(tmp_path):
    """The diagnosis must never be what breaks the run."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    assert dump_transcript([Message.user("x")], "why", directory=str(blocker / "sub")) == ""


# ---------------------------------------------------------------------------
# what a run is allowed to leave on the disk
# ---------------------------------------------------------------------------
def test_an_output_too_large_to_spill_is_refused_out_loud(monkeypatch):
    """A 4 GB aside file is not a way back to anything.

    `read_file` over a huge artifact should not park a copy of it under the
    state directory as a side effect of showing a 24 K character excerpt. The
    refusal is stated in the text, because the useful advice changes with the
    reason: a lost file is worth re-running, a file that is always too big is
    not.
    """
    monkeypatch.setattr(C, "SPILL_MAX_CHARS", 1_000)
    out = bound_output("h" * 5_000, max_chars=2_000, head_chars=1_000,
                       tail_chars=500)

    assert "was not written" in out
    assert "Narrow the request" in out
    assert not os.path.isdir(C.spill_dir())        # nothing landed


def test_old_asides_are_pruned_to_the_budget(monkeypatch):
    """A bounded run must not have an unbounded shadow on disk."""
    monkeypatch.setattr(C, "SPILL_BUDGET_BYTES", 3_000)
    os.makedirs(C.spill_dir(), exist_ok=True)
    made = []
    for i in range(3):
        p = os.path.join(C.spill_dir(), f"old{i}.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("x" * 2_000)
        os.utime(p, (1_000 + i, 1_000 + i))         # deterministic age
        made.append(p)

    removed = C.prune_dumps()

    assert made[0] in removed                       # oldest goes first
    assert os.path.exists(made[-1])                 # newest is still reachable
    left = sum(os.stat(p).st_size for p in made if os.path.exists(p))
    assert left <= 3_000


def test_a_live_runs_own_aside_is_never_pruned(monkeypatch):
    """The footer may name it, so deleting it would break the way back.

    Pruning is oldest-first for this reason, but age is not enough on its own:
    a long run's first spill is the oldest file on disk and the one a model in
    that same run is most likely to have been told to read.
    """
    path = C.spill_oversized("y" * 30_000, "live")
    assert path and os.path.exists(path)

    monkeypatch.setattr(C, "SPILL_BUDGET_BYTES", 0)   # demand everything go
    removed = C.prune_dumps()

    assert path not in removed
    assert os.path.exists(path)


def test_transcripts_share_the_budget(monkeypatch):
    """The diagnostic record is not exempt from the cost of keeping it."""
    monkeypatch.setattr(C, "SPILL_BUDGET_BYTES", 100)
    os.makedirs(C.transcripts_dir(), exist_ok=True)
    stale = os.path.join(C.transcripts_dir(), "turn-0001-old.jsonl")
    with open(stale, "w", encoding="utf-8") as fh:
        fh.write("x" * 5_000)
    os.utime(stale, (1_000, 1_000))

    assert stale in C.prune_dumps()
    assert not os.path.exists(stale)


def test_pruning_with_nothing_to_do_is_a_no_op(monkeypatch):
    monkeypatch.setattr(C, "SPILL_BUDGET_BYTES", 10_000_000)
    assert C.prune_dumps() == []


# ---------------------------------------------------------------------------
# the tail is measured, not counted
# ---------------------------------------------------------------------------
def _groups(n: int, body: int) -> list[Message]:
    msgs = [Message.system("sys"), Message.user("task")]
    for i in range(n):
        msgs.append(Message.assistant(f"step {i}",
                                      [ToolCall(f"c{i}", "read_file", {})]))
        msgs.append(Message.tool("x" * body, f"c{i}", "read_file"))
    return msgs


def test_the_tail_budget_protects_more_than_the_floor():
    """Six groups of tool results are not six groups of prose.

    A counted tail means the same number protects a tenth of the window in one
    transcript and four windows in another. The budget makes "what is still
    live" mean one thing.
    """
    msgs = _groups(40, body=4_000)          # ~1k tokens per group
    floor_only = plan_cut(msgs, keep_head=2, keep_recent_groups=3)
    budgeted = plan_cut(msgs, keep_head=2, keep_recent_groups=3, tail_tokens=25_000)

    assert budgeted < floor_only            # protects more, cuts earlier
    assert estimate_tokens(msgs[budgeted:]) <= 25_000


def test_the_budget_never_cuts_below_the_floor():
    msgs = _groups(40, body=4_000)
    assert (plan_cut(msgs, keep_head=2, keep_recent_groups=3, tail_tokens=1)
            == plan_cut(msgs, keep_head=2, keep_recent_groups=3))


# ---------------------------------------------------------------------------
# and the loop applies it
# ---------------------------------------------------------------------------
def test_the_loop_bounds_a_huge_tool_result_before_the_model_sees_it():
    """The cap has to be applied by the loop, or it is a helper nobody calls.

    Two million characters from one tool call is not a hypothetical: it is a
    recursive grep, a build log, or a whole-file read on a repository that is
    not a toy. Unbounded, that single message walks the transcript past every
    cut, and compaction — which can only cut *between* messages — is left with
    nothing it can do.
    """
    big = 2_000_000
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="dump",
        description="return far too much",
        parameters={"type": "object", "properties": {}},
        fn=lambda: "z" * big,
    ))
    llm = MockLLMClient(script=[
        LLMResponse(content="", tool_calls=[ToolCall("c0", "dump", {})]),
        LLMResponse(content="done"),
    ])
    a = ForgeAgent(llm=llm, registry=reg, max_turns=4)
    result = a.run("dump it")

    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    sent = tool_msgs[0].content
    assert len(sent) < 30_000, len(sent)
    assert "[TRUNCATED]" in sent
    assert sent.startswith("z" * 1_000)     # the beginning survives
    assert sent.rstrip().endswith("z")      # and so does the end
    # The operator's callback saw the whole thing, and so did the disk: bounding
    # is about what the model is sent, not about throwing the result away.
    path = _spilled(sent)
    with open(path, encoding="utf-8") as fh:
        assert len(fh.read()) == big


def test_the_loop_does_not_park_an_unreadable_copy_on_the_disk(monkeypatch):
    """The cap has to reach the wire through the loop, refusal included.

    Same path as above, with the per-output cap turned down to a size a test can
    afford to produce. What matters is which footer the model receives: a
    refusal that says "narrow the request", not a path to a file that was never
    written.
    """
    monkeypatch.setattr(C, "SPILL_MAX_CHARS", 50_000)
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="dump",
        description="return far too much",
        parameters={"type": "object", "properties": {}},
        fn=lambda: "z" * 300_000,
    ))
    llm = MockLLMClient(script=[
        LLMResponse(content="", tool_calls=[ToolCall("c0", "dump", {})]),
        LLMResponse(content="done"),
    ])
    a = ForgeAgent(llm=llm, registry=reg, max_turns=4)
    result = a.run("dump it")

    sent = [m for m in result.messages if m.role == "tool"][0].content
    assert "[TRUNCATED]" in sent
    assert "was not written" in sent
    assert "Narrow the request" in sent
    assert not os.path.isdir(C.spill_dir())     # and nothing landed
