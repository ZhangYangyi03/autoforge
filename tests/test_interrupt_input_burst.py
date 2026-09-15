"""A pasted block is one message, even when the terminal dribbles it out.

The failure this pins down
--------------------------
`_looks_like_paste` tells a block from an Enter by counting logical breaks in
one read burst: one break, at the very end, is Enter. That rule is right for a
fast terminal, which hands the whole block over inside a single burst. It is
wrong for a slow one — a large table, a laggy console, a paste across a remote
session — which hands the block over a line at a time with a gap between the
lines. Every one of those lines is one break, at the end, so every one of them
reads as Enter, and a block the operator pasted once arrives as N steering
messages and N interruptions.

The editor can tell the two apart without guessing: if the next line's bytes
are already on their way when the break lands, the person at the keyboard did
not send them. `SUBMIT_GRACE` is how long a break waits to find out.

The other half is the case where the gap is longer than that wait, so the
fragments really do arrive as separate submits. A burst of lines arriving
while a run is live is folded into the line before it, and the agent is
interrupted once instead of N times.

Timing in these tests is modelled, not slept: `SlowPasteTerm.ready` answers
from the gap it was given, so the relationship under test — gap longer than
the editor's burst window, shorter than `SUBMIT_GRACE` — is exact and the
suite stays fast.
"""
from __future__ import annotations

import io

from autoforge.core.lineedit import (
    SUBMIT_GRACE,
    LineEditor,
    expand_paste_refs,
)
from autoforge.core.steering import Steering

from _terminal import FakeTerm, Out

TABLE = "| a | b |\n| c | d |\n| e | f |\n"


class SlowPasteTerm(FakeTerm):
    """A paste delivered a line at a time, with a gap between the lines.

    The gap is longer than the editor's burst-gathering window (0.02s) and
    shorter than `SUBMIT_GRACE`, which is exactly the real case: the editor
    sees one burst per line, and each line ends in a single break.
    """

    def __init__(self, chunks, columns=60, gap=0.06):
        super().__init__(chunks, columns=columns)
        self.gap = gap

    def ready(self, timeout):
        # Nothing is buffered *yet* — but the next line is on its way, and a
        # caller who is willing to wait `gap` for it will find it there.
        return timeout >= self.gap and bool(self.chunks)


def editor_for(term, **kw):
    editor = LineEditor(term=term, out=Out(), **kw)
    editor.start()
    editor.set_prompt("you> ")
    return editor


# -- the editor: a break with input behind it is not Enter --------------
def test_a_paste_that_arrives_a_line_at_a_time_is_one_message(tmp_path):
    editor = editor_for(SlowPasteTerm(TABLE.splitlines(True)), paste_to=tmp_path)
    line = editor.readline()
    assert line is not None
    # One message, carrying the block, rather than the block's first row.
    assert line.splitlines() == ["| a | b |", "| c | d |", "| e | f |"]


def test_a_big_dribbled_block_is_collapsed_like_any_other_paste(tmp_path):
    """The block still has to leave the input line usable."""
    body = "".join(f"line {i}\n" for i in range(9))     # past PASTE_LINES
    editor = editor_for(SlowPasteTerm(body.splitlines(True)), paste_to=tmp_path)
    line = editor.readline()
    assert line is not None
    assert line.startswith("[Pasted text #")
    # The final break was the Enter that ended the block, so it is not part
    # of the text the agent is handed (`Steering.submit` strips it anyway).
    assert expand_paste_refs(line) == body.rstrip("\n")


def test_enter_is_still_enter_when_nothing_is_behind_it():
    """The fix must not make a deliberate Enter wait for a block."""
    editor = editor_for(FakeTerm(["hello\r"], columns=40))
    assert editor.readline() == "hello"
    editor = editor_for(FakeTerm(["a line typed quickly\n"], columns=40))
    assert editor.readline() == "a line typed quickly"
    # Windows sends CRLF; still one logical break, still one Enter.
    editor = editor_for(FakeTerm(["another line\r\n"], columns=40))
    assert editor.readline() == "another line"


def test_the_wait_before_accepting_a_break_is_not_noticeable():
    assert 0 < SUBMIT_GRACE <= 0.25


# -- the channel: a burst of fragments is one message -------------------
def test_a_burst_of_fragments_is_one_message_and_one_receipt():
    said: list[str] = []
    steering = Steering(stream=io.StringIO(""), printer=said.append)
    steering.begin_run()
    for fragment in TABLE.splitlines():
        steering.submit(fragment)
    # One receipt, not three: the operator pasted once.
    assert len(said) == 1
    assert "heard" in said[0]
    supplements = steering.take_supplements()
    assert len(supplements) == 1
    assert "| a | b |" in supplements[0] and "| e | f |" in supplements[0]


def test_fragments_without_a_run_are_not_folded_together():
    """Nothing is running, so every line is its own next request."""
    said: list[str] = []
    steering = Steering(stream=io.StringIO(""), printer=said.append)
    for fragment in TABLE.splitlines():
        steering.submit(fragment)
    assert len(steering.take_supplements()) == 3


def test_a_later_line_is_not_folded_into_an_older_one():
    """A burst is a burst; a minute apart is two messages."""
    said: list[str] = []
    steering = Steering(stream=io.StringIO(""), printer=said.append)
    steering.begin_run()
    steering.submit("first thing")
    steering._last_at -= 5.0               # the burst window has passed
    steering.submit("second thing")
    supplements = steering.take_supplements()
    assert len(supplements) == 2
    assert len(said) == 2
