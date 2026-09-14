"""The line editor: wrapping, redrawing, and paste handling.

The behaviour under test is the behaviour of a terminal, so the terminal is
faked: a scripted byte source with a fixed width, and a string buffer for the
output. Both are small enough to read, and they let a keystroke sequence be
handed to the editor in exactly the order a person would produce it.
"""
from __future__ import annotations

import io

import pytest

from autoforge.core.lineedit import (
    LineEditor,
    collapse_paste,
    expand_paste_refs,
    paste_dir,
)

from _terminal import FakeTerm, Out, Screen, visible


def build(chunks=(), columns=40, **kw):
    out = Out()
    term = FakeTerm(chunks, columns=columns)
    editor = LineEditor(term=term, out=out, **kw)
    editor.start()
    editor.set_prompt("you> ")
    return editor, term, out


# -- the fallback ------------------------------------------------------
def test_a_stream_with_no_fileno_is_not_editable():
    """A test, a pipe, a redirected file: the old reader keeps working."""
    editor = LineEditor(stream=io.StringIO("hello\n"))
    assert editor.start() is False
    assert editor.available is False


def test_take_line_still_reads_a_pipe():
    from autoforge.core.steering import Steering

    steering = Steering(io.StringIO("hello\nworld\n"))
    assert steering.interactive is False
    assert steering.take_line("you> ") == "hello\n"
    assert steering.take_line("you> ") == "world\n"
    assert steering.take_line("you> ") == ""


# -- drawing -----------------------------------------------------------
def test_a_long_line_wraps_instead_of_running_off_the_screen():
    editor, _, out = build(["x" * 30], columns=20)
    assert editor._consume("x" * 30) is None
    drawn = visible(out.getvalue())
    # 20 columns, a 5-wide prompt, and one column held back: 14 per row.
    assert "you> xxxxxxxxxxxxxx" in drawn
    assert "     xxxxxxxxxxxxxx" in drawn      # continuation, indented
    assert editor._rows == 3


def test_the_cursor_is_never_placed_left_of_the_area():
    """A negative row move would be printed as a literal `[-1A`."""
    for width in (21, 40, 80):
        for n in (0, 1, 14, 15, 40, 41):
            editor, _, out = build([], columns=width)
            editor._consume("y" * n)
            assert "[-" not in visible(out.getvalue()), (width, n)


# -- the area the editor owns ------------------------------------------
def test_a_heartbeat_takes_one_row_and_keeps_it():
    """Five ticks, one line: the row is rewritten, not appended to."""
    editor, _, out = build([], columns=40)
    for i in range(5):
        editor.tick(f"  ~ turn {i}   12s")
    assert out.screen().lines() == ["  ~ turn 4   12s", "you> "]


def test_the_heartbeat_sits_above_the_input_and_below_the_output():
    editor, _, out = build([], columns=40)
    editor._consume("half a sentence")
    editor.write("  -> running bash")
    editor.tick("  ~ turn 3")
    assert out.screen().lines() == ["  -> running bash",
                                    "  ~ turn 3",
                                    "you> half a sentence"]


def test_output_replaces_the_heartbeat_rather_than_stranding_it():
    """Otherwise every beat would leave a dead line above the run's output."""
    editor, _, out = build([], columns=40)
    editor.tick("  ~ turn 3")
    editor.write("  -> running bash")
    assert out.screen().lines() == ["  -> running bash", "you> "]


def test_typing_survives_a_heartbeat_and_a_line_of_output():
    editor, _, out = build([], columns=40)
    for piece in ["a", "b", "c"]:
        editor._consume(piece)
        editor.tick("  ~ turn 1")
        editor.write("  -> bash")
    assert "".join(editor._buf) == "abc"
    assert out.screen().lines()[-1] == "you> abc"


def test_a_wrapped_buffer_still_undoes_its_own_rows():
    """The erase has to climb over every row it drew, not just the first."""
    editor, _, out = build([], columns=20)
    editor._consume("x" * 30)                 # three rows of input
    assert editor._rows == 3
    editor.tick("  ~ turn 1")
    editor.tick("  ~ turn 2")
    lines = out.screen().lines()
    assert lines[0] == "  ~ turn 2"           # one heartbeat row, not two
    assert sum(r.count("x") for r in lines) == 30   # not one x lost
    assert lines[-1].strip().startswith("xx")       # the tail is still shown


def test_a_submitted_line_is_handed_back_whole():
    editor, _, out = build(["first line of input\r", None])
    assert editor.readline() == "first line of input"
    assert editor.readline() is None          # EOF
    assert editor.eof is True


def test_progress_written_mid_typing_does_not_land_on_the_input_line():
    editor, _, out = build([], columns=40)
    editor._consume("half a sentence")
    out.parts.clear()
    editor.write("  -> ran some tool")
    editor.tick("      … waiting on model (3s)")
    drawn = visible(out.getvalue())
    # Each write erases the input area first, prints above it, redraws it after.
    assert drawn.count("<E>[J") == 2
    assert drawn.count("you> half a sentence") == 2
    for chunk in drawn.split("<E>[J")[1:]:
        assert chunk.startswith("  -> ran some tool\n") or \
               chunk.startswith("      … waiting on model (3s)")
    assert drawn.endswith("you> half a sentence\r<E>[20C")


def test_the_heartbeat_keeps_one_row():
    editor, _, out = build([], columns=40)
    editor.tick("… waiting (1s)")
    editor.tick("… waiting (2s)")
    drawn = visible(out.getvalue())
    # Erase, rewrite the row, newline, then redraw the input area below it.
    assert drawn.count("<E>[K\n") == 2
    assert drawn.endswith("you> \r<E>[5C")
    # The second tick erased the first one rather than scrolling it away.
    assert drawn.rstrip().count("… waiting (2s)") == 1


def test_editing_keys_work_on_the_middle_of_the_line():
    editor, _, _ = build([])
    editor._consume("ac")
    editor._consume("\x1b[D")        # left
    editor._consume("b")             # insert in the middle
    assert "".join(editor._buf) == "abc"
    assert editor._cursor == 2
    editor._consume("\x7f")          # backspace — the character before the cursor
    assert "".join(editor._buf) == "ac"
    assert editor._cursor == 1
    editor._consume("\x15")          # ^U — kill to the start, so the tail stays
    assert "".join(editor._buf) == "c"
    assert editor._cursor == 0


def test_kill_to_end_leaves_the_head():
    editor, _, _ = build([])
    editor._consume("keep this")
    editor._consume("\x1b[D\x1b[D\x1b[D\x1b[D\x1b[D")   # back to just past "keep"
    editor._consume("\x0b")          # ^K
    assert "".join(editor._buf) == "keep"


def test_ctrl_d_on_an_empty_line_is_end_of_input():
    editor, _, _ = build([])
    editor._consume("\x04")
    assert editor.eof is True


def test_an_escape_sequence_never_ends_up_in_the_buffer():
    editor, _, _ = build([])
    editor._consume("ab")
    editor._consume("\x1b[D\x1b[D")   # two lefts
    editor._consume("Z")
    assert "".join(editor._buf) == "Zab"


def test_closing_gives_the_terminal_back():
    editor, term, out = build([])
    editor.close()
    assert term.closed is True
    assert editor.available is False
    assert visible(out.getvalue()).endswith("\n")


# -- pastes ------------------------------------------------------------
def test_a_small_paste_is_just_text():
    assert collapse_paste("one\ntwo") == "one\ntwo"


def test_a_big_paste_becomes_a_placeholder_and_a_file(tmp_path):
    body = "\n".join(f"line {i}" for i in range(9))
    ref = collapse_paste(body, directory=tmp_path)
    assert ref.startswith("[Pasted text #1: 9 lines → ")
    assert str(tmp_path) in ref
    assert expand_paste_refs(ref) == body


def test_a_deleted_paste_file_leaves_the_placeholder_alone(tmp_path):
    ref = collapse_paste("\n".join(str(i) for i in range(9)), directory=tmp_path)
    for f in tmp_path.iterdir():
        f.unlink()
    # Better an agent that says it cannot read the file than a silently
    # shorter message than the one that was written.
    assert expand_paste_refs(ref) == ref


def test_paste_numbering_does_not_restart_and_shadow_an_older_paste(tmp_path):
    first = collapse_paste("\n".join(str(i) for i in range(9)), directory=tmp_path)
    assert "#1" in first
    # A new process would start counting at 1 again, and `paste_1_090000.txt`
    # would then shadow today's file for yesterday's placeholder. The
    # directory is what says no.
    import autoforge.core.lineedit as le
    le._counters.clear()
    second = collapse_paste("\n".join(str(i) for i in range(9)), directory=tmp_path)
    assert "#2" in second


def test_a_pasted_block_arrives_as_one_line_not_several(tmp_path):
    editor, _, _ = build([], paste_to=tmp_path)
    editor._consume("\x1b[200~" + "a\nb\nc\nd\ne\nf\n" + "\x1b[201~")
    buffered = "".join(editor._buf)
    assert buffered.startswith("[Pasted text #")
    assert "\n" not in buffered             # one line, as promised


def test_a_pasted_slash_command_keeps_its_newlines(tmp_path):
    """A pasted multi-line command is a script; a file pointer helps nobody."""
    editor, _, _ = build([], paste_to=tmp_path)
    body = "/status\n/status\n/status\n/status\n/status\n/status\n"
    editor._consume("\x1b[200~" + body + "\x1b[201~")
    assert "".join(editor._buf) == body


def test_enter_still_submits_without_bracketed_paste(tmp_path):
    """An older terminal sends no markers, so a block has to be told from Enter."""
    editor, _, _ = build([], paste_to=tmp_path)
    assert editor._consume("hello\r") == "hello"
    # A fast typist's last few keys and the Enter can share one read. That is
    # Enter, not a paste: one break, at the end.
    assert editor._consume("a line typed quickly\n") == "a line typed quickly"
    # On Windows the console sends CRLF for Enter; still one logical break.
    assert editor._consume("another line\r\n") == "another line"
    assert "".join(editor._buf) == ""       # no stray newline left behind


def test_a_block_without_bracketed_paste_is_still_collapsed(tmp_path):
    """No markers, no trailing break: the break in the middle is the tell."""
    editor, _, out = build([], columns=60, paste_to=tmp_path)
    block = "\n".join(f"line {i}" for i in range(9))    # past PASTE_LINES
    assert editor._consume(block) is None
    assert "".join(editor._buf).startswith("[Pasted text #")
    # What the user sees is the one-line placeholder, not nine lines of block.
    screen = out.screen().getvalue()
    assert screen.startswith("you> [Pasted text #")
    assert "line 0" not in screen


def test_a_short_block_keeps_its_own_lines_on_screen():
    """A break in the buffer is a hard break: the cursor must land right."""
    editor, _, out = build([], columns=40)
    editor._consume("one\ntwo\nthree")
    out.parts.clear()
    editor._consume("!")                 # at the very end
    rows, row, col = editor._layout()
    assert rows == ["you> one", "     two", "     three!"]
    assert (row, col) == (2, 11)
    # …and mid-buffer, at the break after "two" (index 7 of "one\ntwo\nthree!").
    editor._cursor = 7
    rows, row, col = editor._layout()
    assert (row, col) == (1, 8)
    assert "[-" not in visible(out.getvalue())


def test_paste_dir_follows_autoforge_home(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    assert paste_dir() == tmp_path / "pastes"


# -- wiring ------------------------------------------------------------
def test_steering_expands_the_placeholder_before_the_model_sees_it(tmp_path):
    from autoforge.core.steering import Steering

    body = "\n".join(f"L{i}" for i in range(9))
    ref = collapse_paste(body, directory=tmp_path)
    steering = Steering(io.StringIO(ref + "\n"))
    steering.submit(steering.take_line("you> ").strip())
    (sent,) = steering.take_supplements()
    assert body in sent
    assert "[Pasted text #" not in sent


@pytest.mark.parametrize("lines", [1, 4])
def test_short_pastes_reach_the_model_unchanged(tmp_path, lines):
    from autoforge.core.steering import Steering

    body = "\n".join(f"L{i}" for i in range(lines))
    steering = Steering(io.StringIO(body + "\n"))
    steering.submit(steering.take_line("you> ").strip())
    (sent,) = steering.take_supplements()
    assert body.split("\n")[0] in sent
