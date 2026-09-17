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


def test_a_prompt_that_opens_with_a_break_does_not_drift():
    """`chat`'s prompt starts with a newline, so the input sits one row down.

    Charged as a column instead of as a row, every keystroke re-emitted that
    break and the input line walked down the screen one row per character
    typed: three keystrokes, three rows of drift.
    """
    editor, _, out = build([], columns=40)
    editor.set_prompt("\nyou> ")
    typed = ""
    for ch in "abc":
        typed += ch
        editor._consume(ch)
        assert out.screen().lines() == ["", "you> " + typed], out.getvalue()
    assert editor._rows == 2 and editor._cur_row == 1


def test_the_prompt_may_carry_colour_and_a_break_at_once():
    """The real prompt is both, and the cursor still lands after the input."""
    editor, _, out = build([], columns=40)
    editor.set_prompt("\n\x1b[36myou>\x1b[0m ")
    editor._consume("hi")
    assert out.screen().lines() == ["", "you> hi"]
    assert out.getvalue().rstrip().endswith("\x1b[7C")   # 5 prompt + 2 typed


def test_a_break_in_the_prompt_survives_the_heartbeat():
    """The prompt's own rows are part of the area a tick has to climb over."""
    editor, _, out = build([], columns=40)
    editor.set_prompt("\nyou> ")
    editor._consume("hi")
    editor.tick("  ~ turn 1")
    editor.tick("  ~ turn 2")
    assert out.screen().lines() == ["  ~ turn 2", "", "you> hi"]


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


def test_a_windows_paste_counts_its_cr_breaks(tmp_path):
    # A Windows terminal sends a pasted block's breaks as a bare CR, so the
    # counter has to normalise before it counts: reading "\n" alone saw
    # this as one line, missed both thresholds, and inserted the block
    # verbatim instead of collapsing it.
    body = "\r".join(f"line {i}" for i in range(9))
    ref = collapse_paste(body, directory=tmp_path)
    assert ref.startswith("[Pasted text #1: 9 lines → ")
    # What lands on disk is normalised too, so the agent reads real lines.
    assert expand_paste_refs(ref) == body.replace("\r", "\n")


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


def _fake_term(cols):
    class _T:
        columns = cols

    return _T()

@pytest.mark.parametrize('cols', [20, 40])
def test_wide_chars_never_overflow_a_row(cols):
    '''A row holding CJK text must fit its terminal width.

    ASCII and CJK are not one column each, so slicing a row by character
    count overflowed the terminal and the driver wrapped the row itself --
    which put the cursor somewhere other than the insertion point. Every row
    built by _layout() has to be no wider than the terminal.
    '''
    from autoforge.core.lineedit import LineEditor, _visible_len

    editor = LineEditor(term=_fake_term(cols))
    editor.set_prompt('> ')
    text = '你好世界' * 6
    editor._buf = [text]
    editor._cursor = len(text)
    rows, row, col = editor._layout()
    for line in rows:
        assert _visible_len(line) <= cols, repr(line)
    assert col >= _visible_len('> '), repr(col)




# -- the selection -----------------------------------------------------
def sgr(code, col, row, release=False):
    """A mouse report as the editor reads it: `\x1b[<code;col;rowM|m`."""
    return f"\x1b[<{code};{col};{row}{'m' if release else 'M'}"


def cell(editor, index, top):
    """The screen cell a buffer position occupies, 1-based, absolute.

    `top` is the physical row the input area starts on. A real console
    reports the cursor's physical row and the editor turns it back into a
    cell by subtraction, so a test has to invent one: any number will do, as
    long as every report in a gesture is measured from the same row.
    """
    from autoforge.core.lineedit import _visible_len

    rows, starts, prefixes, _leading, _row, _col = editor._layout_detail()
    r = 0
    for n, start in enumerate(starts):
        if index >= start:
            r = n
    col = _visible_len(prefixes[r]) + (index - starts[r]) + 1
    return top + r + 1, col


def select(editor, a, b, top=10):
    """Drag from buffer position a to b, the way a person would.

    Two reports: a press, then motion with the button held -- the pair that
    tells the editor a selection is being made rather than two clicks. Each
    report carries the row the cursor is on *at that moment*, because the
    editor redraws between the two and the console's cursor moves with it.
    """
    rows, _s, _p, _l, crow, _c = editor._layout_detail()
    editor._term.screen_row = top + crow
    y, x = cell(editor, a, top)
    editor._consume(sgr(0, x, y))
    rows, _s, _p, _l, crow, _c = editor._layout_detail()
    editor._term.screen_row = top + crow
    y, x = cell(editor, b, top)
    editor._consume(sgr(32, x, y))


def test_selecting_the_middle_and_pressing_delete_removes_the_selection():
    """The failure this whole feature exists for.

    A drag over four characters in the middle of the line, then Delete: it
    used to delete the last character, because the key reached a cursor that
    was still at the end and the console kept the selection to itself.
    """
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    select(editor, 2, 6)                     # "cdef"
    editor._consume("\x1b[3~")               # delete
    assert "".join(editor._buf) == "abgh"
    assert editor._cursor == 2
    assert editor.selection() is None


def test_backspace_removes_the_selection_instead_of_one_character():
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    select(editor, 2, 6)
    editor._consume("\x7f")                 # backspace
    assert "".join(editor._buf) == "abgh"


def test_typing_over_a_selection_replaces_it():
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    select(editor, 2, 6)
    editor._consume("XY")
    assert "".join(editor._buf) == "abXYgh"


def test_the_selection_is_drawn_in_reverse_video():
    editor, _, out = build([], columns=40)
    editor._consume("abcdef")
    out.parts.clear()
    select(editor, 1, 4)
    drawn = out.getvalue()
    assert "\x1b[7mbcd\x1b[27m" in drawn


def test_cut_puts_the_selection_on_the_clipboard_and_removes_it():
    copied = []
    editor, _, _ = build([], columns=40, clipboard_read=lambda: "",
                         clipboard_write=lambda text: copied.append(text) or True)
    editor._consume("abcdefgh")
    select(editor, 2, 6)
    editor._consume("\x18")                 # ^X
    assert copied == ["cdef"]
    assert "".join(editor._buf) == "abgh"


def test_copy_keeps_the_text_and_drops_the_highlight():
    """Ctrl+Insert, not ^C: ^C has to keep stopping a run."""
    copied = []
    editor, _, _ = build([], columns=40, clipboard_read=lambda: "",
                         clipboard_write=lambda text: copied.append(text) or True)
    editor._consume("abcdefgh")
    select(editor, 2, 6)
    editor._consume("\x1b[2;5~")            # Ctrl+Insert
    assert copied == ["cdef"]
    assert "".join(editor._buf) == "abcdefgh"       # text untouched
    assert editor.selection() is None               # and nothing to delete next


def test_ctrl_delete_cuts_and_shift_insert_pastes():
    copied = []
    editor, _, _ = build([], columns=40, clipboard_read=lambda: "QQ",
                         clipboard_write=lambda text: copied.append(text) or True)
    editor._consume("abcdefgh")
    select(editor, 2, 6)
    editor._consume("\x1b[3;5~")            # Ctrl+Delete
    assert copied == ["cdef"]
    assert "".join(editor._buf) == "abgh"
    editor._consume("\x1b[2;2~")            # Shift+Insert
    assert "".join(editor._buf) == "abQQgh"


def test_ctrl_c_is_not_a_copy():
    """A terminal that copies but cannot be interrupted cannot be left."""
    copied = []
    editor, _, _ = build([], columns=40, clipboard_read=lambda: "",
                         clipboard_write=lambda text: copied.append(text) or True)
    editor._consume("abcdefgh")
    select(editor, 2, 6)
    editor._consume("\x03")                 # ^C
    assert copied == []
    assert editor.selection() == (2, 6)      # untouched, still there to delete


def test_the_mouse_belongs_to_the_console_by_default():
    """The wheel and drag-to-copy work unless someone asks otherwise.

    Taking the mouse away fixes a real thing -- a drag over the middle of the
    line then Delete used to delete the last character -- but it costs the
    whole session its scrolling, and that trade was reported as worse than
    the bug: "the mouse is dead, the wheel does nothing". So the console
    keeps it, and the editor selects on the keys instead.
    """
    import os as _os
    from autoforge.core import lineedit as le

    old = _os.environ.get("AUTOFORGE_MOUSE")
    _os.environ.pop("AUTOFORGE_MOUSE", None)
    try:
        assert le.mouse_enabled() is False
        editor, term, _ = build([], columns=40)
        assert term.mouse is False            # nothing was taken
    finally:
        if old is not None:
            _os.environ["AUTOFORGE_MOUSE"] = old


def test_the_editor_can_still_ask_for_the_mouse():
    """`AUTOFORGE_MOUSE=1` hands it over, for anyone who prefers that."""
    import os as _os
    from autoforge.core import lineedit as le

    old = _os.environ.get("AUTOFORGE_MOUSE")
    _os.environ["AUTOFORGE_MOUSE"] = "1"
    try:
        assert le.mouse_enabled() is True
        editor, term, _ = build([], columns=40)
        assert term.mouse is True             # and it was taken
    finally:
        if old is None:
            _os.environ.pop("AUTOFORGE_MOUSE", None)
        else:
            _os.environ["AUTOFORGE_MOUSE"] = old


# -- selecting with the keys, which is what replaced the mouse ---------
def test_shift_arrow_selects_and_delete_removes_the_span():
    """The gesture that has to survive without the mouse.

    Shift+Left walks back over the line, and it is what a person uses to say
    "these characters, not one" once the console owns the drag again.
    """
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    for _ in range(3):
        editor._consume("\x1b[1;2D")          # shift+left
    assert editor.selection() == (5, 8)        # "fgh"
    editor._consume("\x1b[3~")                # delete
    assert "".join(editor._buf) == "abcde"
    assert editor.selection() is None


def test_shift_home_selects_to_the_start_of_the_line():
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    editor._consume("\x1b[1;2H")              # shift+home
    assert editor.selection() == (0, 8)
    editor._consume("\x7f")                   # backspace eats the span
    assert "".join(editor._buf) == ""


def test_shift_end_selects_to_the_end_of_the_line():
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    editor._cursor = 2
    editor._consume("\x1b[1;2F")              # shift+end
    assert editor.selection() == (2, 8)


def test_a_second_shift_arrow_extends_the_same_selection():
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    editor._consume("\x1b[1;2D")
    editor._consume("\x1b[1;2D")
    assert editor.selection() == (6, 8)


def test_a_plain_arrow_drops_the_selection():
    """An unshifted move must not leave a highlight lying to the next key."""
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    editor._consume("\x1b[1;2D")
    assert editor.selection() is not None
    editor._consume("\x1b[D")                 # plain left
    assert editor.selection() is None


def test_shift_then_plain_arrow_in_one_burst_still_drops_it():
    """A fast typist produces both in one read; the gesture is still two."""
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    editor._consume("\x1b[1;2D\x1b[D")
    assert editor.selection() is None
    assert editor._cursor == 6


def test_copy_acts_on_a_keyboard_selection_too():
    editor, _, _ = build([], columns=40, clipboard_read=lambda: "",
                         clipboard_write=lambda text: True)
    editor._consume("abcdefgh")
    for _ in range(3):
        editor._consume("\x1b[1;2D")
    editor._consume("\x1b[2;5~")              # Ctrl+Insert
    assert "".join(editor._buf) == "abcdefgh"
    assert editor.selection() is None


def test_a_shift_arrow_is_recognised_from_a_console_record():
    """The decoder must spell the shift, or the key arrives as a plain move.

    A console puts the modifier in the record's key state and the same
    virtual key carries both meanings, so this is the join that decides
    whether Shift+Left selects or just moves.
    """
    from autoforge.core.lineedit import _console_records, _record_text

    INPUT_RECORD, _ = _console_records()

    def record(vk, shift):
        r = INPUT_RECORD()
        r.EventType = 1
        r.Event.KeyEvent.bKeyDown = 1
        r.Event.KeyEvent.wRepeatCount = 1
        r.Event.KeyEvent.wVirtualKeyCode = vk
        r.Event.KeyEvent.uChar = "\x00"
        r.Event.KeyEvent.dwControlKeyState = 0x10 if shift else 0
        return r

    assert _record_text(record(0x25, True)) == "\x1b[1;2D"     # shift+left
    assert _record_text(record(0x25, False)) == "\x1b[D"      # left
    assert _record_text(record(0x24, True)) == "\x1b[1;2H"     # shift+home
    assert _record_text(record(0x23, True)) == "\x1b[1;2F"     # shift+end
    assert _record_text(record(0x27, True)) == "\x1b[1;2C"     # shift+right


def test_paste_inserts_the_clipboard_at_the_cursor():
    editor, _, _ = build([], columns=40, clipboard_read=lambda: "ZZ",
                         clipboard_write=lambda text: True)
    editor._consume("ab")
    editor._consume("\x16")                 # ^V
    assert "".join(editor._buf) == "abZZ"


def test_paste_replaces_the_selection():
    editor, _, _ = build([], columns=40, clipboard_read=lambda: "ZZ",
                         clipboard_write=lambda text: True)
    editor._consume("abcdef")
    select(editor, 1, 4)
    editor._consume("\x16")
    assert "".join(editor._buf) == "aZZef"


def test_a_click_without_a_drag_just_places_the_cursor():
    editor, _, _ = build([], columns=40)
    editor._consume("abcdef")
    editor._term.screen_row = editor._rows - 1
    editor._consume(sgr(0, 5 + 3 + 1, editor._term.screen_row + 1))
    assert editor._cursor == 3
    assert editor.selection() is None        # an empty range is not a selection


def test_moving_with_the_arrow_keys_drops_the_selection():
    """Otherwise the next keystroke deletes a span the person has left behind."""
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    select(editor, 2, 6)
    editor._consume("\x1b[D")               # left, from the selection's end
    assert editor.selection() is None
    assert editor._cursor == 5
    editor._consume("\x7f")
    assert "".join(editor._buf) == "abcdfgh"        # one character, not the span


def test_a_selection_survives_a_heartbeat():
    editor, _, out = build([], columns=40)
    editor._consume("abcdef")
    select(editor, 1, 4)
    editor.tick("  ~ turn 1")
    editor.write("  -> bash")
    assert editor.selection() == (1, 4)
    assert "\x1b[7mbcd\x1b[27m" in out.getvalue()


def test_selecting_backwards_works_the_same():
    editor, _, _ = build([], columns=40)
    editor._consume("abcdefgh")
    select(editor, 6, 2)                     # drag right to left
    assert editor.selection() == (2, 6)
    editor._consume("\x1b[3~")
    assert "".join(editor._buf) == "abgh"


def test_a_selection_that_spans_a_wrapped_row_is_drawn_on_both():
    editor, _, out = build([], columns=22)
    editor._consume("x" * 24)
    out.parts.clear()
    select(editor, 10, 20)          # 16 characters fit a row, so 10-20 spans both
    drawn = out.getvalue()
    # The last draw is the one that is on the screen: two rows carry a
    # reversed run, and the reversal is closed on each of them. Counted on
    # the final draw rather than the whole stream, because the press and the
    # drag each drew once and each of those draws is complete in itself.
    last = drawn.rsplit("\r\x1b[J", 1)[-1]
    assert last.count("\x1b[7m") == 2
    assert last.count("\x1b[27m") == 2


def test_deleting_a_selection_clears_the_highlight_from_the_screen():
    editor, _, out = build([], columns=40)
    editor._consume("abcdefgh")
    select(editor, 2, 6)
    editor._consume("\x1b[3~")
    assert out.screen().lines()[-1] == "you> abgh"


def test_a_mouse_report_never_lands_in_the_buffer():
    """The escape sequence is decoded, not typed: a stray `\x1b[<0;7;3M` in a
    steering message would be a strange thing to send the model."""
    editor, _, _ = build([], columns=40)
    editor._consume("ab")
    select(editor, 0, 2)
    assert "".join(editor._buf) == "ab"


def test_the_clipboard_pair_survives_a_clipboard_that_cannot_answer():
    """A locked or absent clipboard must not stop the editor working."""
    def boom(*_a, **_k):
        raise RuntimeError("clipboard busy")

    editor, _, _ = build([], columns=40, clipboard_read=boom, clipboard_write=boom)
    editor._consume("abc")
    select(editor, 0, 1)
    editor._consume("\x16")                 # ^V with a clipboard that throws
    assert "".join(editor._buf) == "abc"
    editor._consume("\x18")                 # ^X, same
    assert "".join(editor._buf) == "bc"
