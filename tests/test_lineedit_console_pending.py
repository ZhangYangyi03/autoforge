"""A key coming up is not input, and the editor must not think it is.

The failure this pins down
--------------------------
Every session launched after `34f637f` (2026-09-17 16:18) started, printed
its banner, and then ignored everything typed into it. Reported from the
outside as exactly that: a fresh `auto` session that does not reply and does
not run.

`_consume` has to decide, at a break, whether the break is Enter or the first
line of a block that is still arriving. It asks `_more_coming`, which asks
the terminal whether more input is already buffered. On a Windows console the
answer was taken from `GetNumberOfConsoleInputEvents(...) > 0` -- and a
console leaves TWO records behind for every keystroke, the key going down and
the same key coming up. The count is therefore never zero in the moment after
a key: the Enter's own key-up record was read as "a block is still arriving",
the Enter was filed as a break inside a paste, and the line never submitted.

Measured on a real console with keys injected one record at a time as a
keyboard sends them: `_consume` saw `'hi\r'`, `_more_coming` returned True,
and `'hi'` sat in the buffer with no ledger event and an empty-looking prompt.
The same code tested through a single batched `WriteConsoleInput` call passes,
because a batch leaves no key-up record behind to be seen -- which is how the
regression got in under a passing suite.

So the decision is made from the records instead of the count, and these
tests pin the decision. `_record_is_text` is the predicate; the records here
are the real structs, filled as the console fills them.
"""
from __future__ import annotations

import ctypes

import pytest

from autoforge.core.lineedit import _console_records

INPUT_RECORD, _ = _console_records()

KEY_EVENT = 0x0001
MOUSE_EVENT = 0x0002
WINDOW_BUFFER_SIZE_EVENT = 0x0004
FOCUS_EVENT = 0x0010

ctypes.windll.kernel32


def key_record(ch, *, down, vk=0x00, ctrl=0):
    record = INPUT_RECORD()
    record.EventType = KEY_EVENT
    record.Event.KeyEvent.bKeyDown = 1 if down else 0
    record.Event.KeyEvent.wRepeatCount = 1
    record.Event.KeyEvent.wVirtualKeyCode = vk
    record.Event.KeyEvent.wVirtualScanCode = 0
    record.Event.KeyEvent.uChar = ch
    record.Event.KeyEvent.dwControlKeyState = ctrl
    return record


def mouse_record(x, y, *, buttons=0, flags=0):
    record = INPUT_RECORD()
    record.EventType = MOUSE_EVENT
    record.Event.MouseEvent.dwMousePosition.X = x
    record.Event.MouseEvent.dwMousePosition.Y = y
    record.Event.MouseEvent.dwButtonState = buttons
    record.Event.MouseEvent.dwControlKeyState = 0
    record.Event.MouseEvent.dwEventFlags = flags
    return record


def window_record():
    record = INPUT_RECORD()
    record.EventType = WINDOW_BUFFER_SIZE_EVENT
    return record


def focus_record():
    record = INPUT_RECORD()
    record.EventType = FOCUS_EVENT
    return record


# -- the predicate ------------------------------------------------------
def test_a_key_going_down_is_input():
    from autoforge.core.lineedit import _record_is_text

    assert _record_is_text(key_record("h", down=True))


def test_enter_going_down_is_input():
    from autoforge.core.lineedit import _record_is_text

    assert _record_is_text(key_record("\r", down=True, vk=0x0D))


def test_a_key_coming_up_is_not_input():
    """The whole bug, in one assertion: the echo of a key is not a key."""
    from autoforge.core.lineedit import _record_is_text

    assert not _record_is_text(key_record("\r", down=False, vk=0x0D))
    assert not _record_is_text(key_record("h", down=False))


def test_a_modifier_is_not_input_either_way():
    """A bare modifier is a record with no character in it, and no text."""
    from autoforge.core.lineedit import _record_is_text

    # `uChar` is a single character by construction; a modifier key carries
    # no character at all, and the console sets it to NUL.
    assert not _record_is_text(key_record("\x00", down=False, vk=0x10))   # Shift up
    assert not _record_is_text(key_record("\x00", down=True, vk=0x10))    # Shift down


def test_a_mouse_movement_is_not_input_but_a_press_is():
    """A mouse at rest on the window must not swallow every Enter."""
    from autoforge.core.lineedit import _record_is_text

    MOUSE_MOVED = 0x0001
    assert not _record_is_text(mouse_record(7, 2, flags=MOUSE_MOVED))
    assert _record_is_text(mouse_record(7, 2, buttons=1, flags=MOUSE_MOVED))
    assert _record_is_text(mouse_record(7, 2, buttons=0))          # a click


def test_records_that_carry_no_text_are_not_input():
    from autoforge.core.lineedit import _record_is_text

    assert not _record_is_text(window_record())
    assert not _record_is_text(focus_record())


# -- the decision the predicate feeds ------------------------------------
def test_a_keyboard_enter_does_not_look_like_a_paste_in_flight():
    """The exact pair a console produces for Enter: down, then up.

    `_text_pending` answers "is more of this input already here" by looking at
    what is buffered. Buffered with the key-up record in it, the answer has to
    be no, or the Enter is filed as a paste. The check is stated on the
    predicate because that is where the decision now lives; the console
    round-trip is covered by the integration run recorded in the commit.
    """
    from autoforge.core.lineedit import _record_is_text

    buffered = [key_record("\r", down=False, vk=0x0D)]        # Enter's own key-up
    assert not any(_record_is_text(r) for r in buffered)

    buffered = [key_record("\r", down=True, vk=0x0D)]         # a real next line
    assert any(_record_is_text(r) for r in buffered)


def test_a_pasted_second_line_still_reads_as_more_coming():
    """The fix must not go the other way: a block in flight is still a block."""
    from autoforge.core.lineedit import _record_is_text

    buffered = [key_record(c, down=True) for c in "and more\r"]
    assert any(_record_is_text(r) for r in buffered)
