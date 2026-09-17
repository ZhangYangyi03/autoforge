"""A line editor, so `chat` can be typed into while the agent works.

The failure this fixes
----------------------
`autoforge chat` runs a reader thread that takes lines off stdin while the
agent works, and the agent's progress lines are written to stdout from the
main thread. With the terminal left in cooked mode, the *kernel* echoed your
keystrokes and the process wrote progress onto the same rows the echo was
using. The two collided: typing while a run was in flight produced a line that
was shredded into several, the cursor landing wherever the last `\\r` left it.

Cooked mode also means the terminal owns the line: a pasted block of text
arrives as N separate lines, so it arrives as N separate steering messages, and
anything wider than the window is truncated by the driver rather than wrapped
by us.

What this module does instead
-----------------------------
Own the bottom line of the terminal. `LineEditor` puts the terminal into cbreak
(raw input, no echo, but `^C` still signals, as it does today), draws
`prompt + buffer` itself, and wraps that buffer across as many rows as the
terminal is wide. Every other writer in the process goes through `write()` or
`tick()`, which erase the input area first and redraw it afterwards — so
progress and typing never occupy the same row.

The other half is paste handling. A pasted block is recognised (bracketed paste
where the terminal supports it, a multi-line burst otherwise), and a large one
is written to `pastes/` and replaced by a `[Pasted text #N: L lines -> path]`
placeholder, the way the Hermes CLI does it. `expand_paste_refs` puts the text
back before the agent sees it.

A break is the other tell. A terminal that is not sending bracketed-paste
markers hands a slow paste over a line at a time, and every one of those lines
ends in exactly one break — which is what Enter looks like too. Per line the
two are the same shape, so the shape cannot decide it; what can is whether the
next line's bytes are already on their way. A break that has input behind it
is a break inside a block, and `SUBMIT_GRACE` is how long it waits to find
out. Without this, one pasted table arrived as N steering messages and N
interruptions.

The mouse is the other half of that. In a console that keeps quick-edit
mode on -- the Windows default -- a drag paints the console's own highlight,
that highlight copies to the clipboard, and the selection is then the
console's business rather than the process's. Nothing that follows a drag is
delivered: no report of where the drag went, and the Delete key arrives at a
cursor still sitting at the end of the line, which is why selecting four
characters in the middle and pressing Delete deleted the last one instead.
The selection was made, and invisible to the only code that could act on it.

This module takes the mouse away from the console, asks for mouse reports,
and keeps the selection itself: an anchor plus a cursor, the characters
between them drawn in reverse video. Backspace, Delete and ^X remove exactly
that span, ^C copies it, ^V/^Y and a right-click put the clipboard back in,
and the console has no say in any of it. On Windows all of that reads the
console *input records* directly rather than the ANSI sequences a
`?1006h`-style request would produce, and that is deliberate: those escape
sequences take the console's line editing and its ^C handling away with
them, and both are things this editor must not lose.

Not every terminal can be driven this way — a pipe, a dumb `TERM`, a `stdin`
whose `fileno()` is not a tty. `available` is False there, and `Steering` falls
back to the cooked-mode reader it used before. The fallback is the old
behaviour, not a broken new one.
"""
from __future__ import annotations

import codecs
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
try:
    from wcwidth import wcwidth as _wcwidth
except Exception:
    def _wcwidth(ch):
        return 2 if ord(ch) > 0x2E7F else 1


__all__ = [
    "LineEditor",
    "mouse_enabled",
    "read_clipboard",
    "write_clipboard",
    "collapse_paste",
    "expand_paste_refs",
    "paste_dir",
    "PASTE_CHARS",
    "PASTE_LINES",
    "SUBMIT_GRACE",
]

#: A paste of at least this many lines, or this many characters, is collapsed
#: to a file. Below both, it is simply inserted — a three-line snippet is
#: easier to read in the input line than as a pointer to a temp file.
PASTE_LINES = 5
PASTE_CHARS = 2000

#: How long a break waits to find out whether it is Enter or a break inside a
#: block that is still arriving. Enter is the common case and this is paid on
#: every one of them, so it is kept well under the ~200ms at which a pause
#: starts to read as lag — and it only has to cover the gap between two lines
#: of the same paste, which is a fraction of that.
SUBMIT_GRACE = 0.15

#: `[Pasted text #1: 9 lines -> /path]`. The arrow is a real U+2192.
PASTE_REF_RE = re.compile(r"\[Pasted text #(\d+): (\d+) lines \u2192 (.+?)\]")
PASTE_MARK = "[Pasted text #"

_BRACKET_START = "\x1b[200~"
_BRACKET_END = "\x1b[201~"

# Erase from the cursor to the end of the screen. Everything below the cursor
# belongs to the input area, so this is exactly "clear the input area".
_ERASE_TO_EOS = "\x1b[J"
_ERASE_TO_EOL = "\x1b[K"

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

#: Sequences we decode by hand. Ordered longest-first by the lookup below.
_KEYS = {
    "\x1b[D": "left",
    "\x1b[C": "right",
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[H": "home",
    "\x1b[F": "end",
    "\x1b[1~": "home",
    "\x1b[4~": "end",
    "\x1b[7~": "home",
    "\x1b[8~": "end",
    "\x1bOD": "left",
    "\x1bOC": "right",
    "\x1bOH": "home",
    "\x1bOF": "end",
    "\x1b[3~": "delete",
    "\x1b[2;5~": "copy",     # Ctrl+Insert
    "\x1b[3;5~": "cut",      # Ctrl+Delete
    "\x1b[2;2~": "paste",    # Shift+Insert
}
_KEY_ORDER = sorted(_KEYS, key=len, reverse=True)

#: Control characters that mean something to the editor.
_CTRL_U = "\x15"   # kill to start of line
_CTRL_K = "\x0b"   # kill to end of line
_CTRL_W = "\x17"   # kill previous word
_CTRL_A = "\x01"   # home
_CTRL_E = "\x05"   # end
_CTRL_L = "\x0c"   # redraw
_CTRL_X = "\x18"   # cut the selection
_CTRL_C = "\x03"   # copy the selection, or interrupt
_CTRL_V = "\x16"   # paste
_CTRL_Y = "\x19"   # paste
_CTRL_D = "\x04"   # EOF when the buffer is empty
_BACKSPACE = ("\x7f", "\x08")

#: Windows scan codes -> the escape sequence the rest of this module decodes.
#:
#: `msvcrt.getwch()` reports a special key as two code units: a lead byte
#: (`\x00` for F-keys, `\xe0` for the navigation cluster) followed by a scan
#: code. The scan code is the key's only identity on Windows — there is no
#: escape sequence to read — so it has to be translated here or the key is
#: lost. These are the codes for the navigation cluster, which is the set a
#: line editor needs: the arrows, Home, End and Delete.
_WIN_SCAN = {
    "H": "\x1b[A",   # up
    "P": "\x1b[B",   # down
    "M": "\x1b[C",   # right
    "K": "\x1b[D",   # left
    "G": "\x1b[H",   # home
    "O": "\x1b[F",   # end
    "S": "\x1b[3~",  # delete
}

#: A mouse report, SGR form: an escape, `[` and `<`, the button, the column,
#: the row, and `M` for a press or `m` for a release.
_MOUSE_SGR = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")

#: Ask for press/release (1000), drag (1002) and SGR coordinates (1006).
#: 1002 rather than 1003: motion is wanted while a button is held, because
#: that is what a drag is, and not while nothing is held -- 1003 would hand
#: the editor every mouse move in the room as input.
_MOUSE_ON = "\x1b[?1000h\x1b[?1002h\x1b[?1006h"
_MOUSE_OFF = "\x1b[?1006l\x1b[?1002l\x1b[?1000l"

#: Reverse video, for the selected span. Not the console's highlight: that
#: one is painted by the console, which is the component that could not see
#: the input line to begin with.
_REVERSE_ON = "\x1b[7m"
_REVERSE_OFF = "\x1b[27m"


_CF_UNICODETEXT = 13


def read_clipboard() -> str:
    """The system clipboard as text, or "" when it cannot be read.

    Guarded on purpose. A terminal with no clipboard is a terminal where
    copy and paste are not offered; it is not one where the editor dies.
    """
    if os.name == "nt":
        return _read_clipboard_win()
    for argv in (["wl-paste", "-n"], ["xclip", "-selection", "clipboard", "-o"],
                 ["xsel", "-b"], ["pbpaste"]):
        exe = shutil.which(argv[0])
        if not exe:
            continue
        try:
            done = subprocess.run([exe, *argv[1:]], capture_output=True, timeout=2)
        except Exception:                     # noqa: BLE001 - a clipboard that cannot answer
            continue
        if done.returncode == 0:
            return done.stdout.decode("utf-8", "replace")
    return ""


def write_clipboard(text: str) -> bool:
    """Put text on the system clipboard. False when there is none to put it on."""
    if os.name == "nt":
        return _write_clipboard_win(text)
    for argv in (["wl-copy"], ["xclip", "-selection", "clipboard", "-i"],
                 ["xsel", "-b", "-i"], ["pbcopy"]):
        exe = shutil.which(argv[0])
        if not exe:
            continue
        try:
            done = subprocess.run([exe, *argv[1:]], input=text.encode("utf-8"),
                                  capture_output=True, timeout=2)
        except Exception:                     # noqa: BLE001 - same
            continue
        if done.returncode == 0:
            return True
    return False


def _clipboard_api():
    """The two libraries the clipboard needs, with their pointers typed.

    Typed, not defaulted: an HGLOBAL handed over as a plain Python int is
    truncated to 32 bits, and `SetClipboardData` then fails with a handle
    that looks perfectly fine in the source.
    """
    import ctypes
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [wt.HWND]
    user32.OpenClipboard.restype = wt.BOOL
    user32.SetClipboardData.argtypes = [wt.UINT, wt.HANDLE]
    user32.SetClipboardData.restype = wt.HANDLE
    user32.GetClipboardData.argtypes = [wt.UINT]
    user32.GetClipboardData.restype = wt.HANDLE
    kernel32.GlobalAlloc.restype = wt.HGLOBAL
    kernel32.GlobalLock.argtypes = [wt.HGLOBAL]
    # The return value is a pointer, so it has to be typed as one. Left at
    # ctypes' default it comes back as a 32-bit int and is truncated in a
    # 64-bit process -- and the memmove that follows then writes somewhere
    # that is not the block at all. That is an access violation at best.
    kernel32.GlobalLock.restype = wt.LPVOID
    kernel32.GlobalUnlock.argtypes = [wt.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wt.HGLOBAL]
    return user32, kernel32


def _open_clipboard(user32, attempts: int = 8, pause: float = 0.02) -> bool:
    """Take the clipboard, waiting briefly for whoever else has it.

    The clipboard is a single global resource and it is held constantly by
    whatever else is on the machine -- an input-method helper, a chat client,
    a clipboard history. A single attempt fails whenever one of those happens
    to be in the middle of a read, which is most of the time on a busy
    desktop, and a copy that silently does nothing is worse than a copy that
    takes forty milliseconds.
    """
    for n in range(max(1, attempts)):
        if user32.OpenClipboard(None):
            return True
        if n + 1 < attempts:
            time.sleep(pause)
    return False


def _read_clipboard_win() -> str:
    import ctypes

    try:
        user32, kernel32 = _clipboard_api()
        if not _open_clipboard(user32):
            return ""
        try:
            if not user32.IsClipboardFormatAvailable(_CF_UNICODETEXT):
                return ""
            handle = user32.GetClipboardData(_CF_UNICODETEXT)
            if not handle:
                return ""
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                return ""
            try:
                return ctypes.c_wchar_p(ptr).value or ""
            finally:
                kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()
    except Exception:                         # noqa: BLE001 - a clipboard that cannot answer
        return ""


def _write_clipboard_win(text: str) -> bool:
    import ctypes

    try:
        user32, kernel32 = _clipboard_api()
        if not _open_clipboard(user32):
            return False
        try:
            if not user32.EmptyClipboard():
                return False
            data = text.encode("utf-16-le") + b"\x00\x00"
            handle = kernel32.GlobalAlloc(0x0002, len(data))   # GMEM_MOVEABLE
            if not handle:
                return False
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                kernel32.GlobalFree(handle)
                return False
            ctypes.memmove(ptr, data, len(data))
            kernel32.GlobalUnlock(handle)
            if not user32.SetClipboardData(_CF_UNICODETEXT, handle):
                kernel32.GlobalFree(handle)
                return False
            # Ownership of the block went to the clipboard with the call: the
            # process must not free it, and must not touch that memory again.
            return True
        finally:
            user32.CloseClipboard()
    except Exception:                         # noqa: BLE001 - same
        return False


def paste_dir() -> Path:
    """Where collapsed pastes live. `AUTOFORGE_HOME` wins, else `~/.autoforge`."""
    home = os.environ.get("AUTOFORGE_HOME")
    base = Path(home) if home else Path.home() / ".autoforge"
    return base / "pastes"


_counter_lock = threading.Lock()
#: Last index handed out, per directory. Per directory because paste numbers
#: only have to be unique among the files they name.
_counters: dict[str, int] = {}


def _next_index(directory: Path) -> int:
    """The next free paste number, seeded from what is already on disk.

    Seeding matters: numbering that restarts at 1 every run would make
    `paste_1_112625.txt` from today shadow the one from last week, and a
    placeholder in a transcript would then expand to the wrong text.
    """
    key = str(directory)
    with _counter_lock:
        if not _counters.get(key):
            best = 0
            try:
                for f in directory.glob("paste_*_*.txt"):
                    parts = f.name.split("_")
                    if len(parts) > 1 and parts[1].isdigit():
                        best = max(best, int(parts[1]))
            except OSError:
                best = 0
            _counters[key] = best
        _counters[key] += 1
        return _counters[key]


def collapse_paste(text: str, *, lines: int = PASTE_LINES, chars: int = PASTE_CHARS,
                   directory: Path | str | None = None) -> str:
    """Return `text`, or a placeholder naming a file that holds it.

    Big pastes are the ones that make the input line unusable: they are taller
    than the terminal, and the useful part of them is not on screen anyway.
    """
    if not isinstance(text, str) or not text:
        return text or ""
    # Normalise line endings first. A Windows terminal sends a pasted block's
    # breaks as a bare CR (an old Mac terminal as CR only), so counting
    # "\n" alone sees a sixty-line paste as one line: neither threshold
    # is met and the block is inserted verbatim instead of collapsed. The
    # counter has to see the same breaks the detector did.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    count = text.count("\n") + 1
    if count < lines and len(text) < chars:
        return text
    where = Path(directory) if directory is not None else paste_dir()
    try:
        where.mkdir(parents=True, exist_ok=True)
        index = _next_index(where)
        path = where / f"paste_{index}_{time.strftime('%H%M%S')}.txt"
        path.write_text(text, encoding="utf-8")
    except OSError:
        # A paste we cannot store is still a paste we can send. Failing open
        # here keeps the user's text; failing closed would throw it away.
        return text
    return f"[Pasted text #{index}: {count} lines \u2192 {path}]"


def expand_paste_refs(text: str) -> str:
    """Put collapsed paste text back. Unreadable file -> placeholder stays."""
    if not isinstance(text, str) or PASTE_MARK not in text:
        return text or ""

    def _expand(match: re.Match) -> str:
        try:
            return Path(match.group(3)).read_text(encoding="utf-8")
        except OSError:
            # Deleted between the placeholder and now. Keeping the placeholder
            # is honest: the agent can say it cannot read the file, which is
            # better than silently sending a shorter message than you wrote.
            return match.group(0)

    return PASTE_REF_RE.sub(_expand, text)


def _visible_len(text: str) -> int:
    """Length as the terminal counts it — colour codes take no columns.

    Neither do control characters. A break inside a prompt is a row of its own,
    which the layout deals with separately; counting it here as well would put
    the cursor one column past where it actually is.
    """
    total = 0
    for ch in _ANSI.sub("", text):
        if ch < " ":
            continue
        w = _wcwidth(ch)
        total += w if w and w > 0 else 0
    return total


#: Virtual-key codes whose identity has to be carried separately. A console
#: hands a key over as a virtual key plus a character, and a navigation key
#: carries no character at all — so without this table left, right, home,
#: end and delete arrive as "nothing was pressed" and the cursor cannot be
#: moved, which is the failure this table exists to prevent.
_WIN_VK = {
    # Enter is in the table because a console records the key and its
    # character separately, and a record with the character left empty --
    # which is what a remote or a synthesised Enter looks like -- would
    # otherwise be read as "a key with nothing on it" and dropped. The
    # editor would then be unable to submit a line.
    0x0D: "\r",
    0x25: "\x1b[D",   # left
    0x26: "\x1b[A",   # up
    0x27: "\x1b[C",   # right
    0x28: "\x1b[B",   # down
    0x24: "\x1b[H",   # home
    0x23: "\x1b[F",   # end
    0x2E: "\x1b[3~",  # delete
    0x2D: "\x1b[2~",  # insert
}

#: A console mouse event, reduced to the handful of gestures that mean
#: something. `dwEventFlags` says whether a button is held down and moving,
#: and `dwButtonState` says which button; the pair is what tells a drag from
#: a fresh click, which is the difference between extending a selection and
#: starting one.
_MOUSE_FLAG_MOVED = 0x0001
_MOUSE_FLAG_DOUBLE = 0x0002
_MOUSE_FLAG_WHEEL = 0x0004
_BUTTON_LEFT = 0x0001
_BUTTON_RIGHT = 0x0002
_BUTTON_MIDDLE = 0x0004

#: Control characters the editor acts on. A console reports them as a key
#: carrying a character below 0x20, and dropping those wholesale -- the
#: obvious way to keep a stray escape byte out of the buffer -- would take
#: ^A, ^D, ^K, ^L, ^U, ^V, ^W, ^X and ^Y with it.
_CONTROL_CHARS = "".join(chr(c) for c in range(1, 27))

#: Control-key-state bits, for the gestures that differ only by a modifier.
_LEFT_CTRL = 0x0008
_RIGHT_CTRL = 0x0004
_SHIFT = 0x0010


def mouse_enabled() -> bool:
    """Whether the editor may take the mouse away from the console.

    On by default, because a console that keeps the selection is a console
    where selecting the middle of the line and pressing Delete deletes the
    last character. `AUTOFORGE_MOUSE=0` is the way out for anyone who would
    rather keep the console's own drag-to-copy and its mouse-wheel
    scrolling: with it set, nothing about the mouse is changed and the
    editor behaves as it did before.
    """
    value = os.environ.get("AUTOFORGE_MOUSE", "").strip().lower()
    return value not in ("0", "off", "false", "no")


def _mouse_report(event) -> str:
    """A console mouse event, as the SGR report a VT terminal would send.

    The conversion is not decoration. A console reports the *button state*
    plus a flag saying a button was held while moving, while a VT report
    spells "press", "drag" and "release" out in the button number. The editor
    acts on the VT form, so a gesture has to be expressed in it: read as
    plain presses, a drag is a sequence of fresh clicks and the selection
    never grows past its first character.

    A wheel event has no useful VT equivalent here -- the editor has nothing
    to scroll -- so it is dropped rather than mis-mapped onto a click that
    would move the cursor.
    """
    flags = int(event.dwEventFlags)
    if flags & _MOUSE_FLAG_WHEEL:
        return ""
    state = int(event.dwButtonState)
    if flags & _MOUSE_FLAG_MOVED:
        if state & _BUTTON_LEFT:
            code = 32
        elif state & _BUTTON_MIDDLE:
            code = 33
        elif state & _BUTTON_RIGHT:
            code = 34
        else:
            return ""
    elif state & _BUTTON_LEFT:
        code = 0
    elif state & _BUTTON_MIDDLE:
        code = 1
    elif state & _BUTTON_RIGHT:
        code = 2
    else:
        code = 3                                    # release: no button left
    x = int(event.dwMousePosition.X) + 1            # SGR columns are 1-based
    y = int(event.dwMousePosition.Y) + 1
    return f"\x1b[<{code};{x};{y}{'m' if code == 3 else 'M'}"

#: Control-key-state bits, for the gestures that only differ by modifier.
_SHIFT_PRESSED = 0x0010


def _k32():
    """kernel32 with every handle-taking function typed.

    Typed rather than left to ctypes' defaults, because a console HANDLE
    handed over as a Python int is truncated to 32 bits and the call then
    fails with a handle that looks perfectly fine in the source — an
    `[Error 6] The handle is invalid` on a handle that is 0x1a4.
    """
    import ctypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetStdHandle.argtypes = [ctypes.c_uint]
    k.GetStdHandle.restype = ctypes.c_void_p
    k.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    k.GetConsoleMode.restype = ctypes.c_int
    k.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    k.SetConsoleMode.restype = ctypes.c_int
    k.ReadConsoleInputW.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    k.GetNumberOfConsoleInputEvents.argtypes = [ctypes.c_void_p,
                                                ctypes.POINTER(ctypes.c_uint)]
    k.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint,
                              ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                              ctypes.c_void_p]
    k.CreateFileW.restype = ctypes.c_void_p
    k.WriteConsoleW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
                                ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
    k.GetConsoleScreenBufferInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k.GetConsoleWindow.restype = ctypes.c_void_p
    k.AllocConsole.restype = ctypes.c_int
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    return k


def _console_records():
    """The two console structures, laid out as the console lays them out."""
    import ctypes

    class COORD(ctypes.Structure):
        _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

    class KEY_EVENT_RECORD(ctypes.Structure):
        _fields_ = [("bKeyDown", ctypes.c_int),
                    ("wRepeatCount", ctypes.c_ushort),
                    ("wVirtualKeyCode", ctypes.c_ushort),
                    ("wVirtualScanCode", ctypes.c_ushort),
                    ("uChar", ctypes.c_wchar),
                    ("dwControlKeyState", ctypes.c_uint)]

    class MOUSE_EVENT_RECORD(ctypes.Structure):
        _fields_ = [("dwMousePosition", COORD),
                    ("dwButtonState", ctypes.c_uint),
                    ("dwControlKeyState", ctypes.c_uint),
                    ("dwEventFlags", ctypes.c_uint)]

    class EVENT(ctypes.Union):
        _fields_ = [("KeyEvent", KEY_EVENT_RECORD),
                    ("MouseEvent", MOUSE_EVENT_RECORD),
                    ("_pad", ctypes.c_byte * 16)]

    class INPUT_RECORD(ctypes.Structure):
        _fields_ = [("EventType", ctypes.c_ushort), ("Event", EVENT)]

    class SMALL_RECT(ctypes.Structure):
        _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short),
                    ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]

    class SCREEN_BUFFER_INFO(ctypes.Structure):
        _fields_ = [("dwSize", COORD), ("dwCursorPosition", COORD),
                    ("wAttributes", ctypes.c_ushort), ("srWindow", SMALL_RECT),
                    ("dwMaximumWindowSize", COORD)]

    return INPUT_RECORD, SCREEN_BUFFER_INFO


def _open_own_console() -> tuple[int, int]:
    """(input, output) handles for a console this process can read itself.

    A process whose stdin is a pipe — which is what a terminal that
    multiplexes its children hands down, and what a captured child gets —
    has no console on its stdin: `GetConsoleMode` fails on it and no mouse
    event can ever arrive. `CONIN$` names the console the process is
    attached to regardless of what its handles point at, and only if there
    is none does the process need one of its own. Its own is created with
    the window hidden: the text still reaches the operator, because the
    client attached to that console mirrors the screen buffer.
    """
    import ctypes

    k = _k32()
    GENERIC_READ_WRITE = 0xC0000000
    OPEN_EXISTING = 3
    hin = k.CreateFileW("CONIN$", GENERIC_READ_WRITE, 3, None, OPEN_EXISTING, 0, None)
    hout = k.CreateFileW("CONOUT$", GENERIC_READ_WRITE, 3, None, OPEN_EXISTING, 0, None)
    if not hin or not hout:
        if not k.AllocConsole():
            return 0, 0
        hwnd = k.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(ctypes.c_void_p(hwnd), 0)   # SW_HIDE
        hin = k.CreateFileW("CONIN$", GENERIC_READ_WRITE, 3, None, OPEN_EXISTING, 0, None)
        hout = k.CreateFileW("CONOUT$", GENERIC_READ_WRITE, 3, None, OPEN_EXISTING, 0, None)
    return (hin or 0), (hout or 0)


class _RawTerminal:
    """The terminal as a byte source, plus the width to lay out against.

    This is deliberately the only platform-specific part of the module.
    """

    def __init__(self, stream=None, out=None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self.out = out if out is not None else sys.stdout
        self.ok = False
        self.fd: int | None = None
        self._saved_in = None
        self._saved_out = None
        self._in_handle = None
        self._out_handle = None
        self._conin = None
        self._conout = None
        #: The handle a keyboard or a mouse gesture is read from, and the
        #: handle the screen is measured through. Kept apart from `stream`
        #: and `out`, because a process whose stdout is a pipe still has a
        #: console to read the mouse from.
        self._input_handle = None
        self._screen_handle = None
        self._own_console = False
        self.mouse = False

    @property
    def columns(self) -> int:
        width = self._screen_width()
        if width:
            return width
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 80

    # -- lifecycle -----------------------------------------------------
    def open(self) -> bool:
        if not (getattr(self.stream, "isatty", lambda: False)()
                and getattr(self.out, "isatty", lambda: False)()):
            return False
        try:
            if os.name == "nt":
                self._open_win()
            else:
                self._open_posix()
        except Exception:
            self.ok = False
            return False
        return self.ok

    def _open_posix(self) -> None:
        import termios
        import tty

        self.fd = self.stream.fileno()
        self._saved_in = termios.tcgetattr(self.fd)
        # cbreak, not raw: no echo and no line buffering, but ISIG stays on, so
        # ^C still raises KeyboardInterrupt exactly as it does today. Taking
        # that away would be a silent regression in how you stop a run.
        tty.setcbreak(self.fd)
        self.ok = True

    def _set_input_mode(self, mouse: bool) -> bool:
        """Own the console line, and take the mouse if asked for it.

        With `mouse` False the console keeps its own quick-edit selection,
        which is what `AUTOFORGE_MOUSE=0` asks for: the editor will then not
        hear about a selection, and the console's drag-to-copy and wheel
        scrolling go on working as they always did.
        """
        import ctypes

        k = _k32()
        mode = ctypes.c_uint()
        if not k.GetConsoleMode(ctypes.c_void_p(self._input_handle), ctypes.byref(mode)):
            return False
        own = mode.value
        # ENABLE_LINE_INPUT (0x2) | ENABLE_ECHO_INPUT (0x4): the console must
        # not own the line, or every keystroke is echoed beneath the one the
        # editor drew and a pasted block arrives as one line per row.
        own &= ~0x0002 & ~0x0004
        # ENABLE_QUICK_EDIT_MODE (0x40) is the bit that matters for the mouse.
        # While it is on -- and it is on by default -- a drag selects
        # *console* text, the console keeps that selection, and everything
        # that follows is the console's business: no report of where the drag
        # went, and the next keystroke goes wherever the terminal's own cursor
        # happens to sit. It is exactly why selecting four characters in the
        # middle of a line and pressing Delete deleted the last one instead --
        # the selection existed and was never delivered anywhere.
        if mouse:
            own &= ~0x0040
        else:
            own |= 0x0040
        # ENABLE_EXTENDED_FLAGS (0x80) has to be set for the quick-edit bit to
        # be writable at all; without it the change above is silently ignored
        # and the mouse stays as the console had it.
        own |= 0x0080
        # ENABLE_PROCESSED_INPUT (0x1) stays on: ^C keeps raising
        # KeyboardInterrupt, which is how a run is stopped. ENABLE_MOUSE_INPUT
        # (0x10) is the channel a gesture arrives through, and it is opened
        # only when there is a console to hear it on.
        if mouse:
            own |= 0x0010
        else:
            own &= ~0x0010
        if not k.SetConsoleMode(ctypes.c_void_p(self._input_handle), own):
            return False
        self.mouse = mouse
        return True

    def _open_win(self) -> None:
        import ctypes

        k = _k32()
        std_in = k.GetStdHandle(-10)
        std_out = k.GetStdHandle(-11)
        probe = ctypes.c_uint()
        attached = bool(std_in) and bool(
            k.GetConsoleMode(ctypes.c_void_p(std_in), ctypes.byref(probe)))
        if attached:
            # The ordinary case: this process has a console on its standard
            # handles. Nothing is created and nothing extra is held.
            self._input_handle = std_in
            self._screen_handle = std_out
        elif not self._own_console_handles():
            raise OSError("no console input handle")
        mode = ctypes.c_uint()
        if not k.GetConsoleMode(ctypes.c_void_p(self._input_handle), ctypes.byref(mode)):
            raise OSError("no console input mode")
        self._saved_in = mode.value
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x4) is what makes the escape
        # sequences below colour and cursor moves rather than literal noise.
        # DISABLE_NEWLINE_AUTO_RETURN (0x8) is not optional: with VT on, the
        # console translates a bare newline into CRLF itself and the editor
        # already writes CRLF in _draw, so every row it drew advanced the
        # cursor two rows while _rows/_cur_row counted one. 0x000C is both.
        for handle in (std_out, self._screen_handle):
            out_mode = ctypes.c_uint()
            if handle and k.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(out_mode)):
                if self._saved_out is None:
                    self._saved_out = out_mode.value
                k.SetConsoleMode(ctypes.c_void_p(handle), out_mode.value | 0x000C)
        self._set_input_mode(mouse=mouse_enabled())
        self.ok = True

    def _own_console_handles(self) -> bool:
        """Open this process's console, creating one only if there is none.

        A process whose stdin is a pipe -- what a terminal that multiplexes
        its children hands down, what a captured child gets -- has no console
        there, and no mouse event can ever arrive. `CONIN$` names the console
        the process is attached to whatever its handles point at; only if
        there is genuinely none does one get created, window hidden. The text
        still reaches the operator, because the client attached to that
        console mirrors its screen buffer.
        """
        import ctypes

        k = _k32()
        GENERIC_READ_WRITE = 0xC0000000
        OPEN_EXISTING = 3
        hin = k.CreateFileW("CONIN$", GENERIC_READ_WRITE, 3, None, OPEN_EXISTING, 0, None)
        hout = k.CreateFileW("CONOUT$", GENERIC_READ_WRITE, 3, None, OPEN_EXISTING, 0, None)
        if not hin or not hout:
            if not k.AllocConsole():
                return False
            hwnd = k.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(ctypes.c_void_p(hwnd), 0)   # SW_HIDE
            hin = k.CreateFileW("CONIN$", GENERIC_READ_WRITE, 3, None, OPEN_EXISTING, 0, None)
            hout = k.CreateFileW("CONOUT$", GENERIC_READ_WRITE, 3, None, OPEN_EXISTING, 0, None)
            if not hin or not hout:
                return False
        self._conin, self._conout = hin, hout
        self._input_handle, self._screen_handle = hin, hout
        self._own_console = True
        return True

    def mouse_on(self) -> None:
        """Ask for gestures. The mode bit above is the half that works here."""
        self._write_mouse_request(_MOUSE_ON)

    def mouse_off(self) -> None:
        self._write_mouse_request(_MOUSE_OFF)

    def _write_mouse_request(self, text: str) -> None:
        """Send the request where the console will actually read it.

        A console has to be asked through the console API: its input buffer is
        what decides whether a gesture is reported, and a VT terminal in front
        of a real console wants the same text as bytes on its stream. One or
        the other is a no-op, so both are sent and whichever applies answers.
        """
        if os.name == "nt" and self._screen_handle:
            import ctypes

            k = _k32()
            written = ctypes.c_uint()
            try:
                k.WriteConsoleW(ctypes.c_void_p(self._screen_handle), text, len(text),
                                ctypes.byref(written), None)
            except Exception:                 # noqa: BLE001 - a console that will not take it
                pass
        self._raw_escape(text)

    def _raw_escape(self, text: str) -> None:
        try:
            self.out.write(text)
            self.out.flush()
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        if not self.ok:
            return
        try:
            self.mouse_off()
            if os.name == "nt":
                import ctypes

                k = _k32()
                if self._saved_in is not None:
                    k.SetConsoleMode(ctypes.c_void_p(self._input_handle), self._saved_in)
                if self._saved_out is not None:
                    for handle in (k.GetStdHandle(-11), self._screen_handle):
                        if handle:
                            k.SetConsoleMode(ctypes.c_void_p(handle), self._saved_out)
                for handle in (self._conin, self._conout):
                    if handle:
                        k.CloseHandle(ctypes.c_void_p(handle))
            elif self._saved_in is not None and self.fd is not None:
                import termios

                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved_in)
        except Exception:
            pass
        self.ok = False

    # -- the screen, for laying a gesture out --------------------------
    def _screen_info(self):
        import ctypes

        if not self._screen_handle:
            return None
        k = _k32()
        _, SCREEN_BUFFER_INFO = _console_records()
        info = SCREEN_BUFFER_INFO()
        if not k.GetConsoleScreenBufferInfo(ctypes.c_void_p(self._screen_handle),
                                            ctypes.byref(info)):
            return None
        return info

    def _screen_width(self) -> int:
        info = self._screen_info()
        if info is None:
            return 0
        left, right = info.srWindow.Left, info.srWindow.Right
        if right >= left:
            return right - left + 1
        return int(info.dwSize.X)

    def cursor_row(self) -> int | None:
        """The row the console's own cursor is on, right now.

        A gesture arrives with absolute buffer coordinates, and the input
        area is somewhere inside them; the difference between the two is the
        cursor's position on the last draw. So the cursor is the anchor the
        whole mapping hangs from, and it is read at the moment of the
        gesture rather than remembered, because everything above the input
        area may have scrolled since.
        """
        info = self._screen_info()
        return None if info is None else int(info.dwCursorPosition.Y)

    # -- input ---------------------------------------------------------
    def read_chunk(self) -> bytes | None:
        """Everything the terminal has right now, blocking for at least one byte.

        Returning the whole buffer rather than one byte matters twice over:
        a UTF-8 character survives the trip, and a paste arrives as one chunk
        instead of a thousand keystrokes.
        """
        if os.name == "nt":
            if self._input_handle:
                return self._read_records()
            import msvcrt

            try:
                ch = msvcrt.getwch()
            except (EOFError, KeyboardInterrupt):
                return None
            if ch in ("\x00", "\xe0"):
                # A function key: two code units. The second is a scan code,
                # and on Windows it is the *only* place the key's identity
                # exists. Discarding it, which is what this once did, silently
                # killed left/right/home/end/delete: the editor's
                # `_apply_key` was correct and simply never called.
                try:
                    code = msvcrt.getwch()
                except (EOFError, KeyboardInterrupt):
                    return None
                return _WIN_SCAN.get(code, "").encode("ascii")
            return ch.encode("utf-8", "replace")
        try:
            data = os.read(self.fd, 65536)
        except (OSError, ValueError):
            return None
        return data or None

    def _read_records(self) -> bytes | None:
        """One burst of console input, decoded to the shape the editor reads.

        Reading the console's own input records rather than asking for ANSI
        mouse reports is a deliberate choice, and it is the one that keeps ^C
        working. `ENABLE_VIRTUAL_TERMINAL_INPUT` would turn a mouse gesture
        into an escape sequence -- which is convenient -- but it takes the
        console's line editing and its ^C handling away with it, and it
        re-encodes UTF-8 into whatever the console's code page happens to be.
        The records carry the mouse *and* leave the keyboard exactly as it
        is.

        A mouse record is turned into the SGR report a VT terminal would have
        sent, on the spot. One decoder then covers both sources, and the
        editor cannot tell which terminal it is talking to.
        """
        import ctypes

        k = _k32()
        INPUT_RECORD, _ = _console_records()
        handle = ctypes.c_void_p(self._input_handle)
        count = ctypes.c_uint()
        while True:
            if not k.GetNumberOfConsoleInputEvents(handle, ctypes.byref(count)):
                return None
            if count.value:
                break
            # Console input handles are waitable: signalled when the buffer is
            # no longer empty. The timeout is only there so a console that
            # does not behave that way costs a poll every 50ms instead of a
            # session that hangs.
            k.WaitForSingleObject(handle, 50)
        batch = (INPUT_RECORD * 64)()
        read = ctypes.c_uint()
        if not k.ReadConsoleInputW(handle, ctypes.byref(batch), 64, ctypes.byref(read)):
            return None
        parts: list[str] = []
        for n in range(read.value):
            record = batch[n]
            if record.EventType == 0x0001:            # KEY_EVENT
                key = record.Event.KeyEvent
                if not key.bKeyDown:
                    continue
                repeat = max(1, min(int(key.wRepeatCount), 64))
                vk = int(key.wVirtualKeyCode)
                ctrl = bool(int(key.dwControlKeyState) & (_LEFT_CTRL | _RIGHT_CTRL))
                shift = bool(int(key.dwControlKeyState) & _SHIFT)
                # Ctrl+Insert copies, Ctrl+Delete cuts, Shift+Insert pastes:
                # the three gestures a Windows console has always had for the
                # clipboard. They are separate keys rather than an overload of
                # ^C on purpose -- ^C has to keep stopping a run, and a
                # terminal that can copy but cannot be interrupted is a
                # terminal nobody can get out of.
                if vk == 0x2D and ctrl:
                    text = "\x1b[2;5~"
                elif vk == 0x2E and ctrl:
                    text = "\x1b[3;5~"
                elif vk == 0x2D and shift:
                    text = "\x1b[2;2~"
                else:
                    text = _WIN_VK.get(vk)
                if text is None:
                    ch = key.uChar
                    if ch and (ch >= " " or ch in _CONTROL_CHARS):
                        text = ch * repeat
                if text:
                    parts.append(text)
            elif record.EventType == 0x0002:          # MOUSE_EVENT
                report = _mouse_report(record.Event.MouseEvent)
                if report:
                    parts.append(report)
        if not parts:
            return b""
        return "".join(parts).encode("utf-8", "replace")

    def ready(self, timeout: float) -> bool:
        """Is more input already buffered? Used to gather a paste burst."""
        if os.name == "nt":
            if self._input_handle:
                import ctypes

                k = _k32()
                count = ctypes.c_uint()
                deadline = time.monotonic() + timeout
                while True:
                    if not k.GetNumberOfConsoleInputEvents(
                            ctypes.c_void_p(self._input_handle), ctypes.byref(count)):
                        return False
                    if count.value:
                        return True
                    if time.monotonic() >= deadline:
                        return False
                    time.sleep(0.005)
            import msvcrt

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    if msvcrt.kbhit():
                        return True
                except Exception:
                    return False
                time.sleep(0.005)
            return False
        import select

        try:
            return bool(select.select([self.fd], [], [], timeout)[0])
        except (OSError, ValueError):
            return False


class LineEditor:
    """Owns the bottom line of the terminal.

    One thread reads keys (`readline`); any thread may write above the input
    area (`write`, `tick`). Both take the same lock, so the erase/redraw
    sequence cannot interleave with itself.
    """

    def __init__(self, stream=None, out=None, term=None, *,
                 paste_lines: int = PASTE_LINES,
                 paste_chars: int = PASTE_CHARS,
                 paste_to: Path | str | None = None,
                 submit_grace: float = SUBMIT_GRACE,
                 clipboard_read=None,
                 clipboard_write=None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self.out = out if out is not None else sys.stdout
        self._term = term if term is not None else _RawTerminal(self.stream, self.out)
        self._paste_lines = paste_lines
        self._paste_chars = paste_chars
        self._paste_to = paste_to
        self._submit_grace = max(0.0, float(submit_grace))
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._lock = threading.RLock()
        self._prompt = ""
        self._buf: list[str] = []
        self._cursor = 0
        self._rows = 0
        self._cur_row = 0
        self._ticking = False
        self._open = False
        self.eof = False
        # The selection: an anchor where the drag started, and the cursor as
        # its live end. `None` means nothing is selected, and it is kept as
        # None rather than as an empty range because "no selection" and "an
        # empty selection" have to stay distinguishable: the first is the
        # ordinary state, the second is a gesture nothing should act on.
        self._anchor: int | None = None
        self._mouse_live = False
        self._mouse_ok = mouse_enabled()
        self._clip_read = clipboard_read or read_clipboard
        self._clip_write = clipboard_write or write_clipboard

    # -- lifecycle -----------------------------------------------------
    @property
    def available(self) -> bool:
        """Can this terminal be driven as a line editor at all?"""
        return self._open

    def start(self) -> bool:
        """Enter cbreak mode, and take the mouse away from the console.

        Taking the mouse is part of starting rather than a separate opt-in: a
        terminal where the console owns the selection is a terminal where
        selecting the middle of the line and pressing Delete deletes the last
        character instead, and that is the behaviour this module exists to
        fix. Where the mouse cannot be taken, nothing changes and nothing is
        lost.
        """
        if self._open:
            return True
        self._open = bool(self._term.open())
        if self._open and self._mouse_ok:
            try:
                self._term.mouse_on()
            except Exception:                 # noqa: BLE001 - a terminal with no mouse
                pass
        return self._open

    def close(self) -> None:
        """Give the terminal back the way we found it, mouse included."""
        if not self._open:
            return
        if self._mouse_ok:
            try:
                self._term.mouse_off()
            except Exception:                 # noqa: BLE001 - nothing to give back
                pass
        with self._lock:
            self._erase(extra_up=1 if self._ticking else 0)
            self._ticking = False
            self._raw("\n")
        self._term.close()
        self._open = False

    # -- output that must not clobber the input line -------------------
    def write(self, text: str = "") -> None:
        """A line of output, above the input area.

        It takes the heartbeat's row if one is up — the beat is about to be
        stale anyway, and leaving it above this line would strand it there.
        """
        if not self._open:
            self._raw(text if text.endswith("\n") else text + "\n")
            return
        with self._lock:
            self._erase(extra_up=1 if self._ticking else 0)
            self._ticking = False
            self._raw(text if text.endswith("\n") else text + "\n")
            self._draw()

    def tick(self, text: str) -> None:
        """A line that rewrites itself in place — the heartbeat.

        It gets a row of its own, immediately above the input area, and keeps
        it: erase what we own, rewrite the row, newline, redraw the area
        below. So a heartbeat that ticks for an hour is still one line.
        """
        if not self._open:
            self._raw(text + "\n")
            return
        with self._lock:
            self._erase(extra_up=1 if self._ticking else 0)
            self._raw(text + _ERASE_TO_EOL + "\n")
            self._ticking = True
            self._draw()

    def set_prompt(self, prompt: str) -> None:
        with self._lock:
            if prompt == self._prompt:
                return
            self._prompt = prompt
            if self._open:
                self._erase()
                self._draw()

    def redraw(self) -> None:
        with self._lock:
            if self._open:
                self._erase()
                self._draw()

    # -- the selection -------------------------------------------------
    def selection(self) -> tuple[int, int] | None:
        """The selected span of the buffer, or None when nothing is selected."""
        if self._anchor is None or self._anchor == self._cursor:
            return None
        return (min(self._anchor, self._cursor), max(self._anchor, self._cursor))

    def _delete_selection(self) -> bool:
        """Remove the selected span. True when there was one to remove.

        This is the whole point of the feature. The keys that used to act at
        the cursor alone act on the selection when there is one, so selecting
        four characters in the middle of a line and pressing Delete deletes
        those four rather than the last one.
        """
        span = self.selection()
        if span is None:
            return False
        start, stop = span
        del self._buf[start:stop]
        self._cursor = start
        self._anchor = None
        return True

    def _copy_selection(self) -> bool:
        """Put the selected span on the system clipboard, and drop the highlight.

        Dropping it is what a terminal does after a copy, and it is the
        deliberate half: a selection left highlighted is a selection the next
        keystroke deletes, which is a surprise arriving one keystroke after a
        copy. True when there was a span to copy.
        """
        span = self.selection()
        if span is None:
            return False
        text = "".join(self._buf[span[0]:span[1]])
        self._anchor = None
        if self._clip_write(text):
            return True
        return False

    def _cut_selection(self) -> bool:
        span = self.selection()
        if span is None:
            return False
        text = "".join(self._buf[span[0]:span[1]])
        try:
            self._clip_write(text)
        except Exception:                     # noqa: BLE001 - a clipboard that cannot answer
            pass
        del self._buf[span[0]:span[1]]
        self._cursor = span[0]
        self._anchor = None
        return True

    def _paste(self, text: str = "") -> bool:
        """Insert the clipboard at the cursor, replacing the selection if any."""
        if text == "":
            try:
                text = self._clip_read() or ""
            except Exception:                 # noqa: BLE001 - a clipboard that cannot answer
                return False
        if not text:
            return False
        self._delete_selection()
        self._insert(text.replace("\r\n", "\n").replace("\r", "\n"))
        return True

    def _mouse(self, code: int, x: int, y: int) -> None:
        """One mouse report, in SGR terms: `code`, 1-based column, 1-based row.

        The row in the report is an absolute console buffer row and the editor
        has no idea which one, so only the *difference* is used: the console's
        own cursor is sitting on the input area's last drawn row, and every
        row above or below it is counted from there. That is also why the
        layout is recomputed here instead of remembered -- whatever was
        printed above the input area may have scrolled the buffer since the
        last draw, and a remembered row would then be wrong by exactly that
        scroll.
        """
        if code & 64:                       # wheel: nothing here scrolls
            return
        button = code & 3
        if button == 3:                     # release
            self._mouse_live = False
            return
        row_now = None
        try:
            row_now = self._term.cursor_row()
        except Exception:                   # noqa: BLE001 - a terminal with no screen
            row_now = None
        if row_now is None:
            return
        layout = self._layout_detail()
        rows, _starts, _prefixes, leading, row, _col = layout
        target = max(0, min(row + (y - 1 - row_now), len(rows) - 1))
        index = self._index_at(layout, target, x - 1)
        if code & 32:                       # motion with a button held: extend
            if not self._mouse_live:
                return
            self._cursor = index
        elif button in (1, 2):
            # Middle or right: paste where the gesture landed, as an X
            # terminal does. It is also the only way to paste into a terminal
            # whose console would otherwise swallow the gesture.
            self._cursor = index
            self._anchor = None
            self._paste()
        else:
            # A press starts a selection even if the drag never moves: a
            # click places the cursor, which is the same gesture with an
            # anchor that ends up equal to it.
            self._mouse_live = True
            self._anchor = index
            self._cursor = index
        self._erase()
        self._draw()

    def _index_at(self, layout, row: int, col: int) -> int:
        """Which buffer position a screen cell is, counted in characters.

        The cell is a column, and a column is not a character: a CJK glyph
        takes two of them and a colour code takes none. So the row's text is
        walked with the same widths it was wrapped with, and the position is
        where that walk passes the column asked for.
        """
        rows, starts, prefixes, leading, _row, _col = layout
        total = len(self._buf)
        if row < leading:
            return 0
        if row >= len(rows):
            return total
        start = starts[row]
        body = rows[row][len(prefixes[row]):]
        offset = col - _visible_len(prefixes[row])
        if offset <= 0:
            return min(start, total)
        width = 0
        for k, ch in enumerate(body):
            w = _wcwidth(ch)
            w = w if w and w > 0 else 0
            if width + w > offset:
                return min(start + k, total)
            width += w
        return min(start + len(body), total)

    # -- input ---------------------------------------------------------
    def readline(self) -> str | None:
        """Block until a line is submitted. None means EOF (^D, or stdin closed)."""
        if not self._open:
            return None
        self.redraw()
        while True:
            burst = self._read_burst()
            if burst is None:
                self.eof = True
                with self._lock:
                    self._erase()
                    self._raw("\n")
                return None
            line = self._consume(burst)
            if line is not None:
                return line
            if self.eof:
                return None

    # -- internals -----------------------------------------------------
    def _raw(self, text: str) -> None:
        try:
            self.out.write(text)
            self.out.flush()
        except (OSError, ValueError):
            pass

    def _read_burst(self) -> str | None:
        chunk = self._term.read_chunk()
        if chunk is None:
            return None
        out = self._decoder.decode(chunk, False)
        # A paste is delivered faster than we can read it: whatever is already
        # buffered is almost certainly the rest of the same paste. The timeout
        # is short enough not to be felt between keystrokes.
        while self._term.ready(0.02):
            more = self._term.read_chunk()
            if more is None:
                break
            out += self._decoder.decode(more, False)
        return out or ""

    def _consume(self, burst: str) -> str | None:
        """Feed one burst of input. Returns a submitted line, or None."""
        with self._lock:
            text = burst
            bracketed = False
            if text.startswith(_BRACKET_START):
                bracketed = True
                text = text[len(_BRACKET_START):]
                if text.endswith(_BRACKET_END):
                    text = text[: -len(_BRACKET_END)]
            if bracketed or _looks_like_paste(text):
                self._insert(self._as_paste(text))
                self.redraw()
                return None
            i = 0
            while i < len(text):
                report = _MOUSE_SGR.match(text, i)
                if report is not None:
                    self._mouse(int(report.group(1)), int(report.group(2)),
                                int(report.group(3)))
                    i = report.end()
                    continue
                key = _match_key(text, i)
                if key is not None:
                    seq, name = key
                    self._apply_key(name)
                    i += len(seq)
                    continue
                ch = text[i]
                i += 1
                if ch in ("\r", "\n"):
                    if self._more_coming():
                        # A break with the next line already on its way is a
                        # break inside a block, not an Enter: the person at
                        # the keyboard cannot have sent bytes that are still
                        # in flight. Keep the break and keep reading; the
                        # Enter is the break that nothing follows.
                        self._buf.insert(self._cursor, "\n")
                        self._cursor += 1
                        # CRLF is one break, not two.
                        if i < len(text) and text[i] == ("\n" if ch == "\r" else "\r"):
                            i += 1
                        continue
                    return self._submit()
                if ch in _BACKSPACE:
                    # With something selected, backspace removes the
                    # selection -- that is what it means in every editor, and
                    # without it the key would nibble at the line's edge while
                    # a highlighted span sat there untouched.
                    if not self._delete_selection() and self._cursor:
                        del self._buf[self._cursor - 1]
                        self._cursor -= 1
                elif ch == _CTRL_D:
                    if not self._buf:
                        self.eof = True
                        self._erase()
                        self._raw("\n")
                        return None
                    if self._cursor < len(self._buf):
                        del self._buf[self._cursor]
                elif ch == _CTRL_X:
                    self._cut_selection()
                elif ch == _CTRL_C:
                    # Reached only when something upstream has left ^C out of
                    # the console's signalling path: with processed input on
                    # the console turns it into an interrupt and never puts it
                    # in the buffer. Copying is deliberately not hung off this
                    # key -- a terminal that can copy but cannot be
                    # interrupted is a terminal nobody can get out of, and
                    # Ctrl+Insert carries the copy instead. So the byte is
                    # passed on, for whatever is listening below.
                    self._raw("\x03")
                elif ch in (_CTRL_V, _CTRL_Y):
                    self._paste()
                elif ch == _CTRL_U:
                    if not self._delete_selection():
                        del self._buf[:self._cursor]
                        self._cursor = 0
                elif ch == _CTRL_K:
                    if not self._delete_selection():
                        del self._buf[self._cursor:]
                elif ch == _CTRL_W:
                    if not self._delete_selection():
                        self._kill_word()
                elif ch == _CTRL_A:
                    self._cursor = 0
                elif ch == _CTRL_E:
                    self._cursor = len(self._buf)
                elif ch == _CTRL_L:
                    self._raw("\x1b[2J\x1b[H")
                elif ch < " " or ch == "\x1b":
                    continue   # unhandled control seq — never insert it raw
                else:
                    # Typing over a selection replaces it, as it does anywhere
                    # else. Without this the character lands inside the
                    # highlighted span and the span survives around it.
                    self._delete_selection()
                    self._buf.insert(self._cursor, ch)
                    self._cursor += 1
            self._erase()
            self._draw()
            return None

    def _more_coming(self) -> bool:
        """Is more of the same input already on its way to us?

        Asked of the terminal rather than of a clock: a burst that has already
        been sent is a fact, and it is the only thing that can tell a break
        inside a block from an Enter. A terminal that cannot answer — a pipe,
        a dumb one — answers no, and the editor behaves as it did before.
        """
        if self._submit_grace <= 0:
            return False
        try:
            return bool(self._term.ready(self._submit_grace))
        except Exception:                 # noqa: BLE001 - a terminal that cannot say
            return False

    def _submit(self) -> str:
        """Hand the assembled line back and leave the input line empty.

        A block that was gathered this way is a paste, so it leaves the same
        way any other paste does: collapsed to a file if it is big enough to
        make the input line unusable, verbatim if it is not.
        """
        line = "".join(self._buf)
        self._buf, self._cursor, self._anchor = [], 0, None
        self._erase()
        self._raw("\n")
        return self._as_paste(line)

    def _apply_key(self, name: str) -> None:
        if name == "left":
            self._cursor = max(0, self._cursor - 1)
            self._anchor = None
        elif name == "right":
            self._cursor = min(len(self._buf), self._cursor + 1)
            self._anchor = None
        elif name == "home":
            self._cursor = 0
            self._anchor = None
        elif name == "end":
            self._cursor = len(self._buf)
            self._anchor = None
        elif name == "copy":
            self._copy_selection()
        elif name == "cut":
            # With nothing selected it degrades to the character under the
            # cursor rather than doing nothing: a key that silently does
            # nothing is a key nobody can trust.
            if not self._cut_selection() and self._cursor < len(self._buf):
                del self._buf[self._cursor]
        elif name == "paste":
            self._paste()
        elif name == "delete":
            # Delete removes the selection first, which is the whole point of
            # this change: selecting four characters in the middle of the
            # line and pressing Delete used to delete the last one, because
            # the key reached a cursor that was still at the end.
            if not self._delete_selection() and self._cursor < len(self._buf):
                del self._buf[self._cursor]

    def _kill_word(self) -> None:
        i = self._cursor
        while i > 0 and self._buf[i - 1].isspace():
            i -= 1
        while i > 0 and not self._buf[i - 1].isspace():
            i -= 1
        del self._buf[i:self._cursor]
        self._cursor = i

    def _as_paste(self, text: str) -> str:
        """A paste, briefly: collapse it, or insert it verbatim."""
        if text.lstrip().startswith("/"):
            # A pasted command keeps its newlines: those are a script, not a
            # paragraph, and hiding them behind a file pointer helps nobody.
            return text
        return collapse_paste(text, lines=self._paste_lines, chars=self._paste_chars,
                              directory=self._paste_to)

    def _insert(self, text: str) -> None:
        for ch in text:
            self._buf.insert(self._cursor, ch)
            self._cursor += 1

    def _erase(self, extra_up: int = 0) -> None:
        """Clear what the editor owns, leaving the cursor at its top row.

        That is the input area, plus — when a heartbeat is currently up — the
        row above it. `extra_up` is that row. Without climbing over it, the
        next heartbeat would stack underneath the last one instead of
        replacing it, and an hour-long run would leave a screen of them.
        """
        if not self._rows and not extra_up:
            return
        up = self._cur_row + extra_up
        parts = []
        if up:
            parts.append(f"\x1b[{up}A")
        parts.append("\r" + _ERASE_TO_EOS)
        self._raw("".join(parts))
        self._rows = 0
        self._cur_row = 0

    def _layout_detail(self):
        """Wrap the buffer: (rows, row starts, row prefixes, leading, row, col).

        Continuation rows are indented to the width of the prompt, so a
        wrapped line still reads as one entry rather than as a new one.

        A break inside the buffer is a hard break, not something to wrap: a
        small block pasted on a terminal without bracketed-paste support keeps
        its own lines, and laying those out as if they were ordinary
        characters would put the cursor on the wrong row.

        A break inside the *prompt* is a row the prompt occupies, and the
        buffer starts on the prompt's last row — `chat`'s prompt opens with a
        newline, so the input line sits one row below it. Those rows are part
        of the area the editor owns, and `row` is counted from the prompt's
        first row rather than from the buffer's, so that erasing climbs back
        over them.

        The three extra lists are what a mouse gesture needs and a cursor
        does not: which buffer position each row starts at, how wide its
        prefix is, and how many rows come before the buffer. A screen cell is
        a row and a column, and turning one back into a character offset
        takes exactly these.
        """
        columns = max(20, int(getattr(self._term, "columns", 80) or 80))
        prompt_lines = self._prompt.split("\n")
        head, last = prompt_lines[:-1], prompt_lines[-1]
        indent = " " * _visible_len(last)
        # One column is held back so the cursor never comes to rest on the
        # right margin: a terminal that wraps at that column would push a
        # blank row under the input area every time the buffer filled a line.
        width = max(1, columns - len(indent) - 1)

        rows: list[str] = list(head)     # the prompt's own earlier rows
        starts: list[int] = [0] * len(head)
        prefixes: list[str] = list(head)
        leading = len(rows)              # rows above the one the buffer starts on
        first = True
        offset = 0
        for segment in "".join(self._buf).split("\n"):
            at = offset
            for i in _col_breaks(segment, width):
                prefix = last if first else indent
                chunk = segment[i:i + _col_step(segment, i, width)]
                rows.append(prefix + chunk)
                prefixes.append(prefix)
                starts.append(at)
                at += len(chunk)
                first = False
            offset += len(segment) + 1  # the break that `split` consumed

        at = min(self._cursor, offset - 1)
        row = leading
        for n, start in enumerate(starts):
            if at >= start:
                row = n
        prefix = prefixes[row]
        col = _visible_len(prefix)
        col += _visible_len(rows[row][len(prefix):][:max(0, at - starts[row])])
        return rows, starts, prefixes, leading, row, col

    def _layout(self) -> tuple[list[str], int, int]:
        """The drawing's view of the layout: rows, cursor row, cursor column."""
        rows, _starts, _prefixes, _leading, row, col = self._layout_detail()
        return rows, row, col

    def _draw(self) -> None:
        rows, starts, prefixes, leading, row, col = self._layout_detail()
        marks = self._selection_marks(rows, starts, prefixes, leading)
        parts: list[str] = []
        for n, text in enumerate(rows):
            if n:
                parts.append("\r\n")
            parts.append(self._paint(text, len(prefixes[n]), marks[n]))
        up = len(rows) - 1 - row
        if up:
            parts.append(f"\x1b[{up}A")
        parts.append("\r")
        if col:
            parts.append(f"\x1b[{col}C")
        self._raw("".join(parts))
        self._rows = len(rows)
        self._cur_row = row

    def _selection_marks(self, rows, starts, prefixes, leading) -> list:
        """Per row, the character span drawn in reverse video.

        A span, per row, rather than one span for the buffer: a selection
        crosses rows when it crosses a wrap, and each row has to be painted
        in its own right. The offsets are into the row's *body*, so the
        prompt and the continuation indent keep their own colours.
        """
        span = self.selection()
        if span is None:
            return [()] * len(rows)
        low, high = span
        marks = []
        for n, start in enumerate(starts):
            body = len(rows[n]) - len(prefixes[n])
            a = max(0, low - start)
            b = min(body, high - start)
            marks.append((a, b) if b > a else ())
        return marks

    def _paint(self, text: str, prefix_len: int, mark) -> str:
        """One row, with the selected part of its body in reverse video."""
        if not mark:
            return text
        a, b = mark
        head = text[:prefix_len + a]
        body = text[prefix_len + a:prefix_len + b]
        tail = text[prefix_len + b:]
        return f"{head}{_REVERSE_ON}{body}{_REVERSE_OFF}{tail}"


def _match_key(text: str, i: int) -> tuple[str, str] | None:
    for seq in _KEY_ORDER:
        if text.startswith(seq, i):
            return seq, _KEYS[seq]
    return None


def _looks_like_paste(text: str) -> bool:
    """A burst that carries its own line breaks is a paste, not typing.

    Count *logical* breaks, so a terminal that sends CRLF for Enter is not
    mistaken for a block. One break at the very end is Enter — possibly with
    the characters typed just before it, which is exactly what a fast typist
    produces in a single read. Anything else (a break in the middle, or two of
    them) cannot come from one keystroke, and is a pasted block an older
    terminal sent without the bracketed-paste markers.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    breaks = normalized.count("\n")
    if breaks == 0:
        return False
    return breaks > 1 or not normalized.endswith("\n")


def _col_step(segment: str, i: int, width: int) -> int:
    """How many characters of `segment` from index i fit in `width` columns."""
    used = 0
    n = 0
    for ch in segment[i:]:
        w = _wcwidth(ch)
        w = w if w and w > 0 else 0
        if n and used + w > width:
            break
        used += w
        n += 1
    return max(1, n)


def _col_breaks(segment: str, width: int) -> list:
    """Character offsets at which `segment` must wrap to a new terminal row."""
    breaks = []
    i = 0
    total = len(segment)
    while i < total:
        breaks.append(i)
        i += _col_step(segment, i, width)
    if not breaks:
        breaks.append(0)
    return breaks


def _visible_len_legacy(s):
    n = 0
    i = 0
    L = len(s)
    while i < L:
        c = s[i]
        if c == "\x1b":
            m = re.match(r"\x1b\[[0-9;?]*[A-Za-z]", s[i:])
            if m:
                i += m.end()
                continue
        o = ord(c)
        if o < 32 or o == 127:
            i += 1
            continue
        w = _wcwidth(c)
        n += w if w and w > 0 else 0
        i += 1
    return n
