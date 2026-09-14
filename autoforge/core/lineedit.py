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
import sys
import threading
import time
from pathlib import Path

__all__ = [
    "LineEditor",
    "collapse_paste",
    "expand_paste_refs",
    "paste_dir",
    "PASTE_CHARS",
    "PASTE_LINES",
]

#: A paste of at least this many lines, or this many characters, is collapsed
#: to a file. Below both, it is simply inserted — a three-line snippet is
#: easier to read in the input line than as a pointer to a temp file.
PASTE_LINES = 5
PASTE_CHARS = 2000

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
}
_KEY_ORDER = sorted(_KEYS, key=len, reverse=True)

#: Control characters that mean something to the editor.
_CTRL_U = "\x15"   # kill to start of line
_CTRL_K = "\x0b"   # kill to end of line
_CTRL_W = "\x17"   # kill previous word
_CTRL_A = "\x01"   # home
_CTRL_E = "\x05"   # end
_CTRL_L = "\x0c"   # redraw
_CTRL_D = "\x04"   # EOF when the buffer is empty
_BACKSPACE = ("\x7f", "\x08")


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
    """Length as the terminal counts it — colour codes take no columns."""
    return len(_ANSI.sub("", text))


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

    @property
    def columns(self) -> int:
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

    def _open_win(self) -> None:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        self._in_handle = kernel32.GetStdHandle(-10)
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(self._in_handle, ctypes.byref(mode)):
            raise OSError("no console input mode")
        self._saved_in = mode.value
        # ENABLE_LINE_INPUT (0x2) | ENABLE_ECHO_INPUT (0x4)
        kernel32.SetConsoleMode(self._in_handle, mode.value & ~0x0006)

        self._out_handle = kernel32.GetStdHandle(-11)
        out_mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(self._out_handle, ctypes.byref(out_mode)):
            self._saved_out = out_mode.value
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x4): without it the escape
            # sequences below print as literal `^[[J` noise in cmd.exe.
            kernel32.SetConsoleMode(self._out_handle, out_mode.value | 0x0004)
        self.ok = True

    def close(self) -> None:
        if not self.ok:
            return
        try:
            if os.name == "nt":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                if self._saved_in is not None and self._in_handle is not None:
                    kernel32.SetConsoleMode(self._in_handle, self._saved_in)
                if self._saved_out is not None and self._out_handle is not None:
                    kernel32.SetConsoleMode(self._out_handle, self._saved_out)
            elif self._saved_in is not None and self.fd is not None:
                import termios

                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved_in)
        except Exception:
            pass
        self.ok = False

    # -- input ---------------------------------------------------------
    def read_chunk(self) -> bytes | None:
        """Everything the terminal has right now, blocking for at least one byte.

        Returning the whole buffer rather than one byte matters twice over:
        a UTF-8 character survives the trip, and a paste arrives as one chunk
        instead of a thousand keystrokes.
        """
        if os.name == "nt":
            import msvcrt

            try:
                if not msvcrt.kbhit():
                    ch = msvcrt.getwch()   # blocks
                else:
                    ch = msvcrt.getwch()
            except (EOFError, KeyboardInterrupt):
                return None
            if ch in ("\x00", "\xe0"):
                # A function key: two code units, and we want neither.
                try:
                    msvcrt.getwch()
                except Exception:
                    pass
                return b""
            return ch.encode("utf-8", "replace")
        try:
            data = os.read(self.fd, 65536)
        except (OSError, ValueError):
            return None
        return data or None

    def ready(self, timeout: float) -> bool:
        """Is more input already buffered? Used to gather a paste burst."""
        if os.name == "nt":
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
                 paste_to: Path | str | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self.out = out if out is not None else sys.stdout
        self._term = term if term is not None else _RawTerminal(self.stream, self.out)
        self._paste_lines = paste_lines
        self._paste_chars = paste_chars
        self._paste_to = paste_to
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

    # -- lifecycle -----------------------------------------------------
    @property
    def available(self) -> bool:
        """Can this terminal be driven as a line editor at all?"""
        return self._open

    def start(self) -> bool:
        """Enter cbreak mode. Returns whether it worked."""
        if self._open:
            return True
        self._open = bool(self._term.open())
        return self._open

    def close(self) -> None:
        """Give the terminal back the way we found it."""
        if not self._open:
            return
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
                key = _match_key(text, i)
                if key is not None:
                    seq, name = key
                    self._apply_key(name)
                    i += len(seq)
                    continue
                ch = text[i]
                i += 1
                if ch in ("\r", "\n"):
                    line = "".join(self._buf)
                    self._buf, self._cursor = [], 0
                    self._erase()
                    self._raw("\n")
                    return line
                if ch in _BACKSPACE:
                    if self._cursor:
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
                elif ch == _CTRL_U:
                    del self._buf[:self._cursor]
                    self._cursor = 0
                elif ch == _CTRL_K:
                    del self._buf[self._cursor:]
                elif ch == _CTRL_W:
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
                    self._buf.insert(self._cursor, ch)
                    self._cursor += 1
            self._erase()
            self._draw()
            return None

    def _apply_key(self, name: str) -> None:
        if name == "left":
            self._cursor = max(0, self._cursor - 1)
        elif name == "right":
            self._cursor = min(len(self._buf), self._cursor + 1)
        elif name == "home":
            self._cursor = 0
        elif name == "end":
            self._cursor = len(self._buf)
        elif name == "delete":
            if self._cursor < len(self._buf):
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

    def _layout(self) -> tuple[list[str], int, int]:
        """Wrap the buffer, and say where the cursor lands.

        Continuation rows are indented to the width of the prompt, so a
        wrapped line still reads as one entry rather than as a new one.

        A break inside the buffer is a hard break, not something to wrap: a
        small block pasted on a terminal without bracketed-paste support keeps
        its own lines, and laying those out as if they were ordinary
        characters would put the cursor on the wrong row.
        """
        columns = max(20, int(getattr(self._term, "columns", 80) or 80))
        indent = " " * _visible_len(self._prompt)
        # One column is held back so the cursor never comes to rest on the
        # right margin: a terminal that wraps at that column would push a
        # blank row under the input area every time the buffer filled a line.
        width = max(1, columns - len(indent) - 1)

        rows: list[str] = []
        starts: list[int] = []          # where in the buffer each row's text begins
        first = True
        offset = 0
        for segment in "".join(self._buf).split("\n"):
            at = offset
            for i in range(0, len(segment), width) or [0]:
                chunk = segment[i:i + width]
                rows.append((self._prompt if first else indent) + chunk)
                starts.append(at)
                at += len(chunk)
                first = False
            offset += len(segment) + 1  # the break that `split` consumed

        at = min(self._cursor, offset - 1)
        row = 0
        for n, start in enumerate(starts):
            if at >= start:
                row = n
        col = _visible_len(self._prompt if row == 0 else indent)
        col += min(at - starts[row], len(rows[row]) - col)
        return rows, row, col

    def _draw(self) -> None:
        rows, row, col = self._layout()
        parts: list[str] = []
        for n, text in enumerate(rows):
            if n:
                parts.append("\r\n")
            parts.append(text)
        up = len(rows) - 1 - row
        if up:
            parts.append(f"\x1b[{up}A")
        parts.append("\r")
        if col:
            parts.append(f"\x1b[{col}C")
        self._raw("".join(parts))
        self._rows = len(rows)
        self._cur_row = row


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
