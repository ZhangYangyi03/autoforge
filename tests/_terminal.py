"""Fakes for the terminal, shared by the tests that need one.

`Screen` is the important one. Asserting on the raw byte stream is the wrong
model for a line editor: it clears and redraws by moving the cursor, so the
same text appears many times in the stream and exactly once on the screen.
Any question of the form "is this one row or five" has to be asked of a
screen, and this is a screen — just enough of one to answer.
"""
from __future__ import annotations

import re


class Screen:
    """A terminal, reduced to what the editor's output can mean.

    Handles only what the editor emits: cursor-up, erase-from-cursor-down,
    erase-to-end-of-line, CR, LF, and printable text. Anything else is
    dropped, which is itself the assertion for "no escape sequence leaked
    into the display as literal characters".
    """

    def __init__(self, height: int = 200):
        self.rows = [""] * height
        self.row = 0
        self.col = 0
        self._esc = ""

    def feed(self, text: str) -> "Screen":
        for ch in text:
            if self._esc:
                self._esc += ch
                if ch.isalpha():
                    self._apply(self._esc)
                    self._esc = ""
                continue
            if ch == "\x1b":
                self._esc = ch
            elif ch == "\r":
                self.col = 0
            elif ch == "\n":                  # ONLCR: what a terminal normally does
                self.row += 1
                self.col = 0
            elif ch >= " ":
                self.rows[self.row] = self.rows[self.row].ljust(self.col) + ch
                self.col += 1
        return self

    def _apply(self, seq: str) -> None:
        m = re.fullmatch(r"\x1b\[(\d*)([A-Za-z])", seq)
        if not m:
            return
        n, op = m.group(1), m.group(2)
        if op == "A":
            self.row = max(0, self.row - int(n or 1))
        elif op == "B":
            self.row += int(n or 1)
        elif op == "J":                       # cursor down to the end
            self.rows[self.row] = self.rows[self.row][:self.col]
            for r in range(self.row + 1, len(self.rows)):
                self.rows[r] = ""
        elif op == "K":                       # cursor to end of line
            self.rows[self.row] = self.rows[self.row][:self.col]

    def getvalue(self) -> str:
        rows = list(self.rows)
        while rows and not rows[-1]:
            rows.pop()
        return "\n".join(rows)

    def lines(self) -> list[str]:
        text = self.getvalue()
        return text.split("\n") if text else []


class Out:
    """A write-only stream that remembers, and can render itself."""

    def __init__(self):
        self.parts: list[str] = []

    def write(self, text):
        self.parts.append(text)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self.parts)

    def screen(self) -> Screen:
        return Screen().feed(self.getvalue())


class FakeTerm:
    """A terminal that hands the editor the bursts we tell it to, when we say.

    `columns` is a lie it tells convincingly: the real width comes from the
    environment, which a test cannot set per-case.
    """

    def __init__(self, chunks=(), columns=40):
        self.chunks = [c.encode() if isinstance(c, str) else c for c in chunks]
        self.columns = columns
        self.ok = False
        self.closed = False
        self.mouse = False
        #: What `cursor_row()` answers. A test sets it to the row the editor
        #: believes its cursor is on, which is what a real console reports and
        #: what a mouse report is measured against.
        self.screen_row = None

    def open(self):
        self.ok = True
        return True

    def close(self):
        self.ok = False
        self.closed = True

    def mouse_on(self):
        self.mouse = True

    def mouse_off(self):
        self.mouse = False

    def cursor_row(self):
        return self.screen_row

    def read_chunk(self):
        if not self.chunks:
            return None
        return self.chunks.pop(0)

    def ready(self, timeout):
        return False


def visible(text: str) -> str:
    """Escape sequences made legible, for assertions about them not leaking."""
    return text.replace("\x1b", "<E>")
