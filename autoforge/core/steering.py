"""Mid-run steering: keep talking to the agent while it is working.

A run is not a black box you have to wait out. This is the channel for the
person at the terminal:

  * a plain sentence — delivered to the model mid-run as a user message, at the
    next safe point: before a model call, or right after a tool result.
  * ``/status`` — where it is right now, without entering the conversation.
    Asking for progress must not become part of the task.
  * ``/stop`` — stop after the current step and hand the turn back.

Commands are recognised by the leading ``/`` and are **never** sent to the
model: a line meant for the person running the agent must not be readable as an
instruction by the agent. An unknown slash-word is reported rather than
silently dropped, and a line that starts with a space is sent as text — so
there is a way to say "/usr/bin/env is missing" on purpose.

Reading stdin is a thread's job, not the loop's: the reader blocks on the
stream while the agent works, which is exactly what the loop cannot afford to
do. One reader owns the stream for the whole session — during a run its lines
become steering, between runs they are the next prompt — so a line typed during
a run is never lost, and the agent never competes with ``input()`` for it.
"""
from __future__ import annotations

import queue
import sys
import threading
from typing import Any, Callable, Iterable

from .lineedit import LineEditor, expand_paste_refs

__all__ = ["Steering", "OPERATOR_PREFIX"]


#: Marks a message as coming from the person watching, mid-run. Without it the
#: model reads a mid-task correction as a new task and starts over.
OPERATOR_PREFIX = "[the person running you said this mid-run]"

#: ...and how to act on it. Terse, because the loop is the wrong place for an
#: essay, but specific enough that the model does not restart the task.
OPERATOR_SUFFIX = ("Treat this as a correction to the task in progress, not as a "
                   "new task: act on it within this run.")


def operator_message(text: str) -> str:
    """Wrap one line of mid-run speech for the model."""
    return f"{OPERATOR_PREFIX} {text}\n\n{OPERATOR_SUFFIX}"


#: `/status` and friends — ask where it is. None of these reach the model.
STATUS_WORDS = ("/status", "/progress", "/where", "?")

#: `/stop` — end the run at the next safe point.
STOP_WORDS = ("/stop", "/halt", "/abort")


class Steering:
    """The operator's channel into a running agent.

    Parameters
    ----------
    stream:
        Where lines come from. Defaults to stdin; a non-tty stream (a pipe, a
        cron job, a test) means nobody is watching, so no reader thread is
        started and no mid-run input is possible.
    printer:
        ``printer(text)`` prints one line atomically. The CLI hands it the live
        progress renderer, so steering replies and progress lines share a lock
        and never interleave mid-character.
    status:
        ``status() -> str`` — the snapshot ``/status`` prints. Set per run.
    editor:
        The line editor that owns the bottom line of the terminal. Built here
        if not supplied, and only *opened* in :meth:`start` — a ``StringIO``
        (a test, a pipe) cannot be opened, so such a session keeps the plain
        cooked-mode reader it had before.
    """

    def __init__(
        self,
        stream: Any = None,
        *,
        printer: Callable[[str], None] | None = None,
        status: Callable[[], str] | None = None,
        editor: LineEditor | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self.printer = printer or (lambda text: print(text, flush=True))
        self.status = status
        self.editor = editor if editor is not None else LineEditor(stream=self.stream)
        self._pending: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._eof = threading.Event()
        self._thread: threading.Thread | None = None
        self.delivered = 0
        self.refused: list[str] = []

    # -- lifecycle ------------------------------------------------------
    @property
    def interactive(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    @property
    def editing(self) -> bool:
        """True once the terminal is under the editor's control."""
        return bool(self.editor.available)

    def start(self) -> "Steering":
        """Start reading the stream, if there is a person on the other end."""
        if self._thread is not None:
            return self
        if self.interactive and self.editor.start():
            self._thread = threading.Thread(target=self._read_keys, daemon=True)
        elif self.interactive:
            self._thread = threading.Thread(target=self._read, daemon=True)
        else:
            return self
        self._thread.start()
        return self

    def watch(self, live: Any) -> "Steering":
        """Point the replies at the run currently in progress.

        Called once per task: the printer and the snapshot belong to that run,
        and a stale one would report where the agent *was*.
        """
        self.printer = getattr(live, "say", self.printer)
        self.status = getattr(live, "snapshot", self.status)
        return self

    def close(self) -> None:
        self._closed.set()
        # Give the terminal back before the process exits: leaving it in
        # cbreak would hand the shell a prompt that does not echo.
        self.editor.close()

    def emit(self, text: str = "") -> None:
        """Print a line of the session's own output.

        With the editor running, every line the session prints has to go
        *above* the input area, or it lands on top of what is being typed.
        Without it this is ``print``, unchanged.
        """
        if self.editing:
            self.editor.write(text)
        else:
            print(text, flush=True)

    def _read(self) -> None:
        try:
            for line in self.stream:
                if self._closed.is_set():
                    break
                self.submit(line)
        except (ValueError, OSError):     # stream closed under us
            pass
        finally:
            self._eof.set()

    def _read_keys(self) -> None:
        """:meth:`_read`, for a terminal the editor has taken over."""
        try:
            while not self._closed.is_set():
                line = self.editor.readline()
                if line is None:          # ^D
                    break
                self.submit(line)
        except Exception:                 # noqa: BLE001 - a dead reader must not kill the session
            pass
        finally:
            self._eof.set()

    # -- what the operator typed ----------------------------------------
    def submit(self, line: str) -> None:
        """Route one typed line: a command, or something to tell the agent."""
        text = line.rstrip("\r\n")
        if not text.strip():
            return
        # A leading space is an escape hatch: " /usr/bin/env is missing" is
        # text, not a command. Without it there would be no way to say it.
        if text.startswith(" ") or not text.startswith("/") and text != "?":
            self._queue(text)
            return

        word, _, rest = text.partition(" ")
        word = word.lower()
        if word in STATUS_WORDS:
            self._say(self._snapshot())
            return
        if word in STOP_WORDS:
            self._stop.set()
            self._say("stopping after the current step — sending a /stop note")
            return
        if word == "/help":
            self._say("mid-run: plain text reaches the agent, "
                      "/status asks where it is, /stop ends the run.")
            return
        self.refused.append(text)
        self._say(f"unknown mid-run command {word!r} — not sent to the agent. "
                  f"Start the line with a space to send it as text.")

    def _queue(self, text: str) -> None:
        self._pending.put(text)
        self._say(f"will reach the agent at the next step: {text[:60]}")

    def _say(self, text: str) -> None:
        try:
            self.printer(text)
        except Exception:                 # noqa: BLE001 - a reporter, not a gate
            pass

    def _snapshot(self) -> str:
        if self.status is None:
            return "no run in progress"
        try:
            return self.status()
        except Exception as exc:          # noqa: BLE001
            return f"(no snapshot: {type(exc).__name__}: {exc})"

    # -- what the loop consumes ----------------------------------------
    def take_supplements(self) -> list[str]:
        """Everything the operator said since the last check, ready for the model."""
        out: list[str] = []
        while True:
            try:
                # A collapsed paste leaves a placeholder on the input line; the
                # model gets the text. That is the point of collapsing it — the
                # line stays short without the message getting shorter.
                out.append(operator_message(expand_paste_refs(self._pending.get_nowait())))
            except queue.Empty:
                break
        self.delivered += len(out)
        return out

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    # -- the next prompt ------------------------------------------------
    def take_line(self, prompt: str) -> str:
        """One line for the REPL prompt. `""` means end of input.

        Lines typed during a run that the loop never reached are waiting here,
        so an instruction that arrived too late is the next thing submitted
        rather than a line that vanished.

        The prompt is printed without a newline and the tty echoes what is
        typed, so this reads like `input()` to the person at the terminal —
        while the line itself still arrives through the queue, which is what
        lets a reader thread own the stream for the whole session.

        With the editor running there is nothing to echo and nothing to print:
        the editor has already drawn the prompt, and it redraws the line under
        whatever the run writes while you are in the middle of it.
        """
        try:
            return expand_paste_refs(self._pending.get_nowait())
        except queue.Empty:
            pass
        if self.editing:
            self.editor.set_prompt(prompt)
        elif not self.interactive:
            return expand_paste_refs(self.stream.readline())
        else:
            print(prompt, end="", flush=True)
        # A queue has no EOFError, so end-of-input has to be noticed by hand.
        # Without this the REPL would block forever after a closed stdin
        # instead of exiting, which is how a control-D turns into a hung shell.
        while True:
            try:
                return expand_paste_refs(self._pending.get(timeout=0.2))
            except queue.Empty:
                if self._eof.is_set():
                    return ""

    def feed(self, lines: Iterable[str]) -> None:
        """Route several lines, as if typed. Used by tests and by playback."""
        for line in lines:
            self.submit(line)
