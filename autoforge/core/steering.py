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
import time
from typing import Any, Callable, Iterable

from .lineedit import LineEditor, expand_paste_refs

__all__ = ["Steering", "OPERATOR_PREFIX"]


#: Marks a message as coming from the person watching, mid-run. Without it the
#: model reads a mid-task correction as a new task and starts over.
OPERATOR_PREFIX = "[the person running you said this mid-run]"

#: ...and how to act on it. Terse, because the loop is the wrong place for an
#: essay, but specific enough that the model does not restart the task -- and
#: specific that answering comes *first*.
#:
#: The answer is the half that was missing. Measured on this host: 46 steered
#: lines, and the one that asked "what are you doing?" was absorbed and then
#: followed by sixty more tool calls without a word. The wrapper only said to
#: act on the line within the run, and a question is not something you act on --
#: so the instruction left replying as optional, and the run took the option.
#: The harness saying "heard" is not the agent answering, and from the chair at
#: the terminal those two are only distinguishable by whether words come back.
OPERATOR_SUFFIX = ("They are watching this run and waiting to hear from you: "
                   "answer them in one short line as the very next thing you do, "
                   "then treat this as a correction to the task in progress, not "
                   "as a new task -- act on it within this run.")


def operator_message(text: str) -> str:
    """Wrap one line of mid-run speech for the model."""
    return f"{OPERATOR_PREFIX} {text}\n\n{OPERATOR_SUFFIX}"


#: `/status` and friends — ask where it is. None of these reach the model.
STATUS_WORDS = ("/status", "/progress", "/where", "?")

#: `/stop` — end the run at the next safe point.
STOP_WORDS = ("/stop", "/halt", "/abort")

#: Lines closer together than this are one burst, not several messages. A
#: burst is what a paste looks like when the terminal is too slow to hand the
#: block over in one read: the fragments arrive as separate submits, and each
#: one on its own is indistinguishable from a deliberate sentence. Nobody
#: types two intended messages inside a second, so the window only ever fires
#: on a burst — and the editor already folds the common case before it gets
#: here, which is why this is the safety net rather than the mechanism.
BURST_WINDOW = 1.0


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
        #: Set by `with_editor`. The point of taking one from outside is that the
        #: editor is also the *producer* of a paste: the same object that reads
        #: the line is the one that knows a picture was pasted into it, and only
        #: it can hand those bytes on. A channel that built its own editor would
        #: be routing around the very object doing the reading.
        self._pending_editor: LineEditor | None = None
        self.status = status
        self.editor = editor if editor is not None else LineEditor(stream=self.stream)
        self._pending: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        # Whether a run is live right now. Held by the loop that owns the
        # steps -- ``Agent.run`` -- and not by whoever happens to be printing:
        # the reply to the operator promises that the *step in progress* will
        # yield, and only the loop can know whether there is one. Read by
        # ``_queue``; see ``running``.
        self._run_lock = threading.Lock()
        self._run_depth = 0
        self._closed = threading.Event()
        self._eof = threading.Event()
        self._thread: threading.Thread | None = None
        self.delivered = 0
        self.refused: list[str] = []
        # The burst guard's memory: the last line taken in, and when. Read and
        # written under `_burst_lock`; `_pending`'s own mutex is taken inside
        # that to rewrite the queued line, so the loop can never be served a
        # half-merged message.
        self._burst_lock = threading.Lock()
        self._tail: str | None = None
        self._last_at = 0.0

    # -- lifecycle ------------------------------------------------------
    @property
    def interactive(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    @property
    def editing(self) -> bool:
        """True once the terminal is under the editor's control."""
        return bool(self.editor.available)

    @property
    def running(self) -> bool:
        """True while a run is live, i.e. while there is a step to yield."""
        with self._run_lock:
            return self._run_depth > 0

    def begin_run(self) -> None:
        """Mark a run as live. Called by the loop, cancelled by `end_run`.

        Counted rather than a flag so a nested run (a mode that drives an
        agent, which drives another) cannot end the outer one's life by
        finishing first.
        """
        with self._run_lock:
            self._run_depth += 1

    def end_run(self) -> None:
        """Mark a run as over. Idempotent, so an exit path may call it twice."""
        with self._run_lock:
            if self._run_depth:
                self._run_depth -= 1

    def with_editor(self, editor: LineEditor | None) -> "Steering":
        """Adopt `editor` (if any) as the terminal owner, before :meth:`start`."""
        self._pending_editor = editor
        return self

    def start(self) -> "Steering":
        """Start reading the stream, if there is a person on the other end."""
        if self._thread is not None:
            return self
        if self.interactive and self._pending_editor is not None:
            self.editor = self._pending_editor
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
        # Printed before the line is routed, and before the early return below:
        # a paste that was then submitted as an empty line -- which is what
        # pressing Enter on an attached screenshot alone does -- still gets its
        # receipt. Silence there reads as a paste that failed.
        self._receipt()
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
        if self._fold_into_pending(text):
            # Part of a burst whose receipt has already been given. Saying
            # "heard" again would turn one paste into a column of receipts,
            # which is the noise this is here to remove.
            return
        self._pending.put(text)
        with self._burst_lock:
            self._tail, self._last_at = text, time.monotonic()
        # "At the next step" was the truth when the step boundary was the only
        # way in, and it read as a brush-off: the next step was minutes away, so
        # the honest-sounding line was the one the operator heard as being
        # ignored. Now the loop also asks this queue from inside its long steps
        # (`Agent._operator_wants_the_floor`) and a running tool does too, so
        # the step in progress is cut short instead of being waited out. The
        # reply says that, because it is the difference the operator is
        # watching for.
        #
        # But that is only true when there *is* a step. A line typed between
        # runs -- or into a second process, or after the loop already returned
        # -- used to get the same sentence, promising a yield that nothing
        # could perform. Whether it is true is decidable, because the loop says
        # so when it starts and when it stops, so ask rather than assume.
        if self.running:
            # Deliberately not "the step in progress will yield to it": that was
            # written when a pending line aborted whatever was running, and the
            # price of making it true was that a question killed the download.
            # A long job is not interrupted now -- only /stop is -- so the
            # honest sentence is that the line is heard, that it will be
            # answered, and that the work in flight is safe.
            self._say(f"✓ heard — you'll get an answer; the job in flight keeps "
                      f"running: {text[:60]}")
        else:
            self._say("✓ heard — nothing is running right now, so this goes in "
                      f"with your next request: {text[:60]}")

    def _fold_into_pending(self, text: str) -> bool:
        """Fold a fragment of a burst into the line already waiting.

        True when `text` was folded, and then it is *not* queued again: the
        queued line now carries it.

        Only while a run is live, and only into a line the loop has not taken
        yet. Both conditions are about the same thing — the operator's burst
        is one thing to say, and rewriting history after the agent has read
        half of it is not this class's job. Once the loop has taken the line,
        this returns False and the fragment becomes its own supplement, which
        is the honest outcome: it did arrive too late to be one message.
        """
        with self._burst_lock:
            now = time.monotonic()
            tail = self._tail
            fresh = tail is not None and now - self._last_at <= BURST_WINDOW
            # Measured from this line either way: a fragment that turns out
            # not to fold is still what the next fragment is a burst with.
            self._last_at = now
        if not fresh or not self.running:
            return False
        # The queue's own mutex, not ours: taking it is what makes the rewrite
        # atomic against `take_supplements`. Without it the loop could be
        # handed the line in the instant between reading and replacing it.
        with self._pending.mutex:
            queue = self._pending.queue
            # The line it was a burst with has to still be *there*. If the
            # loop has taken it, this fragment arrived too late to be part of
            # the same message, and saying so by sending it separately is
            # better than rewriting what the agent has already read.
            if not queue or queue[-1] != tail:
                return False
            merged = tail + "\n" + text
            queue[-1] = merged
        with self._burst_lock:
            self._tail = merged
        return True

    def _receipt(self) -> None:
        """Print the receipt for the last paste, if there is one."""
        try:
            notice = self.take_notice()
        except Exception:                 # noqa: BLE001 - a receipt is not a gate
            return
        if notice:
            self._say(notice)

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
    def take_notice(self) -> str:
        """The receipt for the last paste, or "" -- printed back at the terminal.

        The person pasting has no other way to know it took: the line editor
        writes a placeholder into the input area, and a screenshot pasted while
        the agent is running scrolls away with the progress line. The queued
        "heard" receipt is not it either, because that one answers the *line*.
        """
        take = getattr(self.editor, "take_notice", None)
        return take() if callable(take) else ""

    def take_images(self) -> list[str]:
        """Image placeholders attached to everything pending, oldest first.

        Read separately from the lines because the two are consumed by
        different things: the text goes into the message list, the pictures go
        into the next request's parts. An editor that is not running answers
        with nothing, which is the honest answer for a pipe.
        """
        take = getattr(self.editor, "take_images", None)
        return list(take()) if callable(take) else []

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

    def has_pending(self) -> bool:
        """Whether the operator has said anything the loop has not consumed.

        This is the question a *long-running* step has to ask about itself. The
        loop already drains the queue at its boundaries, but a forge is minutes
        long and the boundaries can be minutes apart, so a step that can be cut
        short needs to be able to ask mid-step: "did they just say something?"
        A yes there means the answer being computed is answering a question
        that has already changed.

        Cheap and side-effect free by design -- it is polled, so it must not
        consume, print, or move a counter.
        """
        return not self._pending.empty()

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
