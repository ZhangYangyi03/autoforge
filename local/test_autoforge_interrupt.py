"""Regression tests for autoforge's "stuck halfway, ignores my interrupt" defect.

Run against the *patched* package it should be green. Run against the pre-fix
package (PYTHONPATH pointing at the backup copy) every test here should fail —
that is what makes it evidence rather than decoration.

    python test_autoforge_interrupt.py            # patched  -> all pass
    PYTHONPATH=<old-package-root> python test_autoforge_interrupt.py   # old -> all fail

No network, no keys: `requests.post` is replaced with a controllable fake.
"""
from __future__ import annotations

import io
import sys
import threading
import time
import unittest

from autoforge.core import llm as llm_mod
from autoforge.core import lineedit as lineedit_mod
from autoforge.core import steering as steering_mod
from autoforge.core.llm import MockLLMClient
from autoforge.core.message import Message

if not hasattr(llm_mod, "LLMAborted"):
    # The pre-fix package has no such exception. Everything below asserts the
    # mechanism exists, so a single placeholder keeps the module importable and
    # lets each test fail on its own terms instead of at import time.
    class _Missing(Exception):
        pass

    llm_mod.LLMAborted = _Missing  # type: ignore[attr-defined]

LLMAborted = llm_mod.LLMAborted


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakeResponse:
    status_code = 200
    headers: dict = {}
    text = "{}"

    def json(self):
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    def raise_for_status(self):
        return None


class FakeSteer:
    """The operator's channel, in the shape `Agent` consumes."""

    def __init__(self) -> None:
        self.pending: list[str] = []
        self.stopped = False
        self.delivered = 0

    def take_supplements(self) -> list[str]:
        out = list(self.pending)
        self.pending.clear()
        self.delivered += len(out)
        return out

    def stop_requested(self) -> bool:
        return self.stopped

    def has_pending(self) -> bool:
        return bool(self.pending)


class FakeTerm:
    """A terminal that is only a width."""

    def __init__(self, columns: int = 80) -> None:
        self.columns = columns

    def open(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def read_chunk(self):  # pragma: no cover - never called here
        return None

    def ready(self, timeout: float) -> bool:  # pragma: no cover
        return False


class TinyRegistry:
    def __init__(self) -> None:
        self._tools: dict = {}

    def register(self, spec) -> None:
        self._tools[spec.name] = spec

    def schemas(self) -> list:
        return []

    def call(self, name, arguments):
        raise AssertionError("no tool should be called in these tests")

    def names(self) -> list:
        return list(self._tools)


def _patch_requests(handler):
    """Swap `requests.post` for `handler`, returning a restore callable."""
    original = llm_mod.requests.post
    llm_mod.requests.post = handler
    return lambda: setattr(llm_mod.requests, "post", original)


# ---------------------------------------------------------------------------
# A. the model call itself must be interruptible *in flight*
# ---------------------------------------------------------------------------
class TestMidFlightAbort(unittest.TestCase):
    def setUp(self) -> None:
        self._restore_sleep = llm_mod._sleep
        llm_mod._sleep = lambda seconds: None
        self._restores: list = []

    def tearDown(self) -> None:
        llm_mod._sleep = self._restore_sleep
        for r in self._restores:
            r()

    def test_client_attached_predicate_aborts_a_call_in_flight(self):
        """The predicate is attached to the *client*, nothing passed to chat().

        This is precisely the gap that was fixed: calls made by stages that own
        their own client had no predicate at all, so a round could spend minutes
        deaf.
        """
        started = threading.Event()

        def slow_post(url, **kwargs):
            started.set()
            time.sleep(3.0)          # far longer than the patience below
            return FakeResponse()

        self._restores.append(_patch_requests(slow_post))
        client = llm_mod.OpenAICompatClient("m", "http://x", "k", timeout=20.0)
        spoke = {"now": False}
        client.abort_check = lambda: spoke["now"]

        def _speak_soon():
            time.sleep(0.4)
            spoke["now"] = True

        threading.Thread(target=_speak_soon, daemon=True).start()

        t0 = time.monotonic()
        with self.assertRaises(LLMAborted):
            client.chat([Message.user("hi")])          # no should_abort= at all
        elapsed = time.monotonic() - t0
        self.assertTrue(started.is_set(), "the request never went out")
        # The guarantee, not a stopwatch reading: a 3s stall, a 2s deadline.
        self.assertLess(elapsed, 2.0,
                        f"abort took {elapsed:.2f}s — the call was not polled")

    def test_per_call_predicate_aborts_a_call_in_flight(self):
        def slow_post(url, **kwargs):
            time.sleep(3.0)
            return FakeResponse()

        self._restores.append(_patch_requests(slow_post))
        client = llm_mod.OpenAICompatClient("m", "http://x", "k", timeout=20.0)
        spoke = {"now": False}
        threading.Thread(
            target=lambda: (time.sleep(0.4), spoke.__setitem__("now", True)),
            daemon=True).start()

        t0 = time.monotonic()
        with self.assertRaises(LLMAborted):
            client.chat([Message.user("hi")], should_abort=lambda: spoke["now"])
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_watcher_kwargs_never_reach_the_provider(self):
        """`should_abort`/`on_wait` are addressed to the client, not the body."""
        seen: dict = {}

        def capturing_post(url, **kwargs):
            seen.update(kwargs.get("json") or {})
            return FakeResponse()

        self._restores.append(_patch_requests(capturing_post))
        client = llm_mod.OpenAICompatClient("m", "http://x", "k", timeout=20.0)
        client.chat([Message.user("hi")],
                    should_abort=lambda: False, on_wait=lambda s: None)
        self.assertNotIn("should_abort", seen)
        self.assertNotIn("on_wait", seen)

    def test_a_call_that_finishes_first_is_not_cancelled(self):
        """An unwatched call is unchanged, and a fast one is not retro-cancelled."""
        def fast_post(url, **kwargs):
            return FakeResponse()

        self._restores.append(_patch_requests(fast_post))
        client = llm_mod.OpenAICompatClient("m", "http://x", "k", timeout=20.0)
        client.abort_check = lambda: False
        resp = client.chat([Message.user("hi")])
        self.assertEqual(resp.content, "ok")


# ---------------------------------------------------------------------------
# C. an aborted call must not read as a failed one
# ---------------------------------------------------------------------------
class AbortThenAnswer(MockLLMClient):
    """Aborts the first call, planting the operator's line on the way out."""

    def __init__(self, steer: FakeSteer, line: str) -> None:
        super().__init__()
        self.steer = steer
        self.line = line
        self.n = 0

    def chat(self, messages, tools=None, **kw):
        self.n += 1
        if self.n == 1:
            self.steer.pending.append(self.line)
            raise LLMAborted("the operator interrupted while the call was in flight")
        return llm_mod.LLMResponse(content="answered: " + self.line)


class TestAbortIsNotAFailure(unittest.TestCase):
    def test_the_typed_line_is_delivered_and_no_tool_failure_is_recorded(self):
        steer = FakeSteer()
        llm = AbortThenAnswer(steer, "no, the other host")
        from autoforge.core.agent import Agent

        agent = Agent(llm, TinyRegistry(), steer=steer, allow_self_terminate=False)
        result = agent.run("do the thing")

        self.assertIn("no, the other host", result.content)
        self.assertFalse([m for m in result.messages if m.role == "tool"],
                         "an abort was fed back as a tool result")
        self.assertFalse(result.stopped_by_operator)

    def test_a_nested_loop_re_raises_instead_of_spinning(self):
        """Its own channel is empty while the inherited predicate is still true."""
        from autoforge.core.agent import Agent

        llm = MockLLMClient()
        llm.abort_check = lambda: True          # inherited from the parent
        agent = Agent(llm, TinyRegistry(), steer=None, allow_self_terminate=False)
        with self.assertRaises(LLMAborted):
            agent.run("nested work")


# ---------------------------------------------------------------------------
# D. the mode interface must match its caller
# ---------------------------------------------------------------------------
class TestModeInterface(unittest.TestCase):
    def test_minimal_mode_accepts_progress_kwarg(self):
        from autoforge.modes import MinimalAgent

        llm = MockLLMClient(script=[llm_mod.LLMResponse(content="done")])
        agent = MinimalAgent(llm=llm, cwd=".")
        result = agent.run("hi", progress=lambda kind, payload: None)
        self.assertEqual(result.content, "done")


# ---------------------------------------------------------------------------
# E. a forge round is interruptible, and an abort is not a failed round
# ---------------------------------------------------------------------------
class RecordingGenerator:
    def __init__(self, llm, seen: dict) -> None:
        self.llm = llm
        self.seen = seen

    def generate(self, need: str, context: str = ""):
        self.seen["attached_during_round"] = self.llm.abort_check
        raise LLMAborted("the operator interrupted mid-round")


class StubVerifier:
    def __init__(self, llm, adversary=None) -> None:
        self.llm = llm
        self.adversary = adversary
        self.sandbox = None

    def verify(self, spec, sample_args):  # pragma: no cover - never reached
        raise AssertionError("verification must not run after an abort")


class StubGate:
    """The adversary names its client `attacker`, not `llm`."""

    def __init__(self, attacker) -> None:
        self.attacker = attacker


class TestForgeRoundInterruptible(unittest.TestCase):
    def test_predicate_is_attached_for_the_round_then_restored(self):
        from autoforge.forge.pipeline import ForgePipeline

        seen: dict = {}
        llm = MockLLMClient()
        llm.abort_check = None
        gen = RecordingGenerator(llm, seen)
        ver = StubVerifier(llm)
        events: list = []
        pipe = ForgePipeline(gen, ver, TinyRegistry(),
                             on_event=lambda kind, payload: events.append(kind))

        pred = lambda: True
        with self.assertRaises(LLMAborted):
            pipe.forge("some need", should_abort=pred)

        self.assertIs(seen.get("attached_during_round"), pred,
                      "the predicate never reached the round's client")
        self.assertIsNone(llm.abort_check, "the borrowed predicate was not restored")
        self.assertNotIn("forge_error", events,
                         "an abort was recorded as a failed round")
        self.assertNotIn("forge_done", events)

    def test_the_pipeline_finds_clients_by_asking_its_parts(self):
        from autoforge.forge.pipeline import ForgePipeline

        llm = MockLLMClient()
        pipe = ForgePipeline(RecordingGenerator(llm, {}), StubVerifier(llm),
                             TinyRegistry())
        self.assertEqual(pipe._model_clients(), [llm])

    def test_the_adversary_client_is_collected_even_though_it_is_named_attacker(self):
        from autoforge.forge.pipeline import ForgePipeline

        gen_llm, ver_llm, attacker_llm = (MockLLMClient(), MockLLMClient(),
                                         MockLLMClient())
        ver = StubVerifier(ver_llm, adversary=StubGate(attacker_llm))
        pipe = ForgePipeline(RecordingGenerator(gen_llm, {}), ver, TinyRegistry())
        clients = pipe._model_clients()
        self.assertIn(gen_llm, clients)
        self.assertIn(ver_llm, clients)
        self.assertIn(attacker_llm, clients)
        self.assertEqual(len(clients), 3)

        pred = lambda: True
        with self.assertRaises(LLMAborted):
            pipe.forge("need", should_abort=pred)
        for c in clients:
            self.assertIsNone(c.abort_check, "a borrowed predicate was not restored")


# ---------------------------------------------------------------------------
# F. the input line wraps by display columns, not character count
# ---------------------------------------------------------------------------
class TestLineEditorWidth(unittest.TestCase):
    def _editor(self, columns: int, buf: str, cursor: int | None = None):
        ed = lineedit_mod.LineEditor(
            stream=io.StringIO(), out=io.StringIO(),
            term=FakeTerm(columns),
        )
        ed._open = True
        ed._prompt = "you> "
        ed._buf = list(buf)
        ed._cursor = len(ed._buf) if cursor is None else cursor
        return ed

    def test_cjk_is_two_columns(self):
        self.assertEqual(lineedit_mod._visible_len("中文"), 4)
        self.assertEqual(lineedit_mod._visible_len("ab"), 2)
        self.assertEqual(lineedit_mod._visible_len("\x1b[31m红\x1b[0m"), 2)

    def test_eight_cjk_chars_wrap_onto_two_rows_in_a_twenty_column_terminal(self):
        """8 chars, 16 cells, prompt 5 cells -> 21 columns: one row cannot hold it."""
        ed = self._editor(20, "中文中文中文中文")
        rows, row, col = ed._layout()
        self.assertEqual(len(rows), 2,
                         f"expected 2 rows, got {len(rows)}: {rows!r}")
        self.assertEqual(row, 1)
        # "you> " (5) + 7 CJK chars (14) = 19 cells on row 0; the 8th starts row 1
        # at the 5-cell indent, so the cursor sits at column 7.
        self.assertEqual(col, 7)
        for r in rows:
            self.assertLessEqual(lineedit_mod._visible_len(r), 20,
                                 f"row {r!r} is wider than the terminal")

    def test_ascii_layout_is_unchanged(self):
        ed = self._editor(20, "abcdefghijklmnopqrstuvwxyz")
        rows, row, col = ed._layout()
        # width = 20 - 5 - 1 = 14
        self.assertEqual([len(r) - 5 for r in rows], [14, 12])
        self.assertEqual(col, 5 + 12)

    def test_a_cursor_after_cjk_lands_six_columns_in(self):
        ed = self._editor(80, "中文中文", cursor=3)
        rows, row, col = ed._layout()
        self.assertEqual(len(rows), 1)
        self.assertEqual(col, 5 + 6)


# ---------------------------------------------------------------------------
# G. the ack must be conditional on a run actually being in flight
# ---------------------------------------------------------------------------
class TestSteeringChannel(unittest.TestCase):
    def _steering(self, status):
        said: list = []
        s = steering_mod.Steering(
            stream=io.StringIO(), printer=said.append, status=status)
        return s, said

    def test_has_pending_tracks_the_queue(self):
        s, _ = self._steering(None)
        self.assertFalse(s.has_pending())
        s.submit("hello")
        self.assertTrue(s.has_pending())
        s.take_supplements()
        self.assertFalse(s.has_pending())

    def test_ack_does_not_promise_a_step_that_is_not_running(self):
        s, said = self._steering(None)
        s.submit("hello")
        self.assertIn("nothing is running", said[-1])

        s2, said2 = self._steering(lambda: "turn 1")
        s2.submit("hello")
        self.assertIn("next step", said2[-1])


class TestOperatorLineReachesACallInFlight(unittest.TestCase):
    """The operator's complaint, encoded: "it looks stuck and ignores me".

    A slow call is simulated by a handler that waits for the line to land —
    which is exactly the state a real 600s read timeout used to leave the
    session in, with nothing polling the input buffer.
    """

    def _agent(self, steer, handler):
        from autoforge.core.agent import Agent

        return Agent(MockLLMClient(handler=handler), TinyRegistry(),
                     steer=steer, allow_self_terminate=False)

    def test_a_line_typed_mid_call_is_delivered_without_waiting_for_the_call(self):
        steer = steering_mod.Steering(stream=io.StringIO(), printer=lambda t: None)
        seen = {"n": 0}

        def handler(messages, tools, **kw):
            seen["n"] += 1
            if seen["n"] == 1:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and not steer.has_pending():
                    time.sleep(0.02)
                raise LLMAborted("interrupted while the call was in flight")
            return llm_mod.LLMResponse(content="acted on: " + messages[-1].content)

        threading.Timer(0.25, lambda: steer.submit("no, the other host")).start()
        t0 = time.monotonic()
        result = self._agent(steer, handler).run("do the thing")
        elapsed = time.monotonic() - t0

        self.assertLess(elapsed, 3.0,
                        f"the run waited {elapsed:.2f}s for the call it should "
                        f"have abandoned")
        self.assertIn("no, the other host", result.content)
        self.assertFalse([m for m in result.messages if m.role == "tool"])

    def test_stop_lands_mid_call_and_ends_the_run(self):
        steer = steering_mod.Steering(stream=io.StringIO(), printer=lambda t: None)

        def handler(messages, tools, **kw):
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not steer.stop_requested():
                time.sleep(0.02)
            raise LLMAborted("interrupted while the call was in flight")

        threading.Timer(0.25, lambda: steer.submit("/stop")).start()
        t0 = time.monotonic()
        result = self._agent(steer, handler).run("do the thing")
        elapsed = time.monotonic() - t0

        self.assertLess(elapsed, 3.0)
        self.assertTrue(result.stopped_by_operator)
        self.assertIn("stopped by the operator", result.termination_reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
