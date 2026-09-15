"""The operator's two promises, held to a clock.

There are exactly two things a person at the terminal is owed while the agent
works, and both used to be broken in the same way -- by a blocking call that
owned the thread until it finished:

  1. Say something and get an answer, promptly, without the current work having
     to end first. "Promptly" is a number here, not a feeling: the run has to
     notice within a bounded time even when every individual step is minutes
     long.
  2. Never sit through a long silence. A step that runs for minutes has to
     report as it goes, on a bounded interval, whether or not it has news.

Both promises fail the same way when they fail -- the check lives at a
boundary (before a model call, after a tool result) and the boundaries are
minutes apart -- so both are tested here against the same fake: a step that
takes far longer than the operator's patience.

The clocks are deliberately loose. The point is to prove the mechanism can
react inside the budget, not to assert a stopwatch reading on a busy CI box.
"""
from __future__ import annotations

import threading
import time

import pytest

from autoforge.core.llm import LLMAborted, LLMResponse, OpenAICompatClient
from autoforge.core.message import Message
from autoforge.core.steering import Steering
from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
from autoforge.forge.sandbox import Sandbox
from autoforge.tools.registry import ToolRegistry


#: How long the operator is willing to wait for the run to notice them. The
#: real budget is 30s; the tests assert far tighter so a pass means the
#: mechanism reacted, not that the machine happened to be fast.
PATIENCE = 2.0


class _SlowEndpoint:
    """A `requests.post` that takes `seconds`, recording how it was called."""

    def __init__(self, seconds: float, body: dict | None = None) -> None:
        self.seconds = seconds
        self.body = body or {
            "choices": [{"message": {"content": "done", "role": "assistant"}}]
        }
        self.started = threading.Event()
        self.returned = threading.Event()
        self.calls = 0

    def __call__(self, url, headers=None, json=None, timeout=None, proxies=None):
        self.calls += 1
        self.started.set()
        time.sleep(self.seconds)
        self.returned.set()

        class _Resp:
            status_code = 200

            def json(self_inner):
                return self.body

            def raise_for_status(self_inner):
                return None

        return _Resp()


@pytest.fixture
def slow_post(monkeypatch):
    """Swap the module-level POST for one that can be told to be slow."""
    holder: dict[str, _SlowEndpoint] = {}

    def _install(seconds: float, body: dict | None = None) -> _SlowEndpoint:
        ep = _SlowEndpoint(seconds, body)
        monkeypatch.setattr("autoforge.core.llm.requests.post", ep)
        holder["ep"] = ep
        return ep

    return _install


# -- promise 1: an interjection is noticed while the call is in flight -------

def test_a_call_in_flight_releases_the_thread_when_the_operator_speaks(slow_post):
    """The headline case: a 20s model call must not hold the run for 20s.

    This is the 120s read timeout in the real client, shrunk to something a
    test can afford. The operator's line arrives 0.2s in; the call has to give
    up its thread almost immediately rather than ride the call out.
    """
    ep = slow_post(seconds=20.0)
    client = OpenAICompatClient("m", "http://x", "k", timeout=20.0)
    asked = threading.Event()

    def _operator_said_something() -> bool:
        # Stands in for "is there a line waiting in the steering queue": false
        # until the reader thread reports one, true forever after.
        if not asked.is_set() and ep.started.is_set():
            time.sleep(0.2)
            asked.set()
        return asked.is_set()

    started = time.time()
    with pytest.raises(LLMAborted):
        client.chat([Message.user("hi")], should_abort=_operator_said_something)
    elapsed = time.time() - started

    assert elapsed < PATIENCE, (
        f"the call held the thread for {elapsed:.1f}s; the operator would have "
        "been ignored for that long"
    )
    # The request itself is not cancelled -- it is abandoned. A completion has
    # nothing to undo, and blocking the exit on a socket would trade one hang
    # for another.
    assert ep.returned.wait(timeout=25.0), "the abandoned worker never finished"


def test_a_normal_call_still_returns_its_answer(slow_post):
    """The abort path must not become the path: an unwatched call is unchanged."""
    slow_post(seconds=0.1)
    client = OpenAICompatClient("m", "http://x", "k", timeout=5.0)
    resp = client.chat([Message.user("hi")])
    assert isinstance(resp, LLMResponse)
    assert resp.content == "done"


def test_a_call_that_finishes_first_wins_the_race(slow_post):
    """When the answer beats the interjection, the answer is returned.

    The operator typing at second three must not retroactively cancel a reply
    that landed at second one -- that would throw away real work and make the
    run re-ask a question it already had an answer to.
    """
    slow_post(seconds=0.05)
    client = OpenAICompatClient("m", "http://x", "k", timeout=5.0)
    already_spoke = threading.Event()
    already_spoke.set()
    resp = client.chat([Message.user("hi")], should_abort=already_spoke.is_set)
    assert resp.content == "done"


def test_the_wait_is_reported_while_it_lasts(slow_post):
    """`on_wait` is how the progress line stays honest during a long call."""
    slow_post(seconds=1.2)
    client = OpenAICompatClient("m", "http://x", "k", timeout=5.0)
    seen: list[float] = []
    client.chat([Message.user("hi")], on_wait=seen.append)
    assert seen, "a 1.2s call reported nothing while it was in flight"
    assert seen[-1] > seen[0], "the reported elapsed time never advanced"


def test_abandoning_is_not_retried(slow_post):
    """A retry here would re-ask the question the operator just changed."""
    ep = slow_post(seconds=20.0)
    client = OpenAICompatClient("m", "http://x", "k", timeout=20.0, max_attempts=5)
    with pytest.raises(LLMAborted):
        client.chat([Message.user("hi")], should_abort=lambda: True)
    assert ep.calls == 1, f"the abandoned call was retried {ep.calls} times"


def test_watcher_kwargs_never_reach_the_provider(slow_post):
    """They are addressed to the client. Sent on, they are unknown body fields."""
    ep = slow_post(seconds=0.05)
    sent: dict = {}
    original = ep.__call__

    def _capture(url, headers=None, json=None, timeout=None, proxies=None):
        sent.update(json or {})
        return original(url, headers, json, timeout, proxies)

    ep.__call__ = _capture
    client = OpenAICompatClient("m", "http://x", "k", timeout=5.0)
    client.chat([Message.user("hi")], should_abort=lambda: False,
                on_wait=lambda _s: None)
    assert "should_abort" not in sent and "on_wait" not in sent


# -- promise 1, second half: the steering channel can be asked --------------

def test_has_pending_is_true_exactly_while_a_line_waits():
    st = Steering(stream=None)
    assert st.has_pending() is False
    st.submit("keep going but use the other endpoint")
    assert st.has_pending() is True
    taken = st.take_supplements()
    assert len(taken) == 1
    assert st.has_pending() is False, (
        "has_pending is polled by a running step, so it must not consume -- "
        "but once the loop takes the line it must stop reporting one"
    )


def test_has_pending_ignores_commands() -> None:
    """`/status` is for the person, not the agent, and must not read as speech."""
    st = Steering(stream=None)
    st.submit("/status")
    assert st.has_pending() is False


# -- promise 2: a long step reports, on a bounded interval -------------------

def test_a_forge_reports_before_the_operator_has_to_wonder(slow_post):
    """A forge is the longest silence in a run; it must narrate itself.

    The pipeline emits a `forge_start`, then a per-round event once a round
    ends. A round is minutes long, so anything watching only those two goes
    quiet for minutes. The heartbeat that fills the gap is driven by `on_wait`
    (which the model call ticks) and by the round events -- this asserts the
    forge produces ticks *inside* a round, not only at its edges.
    """
    slow_post(seconds=1.5)
    client = OpenAICompatClient("m", "http://x", "k", timeout=20.0)

    ticks: list[float] = []

    class _Gen:
        def __init__(self) -> None:
            self.n = 0

        def generate(self, need, context=""):
            from autoforge.forge.generator import GeneratedTool
            self.n += 1
            client.chat([Message.user(need)], on_wait=ticks.append)
            return GeneratedTool(
                name=f"t{self.n}", description="d",
                code="def t1(x=0):\n    return x\n", parameters={},
            )

    pipeline = ForgePipeline(
        _Gen(), _Verifier(), ToolRegistry(),
        config=ForgeConfig(max_rounds=1, require_execution=False,
                           require_trigger=False, require_negative=False),
        sandbox=Sandbox(timeout=5.0),
    )
    started = time.time()
    pipeline.forge("add a thing")
    elapsed = time.time() - started

    assert ticks, (
        f"a {elapsed:.1f}s forge produced no progress ticks between its "
        "round events -- the operator would see one line and then silence"
    )
    assert max(ticks) > 0.0, "the ticks carried no elapsed time to report"


class _Verifier:
    """A verifier that passes whatever it is handed."""

    class _Report:
        passed = True

        def to_dict(self):
            return {"passed": True}

        @property
        def failed(self):
            return []

    sandbox = None

    def verify(self, spec, sample_args=None):
        return self._Report()
