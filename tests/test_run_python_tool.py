"""Running a snippet: the tool whose absence sent the agent through a browser.

On 2026-09-15 one run made 126 `browser_eval` calls because there was no way to
execute Python: `forge_tool` was dead (the upstream code model answered 503
sixty-three times), and the remaining route was to drive a browser at a local
service exposing a run-a-script endpoint. A library that can build code but not
run it will find a worse way to run code, so these tests are about the plain way
being present, honest, and interruptible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.agent import BUILTIN_SCOPES, ForgeAgent, _render_run  # noqa: E402
from autoforge.autonomy.policy import AutonomyPolicy                 # noqa: E402
from autoforge.core.llm import MockLLMClient                         # noqa: E402
from autoforge.forge.sandbox import SandboxResult                    # noqa: E402


def make_agent(policy=None):
    kw = {"policy": policy} if policy is not None else {}
    return ForgeAgent(MockLLMClient(), enable_evolution=False, **kw)


def call(agent, **kwargs):
    return agent.registry.call("run_python", kwargs)


# -- the tool is there, and declared ----------------------------------------
def test_the_agent_can_actually_run_a_snippet():
    a = make_agent()
    assert "run_python" in set(a.registry.names())


def test_it_declares_the_widest_scope_it_actually_needs():
    """Executing code spawns a process that can reach anything this one can."""
    assert BUILTIN_SCOPES.get("run_python") == "system"


# -- and it runs code -------------------------------------------------------
def test_a_snippet_runs_and_returns_what_it_printed():
    r = call(make_agent(), code="print('hello from the sandbox')")

    assert r.ok, r.error
    assert "hello from the sandbox" in r.output


def test_the_last_expression_is_an_answer_too():
    """A snippet with no print() in it still computed something.

    An agent asking "how many lines is this" writes an expression, not a
    `print` around it. A tool that answers nothing there is a tool the agent
    stops trusting and starts working around.
    """
    r = call(make_agent(), code="1 + 1")

    assert r.ok, r.error
    assert "2" in r.output


def test_a_last_expression_is_returned_even_after_other_statements():
    r = call(make_agent(), code="rows = [1, 2, 3]\nlen(rows) * 10")

    assert r.ok, r.error
    assert "30" in r.output


def test_it_can_import_the_standard_library():
    """The stance in DESIGN.sandbox: bound the blast radius, not the capability."""
    r = call(make_agent(), code="import json\nprint(json.dumps({'a': 1}))")

    assert r.ok, r.error
    assert '{"a": 1}' in r.output


# The convention these four follow: the call succeeded, the *code* did not.
# A tool reports that in the text it returns -- the same way `cpu_compile`
# returns "Could not compile: ..." -- and records the verdict in the ledger
# itself. Failing the call instead would teach the registry that the tool is
# unreliable, which is not what happened.
def test_a_snippet_that_raises_reports_the_error_rather_than_crashing_the_agent():
    r = call(make_agent(), code="raise ValueError('nope')")

    assert "[error]" in r.output
    assert "ValueError" in r.output and "nope" in r.output


def test_a_syntax_error_is_reported_as_a_result_not_raised():
    # "this is not python" is a valid Python comparison, so it is not the test:
    # an unfinished statement is.
    r = call(make_agent(), code="def broken(:\n    pass")

    assert "SyntaxError" in r.output


def test_an_empty_snippet_is_refused_in_words():
    r = call(make_agent(), code="   \n  ")

    assert "empty" in r.output


def test_a_snippet_that_overruns_its_timeout_is_killed_and_says_so():
    r = call(make_agent(), code="import time\ntime.sleep(30)", timeout=1)

    assert "timed out" in r.output


def test_the_timeout_cap_is_honoured_per_call_not_written_on_the_shared_box():
    """The sandbox is shared with the forge and the verifier.

    A per-call timeout written onto it would outlive the call that asked for it
    and rewrite somebody else's budget, so the tool derives a copy instead.
    """
    a = make_agent()
    before = a.sandbox.timeout

    call(a, code="import time\ntime.sleep(30)", timeout=1)

    assert a.sandbox.timeout == before


# -- the operator can stop it ------------------------------------------------
class _Speaks:
    """A steering channel that has something to say the moment it is asked."""

    def has_pending(self) -> bool:
        return True

    def stop_requested(self) -> bool:
        return False


def test_a_long_snippet_yields_to_the_operator_rather_than_going_deaf():
    """The one thing that must not be the only step nobody can interrupt."""
    a = make_agent()
    a.steer = _Speaks()

    r = call(a, code="import time\ntime.sleep(30)", timeout=60)

    assert "operator" in r.output, "the snippet ran to completion while the operator was talking"


# -- and the policy can switch it off ---------------------------------------
def test_a_policy_that_switches_off_arbitrary_code_refuses_by_name():
    a = make_agent(AutonomyPolicy(may_run_arbitrary_code=False))

    r = call(a, code="print('should never run')")

    assert "may_run_arbitrary_code" in r.output
    assert "should never run" not in r.output


# -- the prompt names it, so the detour does not come back -------------------
def test_the_prompt_points_one_off_lookups_at_it():
    """The incident's shape, pinned as text.

    The prompt named no way to execute code -- it said forge_tool *was* that
    access -- so when the forge died the agent invented a route (a browser
    driving a run-script endpoint, 126 times in one run). Pinned the same way
    test_reach pins the reach paragraph, and for the same reason: deleting the
    rule silently restores the detour.
    """
    from autoforge.agent import AUTONOMOUS_SYSTEM as sp

    assert "run_python" in sp
    assert "One-off lookups" in sp
    assert "do not drive a browser" in sp.lower()


# -- the rendering, which is what the model actually reads ------------------
def test_the_three_outcomes_are_told_apart():
    """Stopped by a person, out of time, and failed are different facts.

    Collapsing the first into either of the others is how an agent learns to
    retry something a person deliberately stopped.
    """
    aborted = _render_run(SandboxResult(ok=False, aborted=True, stdout="part"), 5)
    timed_out = _render_run(SandboxResult(ok=False, timed_out=True,
                                          stdout="part"), 5)
    failed = _render_run(SandboxResult(ok=False, error="ValueError: x"), 5)

    assert "operator" in aborted
    assert "timed out" in timed_out and "part" in timed_out
    assert failed.startswith("[error]")
    assert "part" not in failed
