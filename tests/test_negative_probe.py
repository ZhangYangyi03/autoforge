"""The negative probe: a majority verdict, and not blaming the tool for a bad test.

`check_negative` used to run once and seal on that one draw. Measured on
2026-10-26 with `is_palindrome` and its own negative probe:

    eight runs of the same tool, same code, same library
    five stayed quiet, three called the tool

`ForgePipeline` seals on `report.passed`, so the same good tool was sent back to
DRAFT roughly one time in eight because a coin came up wrong. That is the defect
these tests pin: a single observation is not a measurement of a boundary.

The second half is the reason it looked like a coin at all. "What is the string
length of 'hello'?" is one noun away from "Does the string 'hello' read the same
backwards?" -- both ask about the same literal string. Reaching for
`is_palindrome` there is a reasonable reading, so the tool was not at fault: the
*generated probe* was. That is reported as a warning, not by failing the tool,
for the same reason `check_execution` reports a synthetic-probe failure as a
warning: failing a library for its own test suite is how tools get lost.

The tests here are deterministic by construction. `MajorityModel` is a scripted
LLM whose answers are chosen by the test, so nothing depends on sampling -- what
is under test is the counting and the verdict, not the model.
"""
from __future__ import annotations

import pytest

from autoforge.core.llm import LLMResponse, MockLLMClient, ToolCall
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.tools.spec import ToolSpec, TriggerProbe


def majority_model(tool_name: str, fire_on: set[int]) -> MockLLMClient:
    """An LLM that calls the tool on a scripted subset of trials.

    `fire_on` is a set of 1-based trial indices. Everything else answers without
    a tool call, which is what "stayed quiet" looks like from the verifier's
    side.

    A trial is counted from the first call of a conversation, and a trial that
    fires gets one more call (the agent's follow-up turn after the tool result).
    Distinguishing the two by counting user messages does not work -- both have
    exactly one -- so the handler keys on the conversation's identity: the
    verifier builds a fresh Agent per trial, each with a fresh message list,
    whose first object is therefore a new instance.
    """
    state = {"trial": 0, "seen_first": None}

    def handler(messages, tools=None, **kwargs):
        first = messages[0] if messages else None
        if first is not state["seen_first"]:
            state["seen_first"] = first
            state["trial"] += 1
        if state["trial"] in fire_on:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id=f"c{state['trial']}", name=tool_name,
                                     arguments={"text": "x"})],
            )
        return LLMResponse(content="That is not something I need a tool for.",
                           tool_calls=[])

    return MockLLMClient(handler=handler)


def spec_with(positive: str, negative: str) -> ToolSpec:
    return ToolSpec(
        name="is_palindrome",
        description="Check whether a string reads the same backwards.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
        code="def is_palindrome(text=''):\n    return str(text) == str(text)[::-1]\n",
        fn=lambda **_: "",
        source="human",
        probes=[TriggerProbe(query=positive, expect="call", negative_query=negative)],
    )


FAR_POSITIVE = "Is 'A man, a plan, a canal: Panama' a palindrome?"
FAR_NEGATIVE = "Convert 30 celsius to fahrenheit."
NEAR_NEGATIVE = "How many words are in 'A man, a plan, a canal: Panama'?"


def verifier_for(fire_on: set[int], trials: int = 0) -> ToolVerifier:
    kw = {"negative_trials": trials} if trials else {}
    return ToolVerifier(majority_model("is_palindrome", fire_on),
                        sandbox=Sandbox(timeout=10.0), **kw)


# -- the verdict is a majority, not a draw --------------------------------

def test_three_quiet_runs_pass():
    v = verifier_for(fire_on=set())
    r = v.check_negative(spec_with(FAR_POSITIVE, FAR_NEGATIVE), FAR_NEGATIVE)
    assert r.passed
    assert r.evidence["quiet"] == 3
    assert r.evidence["runs"] == 3


def test_one_quiet_run_out_of_three_is_noise_not_a_pass():
    """The old behaviour in one case: a single draw decided the tool's fate."""
    v = verifier_for(fire_on={1, 2})
    r = v.check_negative(spec_with(FAR_POSITIVE, FAR_NEGATIVE), FAR_NEGATIVE)
    assert not r.passed
    assert r.evidence["fired"] == 2


def test_two_quiet_runs_out_of_three_passes_and_says_so():
    """The case that used to be a coin flip, now a recorded 2-1."""
    v = verifier_for(fire_on={1})
    r = v.check_negative(spec_with(FAR_POSITIVE, FAR_NEGATIVE), FAR_NEGATIVE)
    assert r.passed
    assert r.evidence["quiet"] == 2
    assert r.evidence["fired"] == 1
    assert "2/3" in r.detail


def test_an_even_split_does_not_pass_by_rounding():
    """`>` rather than `>=`: 2-2 on four trials is not a majority."""
    v = verifier_for(fire_on={1, 2}, trials=4)
    r = v.check_negative(spec_with(FAR_POSITIVE, FAR_NEGATIVE), FAR_NEGATIVE)
    assert not r.passed


def test_the_trial_count_is_configurable_and_defaults_to_three():
    assert ToolVerifier(None).negative_trials == 3
    v = verifier_for(fire_on=set(), trials=5)
    r = v.check_negative(spec_with(FAR_POSITIVE, FAR_NEGATIVE), FAR_NEGATIVE)
    assert r.evidence["runs"] == 5


# -- and a near-miss probe is reported as adjacency, not over-firing ------

def test_a_near_probe_waives_the_check_instead_of_failing_the_tool():
    """The probe that caused the original flake asks about the same subject.

    It cannot fail the tool: it is testing "does the model phrase this
    differently", not "does the tool stay out of unrelated work". The check
    reports itself waived, which removes it from the verdict without pretending
    it was evidence.
    """
    v = verifier_for(fire_on={1})
    r = v.check_negative(spec_with(FAR_POSITIVE, NEAR_NEGATIVE), NEAR_NEGATIVE)
    assert r.waived is True
    assert r.passed is False              # a waiver is not a pass
    assert r.judged is False              # ... and it does not count
    assert r.evidence["probe_overlap"] >= 3
    assert "content word" in r.detail


def test_a_distant_probe_is_not_excused():
    """A tool that fires on a genuinely unrelated query has no such cover."""
    v = verifier_for(fire_on={1, 2})
    distant = "Convert 30 celsius to fahrenheit."
    r = v.check_negative(spec_with(FAR_POSITIVE, distant), distant)
    assert not r.passed
    assert r.evidence["probe_overlap"] == 0
    assert "content word" not in r.detail


def test_the_overlap_proxy_counts_a_quoted_payload():
    """Words alone missed the pair that caused this; the payload term catches it."""
    spec = spec_with("Check if 'racecar' is a palindrome.", "Reverse the characters in 'racecar'.")
    v = ToolVerifier(None)
    # one content word (`characters`) plus the shared payload `racecar`
    assert v._probe_overlap(spec, "Reverse the characters in 'racecar'.") >= 4


def test_bare_numbers_are_not_a_shared_payload():
    """`Compute 5 + 3` does not share a payload with `Compute 0!`."""
    spec = spec_with("Compute 0!", "Compute 5 + 3")
    v = ToolVerifier(None)
    assert v._probe_overlap(spec, "Compute 5 + 3") < 3


# -- the warning never changes the verdict --------------------------------

def test_the_warning_is_a_detail_not_a_grade(monkeypatch):
    """Overlap only edits the sentence; `passed` comes from the counts alone.

    Worth its own test because the temptation is to let a near-miss probe
    rescue a failing tool. It must not: a model that calls the tool on every
    draw has failed the check whatever the probe looked like.
    """
    v = verifier_for(fire_on={1, 2, 3})
    r = v.check_negative(spec_with(FAR_POSITIVE, NEAR_NEGATIVE), NEAR_NEGATIVE)
    # Three draws all fired, and the probe is near: the verdict is a waiver,
    # which is the check admitting it learned nothing. A nearer probe must not
    # be able to rescue a tool that failed on a distant one.
    assert r.waived is True
    assert r.evidence["probe_overlap"] >= 3
    distant = verifier_for(fire_on={1, 2, 3})
    rd = distant.check_negative(spec_with(FAR_POSITIVE, FAR_NEGATIVE), FAR_NEGATIVE)
    assert rd.waived is False
    assert rd.passed is False
