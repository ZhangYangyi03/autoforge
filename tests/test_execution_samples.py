"""The execution check must have something to be right about.

The bug these tests pin down was not a crash. `check_execution` passed on
`result.ok` alone -- the tool exited 0 -- while `_infer_args` handed every
forged tool the literal "978-0-306-40615-7", an ISBN left over from the first
tool this framework was ever built for. The generator is separately instructed
to answer "INVALID: <reason>" instead of raising. Put together, every tool was
fed input it could not parse, refused, exited 0, and was marked verified.

The damage was an inverted incentive, not a missed bug. A tool that did nothing
scored best of all: it survived every fuzz probe by refusing all of them, and
it passed the execution check by exiting cleanly, while a tool that genuinely
tried the job and failed on a real input looked worse. The agent then took
"verified" at its word, and when the tool failed in use it blamed the
environment -- ninety minutes spent diagnosing a network that was never broken.

So these tests assert the incentive, not just the code path:
  * refusing everything must FAIL, not pass quietly;
  * answering must PASS, and say what it answered;
  * "never probed" must not print the same as "proved correct".
"""

from __future__ import annotations

import pytest

from autoforge.core.llm import LLMResponse, MockLLMClient
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import CheckResult, ToolVerifier
from autoforge.tools.spec import ToolSpec

PARAMS = {"type": "object", "properties": {"raw": {"type": "string"}}, "required": ["raw"]}

# The shape of failure that used to pass. It is total (never raises, never
# returns None), it differs from a valid answer, and it does nothing at all.
REFUSES_EVERYTHING = (
    "def isbn10(raw=''):\n"
    "    return 'INVALID: not a 10- or 13-digit ISBN'\n"
)

# The same contract, but it actually does the subtask.
ANSWERS = (
    "def isbn10(raw=''):\n"
    "    s = str(raw).strip().replace('-', '').replace(' ', '').upper()\n"
    "    if s.startswith('ISBN'):\n"
    "        s = s[4:].lstrip(': ').strip()\n"
    "    if len(s) != 10 or not s[:9].isdigit():\n"
    "        return 'INVALID: not a 10-digit ISBN'\n"
    "    return s\n"
)

# A real ISBN-10, and the answer a correct tool must produce for it.
VALID_CALL = {"raw": "0-306-40615-2"}
VALID_DIGITS = "0306406152"


def verifier(**kw) -> ToolVerifier:
    llm = MockLLMClient(handler=lambda *a, **k: LLMResponse(content="{}"))
    return ToolVerifier(llm, sandbox=Sandbox(timeout=8), **kw)


def spec(code: str, **kw) -> ToolSpec:
    return ToolSpec(
        name="isbn10",
        description="Normalise an ISBN-10 to its bare digits.",
        parameters=PARAMS,
        fn=lambda **_: "",
        code=code,
        source="generated",
        **kw,
    )


def run(code: str, **kw) -> CheckResult:
    sample = kw.pop("sample_call", VALID_CALL)
    expect = kw.pop("sample_expect", "")
    return verifier().check_execution(
        spec(code, sample_call=sample, sample_expect=expect)
    )


class TestTheIncentive:

    def test_a_tool_that_refuses_everything_fails(self):
        """The exact tool that used to sail through, now stopped.

        This is the regression that mattered. Nothing about the code changed --
        it still runs, still exits 0, still never raises. Only the question
        asked of it changed, from "did it survive" to "did it work".
        """
        r = run(REFUSES_EVERYTHING)
        assert not r.passed
        assert "rejected its own valid sample call" in r.detail
        assert r.evidence["sample_call"] == VALID_CALL

    def test_answering_passes_and_says_what_it_answered(self):
        r = run(ANSWERS)
        assert r.passed, r.detail
        # The detail must carry the answer, not just a green tick: a verdict
        # nobody can audit is how the original bug survived so long.
        assert VALID_DIGITS in r.detail
        assert r.evidence["output"].strip() == VALID_DIGITS

    def test_the_two_differ_only_in_answering(self):
        """Same signature, same totality, opposite verdicts.

        If both of these ever pass together, the check has gone vacuous again.
        """
        refused = run(REFUSES_EVERYTHING)
        answered = run(ANSWERS)
        assert refused.passed != answered.passed

    def test_unprobed_is_not_a_pass(self):
        """No sample_call must read as un-probed, never as 'ran clean'."""
        r = run(ANSWERS, sample_call={})
        assert r.evidence.get("unprobed") is True
        assert "UNPROBED" in r.detail.upper()
        assert "ran clean" not in r.detail

    def test_a_wrong_answer_fails_when_expectation_is_declared(self):
        r = run(ANSWERS, sample_expect="9999999999")
        assert not r.passed
        assert "does not contain" in r.detail

    def test_a_right_answer_passes_when_expectation_is_declared(self):
        r = run(ANSWERS, sample_expect=VALID_DIGITS)
        assert r.passed, r.detail

    def test_none_returning_tool_fails(self):
        """Returning None is the other way to look empty-handed."""
        r = run("def isbn10(raw=''):\n    return None\n")
        assert not r.passed

    def test_empty_string_return_fails(self):
        r = run("def isbn10(raw=''):\n    return ''\n")
        assert not r.passed

    def test_raising_on_a_valid_call_fails(self):
        r = run("def isbn10(raw=''):\n    raise ValueError('boom')\n")
        assert not r.passed


class TestEvidence:

    def test_evidence_names_the_call_belongs_to_the_verdict(self):
        """The verdict cites the input, so it can be checked by hand."""
        r = run(ANSWERS)
        assert r.evidence["sample_call"] == VALID_CALL
        assert "duration_ms" in r.evidence

    def test_failure_evidence_is_present_too(self):
        r = run(REFUSES_EVERYTHING)
        assert r.evidence["sample_call"] == VALID_CALL
        assert "INVALID" in r.evidence["output"]

    def test_synthetic_probe_raising_is_only_a_warning(self):
        """Failing invented garbage is a robustness note, not a wrong answer.

        A good tool that is not total on junk should still be recorded as
        having answered correctly -- the robustness check owns that complaint,
        and mixing the two is what let "did it run" stand in for "did it work".
        """
        code = (
            "def isbn10(raw=''):\n"
            "    s = str(raw).strip().replace('-', '')\n"
            "    if len(s) != 10:\n"
            "        return 'INVALID: length'\n"
            "    return s[:9] + str(int(s[9]) * 2 % 10)\n"
        )
        r = run(code)
        assert r.passed, r.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── the three-state verdict, and the defect it closes ────────────────────────
#
# `test_unprobed_is_not_a_pass` above asserts on `evidence["unprobed"]` and on
# the detail text, but never on `r.passed`. The name promised more than the
# assertions checked, and the gap was real: `check_execution` returned
# `result.ok` for a tool that declared no sample, so an un-probed check read as
# a pass in every machine-readable field while only the prose said otherwise.
#
# Measured on this host before the fix: a tool that answered 212F as 180C
# (correct: 100C) and declared no sample was sealed ACTIVE in 4 of 6 runs. The
# only thing that ever stopped it was the LLM attacker guessing the boundary.
#
# These tests pin the invariant at the level that decides a tool's fate:
# `VerificationReport.passed`, and the pipeline transition it gates.

def test_an_unprobed_check_is_not_passed_in_the_data():
    """The promise `test_unprobed_is_not_a_pass` makes, checked on `passed`."""
    r = run(ANSWERS, sample_call={})
    assert r.unprobed is True
    assert r.passed is False, (
        "an un-probed execution check must not report passed=True: nothing "
        "compared an answer to a question, so there is nothing to pass on")
    assert r.judged is False


def test_unprobed_and_passed_cannot_both_be_true():
    """The invariant is enforced in code, so a future check cannot forget it."""
    from autoforge.forge.verifier import CheckResult

    with pytest.raises(ValueError):
        CheckResult("execution", True, "looks fine", {}, unprobed=True)


def test_the_report_separates_unprobed_from_failed():
    """A report must not let a reader confuse "did not run" with "ran and failed"."""
    from autoforge.forge.sandbox import Sandbox
    from autoforge.forge.verifier import ToolVerifier
    from autoforge.tools.spec import ToolSpec

    verifier = ToolVerifier(None, sandbox=Sandbox(timeout=20.0),
                            run_trigger_check=False, run_negative_check=False,
                            run_claim_check=False, run_adversarial_check=False,
                            run_robustness_check=False)
    spec = ToolSpec(
        name="isbn10", description="d",
        parameters={"type": "object", "properties": {"raw": {"type": "string"}}},
        fn=lambda **_: "", code=ANSWERS, source="human",
        sample_call={}, sample_expect="",
    )
    report = verifier.verify(spec)
    assert [c.name for c in report.unprobed] == ["execution"]
    assert [c.name for c in report.failed] == ["execution"]     # unprobed also fails
    # The summary must name it rather than let "0/1 passed" imply a wrong answer.
    assert "unprobed" in report.summary().lower(), report.summary()


def test_a_sample_less_tool_is_never_sealed_active():
    """The transition the whole framework exists to protect.

    `ForgePipeline` seals on `report.passed`. If an un-probed execution check
    counted as a pass, a wrong tool with no declared sample became ACTIVE.
    """
    from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
    from autoforge.forge.sandbox import Sandbox
    from autoforge.forge.verifier import ToolVerifier
    from autoforge.tools.registry import ToolRegistry
    from autoforge.tools.spec import ToolState

    class _Answers:
        """A generator that hands back one wrong tool, once per round."""
        def __init__(self) -> None:
            self.rounds = 0

        def generate(self, need, context=""):
            from autoforge.forge.generator import GeneratedTool
            self.rounds += 1
            return GeneratedTool(
                name="fahrenheit_to_celsius",
                description="Convert Fahrenheit to Celsius.",
                # 212F is 100C; this returns 180. Wrong, and total.
                code="def fahrenheit_to_celsius(f):\n    return float(f) - 32\n",
                parameters={"type": "object",
                            "properties": {"f": {"type": "number"}},
                            "required": ["f"]},
                sample_call={},            # ← the defect's precondition
                sample_expect="",
            )

    sandbox = Sandbox(timeout=20.0)
    verifier = ToolVerifier(
        None, sandbox=sandbox,
        run_execution_check=True,
        run_robustness_check=False,
        run_adversarial_check=False,
        run_trigger_check=False,
        run_negative_check=False,
        run_claim_check=False,
    )
    registry = ToolRegistry()
    pipeline = ForgePipeline(_Answers(), verifier, registry, sandbox=sandbox,
                             config=ForgeConfig(max_rounds=1))
    result = pipeline.forge("convert fahrenheit to celsius")
    assert not result.ok, (
        "a tool with no declared sample was sealed despite its correctness "
        "never having been examined")
    sealed = registry.get("fahrenheit_to_celsius")
    assert sealed is None or sealed.state is not ToolState.ACTIVE


def test_an_unprobed_check_blocks_even_when_another_check_passes():
    """The exact shape that reintroduced the defect on 2026-10-26.

    `execution` was unprobed and `claim` passed. The first version of the
    three-state fix filtered on `judged` -- which excludes unprobed -- and then
    asked `all(c.passed for c in judged)`. With `claim` in that list and passing,
    the report came out `passed=True` and a tool with no declared sample was
    sealed ACTIVE. The belt-and-braces `all(...)` did not save it, because the
    unprobed check had been filtered out of the list it iterated.

    A single-check test cannot catch that: with only `execution` present, its
    `passed=False` fails the `all(...)` whichever way the unprobed term goes.
    This test needs a SECOND, passing check to reproduce it.
    """
    from autoforge.core.llm import MockLLMClient
    from autoforge.forge.sandbox import Sandbox
    from autoforge.forge.verifier import ToolVerifier
    from autoforge.tools.spec import ToolSpec

    v = ToolVerifier(
        MockLLMClient(handler=lambda *a, **k: None),
        sandbox=Sandbox(timeout=20.0),
        run_execution_check=True,
        run_robustness_check=False,
        run_adversarial_check=False,
        run_trigger_check=False,
        run_negative_check=False,
        run_claim_check=True,          # ← the second, passing check
    )
    spec = ToolSpec(
        name="fahrenheit_to_celsius", description="Convert F to C.",
        parameters={"type": "object", "properties": {"f": {"type": "number"}},
                    "required": ["f"]},
        fn=lambda **_: "",
        # 212F is 100C; this returns 180, and declares no sample.
        code="def fahrenheit_to_celsius(f):\n    return float(f) - 32\n",
        source="generated", sample_call={}, sample_expect="",
    )
    report = v.verify(spec)
    assert [c.name for c in report.unprobed] == ["execution"]
    assert any(c.passed for c in report.checks), (
        "this test is only meaningful when some other check passes")
    assert report.passed is False, (
        "an unprobed execution check must block the battery even when another "
        "check passes: the report is a claim that the tool was TESTED, and "
        "nothing compared an answer to a question")
