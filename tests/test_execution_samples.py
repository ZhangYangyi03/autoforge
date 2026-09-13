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
