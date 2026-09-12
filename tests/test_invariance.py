"""Tests for the metamorphic oracle layer. All offline.

Each test names the exploit it closes. The exploit is reproduced first, then
asserted dead: `run_robustness_checks` used to score a probe as "survived"
iff the tool did not raise, so wrong-but-total functions scored 100%.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.forge.fuzzer import (
    generate_robustness_probes,
    run_robustness_checks,
)
from autoforge.forge.invariance import check as check_invariances
from autoforge.forge.invariance import derive, normalisers_for
from autoforge.tools.spec import ToolSpec


# -- helpers ----------------------------------------------------------
ISBN_PARAMS = {
    "type": "object",
    "properties": {"isbn": {"type": "string"}},
    "required": ["isbn"],
}

FREE_TEXT_PARAMS = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
}

VALID = "978-0-306-40615-7"


def spec(fn, params=ISBN_PARAMS, code="", **kw) -> ToolSpec:
    return ToolSpec(
        name="isbn_valid", description="Validate an ISBN-13.",
        parameters=params, fn=fn, code=code, source="generated", **kw,
    )


def isbn_with_prefix_bug(isbn: str = "") -> bool:
    """Correct on clean input, wrong the moment a user writes 'ISBN ...'."""
    digits = isbn.replace("-", "")
    return len(digits) == 13 and digits.isdigit()


def isbn_correct(isbn: str = "") -> bool:
    """Normalises the decoration, then validates."""
    cleaned = re.sub(r"^\s*isbn(-?1[03])?\s*:?\s*", "", isbn.strip(), flags=re.I)
    digits = cleaned.replace("-", "").replace(" ", "")
    return len(digits) == 13 and digits.isdigit()


# ======================================================================
# the probe was always there — the oracle was not
# ======================================================================
class TestProbeWasNeverTheGap:

    def test_prefix_probe_exists(self):
        """The ISBN probe is generated; the gap was never probe coverage."""
        labels = [p.label for p in generate_robustness_probes(spec(isbn_correct))]
        assert any("ISBN" in label for label in labels)

    def test_prefix_bug_survives_every_probe(self):
        """19/19 "survived" — because surviving only meant not raising."""
        result = run_robustness_checks(spec(isbn_with_prefix_bug))
        assert result.survived == result.total == 19
        assert not result.failures


# ======================================================================
# the oracle now refuses wrong-but-total functions
# ======================================================================
class TestDegeneracyOracle:

    def test_prefix_bug_is_now_caught(self):
        """The motivating case: wrong on the very input the probe supplies."""
        result = run_robustness_checks(spec(isbn_with_prefix_bug))
        assert not result.passed, result.summary()
        relations = {v["relation"] for v in result.invariance.violations}
        assert "normalise:isbn_prefix" in relations, relations

    def test_constant_emitter_fails(self):
        """`lambda: "nope"` used to score 19/19. It validates nothing."""
        result = run_robustness_checks(spec(lambda isbn="": "nope"))
        assert not result.passed
        assert any(v["relation"] == "non_degenerate"
                   for v in result.invariance.violations)

    def test_constant_true_fails(self):
        """A validator that accepts everything is not a validator."""
        result = run_robustness_checks(spec(lambda isbn="": True))
        assert not result.passed
        assert any(v["relation"] == "non_degenerate"
                   for v in result.invariance.violations)

    def test_correct_tool_still_passes(self):
        """The oracle must not be a blunt 'reject everything'."""
        result = run_robustness_checks(spec(isbn_correct))
        assert result.passed, result.summary()
        assert result.invariance.passed

    def test_none_returning_tool_still_fails(self):
        """Pre-existing definedness behaviour is preserved."""
        result = run_robustness_checks(spec(lambda isbn="": None))
        assert not result.passed


# ======================================================================
# deterministic and defined
# ======================================================================
class TestDefinedAndDeterministic:

    def test_nondeterministic_tool_fails(self):
        counter = {"n": 0}

        def flaky(isbn: str = "") -> bool:
            counter["n"] += 1
            return counter["n"] % 2 == 0

        result = check_invariances(spec(flaky))
        assert not result.passed
        assert any(v["relation"] == "deterministic" for v in result.violations)

    def test_raising_tool_fails_defined(self):
        def boom(isbn: str = ""):
            raise ValueError("nope")

        assert not check_invariances(spec(boom)).passed

    def test_clean_tool_passes_all(self):
        result = check_invariances(spec(isbn_correct))
        assert result.passed, result.summary()
        assert result.violations == []


# ======================================================================
# the oracle is scoped — free text is not a token
# ======================================================================
class TestScopeDiscipline:

    def test_free_text_params_get_no_token_relations(self):
        """`title` is content: whitespace and casing are part of the value."""
        assert normalisers_for("title") == []
        assert normalisers_for("text") == []

    def test_token_params_get_their_relations(self):
        labels = {label for label, _ in normalisers_for("isbn")}
        assert "isbn_prefix" in labels and "isbn_case" in labels

    def test_declared_token_on_generic_param(self):
        """Declaration lets an oddly-named parameter opt into token semantics."""
        labels = {label for label, _ in normalisers_for("value", ["isbn"])}
        assert "isbn_prefix" in labels

    def test_text_tool_is_not_flagged(self):
        """A tool whose whitespace matters must not be punished for it."""
        def count_chars(title: str = "") -> int:
            return len(title)

        result = check_invariances(spec(count_chars, FREE_TEXT_PARAMS))
        assert result.passed, result.summary()

    def test_derive_lists_computed_relations(self):
        relations = derive(spec(isbn_correct))
        assert "defined" in relations
        assert "non_degenerate" in relations
        assert "deterministic" in relations

    def test_derive_includes_declared(self):
        relations = derive(spec(isbn_correct, invariances=["isbn"]))
        assert "normalise:isbn" in relations


# ======================================================================
# the verifier's own check consumes the oracle
# ======================================================================
class TestVerifierIntegration:

    def _verifier(self):
        from autoforge.core.llm import LLMResponse, MockLLMClient
        from autoforge.forge.sandbox import Sandbox
        from autoforge.forge.verifier import ToolVerifier

        llm = MockLLMClient(handler=lambda *a, **k: LLMResponse(content="{}"))
        return ToolVerifier(llm, sandbox=Sandbox(timeout=8))

    def test_robustness_check_reports_invariance(self):
        check_result = self._verifier().check_robustness(spec(isbn_correct))
        assert check_result.passed
        assert "invariance" in check_result.evidence

    def test_robustness_check_fails_prefix_bug(self):
        check_result = self._verifier().check_robustness(spec(isbn_with_prefix_bug))
        assert not check_result.passed

    def test_constant_emitter_fails_full_verify(self):
        """The headline exploit: forge a do-nothing tool, fail verification."""
        do_nothing = ToolSpec(
            name="isbn_valid", description="Validate an ISBN-13.",
            parameters=ISBN_PARAMS,
            fn=lambda isbn="": "nope",
            code="def isbn_valid(isbn=''):\n    return 'nope'\n",
            source="generated",
        )
        report = self._verifier().verify(do_nothing)
        assert not report.passed
        assert not any(c.name == "robustness" and c.passed for c in report.checks)


@pytest.mark.parametrize("tool_name", ["isbn", "isbn13", "isbn_13", "isbn10"])
def test_token_name_variants_all_fire(tool_name):
    """Common spellings of the param name all opt into token semantics."""
    labels = {label for label, _ in normalisers_for(tool_name)}
    assert "isbn_prefix" in labels
