"""The claim gate: does it fire on a tool that is wrong, and stay quiet on one
that is right?

A gate is only worth its runtime if it has been shown to fail. Every check in
this file is written as a pair -- the broken tool and the honest one -- because
a suite that only ever asserts `passed` is a suite that passes on a gate that
returns True unconditionally, which is exactly the failure the gate exists to
catch one level up.
"""
from __future__ import annotations

import json

import pytest

from autoforge.forge.claimgate import ClaimGate, check_claim, find_kind, find_rail
from autoforge.forge.sandbox import Sandbox
from autoforge.tools.spec import ToolSpec

#: A sample a real auditor would be handed: the prediction carries a genuine
#: (noisy) signal about the label, so an honest tool reports PASS on it and the
#: controls have something to move. A fixture with no signal at all would make
#: the "honest" tool REJECT its own sample, and a control demanding a collapse
#: would then have nothing to collapse -- which is how the first version of this
#: suite failed, for the wrong reason.
CSV = "scaffold,pred,label\n" + "\n".join(
    f"{i % 8},{0.84 + 0.03 * ((i * 7) % 3) if (i * 3) % 5 < 2 else 0.16 - 0.03 * ((i * 7) % 3)},"
    f"{1 if (i * 3) % 5 < 2 else 0}"
    for i in range(40)
)

AUDIT_PARAMS = {
    "type": "object",
    "properties": {
        "csv_text": {"type": "string"},
        "target_col": {"type": "string"},
        "pred_col": {"type": "string"},
        "higher_is_better": {"type": "boolean", "default": True},
    },
}

#: A deliberately dishonest auditor: it reports PASS and a fixed correlation
#: without ever reading the target column. Sized to sail through every check
#: A-E: it runs, it never crashes, it is trigger-agnostic, it prints a verdict.
BROKEN_AUDIT = """\
import csv, io
def broken_audit(csv_text='', target_col='label', pred_col='pred', higher_is_better=True):
    rows = [r for r in csv.reader(io.StringIO(csv_text)) if any(c.strip() for c in r)]
    return ('VERDICT: PASS | GATE1 reproduce ok (r=0.812, order_dev=0.0e+00) | '
            'GATE2 perturb ok | GATE3 distribution ok | GATE4 coverage ok')
"""

#: An auditor that actually reads both columns and reports what it finds.
HONEST_AUDIT = """\
import csv, io, math
def _rho(pairs):
    n = len(pairs)
    if n < 3: return None
    mx = sum(p for p, _ in pairs) / n; my = sum(q for _, q in pairs) / n
    sx = math.sqrt(sum((p - mx) ** 2 for p, _ in pairs)); sy = math.sqrt(sum((q - my) ** 2 for _, q in pairs))
    if sx == 0 or sy == 0: return None
    return sum((p - mx) * (q - my) for p, q in pairs) / (sx * sy)
def honest_audit(csv_text='', target_col='label', pred_col='pred', higher_is_better=True):
    rows = [r for r in csv.reader(io.StringIO(csv_text)) if any(c.strip() for c in r)]
    if len(rows) < 3: return 'VERDICT: FAIL | too few rows'
    head = [h.strip().lower() for h in rows[0]]
    ti = head.index(target_col.strip().lower()); pi = head.index(pred_col.strip().lower())
    pairs = [(float(r[pi]), float(r[ti])) for r in rows[1:]]
    r0 = _rho(pairs)
    if r0 is None: return 'VERDICT: FAIL | GATE1 reproduce FAIL (r=0.000)'
    return 'VERDICT: %s | GATE1 reproduce ok (r=%.3f)' % ('PASS' if abs(r0) >= 0.2 else 'REJECT', r0)
"""

#: No rail at all: a converter. The gate must say "not applicable", not "pass".
PLAIN = """\
def double(n=1):
    return int(n) * 2
"""

#: Non-deterministic: the same call gives a different answer every time.
JITTERY = """\
import random
def jittery(x='978-0-306-40615-7', higher_is_better=True):
    return {'score': 0.5 + random.random()}
"""


# NOTE: the entry point of a forged tool is looked up by the tool's own name,
# so each snippet below defines a function named exactly like the spec. A
# mismatch is not a subtle bug -- it fails as "entry not found", which is worth
# knowing: the first version of this file got it wrong and every test read as
# a broken tool rather than a broken test.
def _spec(name, code, params, sample, expect=""):
    return ToolSpec(name=name, description="test", parameters=params,
                    fn=lambda **k: None, code=code, source="generated",
                    sample_call=sample, sample_expect=expect)


def _gate(spec, sample=None):
    return check_claim(spec, Sandbox(timeout=20.0), sample_args=sample)


def test_kind_detection():
    assert find_kind(_spec("audit_x", "", AUDIT_PARAMS, {})) == "audit"
    assert find_rail(_spec("audit_x", "", AUDIT_PARAMS, {}))[0] == "higher_is_better"
    assert find_kind(_spec("double", "", {"type": "object", "properties": {"n": {"type": "integer"}}}, {})) == "none"


def test_dishonest_auditor_is_caught():
    """The case this gate was written for: it says PASS and never reads a label.

    Measured on 2026-10-02, `audit_predictive_claim_csv` did exactly this and
    passed all five checks it had.
    """
    spec = _spec("broken_audit", BROKEN_AUDIT, AUDIT_PARAMS,
                 {"csv_text": CSV, "target_col": "label", "pred_col": "pred"})
    rep = _gate(spec)
    assert rep.kind == "audit"
    assert not rep.passed, rep.summary()
    failed = {f.name for f in rep.failed}
    # The constant answer must fail the controls that inject a known answer.
    assert "label_permutation" in failed, rep.to_dict()
    assert "control_pair" in failed, rep.to_dict()
    assert "spike_recovery" in failed, rep.to_dict()


def test_honest_auditor_passes():
    spec = _spec("honest_audit", HONEST_AUDIT, AUDIT_PARAMS,
                 {"csv_text": CSV, "target_col": "label", "pred_col": "pred"})
    rep = _gate(spec)
    assert rep.kind == "audit"
    assert rep.passed, [f.to_dict() for f in rep.failed]
    assert {f.name for f in rep.findings} >= {"baseline", "deterministic",
                                              "order_invariance", "label_permutation",
                                              "control_pair", "spike_recovery"}


def test_non_deterministic_tool_is_caught():
    spec = _spec("jittery", JITTERY, {"type": "object", "properties": {
        "x": {"type": "string"}, "higher_is_better": {"type": "boolean"}}},
        {"x": "abc", "higher_is_better": True})
    rep = _gate(spec)
    assert not rep.passed, rep.summary()
    assert "deterministic" in {f.name for f in rep.failed}


def test_no_rail_is_not_a_pass_and_not_a_failure():
    """A converter has no rail. The record must say "we did not look"."""
    spec = _spec("double", PLAIN, {"type": "object", "properties": {"n": {"type": "integer"}}},
                 {"n": 3})
    rep = _gate(spec)
    assert rep.applicable is False
    assert rep.kind == "none"
    assert rep.passed  # nothing was claimed, so nothing is contradicted
    assert "not applicable" in rep.summary()


def test_audit_without_a_sample_is_un_probed_not_passed():
    spec = _spec("audit_x", HONEST_AUDIT, AUDIT_PARAMS, {})
    rep = _gate(spec)
    assert rep.applicable is True
    assert not rep.passed
    assert "un_probed" in {f.name for f in rep.failed}


def test_gate_is_deterministic_across_runs():
    spec = _spec("honest_audit", HONEST_AUDIT, AUDIT_PARAMS,
                 {"csv_text": CSV, "target_col": "label", "pred_col": "pred"})
    a = _gate(spec).to_dict()
    b = _gate(spec).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_disabled_gate_does_not_run_code():
    class Boom:
        def run(self, *a, **k):
            raise AssertionError("the gate ran code with run=False")

    spec = _spec("honest_audit", HONEST_AUDIT, AUDIT_PARAMS,
                 {"csv_text": CSV, "target_col": "label", "pred_col": "pred"})
    rep = ClaimGate(Boom(), run=False).gate(spec, {"csv_text": CSV})
    assert rep.passed and rep.kind != "none"
