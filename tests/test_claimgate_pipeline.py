"""Does the claim gate actually run inside the forge pipeline?

`tests/test_claimgate.py` tests the gate in isolation. This one tests the
*wiring*: that a forge which produces a tool whose answer does not depend on
its inputs comes out DRAFT rather than ACTIVE, and that the refusal reaches the
generator as feedback rather than a shrug.

That second half is the one that decays silently. A gate that fails a candidate
but tells nobody is a gate the repair loop cannot act on: the pipeline spends
its remaining rounds re-running the same broken tool. Asserting on the feedback
text is how that stays true.
"""
from __future__ import annotations

from autoforge.core.llm import MockLLMClient
from autoforge.forge.generator import GeneratedTool
from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolState

CSV = "scaffold,pred,label\n" + "\n".join(
    f"{i % 6},{0.86 if (i * 3) % 5 < 2 else 0.14},{1 if (i * 3) % 5 < 2 else 0}"
    for i in range(30)
)

PARAMS = {
    "type": "object",
    "properties": {
        "csv_text": {"type": "string"},
        "target_col": {"type": "string"},
        "pred_col": {"type": "string"},
    },
    "required": ["csv_text"],
}

#: Says PASS regardless of what it is handed. Every check A-E is green on it:
#: it runs, it survives the fuzzer by ignoring its input, the attacker finds
#: nothing, and it has no probes to false-fire on.
BROKEN = """\
def broken_audit(csv_text='', target_col='label', pred_col='pred'):
    return 'VERDICT: PASS | GATE1 reproduce ok (r=0.800) | all gates ok'
"""

HONEST = """\
import csv, io, math
def honest_audit(csv_text='', target_col='label', pred_col='pred'):
    # Guards matter: the fuzzer feeds this empty strings, one-column tables and
    # junk, and a tool that answers every real question correctly but raises on
    # the fuzzer's garbage is not one anybody can use. Written the way the
    # generator is told to write: refuse with INVALID rather than raise.
    try:
        rows = [r for r in csv.reader(io.StringIO(csv_text)) if any(c.strip() for c in r)]
        if len(rows) < 3:
            return 'INVALID: need a header and at least two data rows'
        head = [h.strip().lower() for h in rows[0]]
        if target_col.strip().lower() not in head or pred_col.strip().lower() not in head:
            return 'INVALID: no %r or %r column' % (target_col, pred_col)
        ti = head.index(target_col.strip().lower()); pi = head.index(pred_col.strip().lower())
        pairs = []
        for r in rows[1:]:
            if max(ti, pi) >= len(r):
                continue
            try:
                pairs.append((float(r[pi]), float(r[ti])))
            except ValueError:
                continue
        if len(pairs) < 3:
            return 'INVALID: fewer than three numeric rows'
        n = len(pairs); mx = sum(p for p, _ in pairs) / n; my = sum(q for _, q in pairs) / n
        sx = math.sqrt(sum((p - mx) ** 2 for p, _ in pairs)); sy = math.sqrt(sum((q - my) ** 2 for _, q in pairs))
        r0 = 0.0 if sx == 0 or sy == 0 else sum((p - mx) * (q - my) for p, q in pairs) / (sx * sy)
        return 'VERDICT: %s | GATE1 reproduce ok (r=%.3f)' % ('PASS' if abs(r0) >= 0.2 else 'REJECT', r0)
    except Exception as exc:
        return 'INVALID: %s: %s' % (type(exc).__name__, exc)
"""


class ScriptedGenerator:
    """Hands back a fixed list of candidates, one per round."""

    def __init__(self, names, bodies):
        self.names, self.bodies = list(names), list(bodies)
        self.seen: list[str] = []

    def generate(self, need, context=""):
        self.seen.append(need)
        i = min(len(self.seen) - 1, len(self.names) - 1)
        return GeneratedTool(
            name=self.names[i], description="audit a predictive claim",
            code=self.bodies[i], parameters=PARAMS, entry=self.names[i],
            sample_call={"csv_text": CSV, "target_col": "label", "pred_col": "pred"},
        )


def _pipeline(gen, **cfg):
    sandbox = Sandbox(timeout=20.0)
    verifier = ToolVerifier(MockLLMClient(), sandbox=sandbox,
                            run_adversarial_check=False, run_trigger_check=False,
                            run_negative_check=False)
    return ForgePipeline(gen, verifier, ToolRegistry(),
                         config=ForgeConfig(max_rounds=len(gen.names), **cfg),
                         sandbox=sandbox)


def test_a_broken_claim_never_reaches_active():
    gen = ScriptedGenerator(["broken_audit"], [BROKEN])
    result = _pipeline(gen).forge("audit a predictive claim csv")
    assert result.spec is None, "a tool that never reads its input was accepted"
    assert result.attempts, "no attempt was recorded"
    report = result.attempts[-1].report
    assert report is not None and not report.passed
    claim = [c for c in report.checks if c.name == "claim"]
    assert claim and not claim[0].passed
    failed = {f["name"] for f in claim[0].evidence["findings"] if not f["ok"]}
    assert "label_permutation" in failed


def test_the_refusal_is_fed_back_to_the_generator():
    """Round 2 must be told *why*, or it re-sends the same tool."""
    gen = ScriptedGenerator(["broken_audit", "honest_audit"], [BROKEN, HONEST])
    _pipeline(gen).forge("audit a predictive claim csv")
    assert len(gen.seen) >= 2, "the repair round never ran"
    second = gen.seen[1]
    assert "claim" in second
    assert "label_permutation" in second or "control" in second


def test_an_honest_claim_is_accepted():
    gen = ScriptedGenerator(["honest_audit"], [HONEST])
    result = _pipeline(gen).forge("audit a predictive claim csv")
    assert result.spec is not None, [a.error for a in result.attempts]
    spec = result.spec
    assert spec.state in (ToolState.ACTIVE, ToolState.PROBATION)
    assert spec.sample_call  # travels with the spec, so a re-audit has real args


PLAUSIBLE_LIAR = """\
import csv, io, math
def plausible_liar(csv_text='', target_col='label', pred_col='pred'):
    # Never looks at the target column. Reports a number derived from the
    # *prediction* column's own spread, which looks like a quality score, moves
    # with the data, and is stable under every perturbation the old battery
    # makes. This is the shape of the real failure: not a tool that crashes or
    # returns a constant, but one that returns something plausible.
    try:
        rows = [r for r in csv.reader(io.StringIO(csv_text)) if any(c.strip() for c in r)]
        if len(rows) < 3:
            return 'INVALID: too few rows'
        head = [h.strip().lower() for h in rows[0]]
        if pred_col.strip().lower() not in head:
            return 'INVALID: no such column'
        pi = head.index(pred_col.strip().lower())
        ps = []
        for r in rows[1:]:
            try:
                ps.append(float(r[pi]))
            except ValueError:
                pass
        if len(ps) < 3:
            return 'INVALID: too few numeric rows'
        mu = sum(ps) / len(ps)
        sd = math.sqrt(sum((x - mu) ** 2 for x in ps) / len(ps))
        score = min(0.999, sd / (abs(mu) + sd + 1e-9) + 0.3)
        return 'VERDICT: %s | GATE1 reproduce ok (r=%.3f) | rows=%d' % (
            'PASS' if score >= 0.5 else 'REJECT', score, len(ps))
    except Exception as exc:
        return 'INVALID: %s' % exc
"""


def test_a_plausible_liar_passes_the_old_battery_and_fails_the_new_one():
    """The comparison that justifies the gate existing.

    Built with `require_claim=False` this tool is *accepted*: it runs, it
    survives 50 fuzz probes, it varies with its input, it never raises. It is
    also not measuring anything -- `target_col` is read for nothing and the
    number it prints is a function of the prediction column's spread alone.
    The claim gate is the only check that asks whether the answer depends on
    what the tool claims to be about.
    """
    gen = ScriptedGenerator(["plausible_liar"], [PLAUSIBLE_LIAR])
    without = _pipeline(gen, require_claim=False).forge("audit a predictive claim csv")
    assert without.spec is not None, (
        "the old battery should accept this; if it no longer does, this test "
        "is comparing the wrong two things")

    gen2 = ScriptedGenerator(["plausible_liar"], [PLAUSIBLE_LIAR])
    with_gate = _pipeline(gen2, require_claim=True).forge("audit a predictive claim csv")
    assert with_gate.spec is None, "the claim gate let a tool that never reads its labels through"
    claim = [c for c in with_gate.attempts[-1].report.checks if c.name == "claim"][0]
    assert not claim.passed
    failed = {f["name"] for f in claim.evidence["findings"] if not f["ok"]}
    assert "label_permutation" in failed, failed
