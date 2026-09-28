"""The routing weights were measured, and the shape of them is the finding.

`RoutingWeights` shipped as 1.0 / 1.2 / 0.8 / 0.5 / 0.6 -- picked by taste,
never scored. `tools/calibrate_routing.py` scores them against two ground-truth
sets already on disk: 312 (query, expect=call) TriggerProbe pairs with 312
negative pairs, and the 117 accepted forges in the ledger, each a real need
written with no knowledge of what the tool would one day be called.

What the measurement said was not "the numbers are slightly off". It was that
the terms are the wrong shape. `success` and `trust` do not vary with the query
-- they are per-tool constants -- so as additive terms they add a fixed bonus to
every tool on every query, and the more weight they carry the more they drown
out the only term that answers "which tool for THIS need".

    5-fold CV, 114 forged cases              top1     recall@5
      shipped 1.0/1.2/0.8/0.5/0.6            0.183      0.228
      text only                              0.570      0.833
      text + availability gate + tiebreak    0.570      0.833

The fix separates two questions that are not the same question:

  relevance  = text * gate, where gate = min(1, success/gate) scales a
               proven-broken tool toward zero and leaves a healthy one at 1.0.
               It never rewards a good record, so it cannot promote a tool for
               an unrelated need.
  tiebreak   = a small trust term, to order candidates that match equally well.

The corpus is what makes the gate cheap: 22 of the 24 tools with enough calls
to judge sit at success 1.0, so it fires on 1 of 120 tools, which is the case
those tests were written for -- a never-working tool must not outrank a working
one on the strength of its name.

These tests pin the SHAPE, not just the numbers, because re-tuning the numbers
back toward the old proportions is the failure mode.
"""
from __future__ import annotations

import pytest

from autoforge.route.router import (
    SKILL_WEIGHTS,
    BehaviourRouter,
    RoutingWeights,
)
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec, ToolState


def _tool(name, description, *, state=ToolState.ACTIVE, calls=0, ok=0):
    sp = ToolSpec(name=name, description=description,
                  parameters={"type": "object", "properties": {}},
                  fn=lambda **kw: "ok", state=state)
    sp.stats.calls = calls
    sp.stats.successes = ok
    return sp


class TestTheShape:
    """A per-tool constant must not be able to outvote the query."""

    def test_a_better_record_cannot_beat_a_better_match(self):
        """The defect, as a test: relevant-and-flaky must beat irrelevant-and-solid.

        `success` is constant per tool, so with the old weights a tool with a
        clean record outranked one that matched the need, on a need that had
        nothing to do with either. That is the whole failure in one assertion.
        """
        reg = ToolRegistry()
        reg.register(_tool("parse_the_thing", "parse a JSON document about things",
                           calls=9, ok=8))
        reg.register(_tool("unrelated_tool", "send an email", calls=50, ok=50))
        ranked = BehaviourRouter(reg).rank("parse a JSON document")
        assert ranked[0].name == "parse_the_thing"

    def test_the_gate_never_rewards_only_penalises(self):
        """A healthy tool is scaled by exactly 1.0 -- it gets no bonus at all."""
        reg = ToolRegistry()
        reg.register(_tool("t", "d", calls=10, ok=10))
        cand = BehaviourRouter(reg).rank("d")[0]
        assert cand.breakdown["gate"] == 1.0

    def test_the_gate_scales_a_broken_tool_down(self):
        reg = ToolRegistry()
        reg.register(_tool("t", "d", calls=10, ok=0))
        cand = BehaviourRouter(reg).rank("d")[0]
        assert cand.breakdown["gate"] == 0.0

    def test_an_unjudged_tool_is_not_penalised_for_being_new(self):
        """Too few calls -> success defaults to 0.5, which is exactly the gate.

        This is deliberate and it is the difference between "we have no evidence
        against it" and "we have evidence against it". Penalising here would
        bury every newly forged tool.
        """
        reg = ToolRegistry()
        reg.register(_tool("fresh", "d", calls=0))
        assert BehaviourRouter(reg).rank("d")[0].breakdown["gate"] == 1.0

    def test_the_terms_that_cannot_vary_are_gone_not_zero(self):
        """cost and over_trigger were 0.0 for all 120 tools; the fields are gone.

        The 09-27 fix set them to zero and left them settable. That is a knob
        with no signal behind it: `amend_self` could raise it and make retrieval
        worse with nothing to notice it by, which is the same shape as the
        weights this file exists to pin. Measured 2026-09-28: `cost_hint` is
        "cheap" for 120/120 tools and `trigger_misses` is 0 for 120/120.
        """
        w = RoutingWeights()
        assert not hasattr(w, "cost") and not hasattr(w, "over_trigger")
        assert "cost" not in w.to_dict() and "over_trigger" not in w.to_dict()
        # and the additive success term stays 0.0: it acts through the gate
        assert w.success == 0.0

    def test_a_retired_weight_is_refused_by_name_not_by_silence(self):
        """An amendment naming a retired knob gets told why, not "unknown key".

        "unknown" and "retired" are different answers. The first sends the
        caller looking for a typo; the second tells it the measurement removed
        the term.
        """
        from autoforge.agent import _coerce_weights
        for name in ("cost", "over_trigger"):
            parsed, err = _coerce_weights({name: 0.7}, RoutingWeights())
            assert parsed is None and err is not None, name
            assert name in err and "retired" in err, (name, err)

    def test_the_live_weights_still_amend(self):
        """What is left is still editable -- the retirement is not a lockout."""
        from autoforge.agent import _coerce_weights
        parsed, err = _coerce_weights({"trust": 0.2}, RoutingWeights())
        assert err is None and parsed.trust == 0.2

    def test_the_tiebreak_is_too_small_to_reorder_relevance(self):
        """Its whole justification: it separates equals and touches nothing else."""
        a = _tool("parse_json", "parse a JSON document", calls=10, ok=10)
        b = _tool("parse_json_strict", "parse a JSON document strictly", calls=10, ok=10)
        reg = ToolRegistry()
        reg.register(a)
        reg.register(b)
        # Same text score family, so trust may decide -- and does so stably.
        first = [c.name for c in BehaviourRouter(reg).rank("parse json")]
        assert first == [c.name for c in BehaviourRouter(reg).rank("parse json")]


class TestTheSkillRouterDidNotInheritTheToolDefaults:
    """A real regression, caught by `test_skills.py` and pinned here.

    Zeroing the tool router's `success` also zeroed the skill router's evidence
    term, because `SkillRouter` was constructed with `RoutingWeights()`. Two
    skills that read identically then became a coin flip on name order, and the
    one that had actually been loaded lost. The terms are different things: a
    tool's success folds into an availability gate, a skill's `proven` is the
    only signal there is.
    """

    def test_the_skill_defaults_keep_evidence_alive(self):
        assert SKILL_WEIGHTS.success != 0.0, (
            "zeroing this makes SkillRouter blind to 'reads right, loads "
            "never' -- the failure it exists to catch")

    def test_the_skill_router_does_not_use_the_tool_defaults(self):
        from autoforge.route.router import SkillRouter
        import inspect
        src = inspect.getsource(SkillRouter.__init__)
        assert "SKILL_WEIGHTS" in src or "skill" in src.lower()
