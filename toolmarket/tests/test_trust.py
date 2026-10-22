"""The `earn` edge: what a ledger buys, what it costs, and what it cannot buy.

The claim under test is not "the numbers are right". It is that trust is
*measured* — that a resource's state is a function of what it actually did, that
the function is inspectable when it says no, and that two specific things it
must never do are refused:

  * promote a tool nobody verified, because good behaviour is not a substitute
    for the verify edge; and
  * demote a tool that has done nothing, because a quiet ledger is not a
    confession.

Both are cases where the *tempting* implementation is the wrong one — the first
because "it works, ship it" is always seductive, the second because a naive
`rate < floor` comparison reads 0.0 as catastrophic. So they are pinned here.

Everything except the last section is a pure function of (state, ledger, policy),
which is what lets the console render a decision without changing one.
"""
from __future__ import annotations

import copy

import pytest

from toolmarket.protocol.lifecycle import ResourceState
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore
from toolmarket.trust import (
    TRUST_ACTOR,
    TrustPolicy,
    assess,
    read_ledger,
)

P, A, D, Q = (
    ResourceState.PROBATION.value,
    ResourceState.ACTIVE.value,
    ResourceState.DRAFT.value,
    ResourceState.QUARANTINED.value,
)


def _ledger(calls=0, successes=None, failures=0, consecutive=0, rate=None):
    """A ledger shaped like `ToolStats.to_dict()`.

    `rate` is left out unless asked for, because its *absence* is the branch the
    next test is about.
    """
    if successes is None:
        successes = calls
    out = {"calls": calls, "successes": successes, "failures": failures,
           "consecutive_failures": consecutive}
    if rate is not None:
        out["success_rate"] = rate
    return out


# ------------------------------------------------------------------ the numbers
def test_a_ledger_without_a_rate_recomputes_it_rather_than_claiming_zero():
    """0.0 is a claim about behaviour; a missing field is not.

    Defaulting the rate to 0.0 here would quarantine a tool whose ledger simply
    predates the field — punishing it for a schema change.
    """
    assert read_ledger({"calls": 4, "successes": 4})["success_rate"] == 1.0
    assert read_ledger({"calls": 4, "successes": 2})["success_rate"] == 0.5


def test_a_stored_rate_wins_so_policy_and_console_never_disagree():
    # Both read the same history; a second, independently rounded rate is a
    # chance for the page and the verdict to contradict each other.
    assert read_ledger({"calls": 10, "successes": 10, "success_rate": 0.83})["success_rate"] == 0.83


def test_an_empty_ledger_norms_to_zeros_without_dividing_by_zero():
    ev = read_ledger(None)
    assert ev == {"calls": 0, "successes": 0, "failures": 0,
                  "consecutive_failures": 0, "success_rate": 0.0}


# ------------------------------------------------------------------ earning
def test_a_probation_resource_that_clears_the_bar_is_promoted():
    d = assess("tool:x", P, _ledger(calls=5, rate=1.0))
    assert d.action == "promote" and d.to == A
    assert "earned" in d.reason


def test_the_bar_is_a_bar_not_a_decoration():
    """One call short is short. The bar is uninteresting if it bends at the edge."""
    d = assess("tool:x", P, _ledger(calls=4, rate=1.0))
    assert d.action == "hold" and d.to is None
    assert "4 call(s)" in d.reason and "needs 5" in d.reason


def test_nine_in_ten_is_enough_and_eight_in_ten_is_not():
    """The 0.9-not-1.0 argument, pinned.

    1.0 is unreachable for anything with a stochastic input, and an unreachable
    bar is one that gets ignored — which is the failure this module was written
    to fix. So the boundary is asserted from both sides.
    """
    assert assess("tool:x", P, _ledger(calls=10, successes=9, rate=0.9)).action == "promote"
    assert assess("tool:x", P, _ledger(calls=10, successes=8, rate=0.8)).action == "hold"


def test_a_perfect_rate_does_not_outvote_an_unresolved_failure():
    """`consecutive_failures` is the recent signal; the average is the historical one.

    A tool that succeeded nine times and has failed twice since has a flattering
    rate and is currently broken. Promotion reads both or it reads the wrong one.
    """
    d = assess("tool:x", P, _ledger(calls=10, successes=10, consecutive=1, rate=1.0))
    assert d.action == "hold"
    assert "1 consecutive failure(s) outstanding" in d.reason


def test_every_blocking_reason_is_named_not_summarised():
    """"not yet earned" with no detail is the empty `reason` this replaced."""
    d = assess("tool:x", P, _ledger(calls=1, successes=0, consecutive=2, rate=0.0))
    assert "needs 5" in d.reason
    assert "below 0.9" in d.reason
    assert "2 consecutive failure(s)" in d.reason


# ------------------------------------------------------------------ the unskippable edge
def test_a_draft_with_a_spotless_ledger_cannot_be_promoted():
    """The rule that makes the whole thing honest.

    ACTIVE is reachable by behaving well *from PROBATION*. It is not reachable
    from DRAFT at all — the verify edge is the only door, and no amount of
    good behaviour substitutes for it. A policy that promoted here would let the
    ledger override verification, which is exactly the "trust declared rather
    than earned" bug, with extra steps.
    """
    d = assess("tool:x", D, _ledger(calls=100, successes=100, rate=1.0))
    assert d.action == "hold" and d.to is None
    assert "verify" in d.reason


# ------------------------------------------------------------------ decay
def test_an_active_tool_with_no_calls_is_not_demoted():
    """Absence of evidence is not evidence of decay.

    Without this guard a quiet week reads 0.0 and quarantines every idle tool on
    the shelf — a policy that punishes being unused, which is most of a shelf.
    """
    d = assess("tool:x", A, _ledger())
    assert d.action == "hold" and d.to is None
    assert "absence of evidence" in d.reason


def test_three_consecutive_failures_quarantine_an_active_tool():
    d = assess("tool:x", A, _ledger(calls=10, successes=7, consecutive=3, rate=0.7))
    assert d.action == "quarantine" and d.to == Q
    assert "3 consecutive failures" in d.reason


def test_worse_than_a_coin_flip_quarantines_even_without_a_run():
    d = assess("tool:x", A, _ledger(calls=10, successes=4, rate=0.4))
    assert d.action == "quarantine" and d.to == Q


def test_a_healthy_active_tool_is_left_alone():
    d = assess("tool:x", A, _ledger(calls=50, successes=48, consecutive=1, rate=0.96))
    assert d.action == "hold" and d.to is None
    assert "trust intact" in d.reason


@pytest.mark.parametrize("state", [Q, ResourceState.RETIRED.value])
def test_states_outside_the_evidence_axis_are_inert(state):
    # QUARANTINED returns to PROBATION by rehab, not by the ledger; RETIRED is
    # not a trust candidate. Neither should be moved by a measurement.
    d = assess("tool:x", state, _ledger(calls=99, successes=99, rate=1.0))
    assert d.action == "hold" and d.to is None


# ------------------------------------------------------------------ purity
def test_assess_reads_the_ledger_without_touching_it():
    """It runs on a render path. A getter that mutates is a bug with a delay fuse."""
    led = _ledger(calls=5, rate=1.0)
    before = copy.deepcopy(led)
    assess("tool:x", P, led)
    assert led == before


def test_the_thresholds_are_one_object_an_operator_can_retune():
    strict = TrustPolicy(min_calls=2, min_success_rate=0.5)
    assert assess("tool:x", P, _ledger(calls=2, successes=1, rate=0.5), strict).action == "promote"
    assert assess("tool:x", P, _ledger(calls=2, successes=1, rate=0.5)).action == "hold"


def test_the_decision_carries_the_thresholds_it_was_measured_against():
    """"Why is it still on probation" needs the bar, not just the shortfall."""
    d = assess("tool:x", P, _ledger(calls=1, rate=1.0))
    view = d.to_dict(TrustPolicy().thresholds())
    assert view["thresholds"]["min_calls"] == 5
    assert view["evidence"]["calls"] == 1
    assert view["enabled"] is True


# ------------------------------------------------------------------ the earn edge, wired
CODE = "def slugify(text=''):\n    return '-'.join(text.lower().split())\n"


def _registry(**kw) -> ResourceRegistry:
    from autoforge.tools.spec import ToolSpec, TriggerProbe

    def slugify(text: str = "") -> str:
        return "-".join(text.lower().split())

    reg = ResourceRegistry(store=ResourceStore(":memory:"), **kw)
    reg.register(ToolSpec(
        name="slugify", description="Slugify a string.",
        parameters={"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
        fn=slugify, code=CODE, source="human",
        probes=[TriggerProbe(query="slugify this", expect="call")],
        invariances=["text"], effect_signature="pure",
    ))
    reg.transition("tool:slugify", ResourceState.PROBATION, reason="verified")
    return reg


def _call_n(reg, n):
    for _ in range(n):
        assert getattr(reg.invoke("tool:slugify", {"text": "hello world"}), "ok", False)


def test_invoking_past_the_bar_promotes_without_anyone_asking():
    """The property the whole module exists for: it happens on its own.

    Nothing here promotes the resource. Five real calls do.
    """
    reg = _registry()
    assert reg.get("tool:slugify").state is ResourceState.PROBATION
    _call_n(reg, 4)
    assert reg.get("tool:slugify").state is ResourceState.PROBATION, "promoted early"
    _call_n(reg, 1)
    assert reg.get("tool:slugify").state is ResourceState.ACTIVE


def test_the_promotion_names_the_policy_not_system():
    """The audit trail is the difference between an earned move and a mystery.

    Every ACTIVE resource on this shelf before now arrived with an `actor` of
    "system" and an empty `reason`, which is indistinguishable from a keyboard.
    """
    reg = _registry()
    _call_n(reg, 5)
    moves = [e for e in reg.event_log("tool:slugify")
             if e["data"].get("to") == A and e["actor"] == TRUST_ACTOR]
    assert len(moves) == 1
    # The reason rides in `data`, beside the from/to pair, so the audit row is
    # self-explanatory without cross-referencing the ledger at that seq.
    assert "earned" in moves[0]["data"]["reason"]


def test_disabling_the_policy_leaves_the_ledger_as_a_record_nothing_acts_on():
    reg = _registry(trust_policy=TrustPolicy(enabled=False))
    _call_n(reg, 6)
    assert reg.get("tool:slugify").state is ResourceState.PROBATION


def test_reading_the_trust_view_does_not_move_the_resource():
    """The console renders this. Opening a page must not be a state change."""
    reg = _registry()
    _call_n(reg, 4)
    view = reg.trust_view("tool:slugify")
    assert view["action"] == "hold"
    assert view["state"] == P
    assert reg.get("tool:slugify").state is ResourceState.PROBATION


def test_a_sweep_reports_a_reason_for_every_resource_including_the_held_ones():
    """`reconcile_all` is for the operator asking "why is nothing promoted".

    A sweep that returns only the moves answers that question with silence.
    """
    reg = _registry()
    _call_n(reg, 4)  # one short
    rows = reg.reconcile_all()
    assert len(rows) == 1
    assert rows[0]["action"] == "hold"
    assert "needs 5" in rows[0]["reason"]
    assert reg.get("tool:slugify").state is ResourceState.PROBATION


def test_a_sweep_applies_what_the_invoke_path_could_not():
    """A resource whose ledger was earned before the policy existed.

    The shelf has fourteen of these: PROBATION, real call history, no promotion,
    because there was no reader. The sweep is their way out.
    """
    reg = _registry()
    _call_n(reg, 5)
    rec = reg.get("tool:slugify")
    rec.state = ResourceState.PROBATION  # as if promoted never existed
    reg.save(rec)
    assert reg.reconcile_all()[0]["action"] == "promote"
    assert reg.get("tool:slugify").state is ResourceState.ACTIVE


# ------------------------------------------------------- the two surfaces
def _client(reg):
    from fastapi.testclient import TestClient

    from toolmarket.web import build_app

    return TestClient(build_app(reg))


def test_the_trust_route_answers_with_the_evidence_and_the_bar():
    """The endpoint exists so "why is this still on probation" is inspectable.

    Without it the answer lives only in a Python call, which is no answer for
    whoever is looking at the deployment.
    """
    c = _client(_registry())
    body = c.get("/api/resources/tool:slugify/trust").json()
    assert body["action"] == "hold"
    assert body["state"] == P
    assert body["evidence"]["calls"] == 0
    assert body["thresholds"]["min_calls"] == 5, "the bar must travel with the verdict"


def test_the_trust_route_is_read_only():
    """Rendering a page must not be a state-changing act.

    The route is the console's data source, so if it moved state then every
    refresh would be a write — and a resource one call short of promotion would
    get promoted by somebody opening a browser tab.
    """
    reg = _registry()
    _call_n(reg, 4)
    c = _client(reg)
    for _ in range(3):
        assert c.get("/api/resources/tool:slugify/trust").status_code == 200
    assert reg.get("tool:slugify").state is ResourceState.PROBATION


def test_an_unknown_resource_is_a_404_not_an_empty_verdict():
    # A silent `hold` for a resource that does not exist would look like a
    # well-behaved tool on probation, which is a worse answer than an error.
    c = _client(_registry())
    assert c.get("/api/resources/tool:nope/trust").status_code == 404


def test_the_console_ships_the_panel_and_degrades_when_the_route_is_absent():
    """Two deployments, one console.

    The panel is fetched through a `catch`, because the console is also pointed
    at APIs predating the route — and an unguarded `Promise.all` would reject the
    whole render on that 404, blanking the page rather than dropping one panel.
    """
    page = _client(_registry()).get("/").text
    assert "what the ledger earns" in page, "panel missing from the console"
    assert "trust view unavailable" in page, "no degradation path for an older API"
    assert '.catch(() => null)' in page, "fetch is not guarded"
