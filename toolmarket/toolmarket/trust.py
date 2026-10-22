"""TrustPolicy — the missing `earn` edge of the lifecycle.

`toolmarket/protocol/lifecycle.py` draws five states and these edges::

    DRAFT ──verify──> PROBATION ──earn──> ACTIVE ──decay──> QUARANTINED

Three of them had an owner. `verify` belongs to whoever ran the verifier
(autoforge's forge loop). The `execution` and `evolution` edges belong to the
registry. The two *evidence* edges — `earn` upward and `decay` downward —
belonged to nobody, and this module is what takes them.

What that cost, concretely. autoforge publishes a tool, moves it to PROBATION
when verification passes, and deliberately stops there:

    `autoforge/agent.py::_promote_on_market`: "Not `active`: earning trust is a
    separate step and this code has no evidence for it."

That refusal is correct, and the problem is that the step it was deferring to
did not exist. `ResourceRecord.ledger` was being written on every invoke —
`calls`, `successes`, `failures`, `consecutive_failures`, the whole history —
and read by nothing. So a tool sat at PROBATION forever no matter how well it
behaved, and the only two resources that ever reached ACTIVE in this shelf's
history got there via a hand-typed POST carrying an empty `reason` and an
`actor` of "system". Trust was being *declared*.

`toolmarket/__init__.py` states the rule this module exists to enforce: the
state "can only be *earned*, never declared". Earning needs a measuring stick,
and the ledger was already the measurement. It just had no reader.

So `assess()` reads the ledger and answers one question — given what this
resource has actually done in the wild, what does it deserve? The answer is a
`TrustDecision`: an action, the evidence behind it, and the specific
precondition that failed whenever the action is `hold`. A hold carries a
*reason*, never a shrug, because "why is this still on probation" is exactly
the question the empty `reason` fields left unanswerable.

Two properties are not negotiable, and neither is a threshold:

  * **Promotion cannot skip PROBATION.** `DRAFT -> ACTIVE` is not a legal
    transition in the lifecycle, so a tool nobody verified cannot be promoted
    by good behaviour alone. This module will not pretend otherwise; a DRAFT
    resource with a spotless ledger gets `hold`, naming the missing verify edge.
  * **Absence of evidence is not evidence of decay.** An ACTIVE resource with
    zero calls is not demoted. Nothing was observed, so nothing was observed to
    go wrong, and demoting on an empty ledger would let a quiet week look like a
    failure. The decay edge needs positive evidence of degradation.

The thresholds themselves are *policy*, and they are labelled as such. Nothing
in the protocol dictates "five calls at 90%". That is a chosen bar with an
argument behind it — see `TrustPolicy` — it lives in one dataclass, and it is
meant to be argued with and changed without touching the protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toolmarket.protocol.lifecycle import ResourceState

# Who moves a resource when the policy does. Every transition this module makes
# is stamped with it, which is the difference between a shelf where "system"
# pushed 22 transitions and nobody can say why, and one where a promotion names
# its cause. The value is deliberately not "system": that string is what the
# un-audited manual probes wrote, and it should stay greppably distinct from a
# decision a program actually made.
TRUST_ACTOR = "trust-policy"


@dataclass(frozen=True)
class TrustPolicy:
    """The bar a resource must clear, and the floor it falls through.

    The defaults are a first guess with an argument, not a finding, and they are
    the kind of number an operator should expect to tune per deployment:

    `min_calls = 5`
        Five real invocations is the smallest sample where "it usually works"
        carries information. At three calls you cannot distinguish a reliable
        tool from a two-thirds one, and the whole point of gating on evidence is
        not to hand ACTIVE to something whose ledger cannot yet tell the
        difference.

    `min_success_rate = 0.9`
        Not 1.0. Demanding a perfect record makes the bar unreachable for any
        tool with a stochastic input, and an unreachable bar is a bar nobody
        honours — the failure mode this module was written to fix. Nine in ten
        with a clean recent run is the shape of "trustworthy but not magic".

    `max_consecutive_failures = 3`
        A single failure is a bad argument, not a broken tool. Three in a row is
        a tool that has stopped working, and three is the point where the
        `consecutive` counter in the ledger is unambiguous.

    `floor_success_rate = 0.5`
        Below half, over any sample that cleared `min_calls`, the tool is worse
        than a coin flip and should not be advertised as trusted. It is a floor,
        not a target.
    """

    enabled: bool = True
    min_calls: int = 5
    min_success_rate: float = 0.9
    max_consecutive_failures: int = 3
    floor_success_rate: float = 0.5

    def thresholds(self) -> dict[str, Any]:
        return {
            "min_calls": self.min_calls,
            "min_success_rate": self.min_success_rate,
            "max_consecutive_failures": self.max_consecutive_failures,
            "floor_success_rate": self.floor_success_rate,
        }


@dataclass(frozen=True)
class TrustDecision:
    """What the policy decided, and the evidence it decided from."""

    resource_id: str
    state: str
    action: str            # "promote" | "quarantine" | "hold"
    to: str | None         # target state, None when the action is "hold"
    reason: str
    evidence: dict[str, Any]
    enabled: bool = True

    def to_dict(self, thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "resource_id": self.resource_id,
            "state": self.state,
            "action": self.action,
            "to": self.to,
            "reason": self.reason,
            "evidence": self.evidence,
            "enabled": self.enabled,
        }
        if thresholds is not None:
            out["thresholds"] = thresholds
        return out


def _state_name(state: Any) -> str:
    return getattr(state, "value", None) or (str(state) if state else "")


def read_ledger(ledger: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise a ledger dict into the five numbers the policy reasons about.

    `ToolStats.to_dict()` already carries a rounded `success_rate`; it is
    preferred when present so the policy and the console never disagree about
    the same call history. When it is absent the rate is recomputed from
    `successes / calls` rather than defaulted to 0.0, because 0.0 is a claim
    about behaviour and a ledger that merely lacks the field has made no claim.
    """
    src = ledger or {}
    calls = int(src.get("calls", 0) or 0)
    successes = int(src.get("successes", 0) or 0)
    failures = int(src.get("failures", 0) or 0)
    consecutive = int(src.get("consecutive_failures", 0) or 0)
    rate = src.get("success_rate")
    if rate is None:
        rate = (successes / calls) if calls else 0.0
    return {
        "calls": calls,
        "successes": successes,
        "failures": failures,
        "consecutive_failures": consecutive,
        "success_rate": round(float(rate), 4),
    }


def assess(
    resource_id: str,
    state: Any,
    ledger: dict[str, Any] | None,
    policy: TrustPolicy | None = None,
) -> TrustDecision:
    """Decide what this resource's observed behaviour earns it.

    Pure: reads state and ledger, writes nothing. Every caller that wants the
    decision *applied* goes through `ResourceRegistry.reconcile_trust`, so this
    can be called on a hot read path — the console does exactly that — without
    making the rendering of a page a state-changing act.
    """
    pol = policy or TrustPolicy()
    name = _state_name(state)
    ev = read_ledger(ledger)
    calls, rate, consecutive = ev["calls"], ev["success_rate"], ev["consecutive_failures"]

    # -- earning ACTIVE ---------------------------------------------------
    if name == ResourceState.PROBATION.value:
        short: list[str] = []
        if calls < pol.min_calls:
            short.append(
                f"{calls} call(s) on the ledger, needs {pol.min_calls}")
        if rate < pol.min_success_rate:
            short.append(
                f"success_rate {rate:.3f} below {pol.min_success_rate}")
        if consecutive > 0:
            short.append(
                f"{consecutive} consecutive failure(s) outstanding")
        if short:
            return TrustDecision(
                resource_id, name, "hold", None,
                "not yet earned: " + "; ".join(short), ev, pol.enabled)
        return TrustDecision(
            resource_id, name, "promote", ResourceState.ACTIVE.value,
            f"earned: {calls} call(s) at success_rate {rate:.3f} "
            f"with no consecutive failures",
            ev, pol.enabled)

    # -- losing ACTIVE ----------------------------------------------------
    if name == ResourceState.ACTIVE.value:
        if calls == 0:
            # See the module docstring: an empty ledger is not a verdict.
            return TrustDecision(
                resource_id, name, "hold", None,
                "holding: no calls on the ledger yet — absence of evidence "
                "is not evidence of decay", ev, pol.enabled)
        burned = consecutive >= pol.max_consecutive_failures
        diluted = rate < pol.floor_success_rate
        if burned or diluted:
            why = []
            if burned:
                why.append(
                    f"{consecutive} consecutive failures "
                    f"(limit {pol.max_consecutive_failures})")
            if diluted:
                why.append(
                    f"success_rate {rate:.3f} below floor "
                    f"{pol.floor_success_rate}")
            return TrustDecision(
                resource_id, name, "quarantine",
                ResourceState.QUARANTINED.value,
                "decayed: " + "; ".join(why), ev, pol.enabled)
        return TrustDecision(
            resource_id, name, "hold", None,
            f"holding: {calls} call(s) at success_rate {rate:.3f} — "
            f"trust intact", ev, pol.enabled)

    # -- everything else --------------------------------------------------
    if name == ResourceState.DRAFT.value:
        why = ("the verify edge has not been walked; a resource cannot reach "
               "ACTIVE by behaving well without first being PROBATION")
    elif name == ResourceState.QUARANTINED.value:
        why = ("quarantined; it returns to PROBATION by rehab, and re-earns "
               "ACTIVE from there")
    elif name == ResourceState.RETIRED.value:
        why = "retired; not a candidate for trust"
    else:
        why = f"state {name!r} is not evidence-gated"
    return TrustDecision(resource_id, name, "hold", None,
                         f"holding: {why}", ev, pol.enabled)


__all__ = [
    "TRUST_ACTOR",
    "TrustDecision",
    "TrustPolicy",
    "assess",
    "read_ledger",
]
