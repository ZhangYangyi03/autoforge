"""Trade arbitration: the call site the behavioural router never had.

`BehaviourRouter` was written, weighted, documented and amendable -- and never
asked. The ledger on 2026-09-23 says it plainly: seven references to routing
weights, and zero calls to `route()` or `rank()` from anywhere in the package.
The router had tests, and the tests passed, because a ranking can be asserted
without ever sitting on a path a real turn takes. A scored ranking nothing
consults is an ornament.

This is the same defect the pre-forge market lookup had, one level down: the
principle ("look before you build") was in the prompt and the call site was
missing, and it took 158 forges with zero lookups to notice. Advice appended to
a prompt is not a decision. This module makes the decision.

What it decides: of the tools that already exist, does one of them own this
need? If one does, the need is *attended to* -- routed to what exists -- and the
construction of a new tool is refused. That is the whole mechanism, and it is
not a metaphor: the arithmetical difference is that a forge which would have
produced the ninth silent replacement produces nothing, and the existing tool
gets called instead.

The waste this exists to stop is measured, not hypothesised:
  - 120 tools owned, 293 forge attempts against them (2.4 attempts per tool);
  - 8 accepted forges landed on a name already taken and replaced it silently,
    because `registry.register` defaults to `replace=True` and nothing on the
    forge path asked whether the name was already someone's;
  - `chrome_cdp_drive` reached version 3 this way, each version overwriting the
    last with no refusal and no sign in the reply that a tool was destroyed;
  - on the live shelf, for a need whose tool already existed, the right tool sat
    at median rank 64 of 120 in the comma-separated list the forge was handed.

Ranking is not the same as deciding, so the veto does not fire on rank alone.
It requires a claim about *name*, because a name is a statement of what a tool
is for, while a description is a sentence that happens to contain words -- and
this codebase already paid for that lesson (`repo` matching `report`,
`host_artifact_scan` named as an overlap for a question about processes). So:
the name anchor plus a score floor. Both halves are needed; either one alone
either never fires or cries duplicate on every forge.

The refusal is escapable and says how. A veto is not a wall: if the existing
tool is broken the answer is `evolve_tool`, and if the capability genuinely
differs then the difference is the thing that licenses a new tool, so stating
it in the need changes the need -- and the changed need is what it forges under.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# The name rule, kept here rather than imported from the agent so the arbiter has
# no dependency on the agent (which imports the arbiter). `autoforge.agent` keeps
# its own copy inside `_lexical_name_hits` for the market gate; a test pins the
# two to agree on the cases that codebase already paid for, so they cannot drift
# apart without a failure naming it.
_NAME_STOPWORDS = frozenset(
    "the and for with from into when then your have will does each some more "
    "this that need tool which what over under than make made used using".split())
_SUFFIXES = ("ing", "es", "s", "ed")


def canon_token(token: str) -> str:
    """One word reduced to what it means, not how it ends.

    The endings are a closed list on purpose: an open one is a spell-checker,
    and the job here is equality after reduction, never a prefix test.
    """
    for suffix in _SUFFIXES:
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def name_tokens(text: str) -> list[str]:
    return [canon_token(t) for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]


def content_tokens(text: str) -> set[str]:
    """Name tokens that are about a capability rather than about English."""
    return {t for t in name_tokens(text)
            if len(t) >= 3 and t not in _NAME_STOPWORDS}


def name_anchor(need: str, name: str) -> int:
    """How many of a tool name's own words the need says, 0 when none.

    The whole name appearing in the need counts as every word of it, so a need
    that names the tool verbatim always anchors.
    """
    need_norm = " ".join(name_tokens(need))
    if str(name).lower().replace("_", " ") in need_norm:
        return len(content_tokens(name)) or 1
    return len(content_tokens(name) & content_tokens(need))


@dataclass
class Candidate:
    name: str
    score: float
    why: list[str] = field(default_factory=list)
    state: str = ""
    description: str = ""
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 4),
                "why": list(self.why), "state": self.state}


@dataclass
class Arbitration:
    need: str
    ranked: list[Candidate]
    veto: Candidate | None = None
    reason: str = ""

    @property
    def refused(self) -> bool:
        return self.veto is not None

    def context_block(self) -> str:
        """The ranked competitors, rendered for the forge's context.

        This is the second half of the call site: the router's ranking, in the
        prompt at the moment a tool is being written, ordered by fit rather than
        by insertion order. The generator already received *every* name; what it
        never received was the order, and order is the only thing that makes a
        list of 120 names information.
        """
        if not self.ranked:
            return ""
        lines = ["Tools you already hold, ranked for THIS need (best fit first):"]
        for c in self.ranked:
            who = "+".join(c.why)
            desc = (c.description or "").strip().replace("\n", " ")[:96]
            lines.append("  - %s [%s, score %.2f, %s] %s"
                         % (c.name, c.state or "?", c.score, who, desc))
        lines.append("If one of these serves the need, call it and do not build a "
                     "second one beside it. Build only what none of them does, and "
                     "say in the need what is missing from the best of them.")
        return "\n".join(lines)

    def refusal(self) -> str:
        v = self.veto
        assert v is not None
        others = [c for c in self.ranked[1:4]]
        lines = [
            "Refused before forging, on evidence rather than on principle: "
            "%r already owns this need in this library, so a forge here would be "
            "a second tool beside it -- which is how this shelf reached 120 tools "
            "with 8 accepted forges silently replacing a name that was already "
            "taken." % v.name,
            "",
            "  - %s [%s, score %.2f, %s] %s"
            % (v.name, v.state or "?", v.score, "+".join(v.why),
               (v.description or "")[:110]),
        ]
        for c in others:
            lines.append("  - %s [%s, score %.2f, %s]"
                         % (c.name, c.state or "?", c.score, "+".join(c.why)))
        lines += [
            "",
            "Do one of these instead, and say which in your reply:",
            "  - call %s with its arguments, if it serves the need;" % v.name,
            "  - if it is broken or nearly right, evolve_tool(%r, "
            "'<what is wrong>') rather than building a second one beside "
            "it;" % v.name,
            "  - if the capability genuinely differs, re-state the need as what "
            "is *different* about it -- the difference is what licenses a new "
            "tool, so put it in the need text and forge again.",
            "",
            "This refusal is recorded on the ledger with the name that won. Do "
            "not ask for the same forge again unchanged.",
        ]
        return "\n".join(lines)


class Arbiter:
    """Decide whether an existing tool owns the need, using the live router.

    Scoring is delegated to `BehaviourRouter`, so the weights the agent can
    amend with `amend_self(routing_weights=...)` are the weights that make this
    decision. That is the point: a routing weight that changes nothing is a
    number in a dataclass, and there were seven of them.

    The name anchor is computed here because the router does not model it, and
    because a name is the cheaper claim to defend: `list_agent_processes` is a
    statement about what a tool does, while any sentence may mention processes.
    """

    def __init__(self, router: Any, registry: Any, *,
                 veto_floor: float = 0.12, min_name_words: int = 2) -> None:
        self.router = router
        self.registry = registry
        self.veto_floor = veto_floor
        self.min_name_words = min_name_words

    def _eligible(self, name: str) -> bool:
        spec = self.registry.get(name)
        if spec is None:
            return False
        # A retired or quarantined tool is not evidence of coverage: pointing a
        # need at one would be routing to something the agent already judged
        # unfit, which is worse than forging.
        state = getattr(getattr(spec, "state", None), "value", "") or ""
        return state in ("active", "probation")

    def decide(self, need: str, *, k: int = 5) -> Arbitration:
        try:
            cands = self.router.rank(need)
        except Exception:                      # noqa: BLE001 - never block a forge
            return Arbitration(need=need, ranked=[],
                               reason="the router could not rank this need")

        ranked: list[Candidate] = []
        for c in cands:
            if not self._eligible(c.name):
                continue
            spec = self.registry.get(c.name)
            words = name_anchor(need, c.name)
            why = []
            if words:
                why.append("name")
            if c.breakdown.get("text"):
                why.append("text")
            if c.breakdown.get("success"):
                why.append("success")
            ranked.append(Candidate(
                name=c.name, score=float(c.score), why=why or ["ranked"],
                state=getattr(getattr(spec, "state", None), "value", ""),
                description=getattr(spec, "description", "") or "",
                breakdown=dict(c.breakdown or {})))
            if len(ranked) >= max(k, 5):
                break

        veto = None
        reason = "no existing tool owns this need"
        for c in ranked:
            spec = self.registry.get(c.name)
            words = name_anchor(need, c.name)
            whole = (" ".join(name_tokens(need))
                     == " ".join(name_tokens(c.name)))
            if (whole or words >= self.min_name_words) and c.score >= self.veto_floor:
                veto = c
                reason = ("%r is named by this need and scores %.2f (floor %.2f)"
                          % (c.name, c.score, self.veto_floor))
                break
        return Arbitration(need=need, ranked=ranked, veto=veto, reason=reason)


__all__ = ["Arbiter", "Arbitration", "Candidate", "canon_token", "name_tokens",
           "content_tokens", "name_anchor"]
