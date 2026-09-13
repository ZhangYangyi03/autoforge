"""Behaviour-aligned tool routing.

Mainstream skill libraries retrieve by *text similarity*: embed the need,
embed each skill description, take the nearest. This routinely surfaces a
skill that reads right and runs wrong, because semantic closeness is not
execution success.

The fix (credited to Memento-Skills, generalised here) is to score candidates
on *observed behaviour*:
    score = w_text * text_similarity
          + w_success * success_rate
          + w_trust * state_trust
          - w_cost * cost_penalty
          - w_fire * over_trigger_penalty

No offline RL required — a weighted, inspectable blend. The weights are the
policy, and they live in one dataclass so the agent's retrieval taste is a
tunable, debuggable thing rather than an opaque vector index.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState

_WORD = re.compile(r"[a-zA-Z0-9_]+")

_TRUST = {
    ToolState.ACTIVE: 1.0,
    ToolState.PROBATION: 0.6,
    ToolState.QUARANTINED: 0.1,
    ToolState.RETIRED: 0.0,
    ToolState.DRAFT: 0.2,
}

_COST = {"cheap": 0.0, "moderate": 0.15, "expensive": 0.4}


def _tokens(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD.findall(text or "") if len(w) > 1)


def _cosine(q: Counter, d: Counter) -> float:
    """Shared by tool and skill similarity: no embedding model, no network."""
    if not q or not d:
        return 0.0
    common = set(q) & set(d)
    num = sum(q[t] * d[t] for t in common)
    den = math.sqrt(sum(v * v for v in q.values())) * math.sqrt(sum(v * v for v in d.values()))
    return num / den if den else 0.0


def text_similarity(query: str, spec: ToolSpec) -> float:
    """Lightweight lexical cosine — no embedding model, no network."""
    return _cosine(_tokens(query),
                   _tokens(f"{spec.name} {spec.description} {' '.join(spec.tags)}"))


@dataclass
class RoutingWeights:
    text: float = 1.0
    success: float = 1.2
    trust: float = 0.8
    cost: float = 0.5
    over_trigger: float = 0.6
    min_calls_for_success: int = 3

    def to_dict(self) -> dict[str, float]:
        return {
            "text": self.text, "success": self.success, "trust": self.trust,
            "cost": self.cost, "over_trigger": self.over_trigger,
        }


@dataclass
class RouteCandidate:
    name: str
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    state: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "state": self.state,
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
        }


class BehaviourRouter:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        weights: RoutingWeights | None = None,
        include_states: set[ToolState] | None = None,
    ) -> None:
        self.registry = registry
        self.weights = weights or RoutingWeights()
        self.include_states = include_states or {ToolState.PROBATION, ToolState.ACTIVE}

    def score(self, query: str, spec: ToolSpec) -> RouteCandidate:
        w = self.weights
        st = spec.stats

        text = text_similarity(query, spec)
        success = st.success_rate if st.calls >= w.min_calls_for_success else 0.5
        trust = _TRUST.get(spec.state, 0.0)
        cost = _COST.get(spec.cost_hint, 0.0)
        # over-trigger penalty: high call count relative to distinct needs is a
        # proxy for a tool that fires when it should not.
        over = 0.0
        if st.calls:
            over = min(1.0, st.trigger_misses / max(st.calls, 1))

        total = (
            w.text * text
            + w.success * success
            + w.trust * trust
            - w.cost * cost
            - w.over_trigger * over
        )
        return RouteCandidate(
            name=spec.name,
            score=total,
            breakdown={
                "text": w.text * text,
                "success": w.success * success,
                "trust": w.trust * trust,
                "cost": -w.cost * cost,
                "over_trigger": -w.over_trigger * over,
            },
            state=spec.state.value,
        )

    def rank(self, query: str) -> list[RouteCandidate]:
        cands = [
            self.score(query, s)
            for s in self.registry._tools.values()
            if s.state in self.include_states
        ]
        return sorted(cands, key=lambda c: c.score, reverse=True)

    def route(self, query: str, k: int = 3) -> list[str]:
        """Names of the top-k tools for this query, best first."""
        return [c.name for c in self.rank(query)[:k]]


def skill_similarity(query: str, skill: Any) -> float:
    """Lexical cosine over the fields a skill is described by.

    `when_to_use` carries the weight a description does not: it is written as a
    trigger ("when a deploy needs proving"), which is the shape of the thing
    being matched, rather than as a summary of the thing itself.
    """
    fields = (f"{skill.name} {skill.description} {skill.when_to_use} "
              f"{' '.join(skill.tags)}")
    return _cosine(_tokens(query), _tokens(fields))


class SkillRouter:
    """Rank skills by fit, blended with evidence that they have been used.

    The lexical half is honest about what it is: at routing time, the only
    cheap signal about a skill is how its own words line up with the need. What
    makes the ranking *behavioural* is the second term. `proven` is built from
    `loads` — how many times the agent actually opened the skill — so a
    procedure that keeps getting reached for climbs above one that merely reads
    well. That is the failure this repo exists to argue against: nearest-
    description retrieval returns the skill that reads right and runs wrong.

    Two of the tool router's four terms are deliberately absent. A skill has no
    state lifecycle to score trust from and no cost hint, so including them
    would mean inventing numbers and calling the result a measurement. Where
    the tool router says "not measured yet" (0.5 for too-few calls), the skill
    router says zero: a never-loaded skill has no evidence, and rounding that
    up to neutral would let an unread procedure tie a proven one.
    """

    def __init__(self, library: Any, *, weights: RoutingWeights | None = None) -> None:
        self.library = library
        self.weights = weights or RoutingWeights()

    def score(self, query: str, skill: Any) -> RouteCandidate:
        w = self.weights
        text = skill_similarity(query, skill)
        proven = min(1.0, skill.loads / max(1, w.min_calls_for_success))
        total = w.text * text + w.success * proven
        return RouteCandidate(
            name=skill.name,
            score=total,
            breakdown={"text": w.text * text, "proven": w.success * proven},
            state="used" if skill.loads else "never-used",
        )

    def rank(self, query: str) -> list[RouteCandidate]:
        cands = [self.score(query, s) for s in self.library.all()]
        # Ties go to the more-used skill, then to name order, so the same
        # library always ranks the same way and a diff means something.
        cands.sort(key=lambda c: (-c.score, c.name))
        return cands

    def route(self, query: str, k: int = 3) -> list[str]:
        """Names of the top-k skills for this need, best first."""
        return [c.name for c in self.rank(query)[:k]]


__all__ = [
    "BehaviourRouter", "RoutingWeights", "RouteCandidate", "text_similarity",
    "SkillRouter", "skill_similarity",
]
