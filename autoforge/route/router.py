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


def text_similarity(query: str, spec: ToolSpec) -> float:
    """Lightweight lexical cosine — no embedding model, no network."""
    q = _tokens(query)
    d = _tokens(f"{spec.name} {spec.description} {' '.join(spec.tags)}")
    if not q or not d:
        return 0.0
    common = set(q) & set(d)
    num = sum(q[t] * d[t] for t in common)
    den = math.sqrt(sum(v * v for v in q.values())) * math.sqrt(sum(v * v for v in d.values()))
    return num / den if den else 0.0


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


__all__ = ["BehaviourRouter", "RoutingWeights", "RouteCandidate", "text_similarity"]
