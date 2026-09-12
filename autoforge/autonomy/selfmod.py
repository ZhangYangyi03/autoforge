"""Self-modification: the agent rewrites parts of itself, with a paper trail.

The agent can change:
  - its own system prompt
  - the forge config (max rounds, promotion policy, which checks run)
  - the routing weights (what it values when choosing tools)
  - the autonomy policy itself

Every change is an `Amendment`: a before/after diff, a rationale, a timestamp,
and an accepted/rejected verdict. Amendments are immutable once written. The
log is the accountability mechanism — freedom without memory of what you did
with it is just chaos.

A rejected amendment is kept too. Knowing what the agent *tried* to change is
as informative as knowing what it succeeded in changing.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Amendment:
    """One self-modification, recorded forever."""

    target: str                     # "system_prompt" | "forge_config" | ...
    rationale: str
    before: Any
    after: Any
    accepted: bool = True
    rejected_reason: str | None = None
    timestamp: float = field(default_factory=time.time)

    def is_noop(self) -> bool:
        return self.before == self.after

    def diff_summary(self) -> str:
        b = str(self.before)
        a = str(self.after)
        if len(b) > 120:
            b = b[:120] + "..."
        if len(a) > 120:
            a = a[:120] + "..."
        return f"{self.target}: {b!r} -> {a!r}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "rationale": self.rationale,
            "before": str(self.before)[:500],
            "after": str(self.after)[:500],
            "accepted": self.accepted,
            "rejected_reason": self.rejected_reason,
            "timestamp": self.timestamp,
        }


class SelfModifier:
    """Applies and records self-modifications against a host object."""

    def __init__(
        self,
        *,
        require_rationale: bool = True,
        log_all: bool = True,
        veto: Callable[[Amendment], str | None] | None = None,
    ) -> None:
        self.require_rationale = require_rationale
        self.log_all = log_all
        self.veto = veto
        self.amendments: list[Amendment] = []

    def amend(
        self,
        host: Any,
        target: str,
        new_value: Any,
        rationale: str = "",
        *,
        attr: str | None = None,
        nested: tuple[str, ...] = (),
    ) -> Amendment:
        """Change `target` on `host` to `new_value`, recording the amendment.

        `target` is a label for the log. `attr`/`nested` locate the real
        attribute — left unset, `target` is used as the attribute name.
        """
        if self.require_rationale and not rationale.strip():
            amend = Amendment(
                target=target, rationale=rationale, before=None, after=new_value,
                accepted=False, rejected_reason="rationale required but not provided",
            )
            self.amendments.append(amend)
            return amend

        # Resolve the container that holds the attribute
        holder = host
        for key in nested:
            holder = getattr(holder, key)

        attr_name = attr or target
        before = copy.deepcopy(getattr(holder, attr_name, None))

        amendment = Amendment(
            target=target, rationale=rationale, before=before, after=new_value,
        )

        if amendment.is_noop():
            amendment.accepted = False
            amendment.rejected_reason = "no-op (value unchanged)"
            self.amendments.append(amendment)
            return amendment

        if self.veto is not None:
            reason = self.veto(amendment)
            if reason:
                amendment.accepted = False
                amendment.rejected_reason = reason
                self.amendments.append(amendment)
                return amendment

        setattr(holder, attr_name, new_value)
        self.amendments.append(amendment)
        return amendment

    # -- introspection ---------------------------------------------------
    def log(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.amendments]

    def accepted_count(self) -> int:
        return sum(1 for a in self.amendments if a.accepted)

    def rejected_count(self) -> int:
        return sum(1 for a in self.amendments if not a.accepted)

    def last_rationale(self) -> str | None:
        for a in reversed(self.amendments):
            if a.accepted:
                return a.rationale
        return None


__all__ = ["SelfModifier", "Amendment"]
