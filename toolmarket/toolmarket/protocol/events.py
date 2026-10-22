"""Append-only event log — the audit substrate.

Every state change, evolution step and invocation leaves an immutable record.
The log is the source of truth you replay to reconstruct any resource's history;
the store's current-state tables are a *projection* of it.

Design choices that matter:

  * Monotonic `seq` — a total order, so "before/after" is never ambiguous even
    when timestamps collide under fast execution.
  * `kind` is a closed vocabulary (`EventKind`), so a consumer can switch on it
    exhaustively instead of guessing at free-form strings.
  * Records are frozen. You append; you never mutate. `verify_chain` re-derives
    each event's hash from its content and its predecessor, so tampering with a
    middle event is detectable.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterator, Optional


class EventKind(str, Enum):
    """The closed vocabulary of things that can happen to a resource."""

    REGISTER = "register"        # a resource first entered the substrate
    PROPOSE = "propose"          # an evolution was proposed (a DRAFT candidate)
    ASSESS = "assess"            # a candidate was evaluated (gate + verification)
    COMMIT = "commit"            # a candidate was accepted, a version advanced
    REJECT = "reject"            # a candidate was refused; the incumbent stays
    ROLLBACK = "rollback"        # a version was restored over the current one
    TRANSITION = "transition"    # a lifecycle state change (the enforcement axis)
    INVOKE = "invoke"            # the resource was called
    QUARANTINE = "quarantine"    # decayed trust
    REHAB = "rehab"              # trust partially restored
    RETIRE = "retire"            # withdrawn from service


@dataclass(frozen=True)
class Event:
    """One immutable fact. `prev` chains it to its predecessor."""

    seq: int
    kind: EventKind
    resource_id: str
    at: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    actor: str = "system"
    prev: Optional[str] = None      # hash of the previous event (chain)
    hash: str = ""                  # content hash, filled by EventLog.append

    def _digest(self) -> str:
        payload = {
            "seq": self.seq,
            "kind": self.kind.value,
            "resource_id": self.resource_id,
            "at": round(self.at, 6),
            "data": self.data,
            "actor": self.actor,
            "prev": self.prev,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        d = dict(d)
        d["kind"] = EventKind(d["kind"])
        return cls(**d)


def seal(
    kind: EventKind,
    resource_id: str,
    *,
    seq: int,
    prev: Optional[str],
    data: Optional[dict[str, Any]] = None,
    actor: str = "system",
    at: Optional[float] = None,
) -> Event:
    """Build and hash one event from an already-decided `(seq, prev)` pair.

    Split out of `append` because a store that allocates those two values
    *itself* — from the database, under a lock — still has to seal what it
    allocated, and sealing in two places is how one chain acquires two dialects.
    There is exactly one definition of what an event's hash covers.
    """
    draft = Event(
        seq=seq,
        kind=EventKind(kind),
        resource_id=resource_id,
        at=time.time() if at is None else at,
        data=data or {},
        actor=actor,
        prev=prev,
    )
    return replace(draft, hash=draft._digest())


class EventLog:
    """An append-only log. Persist it by wiring an `on_append` sink — or, when
    more than one process appends to it, an `on_reserve` sink.

    The sink is the single point at which an event becomes durable, so *every*
    append — from the registry, from the evolution operator, from a future
    module — is persisted without each caller remembering to. A log that only
    persisted when you called the right helper would silently lose the events
    that matter most (the ones an evolution produced).

    What the sink does *not* get to do is choose `seq`. This object's `_events`
    is a mirror of what this process has seen, and in a multi-process deployment
    it is a partly stale one; a `seq` derived from its length is a claim about a
    database that this object cannot see. See `append`.
    """

    def __init__(
        self,
        events: Optional[list[Event]] = None,
        *,
        on_append: Optional[Callable[["Event"], None]] = None,
        on_reserve: Optional[Callable[[Callable[[int, Optional[str]], "Event"]], "Event"]] = None,
    ) -> None:
        self._events: list[Event] = list(events or [])
        self.on_append: Optional[Callable[["Event"], None]] = on_append
        self.on_reserve: Optional[
            Callable[[Callable[[int, Optional[str]], "Event"]], "Event"]
        ] = on_reserve

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    @property
    def head(self) -> Optional[str]:
        return self._events[-1].hash if self._events else None

    def append(
        self,
        kind: EventKind,
        resource_id: str,
        *,
        data: Optional[dict[str, Any]] = None,
        actor: str = "system",
    ) -> Event:
        """Append one event, sealed — durable first, mirrored second.

        Who decides `seq` and `prev` is the whole question, and there are exactly
        two answers:

          * `on_reserve` — the *store* decides, inside its own write
            transaction, and calls the builder it is handed with the `(seq, prev)`
            it allocated. This is the only arrangement that is correct when more
            than one process appends: two processes each taking `seq` from their
            own mirror both call themselves event 29, and the second INSERT dies
            on the primary key. With the store allocating, this object's memory
            is a cache of the tail rather than the source of the counter.
          * otherwise the log decides — `seq` is its length, `prev` its head —
            and `on_append` persists the result. Correct for a bare log and for a
            store only one process ever writes, and it costs no round trip.

        The mirror is updated *after* the sink returns, so an append that the
        store refused does not linger in memory pretending to have happened.
        """
        if self.on_reserve is not None:
            sealed = self.on_reserve(
                lambda seq, prev: seal(
                    kind, resource_id, seq=seq, prev=prev, data=data, actor=actor
                )
            )
        else:
            sealed = seal(
                kind,
                resource_id,
                seq=len(self._events),
                prev=self.head,
                data=data,
                actor=actor,
            )
            if self.on_append is not None:
                self.on_append(sealed)
        self._events.append(sealed)
        return sealed

    def for_resource(self, resource_id: str) -> list[Event]:
        return [e for e in self._events if e.resource_id == resource_id]

    def of_kind(self, kind: EventKind) -> list[Event]:
        kind = EventKind(kind)
        return [e for e in self._events if e.kind == kind]

    def tail(self, n: int = 50) -> list[Event]:
        return self._events[-n:]

    def verify_chain(self) -> bool:
        """Re-derive every hash. Returns False if any event was tampered with."""
        prev: Optional[str] = None
        for e in self._events:
            if e.prev != prev:
                return False
            if e.hash != e._digest():
                return False
            prev = e.hash
        return True
