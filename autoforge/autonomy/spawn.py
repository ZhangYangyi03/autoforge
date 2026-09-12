"""Spawning: the agent creates other agents, freely.

No pre-declared agent pool, no fixed orchestration graph. An agent decides at
runtime that a subtask wants its own worker, and spawns one. The child can
share the parent's tool registry (so tools forged by one are usable by all) or
get an isolated one (so experiments don't pollute the parent's library).

Two sharing modes:

  SHARED  — same registry object. A tool the child forges is immediately
            available to the parent. Fast convergence, but a bad tool
            propagates.
  ISOLATED — child gets a copy. It can forge freely; only tools that pass its
            verification AND the parent approves are merged back.

The parent chooses per-spawn. Default is SHARED, because the whole thesis is
that a growing shared library is the point — isolation is for when you want to
A/B a child's tool quality before trusting it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState


class ShareMode(str, Enum):
    SHARED = "shared"
    ISOLATED = "isolated"


@dataclass
class SpawnRecord:
    child_id: str
    task: str
    mode: str
    parent_id: str
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    result: str = ""
    tools_forged: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "task": self.task[:200],
            "mode": self.mode,
            "tools_forged": self.tools_forged,
            "duration_s": round(self.finished - self.started, 2) if self.finished else None,
            "error": self.error,
        }


@dataclass
class Spawner:
    """Creates child agents on demand.

    `agent_factory(parent, registry) -> agent-like` must return an object with
    a `.run(task) -> result` method whose result exposes `.content` and the
    child's registry (if it forged anything).
    """

    registry: ToolRegistry
    agent_factory: Callable[[Any, ToolRegistry], Any]
    parent_id: str = "root"
    spawn_count: int = 0
    history: list[SpawnRecord] = field(default_factory=list)
    max_depth: int = 8              # a limit the agent can raise, not a cage
    _depth: int = 0

    def spawn(
        self,
        task: str,
        *,
        mode: ShareMode = ShareMode.SHARED,
        child_registry: ToolRegistry | None = None,
    ) -> SpawnRecord:
        self.spawn_count += 1
        child_id = f"{self.parent_id}.{self.spawn_count}"

        record = SpawnRecord(
            child_id=child_id, task=task, mode=mode.value, parent_id=self.parent_id,
        )

        if self._depth >= self.max_depth:
            record.error = f"max spawn depth {self.max_depth} reached (raise it if needed)"
            record.finished = time.time()
            self.history.append(record)
            return record

        reg = self.registry if mode == ShareMode.SHARED else (child_registry or ToolRegistry())
        before = set(reg.names())

        try:
            child = self.agent_factory(self, reg)
            result = child.run(task)
            record.result = getattr(result, "content", str(result))
        except Exception as exc:  # noqa: BLE001
            record.error = f"{type(exc).__name__}: {exc}"

        after = set(reg.names())
        record.tools_forged = sorted(after - before)
        record.finished = time.time()
        self.history.append(record)
        return record

    def child_spawner(self, child_registry: ToolRegistry) -> "Spawner":
        """A spawner for a child, one level deeper."""
        return Spawner(
            registry=child_registry,
            agent_factory=self.agent_factory,
            parent_id=f"{self.parent_id}.{self.spawn_count}",
            max_depth=self.max_depth,
            _depth=self._depth + 1,
        )

    # -- merging isolated children back --------------------------------
    def merge_back(
        self,
        child_registry: ToolRegistry,
        *,
        require_state: ToolState = ToolState.ACTIVE,
        approve: Callable[[ToolSpec], bool] | None = None,
    ) -> list[str]:
        """Pull an isolated child's trusted tools into the parent library."""
        merged: list[str] = []
        for name, spec in child_registry._tools.items():
            if spec.state != require_state:
                continue
            if approve is not None and not approve(spec):
                continue
            self.registry.register(spec, replace=True)
            merged.append(name)
        return merged

    def summary(self) -> dict[str, Any]:
        return {
            "spawned": self.spawn_count,
            "max_depth": self.max_depth,
            "children": [r.to_dict() for r in self.history],
        }


__all__ = ["Spawner", "SpawnRecord", "ShareMode"]
