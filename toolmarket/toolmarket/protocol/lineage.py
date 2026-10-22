"""Lineage — who evolved from whom, and why.

AGP calls for "auditable lineage" and "atomic rollback"; this is the data
structure that makes both meaningful. A resource is not a point in time; it is a
node in a DAG whose edges record every accepted evolution.

The graph is intentionally simple and answerable in O(edges):

  * `ancestors(node)`   — the full ancestry of the current version
  * `descendants(node)` — everything a version spawned (what a rollback orphans)
  * `path(a, b)`        — a concrete evolutionary route between two versions
  * `roots()`           — resources with no parent: the original registrations

A rollback moves the *live pointer*; it never deletes a node. The DAG is
append-only, exactly like the event log, so history survives the rollback that
history made necessary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LineageNode:
    """One version of one resource."""

    node_id: str                 # e.g. "tool:slugify@v2"
    resource_id: str             # e.g. "tool:slugify"
    version: str                 # semver string, e.g. "1.2.0"
    parents: list[str] = field(default_factory=list)
    reason: str = ""             # the goal / failure that motivated this version
    mutation: str = "register"   # register | mutate | crossover | rollback
    verification: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "resource_id": self.resource_id,
            "version": self.version,
            "parents": list(self.parents),
            "reason": self.reason,
            "mutation": self.mutation,
            "verification": self.verification,
            "created_at": self.created_at,
        }


class LineageGraph:
    """A DAG of versions. Append-only; rollback moves pointers, never deletes."""

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def add(self, node: LineageNode) -> LineageNode:
        if node.node_id in self._nodes:
            raise ValueError(f"lineage node already exists: {node.node_id}")
        for p in node.parents:
            if p not in self._nodes:
                raise ValueError(
                    f"node {node.node_id} claims unknown parent {p}; "
                    "lineage must be added parents-first"
                )
        self._nodes[node.node_id] = node
        return node

    def get(self, node_id: str) -> Optional[LineageNode]:
        return self._nodes.get(node_id)

    def of_resource(self, resource_id: str) -> list[LineageNode]:
        return sorted(
            (n for n in self._nodes.values() if n.resource_id == resource_id),
            key=lambda n: (n.created_at, n.node_id),
        )

    def roots(self) -> list[LineageNode]:
        return [n for n in self._nodes.values() if not n.parents]

    def children(self, node_id: str) -> list[LineageNode]:
        return [n for n in self._nodes.values() if node_id in n.parents]

    def ancestors(self, node_id: str) -> list[LineageNode]:
        """Every node reachable by walking parents up. Post-order, deduped."""
        seen: dict[str, LineageNode] = {}
        stack = [node_id]
        while stack:
            cur = stack.pop()
            node = self._nodes.get(cur)
            if node is None:
                continue
            for p in node.parents:
                if p not in seen:
                    seen[p] = self._nodes[p]
                    stack.append(p)
        return list(seen.values())

    def descendants(self, node_id: str) -> list[LineageNode]:
        """Everything a rollback of this node away from would orphan."""
        seen: dict[str, LineageNode] = {}
        stack = [node_id]
        while stack:
            cur = stack.pop()
            for child in self.children(cur):
                if child.node_id not in seen:
                    seen[child.node_id] = child
                    stack.append(child.node_id)
        return list(seen.values())

    def path(self, src: str, dst: str) -> Optional[list[str]]:
        """One concrete route src -> dst walking edges *down* (parent->child)."""
        if src not in self._nodes or dst not in self._nodes:
            return None
        queue: list[list[str]] = [[src]]
        visited = {src}
        while queue:
            route = queue.pop(0)
            cur = route[-1]
            if cur == dst:
                return route
            for child in self.children(cur):
                if child.node_id not in visited:
                    visited.add(child.node_id)
                    queue.append(route + [child.node_id])
        return None

    def to_dict(self) -> dict[str, Any]:
        return {nid: n.to_dict() for nid, n in self._nodes.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LineageGraph":
        g = cls()
        g._nodes = {nid: LineageNode(**node) for nid, node in d.items()}
        return g
