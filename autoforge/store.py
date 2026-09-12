"""SQLite-backed persistence for the tool registry.

Every tool version, every forge attempt, every dependency link — survives
restarts. Auto-creates the DB on first use.

Schema is a flat tool table plus an append-only event log. JSON columns for
structured fields avoid schema migrations while keeping the critical fields
(first-class columns) queryable.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools.spec import ToolSpec, ToolState, ToolStats, TriggerProbe

_DEFAULT_DB = "autoforge.db"

# -- column-level constants (used by schema generation) -------------
_COLUMNS = dict(
    name="TEXT NOT NULL",
    version="INTEGER DEFAULT 1",
    description="TEXT NOT NULL DEFAULT ''",
    parameters="TEXT NOT NULL DEFAULT '{}'",
    code="TEXT NOT NULL DEFAULT ''",
    entry="TEXT NOT NULL DEFAULT ''",
    source="TEXT NOT NULL DEFAULT 'human'",
    generator="TEXT NOT NULL DEFAULT ''",
    probes="TEXT NOT NULL DEFAULT '[]'",
    effect_signature="TEXT NOT NULL DEFAULT ''",
    state="TEXT NOT NULL DEFAULT 'draft'",
    tags="TEXT NOT NULL DEFAULT '[]'",
    cost_hint="TEXT NOT NULL DEFAULT 'cheap'",
    created_at="REAL NOT NULL",
    verification="TEXT NOT NULL DEFAULT '{}'",
    stats="TEXT NOT NULL DEFAULT '{}'",
    old_versions="TEXT NOT NULL DEFAULT '[]'",
    updated_at="REAL NOT NULL",
)
_DDL = (
    "CREATE TABLE IF NOT EXISTS tools (\n"
    + "\n".join(f"    {name} {dtype}," for name, dtype in _COLUMNS.items())
    + "\n    PRIMARY KEY (name)\n);"
    "\nCREATE TABLE IF NOT EXISTS forge_events ("
    "\n    id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "\n    timestamp REAL NOT NULL,"
    "\n    kind TEXT NOT NULL,"
    "\n    payload TEXT NOT NULL DEFAULT '{}'"
    "\n);"
    "\nCREATE TABLE IF NOT EXISTS dependencies ("
    "\n    tool TEXT NOT NULL,"
    "\n    depends_on TEXT NOT NULL,"
    "\n    PRIMARY KEY (tool, depends_on)"
    "\n);"
    "\nCREATE TABLE IF NOT EXISTS baselines ("
    "\n    tool TEXT NOT NULL PRIMARY KEY,"
    "\n    frozen_at REAL NOT NULL,"
    "\n    baseline TEXT NOT NULL DEFAULT '{}'"
    "\n);"
    "\nCREATE TABLE IF NOT EXISTS topologies ("
    "\n    id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "\n    timestamp REAL NOT NULL,"
    "\n    task TEXT NOT NULL DEFAULT '',"
    "\n    topology TEXT NOT NULL DEFAULT '{}',"
    "\n    fitness REAL NOT NULL DEFAULT 0.0,"
    "\n    trials INTEGER NOT NULL DEFAULT 0"
    "\n);"
)


def _now() -> float:
    return time.time()


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _unjson(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _probes_from_list(items: list[dict]) -> list[TriggerProbe]:
    return [
        TriggerProbe(
            query=p.get("query", ""),
            expect=p.get("expect", "call"),
            negative_query=p.get("negative_query"),
        )
        for p in items
        if isinstance(p, dict) and p.get("query")
    ]


# ---------------------------------------------------------------------------
@dataclass
class ToolRecord:
    """Flat persisted representation. Converted to/from ToolSpec."""

    name: str
    version: int
    description: str
    parameters: dict[str, Any]
    code: str
    entry: str
    source: str
    generator: str
    probes: list[TriggerProbe]
    effect_signature: str
    state: ToolState
    tags: list[str]
    cost_hint: str
    created_at: float
    verification: dict[str, Any]
    stats: dict[str, Any]
    old_versions: list[dict[str, Any]]
    updated_at: float

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> "ToolRecord":
        return cls(
            name=spec.name,
            version=1,
            description=spec.description,
            parameters=spec.parameters,
            code=spec.code,
            entry="",
            source=spec.source,
            generator=spec.generator,
            probes=spec.probes,
            effect_signature=spec.effect_signature,
            state=spec.state,
            tags=spec.tags,
            cost_hint=spec.cost_hint,
            created_at=spec.created_at,
            verification=spec.verification,
            stats=spec.stats.to_dict(),
            old_versions=[],
            updated_at=_now(),
        )

    def to_spec(self, fn=None, runner=None) -> ToolSpec:
        st = ToolStats()
        for k, v in self.stats.items():
            if hasattr(st, k):
                setattr(st, k, v)
        spec = ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            fn=fn or (lambda **_: ""),
            runner=runner,
            code=self.code,
            source=self.source,
            generator=self.generator,
            probes=self.probes,
            effect_signature=self.effect_signature,
            state=self.state,
            stats=st,
            verification=self.verification,
            tags=self.tags,
            cost_hint=self.cost_hint,
            created_at=self.created_at,
        )
        return spec


# ---------------------------------------------------------------------------
class ToolStore:
    """SQLite-backed tool registry persistence."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = str(db_path or _DEFAULT_DB)
        home = os.environ.get("AUTOFORGE_HOME", ".")
        if not os.path.isabs(self.db_path):
            self.db_path = os.path.join(home, self.db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DDL)

    def close(self) -> None:
        self._conn.close()

    # -- tools ------------------------------------------------------------
    def save_tool(self, spec: ToolSpec) -> None:
        record = ToolRecord.from_spec(spec)
        self._conn.execute(
            """INSERT OR REPLACE INTO tools
            (name, version, description, parameters, code, entry,
             source, generator, probes, effect_signature, state,
             tags, cost_hint, created_at, verification, stats,
             old_versions, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.name, record.version, record.description,
                _j(record.parameters), record.code, record.entry,
                record.source, record.generator, _j([
                    {"query": p.query, "expect": p.expect, "negative_query": p.negative_query}
                    for p in record.probes
                ]),
                record.effect_signature, record.state.value,
                _j(record.tags), record.cost_hint,
                record.created_at, _j(record.verification),
                _j(record.stats), _j(record.old_versions),
                record.updated_at,
            ),
        )
        self._conn.commit()

    def load_all_tools(self) -> dict[str, ToolRecord]:
        rows = self._conn.execute(
            "SELECT * FROM tools ORDER BY name"
        ).fetchall()
        result: dict[str, ToolRecord] = {}
        for r in rows:
            try:
                result[r["name"]] = ToolRecord(
                    name=r["name"],
                    version=r["version"],
                    description=r["description"],
                    parameters=_unjson(r["parameters"]),
                    code=r["code"],
                    entry=r["entry"],
                    source=r["source"],
                    generator=r["generator"],
                    probes=_probes_from_list(_unjson(r["probes"])),
                    effect_signature=r["effect_signature"],
                    state=ToolState(r["state"]),
                    tags=_unjson(r["tags"]),
                    cost_hint=r["cost_hint"],
                    created_at=r["created_at"],
                    verification=_unjson(r["verification"]),
                    stats=_unjson(r["stats"]),
                    old_versions=_unjson(r["old_versions"]),
                    updated_at=r["updated_at"],
                )
            except Exception:  # noqa: BLE001
                continue  # skip corrupted rows
        return result

    def delete_tool(self, name: str) -> None:
        self._conn.execute("DELETE FROM tools WHERE name=?", (name,))
        self._conn.execute("DELETE FROM dependencies WHERE tool=? OR depends_on=?",
                           (name, name))
        self._conn.commit()

    # -- versions ---------------------------------------------------------
    def archive_version(self, name: str, code: str, verification: dict[str, Any]) -> None:
        """Push the current code+verification into old_versions for rollback."""
        row = self._conn.execute("SELECT old_versions, version FROM tools WHERE name=?",
                                 (name,)).fetchone()
        if row is None:
            return
        versions = _unjson(row["old_versions"])
        versions.append({
            "version": row["version"],
            "code": code,
            "verification": verification,
            "archived_at": _now(),
        })
        self._conn.execute(
            "UPDATE tools SET old_versions=?, version=version+1, updated_at=? WHERE name=?",
            (_j(versions), _now(), name),
        )
        self._conn.commit()

    # -- dependencies -----------------------------------------------------
    def save_deps(self, tool: str, depends_on: list[str]) -> None:
        self._conn.execute("DELETE FROM dependencies WHERE tool=?", (tool,))
        for dep in depends_on:
            if dep == tool:
                continue
            self._conn.execute(
                "INSERT OR IGNORE INTO dependencies (tool, depends_on) VALUES (?,?)",
                (tool, dep),
            )
        self._conn.commit()

    def get_deps(self, tool: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT depends_on FROM dependencies WHERE tool=? ORDER BY depends_on",
            (tool,),
        ).fetchall()
        return [r["depends_on"] for r in rows]

    def get_reverse_deps(self, tool: str) -> list[str]:
        """Everything that depends on `tool` (for cascade checks)."""
        rows = self._conn.execute(
            "SELECT tool FROM dependencies WHERE depends_on=? ORDER BY tool",
            (tool,),
        ).fetchall()
        return [r["tool"] for r in rows]

    # -- frozen baselines -------------------------------------------------
    def save_baseline(self, baseline: Any) -> None:
        """Persist a tool's frozen exam. Upsert: the row is the obligation set."""
        self._conn.execute(
            "INSERT INTO baselines (tool, frozen_at, baseline) VALUES (?,?,?)"
            " ON CONFLICT(tool) DO UPDATE SET"
            " frozen_at=excluded.frozen_at, baseline=excluded.baseline",
            (baseline.tool, float(baseline.frozen_at), _j(baseline.to_dict())),
        )
        self._conn.commit()

    def load_baseline(self, tool: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT baseline FROM baselines WHERE tool = ?", (tool,)
        ).fetchone()
        return _unjson(row["baseline"]) if row else None

    # -- topologies -------------------------------------------------------
    def save_topology(self, topology: Any, task: str = "") -> None:
        """Persist a designed multi-agent topology (nodes, edges, rationale)."""
        self._conn.execute(
            "INSERT INTO topologies (timestamp, task, topology, fitness, trials)"
            " VALUES (?,?,?,?,?)",
            (
                _now(),
                (task or "")[:500],
                _j(topology.to_dict()),
                float(getattr(topology, "fitness", 0.0) or 0.0),
                int(getattr(topology, "trials", 0) or 0),
            ),
        )
        self._conn.commit()

    def load_topologies(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent topologies first, newest at index 0."""
        rows = self._conn.execute(
            "SELECT * FROM topologies ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "task": r["task"],
                "fitness": r["fitness"],
                "trials": r["trials"],
                "topology": _unjson(r["topology"]),
            }
            for r in rows
        ]

    # -- events -----------------------------------------------------------
    def log_event(self, kind: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO forge_events (timestamp, kind, payload) VALUES (?,?,?)",
            (_now(), kind, _j(payload)),
        )
        self._conn.commit()

    def get_events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM forge_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for r in reversed(rows):
            entry: dict[str, Any] = {"id": r["id"], "timestamp": r["timestamp"], "kind": r["kind"]}
            entry.update(_unjson(r["payload"]))
            results.append(entry)
        return results


__all__ = ["ToolStore", "ToolRecord"]