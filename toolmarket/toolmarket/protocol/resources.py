"""The RSPL record: a tool as a protocol-registered resource.

AGP's Resource Substrate Protocol Layer models prompts, agents, tools,
environments and memory as *resources* — each with explicit state, lifecycle,
versioned interface and auditable lineage. This module gives the **tool**
resource type its concrete record.

The mapping to `autoforge` is 1:1 on every field that matters, and lossless:
nothing autoforge records is dropped, and nothing here is invented to look
fuller than the backing system can support.

    autoforge ToolSpec          toolmarket ResourceRecord
    --------------------------  -------------------------------------------
    name                        ResourceRecord.name (id = "tool:<name>")
    description                 ResourceRecord.description
    parameters (.schema)        ToolContract.parameters
    code                        ToolContract.code
    probes                      ToolContract.probes
    effect_signature            ToolContract.effect_signature
    invariances                 ToolContract.invariances
    verification                ToolContract.verification
    source / generator          ResourceRecord.provenance
    tags / cost_hint            ResourceRecord.metadata
    state (ToolState)           ResourceRecord.state      (enforcement axis)
    stats (ToolStats)           ResourceRecord.ledger     (the live ledger)
    hash                        ResourceVersion.content_hash

AGP-aligned fields that have no autoforge equivalent (`enable_evolving`,
`permission_mode`, `progress_policy`) are carried explicitly, defaulted to the
conservative value, and documented as AGP-native rather than papered over.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from toolmarket.protocol.lifecycle import ResourceState, VersionStatus


class ResourceType(str, Enum):
    """The resource types AGP's substrate registers.

    This list is taken from AGP's own registry (`autogenesis/registry.py`),
    which registers exactly these categories. `tool` is the one this project
    implements end-to-end; the rest are present so a record can *name* them and
    a future module can own them without a schema change.
    """

    TOOL = "tool"
    AGENT = "agent"
    PROMPT = "prompt"
    MEMORY_SYSTEM = "memory_system"
    ENVIRONMENT = "environment"
    SKILL = "skill"
    KNOWLEDGE = "knowledge"
    COMMAND = "command"
    HOOK = "hook"
    CONSTRAINT = "constraint"
    SANDBOX = "sandbox"
    WORKFLOW = "workflow"
    DATASET = "dataset"
    BENCHMARK = "benchmark"
    PLUGIN = "plugin"
    PROCESS = "process"


# Permission modes AGP defines on a tool. Default is the middle rung: a tool may
# write within its workspace but may not, on its own authority, reach further.
PERMISSION_MODES = ("read_only", "workspace_write", "danger_full_access")

# AGP's no-progress policy vocabulary for repeated calls.
PROGRESS_POLICIES = ("workspace", "external", "polling", "always")


@dataclass
class ToolContract:
    """A tool's birth contract: schema + probes + effect signature + verification.

    This is the part AGP frames as the "versioned interface" and autoforge
    frames as "the contract a tool is born with". Same object, two vocabularies.
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    code: str = ""
    probes: list[dict[str, Any]] = field(default_factory=list)
    effect_signature: str = ""
    invariances: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "code": self.code,
            "probes": self.probes,
            "effect_signature": self.effect_signature,
            "invariances": self.invariances,
            "verification": self.verification,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolContract":
        return cls(
            parameters=d.get("parameters", {}),
            code=d.get("code", ""),
            probes=d.get("probes", []),
            effect_signature=d.get("effect_signature", ""),
            invariances=d.get("invariances", []),
            verification=d.get("verification", {}),
        )


@dataclass
class ResourceVersion:
    """One concretely-identified version of a resource.

    `status` is AGP's three-valued VersionStatus; `content_hash` is autoforge's
    tool hash (the thing a frozen baseline is keyed against).
    """

    version: str                                  # semver, e.g. "1.0.0"
    status: VersionStatus = VersionStatus.ACTIVE
    content_hash: str = ""
    created_at: float = field(default_factory=time.time)
    supersedes: Optional[str] = None              # the version string this replaced
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "supersedes": self.supersedes,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResourceVersion":
        d = dict(d)
        d["status"] = VersionStatus(d["status"])
        return cls(**d)


def _bump(version: str, *, major: bool = False) -> str:
    """Advance a semver string. Minor by default; major for a breaking change."""
    import re

    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version or "0.0.0")
    if not m:
        return "0.1.0"
    maj, mi, pa = (int(x) for x in m.groups())
    if major:
        return f"{maj + 1}.0.0"
    # A patch bump signals "same contract, changed body"; that is what an
    # evolution that passes the gate but keeps the interface is.
    return f"{maj}.{mi}.{pa + 1}"


@dataclass
class ResourceRecord:
    """A tool, as the substrate sees it. The unit AGP calls a resource."""

    id: str
    type: ResourceType = ResourceType.TOOL
    name: str = ""
    description: str = ""
    contract: ToolContract = field(default_factory=ToolContract)

    # Enforcement axis (autoforge).
    state: ResourceState = ResourceState.DRAFT

    # Provenance axis (AGP-shaped).
    version_current: ResourceVersion = field(
        default_factory=lambda: ResourceVersion(version="0.1.0")
    )
    versions: list[ResourceVersion] = field(default_factory=list)

    # Live ledger — autoforge ToolStats, verbatim.
    ledger: dict[str, Any] = field(default_factory=dict)

    # Provenance metadata.
    provenance: dict[str, Any] = field(
        default_factory=lambda: {"source": "human", "generator": ""}
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # AGP-native fields with no autoforge equivalent — carried explicitly.
    enable_evolving: bool = True
    permission_mode: str = "workspace_write"
    progress_policy: Optional[str] = None

    lineage_parents: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # -- identity ---------------------------------------------------------
    @staticmethod
    def make_id(name: str) -> str:
        return f"tool:{name}"

    @property
    def fingerprint(self) -> str:
        """A stable content hash of the contract — the resource's identity."""
        blob = repr(sorted(self.contract.to_dict().items())).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    # -- AGP-shaped projections ------------------------------------------
    def as_capability_schema(self) -> dict[str, Any]:
        """The function-calling schema, in AGP's CapabilitySchema shape.

        AGP requires strict schemas to set `additionalProperties: false`; we
        honour that so the record is directly usable as a model-callable schema.
        """
        params = dict(self.contract.parameters or {})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        params.setdefault("additionalProperties", False)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
            "strict": True,
            "source": self.metadata.get("schema_source", "declared"),
        }

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "contract": self.contract.to_dict(),
            "state": self.state.value,
            "version_current": self.version_current.to_dict(),
            "versions": [v.to_dict() for v in self.versions],
            "ledger": self.ledger,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "enable_evolving": self.enable_evolving,
            "permission_mode": self.permission_mode,
            "progress_policy": self.progress_policy,
            "lineage_parents": list(self.lineage_parents),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResourceRecord":
        d = dict(d)
        d.pop("fingerprint", None)
        d["type"] = ResourceType(d["type"])
        d["state"] = ResourceState(d["state"])
        d["contract"] = ToolContract.from_dict(d.get("contract", {}))
        d["version_current"] = ResourceVersion.from_dict(d["version_current"])
        d["versions"] = [ResourceVersion.from_dict(v) for v in d.get("versions", [])]
        return cls(**d)

    # -- bridging to / from autoforge ------------------------------------
    @classmethod
    def from_toolspec(cls, spec: Any) -> "ResourceRecord":
        """Adopt an `autoforge.ToolSpec` into the substrate, losslessly."""
        probes = [p.to_dict() if hasattr(p, "to_dict") else dict(p)
                  for p in getattr(spec, "probes", [])]
        ledger = spec.stats.to_dict() if getattr(spec, "stats", None) else {}
        state = ResourceState(getattr(spec, "state", ResourceState.DRAFT).value) \
            if hasattr(getattr(spec, "state", None), "value") else ResourceState.DRAFT
        contract = ToolContract(
            parameters=getattr(spec, "parameters", {}) or {},
            code=getattr(spec, "code", "") or "",
            probes=probes,
            effect_signature=getattr(spec, "effect_signature", "") or "",
            invariances=list(getattr(spec, "invariances", []) or []),
            verification=getattr(spec, "verification", {}) or {},
        )
        rec = cls(
            id=cls.make_id(spec.name),
            name=spec.name,
            description=getattr(spec, "description", "") or "",
            contract=contract,
            state=state,
            ledger=ledger,
            provenance={
                "source": getattr(spec, "source", "human"),
                "generator": getattr(spec, "generator", ""),
            },
            metadata={
                "tags": list(getattr(spec, "tags", []) or []),
                "cost_hint": getattr(spec, "cost_hint", "cheap"),
            },
            enable_evolving=getattr(spec, "source", "human") != "human",
        )
        rec.version_current = ResourceVersion(
            version="0.1.0",
            status=VersionStatus.ACTIVE,
            content_hash=getattr(spec, "hash", "") or rec.fingerprint,
            note="initial registration",
        )
        rec.versions = [rec.version_current]
        return rec

    def to_toolspec(self, fn: Any = None, runner: Any = None) -> Any:
        """Rehydrate an `autoforge.ToolSpec` from the record.

        Used to hand a stored resource back to autoforge's registry / verifier
        so the enforcement machinery runs on exactly what was registered.
        """
        from autoforge.tools.spec import ToolSpec, ToolState, ToolStats, TriggerProbe

        probes = [TriggerProbe(**p) for p in self.contract.probes]
        stats = ToolStats(**{
            k: v for k, v in (self.ledger or {}).items()
            if k in ToolStats.__dataclass_fields__
        })
        spec = ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.contract.parameters,
            fn=fn,
            code=self.contract.code,
            source=self.provenance.get("source", "human"),
            generator=self.provenance.get("generator", ""),
            probes=probes,
            effect_signature=self.contract.effect_signature,
            invariances=list(self.contract.invariances),
            state=ToolState(self.state.value),
            stats=stats,
            verification=self.contract.verification,
            tags=list(self.metadata.get("tags", [])),
            cost_hint=self.metadata.get("cost_hint", "cheap"),
            runner=runner,
        )
        return spec

    # -- versioning -------------------------------------------------------
    def advance_version(self, *, major: bool = False, note: str = "",
                        content_hash: str = "") -> ResourceVersion:
        """Roll the current version forward, archiving the old one.

        The superseded version keeps its record but its status moves to
        ARCHIVED — this is the AGP provenance axis doing its job, and it is what
        makes `rollback` possible without re-deriving anything.
        """
        old = self.version_current
        old.status = VersionStatus.ARCHIVED
        nxt = ResourceVersion(
            version=_bump(old.version, major=major),
            status=VersionStatus.ACTIVE,
            content_hash=content_hash or self.fingerprint,
            supersedes=old.version,
            note=note,
        )
        self.versions.append(nxt)
        self.version_current = nxt
        self.updated_at = time.time()
        return nxt


# -- code, compiled once ----------------------------------------------------
def compile_tool_fn(code: str, name: str) -> Any:
    """Compile a contract's code into a callable, without executing it.

    Lives here, beside the record, rather than beside a surface. It has two
    callers — the REST route and the gRPC handler — and no owner; leaving it in
    either one would make the other import a *surface* to get at a rule, which
    is how two doors start to have two behaviours.

    It never raises on bad code, deliberately. A contract whose body does not
    compile is still a registerable resource: the contract is the interface, and
    "the body is broken" is a fact about the tool, not grounds to refuse holding
    it. The returned callable reports the failure when called, so `invoke` is
    where the question "does this run" gets answered — and gets *logged*.
    """
    if not code.strip():
        def _noop(**kwargs: Any) -> str:
            return f"{name}: no code registered"

        return _noop
    ns: dict[str, Any] = {}
    try:
        exec(compile(code, f"<tool:{name}>", "exec"), ns)  # noqa: S102
    except Exception:  # noqa: BLE001 - a tool that will not compile is callable-nowhere
        def _broken(**kwargs: Any) -> str:
            return f"{name}: code failed to compile"

        return _broken
    # Prefer a function named after the tool, else the first callable defined.
    fn = ns.get(name)
    if callable(fn):
        return fn
    for key, val in ns.items():
        if callable(val) and not key.startswith("_"):
            return val

    def _empty(**kwargs: Any) -> None:
        return None

    return _empty
