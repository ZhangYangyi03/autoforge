"""The MCP servers the other agents on this machine already have.

An operator running Hermes, Codex and a Claude-shaped client has described
their MCP servers three times, in three dialects. Asking them to type a fourth
copy into `~/.autoforge/config.json` is not a feature — it is a tax on using
this framework at all. So the servers are read from where they are already
written.

The dialects, as they actually are on disk rather than as documented:

  * Hermes   — YAML, top-level `mcp_servers:`, `timeout` + `connect_timeout`
  * Codex    — TOML, `[mcp_servers.<name>]`, `startup_timeout_sec`,
               `enabled_tools` / `disabled_tools`, `bearer_token_env_var`
  * Claude Code / Desktop, Cursor, Cline, Roo, ds-harness
             — JSON, `mcpServers` (ds-harness ships a server for this shape,
               not a config of its own, so "compatible with ds-harness" means
               speaking the shape it is dropped into)
  * VS Code  — JSON, `servers` rather than `mcpServers`, and `type: stdio`

Four rules hold the module together:

1. A file that cannot be read costs its own servers and nothing else. The
   operator gets a sentence naming the file and the reason, then every other
   source is still read.
2. The parsers for two of these formats are third-party. When one is missing
   the sentence is the command that installs it, because "no module named
   yaml" is a message about this program rather than about what to do.
3. When two sources describe a server with the same name, the first source
   wins and the collision is *reported*. Silently picking one is how an
   operator spends an afternoon wondering why their edit did nothing. A file
   the operator named is labelled by its own file name rather than generically
   "named", so the collision sentence names two files and not one word twice.
4. Only the *name* of a token comes across, never a value. Importing someone's
   config must not copy their secret into a second file that then has to be
   protected too.

Reading is not trusting: nothing here starts a process. The report says what
was found, and `merged_servers` is what an agent asks for when it wants to use
it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .mcp import MCPError, MCPServerConfig, servers_from_config

#: Set to a `os.pathsep`-separated list of files to read those and only those.
#: Mirrors `AUTOFORGE_SKILLS_DIRS`: the escape hatch has to exist, because a
#: machine with five agent installs should not need five code changes to say
#: "read this one".
ENV_SOURCES = "AUTOFORGE_MCP_SOURCES"

#: The key under which each dialect nests its servers, and the format it is
#: written in. One table, so a reader and its test cannot disagree.
DIALECTS: dict[str, tuple[str, str]] = {
    "hermes": ("yaml", "mcp_servers"),
    "codex": ("toml", "mcp_servers"),
    "claude": ("json", "mcpServers"),
    "vscode": ("json", "servers"),
}

#: What to install when a parser is absent. Two of the four dialects need
#: something that is not in the standard library, and `tomllib` itself only
#: exists from 3.11 — so a 3.10 interpreter needs `tomli` for Codex.
INSTALL_HINT = {"yaml": "pip install pyyaml", "toml": "pip install tomli"}

#: The keys a JSON file may keep its servers under, in the order they are
#: tried for a file whose dialect is not knowable from its name. Derived from
#: `DIALECTS` so the two cannot drift.
JSON_KEYS: tuple[str, ...] = tuple(
    key for fmt, key in DIALECTS.values() if fmt == "json")


@dataclass(frozen=True)
class Source:
    """One file, whose servers they are, and where inside it they live."""

    label: str
    path: Path
    fmt: str
    key: str

    def __str__(self) -> str:
        return f"{self.label} ({self.path})"


@dataclass
class EcosystemReport:
    """What the other agents on this machine are willing to tell us."""

    servers: list[MCPServerConfig] = field(default_factory=list)
    #: Files that contributed a server, in order.
    sources: list[str] = field(default_factory=list)
    #: Every file that existed and was examined, whether or not it contributed.
    #: Kept apart from `sources` so "this file has no servers under the key" is
    #: a fact on the `auto mcp` screen rather than a problem in the log.
    scanned: list[str] = field(default_factory=list)
    #: One sentence per file that could not be used, naming it and why.
    problems: list[str] = field(default_factory=list)
    #: Same name from more than one source. Reported, never resolved silently.
    conflicts: list[str] = field(default_factory=list)
    #: label -> the server names that came from it, for `auto mcp` to print.
    by_source: dict[str, list[str]] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.servers]

    def get(self, name: str) -> MCPServerConfig | None:
        return next((c for c in self.servers if c.name == name), None)

    def summary(self) -> str:
        if not self.scanned:
            return "no other agent's MCP config was found"
        line = (f"{len(self.servers)} server(s) from {len(self.sources)} of "
                f"{len(self.scanned)} file(s)")
        if self.conflicts:
            line += f"; {len(self.conflicts)} name collision(s)"
        if self.problems:
            line += f"; {len(self.problems)} file(s) unreadable"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "servers": [c.to_dict() for c in self.servers],
            "sources": list(self.sources),
            "scanned": list(self.scanned),
            "problems": list(self.problems),
            "conflicts": list(self.conflicts),
            "by_source": {k: list(v) for k, v in self.by_source.items()},
        }


# ----------------------------------------------------------------------
# where to look
# ----------------------------------------------------------------------
def _hermes_config(home: Path) -> list[Path]:
    """Hermes keeps its profile in a per-OS data directory, and `HERMES_HOME`
    overrides. All of the places it may be are offered; only one is usually
    there, and the others cost a `stat`."""
    out = []
    override = os.environ.get("HERMES_HOME")
    if override:
        out.append(Path(override) / "config.yaml")
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if local:
        out.append(Path(local) / "hermes" / "config.yaml")
    out += [home / ".hermes" / "config.yaml", home / ".config" / "hermes" / "config.yaml"]
    return out


def _codex_config(home: Path) -> list[Path]:
    override = os.environ.get("CODEX_HOME")
    return ([Path(override) / "config.toml"] if override
            else [home / ".codex" / "config.toml"])


def _claude_desktop(home: Path) -> list[Path]:
    """Claude Desktop's file, per platform. Written three ways because it is
    three files: Windows keeps it in roaming AppData, macOS in Application
    Support, Linux under XDG."""
    out = []
    roaming = os.environ.get("APPDATA")
    if roaming:
        out.append(Path(roaming) / "Claude" / "claude_desktop_config.json")
    out += [
        home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        home / ".config" / "Claude" / "claude_desktop_config.json",
    ]
    return out


def candidates(home: str | Path | None = None, cwd: str | Path | None = None) -> list[Source]:
    """Every place another agent on this machine may keep its MCP servers.

    Ordered most-specific-first, because that order is also the precedence used
    when two sources describe a server of the same name: a project's own file
    beats a global one, and this framework's own config beats everything.
    """
    home = Path(home) if home else Path.home()
    cwd = Path(cwd) if cwd else Path.cwd()
    out: list[Source] = []
    for path in _hermes_config(home):
        out.append(Source("hermes", path, *DIALECTS["hermes"]))
    for path in _codex_config(home):
        out.append(Source("codex", path, *DIALECTS["codex"]))
    # Claude Code: one file, top-level and per-project server maps.
    out.append(Source("claude", home / ".claude.json", *DIALECTS["claude"]))
    for path in _claude_desktop(home):
        out.append(Source("claude-desktop", path, *DIALECTS["claude"]))
    out.append(Source("cursor", home / ".cursor" / "mcp.json", *DIALECTS["claude"]))
    # Project-local, where a repo pins its own servers.
    out.append(Source("project", cwd / ".mcp.json", *DIALECTS["claude"]))
    out.append(Source("vscode", cwd / ".vscode" / "mcp.json", *DIALECTS["vscode"]))
    return out


def discover(home: str | Path | None = None, cwd: str | Path | None = None,
             paths: list[str | Path] | None = None) -> list[Source]:
    """The candidates that actually exist, one entry per distinct file.

    `paths` (or `$AUTOFORGE_MCP_SOURCES`) replaces the search entirely: when the
    operator names files, they mean those files, and a fallback to a
    guessed location would be this module ignoring them.
    """
    named = paths
    if named is None:
        env = os.environ.get(ENV_SOURCES)
        named = env.split(os.pathsep) if env else None
    if named is not None:
        found = [Source(f"named:{Path(p).name}", Path(p), *_by_extension(Path(p)))
                 for p in (str(x).strip() for x in named)
                 if p and Path(p).is_file()]
    else:
        found = [s for s in candidates(home, cwd) if s.path.is_file()]
    return _dedupe(found)


def _dedupe(sources: list[Source]) -> list[Source]:
    """Drop later sources that name a file an earlier one already named.

    Not hypothetical: Hermes sets `HERMES_HOME` to the same directory its
    platform default points at, so both candidate paths resolve to one file.
    Without this the same servers are read twice and every one of them is
    reported as a name collision with itself — an alarming, entirely fictional
    problem that would train the operator to ignore the collision list.
    """
    seen: set[str] = set()
    out: list[Source] = []
    for source in sources:
        try:
            key = str(source.path.resolve()).lower()
        except OSError:
            key = str(source.path).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(source)
    return out


def _by_extension(path: Path) -> tuple[str, str]:
    """A named file still has to be parsed by its own dialect."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return DIALECTS["hermes"]
    if suffix == ".toml":
        return DIALECTS["codex"]
    return DIALECTS["claude"]


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------
def _parse(text: str, fmt: str, source: Source) -> Any:
    """Parse one file, or say in one sentence why it could not be parsed."""
    if fmt == "json":
        try:
            return json.loads(text)
        except ValueError as exc:
            raise MCPError(f"{source}: not valid JSON — {exc}") from None
    if fmt == "toml":
        # `tomllib` is 3.11+; this package supports 3.10, where `tomli` is the
        # same parser under its pre-stdlib name.
        try:
            import tomllib as toml_reader
        except ImportError:
            try:
                import tomli as toml_reader       # type: ignore[no-redef]
            except ImportError:
                raise MCPError(f"{source}: needs a TOML reader — "
                               f"{INSTALL_HINT['toml']}") from None
        try:
            return toml_reader.loads(text)
        except Exception as exc:
            raise MCPError(f"{source}: not valid TOML — {exc}") from None
    if fmt == "yaml":
        try:
            import yaml
        except ImportError:
            raise MCPError(f"{source}: needs a YAML reader — "
                           f"{INSTALL_HINT['yaml']}") from None
        try:
            return yaml.safe_load(text)
        except Exception as exc:
            raise MCPError(f"{source}: not valid YAML — {exc}") from None
    raise MCPError(f"{source}: unknown format {fmt!r}")


def _entries(source: Source, doc: Any,
             key: str) -> tuple[list[tuple[str, str, dict[str, Any]]], bool]:
    """One document's servers, and whether it declared the key at all.

    `(entries, declared)`. The second value is what separates "this file has no
    MCP servers, which is fine" from "this file has MCP servers under a key we
    did not look at, which is a bug worth reporting". An empty table declares
    itself and holds nothing — Codex writes `[mcp_servers]` with no rows — and
    calling that an error would put a permanent complaint on the screen of
    every operator who simply has no MCP servers configured in Codex.

    `key` is passed in rather than read off `source` because for a named file
    it is not the same thing — see `_resolve_key`.

    Claude Code writes a second server map per project inside the same file,
    and those are real servers an operator expects to be found. They are
    labelled by the project's own directory name so a name collision between
    two projects reads as what it is.
    """
    if not isinstance(doc, dict):
        raise MCPError(f"{source}: expected an object at the top level, "
                       f"got {type(doc).__name__}")
    block = doc.get(key)
    out: list[tuple[str, str, dict[str, Any]]] = []
    if block is not None:
        if not isinstance(block, dict):
            raise MCPError(f"{source}: {key!r} is not an object")
        for name, entry in block.items():
            out.append((str(name), source.label, entry))
    declared = block is not None
    if key == "mcpServers" and isinstance(doc.get("projects"), dict):
        for project, body in doc["projects"].items():
            if not isinstance(body, dict):
                continue
            per_project = body.get("mcpServers")
            if not isinstance(per_project, dict):
                continue
            declared = True
            label = f"claude:{Path(str(project)).name or project}"
            for name, entry in per_project.items():
                out.append((str(name), label, entry))
    return out, declared


def _resolve_key(source: Source, doc: Any) -> str:
    """Which key this document actually keeps its servers under.

    Two cases, and they are not the same question.

    A *discovered* file is picked up because of where it sits, so its label
    names its dialect and the key is known. It is used strictly: a Claude file
    with no `mcpServers` is a file whose key is absent, which is exactly the
    thing worth saying, and quietly accepting `servers` there would make a
    mistyped key look like a working one.

    A *named* file is different, because `.json` does not say whether it is
    Claude's `mcpServers` or VS Code's `servers`. The operator who pointed at
    it knows which tool owns it but did not write it down, and guessing wrong
    reports a perfectly correct file as having the wrong key. So both known
    JSON keys are tried, and the declared key is the fallback when neither is
    present — because then the "absent" message should name the key the label
    implied.
    """
    if source.label.startswith("named") and source.fmt == "json" \
            and isinstance(doc, dict):
        for key in JSON_KEYS:
            if isinstance(doc.get(key), dict):
                return key
    return source.key


def read(source: Source) -> tuple[list[MCPServerConfig], list[str]]:
    """Read one file. Returns `(servers, problems)` and never raises.

    Returning rather than raising is what makes rule 1 true at the call site:
    the caller cannot forget to catch, because there is nothing to catch.
    """
    try:
        text = source.path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [f"{source}: unreadable — {exc}"]
    try:
        doc = _parse(text, source.fmt, source)
    except MCPError as exc:
        return [], [str(exc)]

    servers: list[MCPServerConfig] = []
    problems: list[str] = []
    key = _resolve_key(source, doc)
    try:
        entries, declared = _entries(source, doc, key)
    except MCPError as exc:
        return [], [str(exc)]
    for name, label, entry in entries:
        try:
            cfg = MCPServerConfig.from_dict(name, entry)
        except MCPError as exc:
            problems.append(f"{source}: {exc}")
            continue
        if not cfg.enabled:
            continue
        # The scope records where it came from, so a tool call's audit trail
        # can say which agent's config asked for it.
        cfg.scope = f"ecosystem:{label}"
        servers.append(cfg)
    if not declared:
        problems.append(f"{source}: {key!r} absent — is this the right "
                        f"key for this file?")
    return servers, problems


def read_all(home: str | Path | None = None, cwd: str | Path | None = None,
             paths: list[str | Path] | None = None) -> EcosystemReport:
    """Read every source that exists. First one to name a server owns it.

    The precedence is the source order, not a merge: two servers with the same
    name would otherwise both be started, both be prefixed identically, and the
    second would silently overwrite the first in the registry.
    """
    report = EcosystemReport()
    seen: dict[str, str] = {}
    for source in discover(home, cwd, paths):
        report.scanned.append(str(source))
        servers, problems = read(source)
        # A file is listed under `sources` only if it contributed or failed —
        # the ones that merely exist are in `scanned`.
        if servers or problems:
            report.sources.append(str(source))
        report.problems.extend(problems)
        for cfg in servers:
            owner = seen.get(cfg.name)
            if owner is not None:
                report.conflicts.append(
                    f"{cfg.name!r} is described by both {owner} and {source.label}; "
                    f"using {owner}'s")
                continue
            seen[cfg.name] = source.label
            report.servers.append(cfg)
            report.by_source.setdefault(source.label, []).append(cfg.name)
    return report


# ----------------------------------------------------------------------
# the entry point an agent uses
# ----------------------------------------------------------------------
def merged_servers(config: dict[str, Any] | None,
                   home: str | Path | None = None,
                   cwd: str | Path | None = None,
                   ) -> tuple[list[MCPServerConfig], list[str]]:
    """This framework's own servers, plus the other agents' if asked for.

    `mcp.ecosystem: true` in the config file turns the import on:

        {"mcp": {"ecosystem": true, "servers": {...}}}

    Off by default, deliberately. Reading another tool's config is free; the
    servers it names are processes, and starting processes nobody asked for is
    not something an upgrade should begin doing on its own.

    Collisions go to *this* config's server, not the imported one. The operator
    wrote that file to change the behaviour of this program, and an import that
    outranked it would make their edit look broken.
    """
    servers, problems = servers_from_config(config)
    block = (config or {}).get("mcp")
    if not isinstance(block, dict) or not _truthy(block.get("ecosystem")):
        return servers, problems
    report = read_all(home, cwd)
    mine = {c.name for c in servers}
    for cfg in report.servers:
        if cfg.name in mine:
            report.conflicts.append(
                f"{cfg.name!r} is also in this config; using this config's")
            continue
        servers.append(cfg)
    problems = problems + [f"ecosystem: {p}" for p in report.problems + report.conflicts]
    return servers, problems


def _truthy(value: Any) -> bool:
    """`ecosystem: true`, `"true"`, `yes`, `1` — a config file is written by
    hand, and a switch that only accepts one spelling is a switch that gets
    typed wrong once and then distrusted."""
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "no", "off", "none"}
    return bool(value)


__all__ = [
    "DIALECTS", "ENV_SOURCES", "EcosystemReport", "Source", "candidates",
    "discover", "merged_servers", "read", "read_all",
]
