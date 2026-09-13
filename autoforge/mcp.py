"""MCP: the agent reaching tools it did not write and does not own.

Everything else in this framework is about the agent's *own* tools — forged
here, verified here, with an effect signature measured from code we can read.
MCP is the opposite case, and it is worth being precise about why it is not
just another ToolSpec source:

    a forged tool is a claim this framework can check.
    an MCP tool is a claim someone else made, over a pipe, right now.

Three consequences follow, and they are the design:

1. **The scope is `undeclared`, and that is not a placeholder.** A remote
   server's tools touch whatever the remote server touches. We cannot read its
   code, so we cannot derive an effect signature, and guessing one would turn
   the confirmation gate into a rubber stamp. `undeclared` is permissive in
   `SCOPE_ALLOWANCES` but requires *every* disabled freedom in the gate — so an
   MCP tool is gated by anything the operator switched off, while a forged tool
   with a real signature is gated only by what it actually does. A server may
   declare a narrower scope in config; that is the operator vouching for it,
   which is different from us inferring it.

2. **The pipe is the tool's lifetime, not the process's.** A server is started
   lazily, kept while used, and torn down on `close`. If it dies, its tools must
   stop being offered — a tool that still appears on the menu and fails on every
   call is worse than one that is honestly gone. `MCPClient.alive` is checked
   before every call, and a broken server's import is dropped with the reason
   recorded.

3. **Nothing here is verified, so nothing here is trusted.** Imported tools
   arrive in DRAFT and are promoted only because unreachable tools cannot be
   called at all. Their `verification` dict says plainly that no probe ran and
   why — so `evaluate_tool` shows a gap rather than a pass. The framework's
   thesis is that trust must be earned; the honest reading for a tool whose
   implementation you cannot see is that it has earned nothing yet.

The transport is line-delimited JSON-RPC 2.0 over the child's stdin/stdout, per
the MCP stdio transport. A reader thread feeds a queue so reads can time out —
`select` does not work on Windows pipes, and a server that accepts a request and
then hangs must not hang the agent with it.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "autoforge"

#: How long to wait for a single response. Generous enough for a cold server
#: that has to import a large library, short enough that a wedged one is
#: noticed. `call` timeouts are separate: a tool may legitimately take longer
#: than a handshake.
DEFAULT_TIMEOUT = 30.0
DEFAULT_CALL_TIMEOUT = 120.0

#: The scope an imported tool gets unless the operator declares otherwise. See
#: the module docstring: this is the honest answer, not a default we forgot.
IMPORTED_SCOPE = "undeclared"


class MCPError(RuntimeError):
    """Anything that stops a server from being usable, said in one sentence."""


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
@dataclass
class MCPServerConfig:
    """One server, as the operator described it.

    `command` is run directly, never through a shell: a shell would make the
    path a quoting question and would resolve `python` to whatever the platform
    prefers. The operator writes the executable they mean.
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    scope: str = IMPORTED_SCOPE
    cwd: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    call_timeout: float = DEFAULT_CALL_TIMEOUT
    enabled: bool = True

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "MCPServerConfig":
        if not isinstance(d, dict):
            raise MCPError(f"server {name!r}: expected an object, got {type(d).__name__}")
        cmd = d.get("command")
        if not cmd:
            raise MCPError(f"server {name!r}: no 'command' to start it with")
        args = d.get("args") or []
        if not isinstance(args, list):
            raise MCPError(f"server {name!r}: 'args' must be a list")
        env = d.get("env") or {}
        if not isinstance(env, dict):
            raise MCPError(f"server {name!r}: 'env' must be an object")
        return cls(
            name=name,
            command=str(cmd),
            args=[str(a) for a in args],
            env={str(k): str(v) for k, v in env.items()},
            scope=str(d.get("scope") or IMPORTED_SCOPE),
            cwd=d.get("cwd"),
            timeout=float(d.get("timeout", DEFAULT_TIMEOUT)),
            call_timeout=float(d.get("call_timeout", DEFAULT_CALL_TIMEOUT)),
            enabled=bool(d.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "command": self.command, "args": list(self.args),
            "env": dict(self.env), "scope": self.scope, "cwd": self.cwd,
            "timeout": self.timeout, "call_timeout": self.call_timeout,
            "enabled": self.enabled,
        }


def servers_from_config(config: dict[str, Any] | None) -> tuple[list[MCPServerConfig], list[str]]:
    """Read `mcp.servers` out of the config file.

    Returns `(servers, problems)`. A malformed entry is reported and skipped
    rather than aborting startup: one bad server should cost its own tools, not
    every other server's. The problems come back as sentences because they end
    up in a log a human reads.
    """
    block = (config or {}).get("mcp")
    if not isinstance(block, dict):
        return [], []
    raw = block.get("servers")
    if not isinstance(raw, dict):
        return [], ["mcp.servers is not an object; no servers were read"] if raw else []

    servers: list[MCPServerConfig] = []
    problems: list[str] = []
    for name, entry in raw.items():
        try:
            cfg = MCPServerConfig.from_dict(str(name), entry)
        except MCPError as exc:
            problems.append(str(exc))
            continue
        if not cfg.enabled:
            continue
        servers.append(cfg)
    return servers, problems


# ----------------------------------------------------------------------
# the client
# ----------------------------------------------------------------------
class MCPClient:
    """One stdio MCP server, spoken to directly.

    Lifecycle: `start()` on first need, `close()` when done. The reader thread
    is a daemon so a stuck server cannot keep the interpreter alive, and every
    failure path leaves the child killed rather than orphaned.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen[bytes] | None = None
        self._q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._next_id = 0
        self._id_lock = threading.Lock()
        self._closed = False
        self._notes: list[dict[str, Any]] = []   # server-sent notifications
        self._logs: list[str] = []               # non-JSON lines the child printed
        self._stderr = ""
        self.server_info: dict[str, Any] = {}
        self.protocol: str = ""
        self.tools: list[dict[str, Any]] = []    # as the server described them
        self.error: str = ""

    # -- lifecycle -----------------------------------------------------
    def start(self) -> bool:
        """Spawn and handshake. False (with `.error` set) if it could not.

        A client that was explicitly `close`d stays closed: `close` is what a
        shutdown path calls, and a process reappearing after shutdown because
        some later call wanted a tool is a resource leak with good manners.
        A client whose child *crashed* is a different case and does restart —
        that is the resilience the reader thread exists to support.
        """
        if self._closed:
            self.error = f"server {self.config.name!r} was closed; a closed client does not restart"
            return False
        if self.alive:
            return True
        # Reset every per-connection field. A restart reuses this object, and
        # carrying the previous reader's state into it is not harmless: the old
        # thread's `None` sentinel is already sitting in the queue, so the next
        # `_request` would read "server closed the connection" before the fresh
        # child had said anything. The symptom is a startup that always fails
        # after the first one, which is the hardest kind to attribute.
        self._q = queue.Queue()
        self._notes = []
        self._logs = []
        self._stderr = ""
        self.error = ""
        exe = shutil.which(self.config.command) or self.config.command
        if not (os.path.isabs(exe) and os.path.exists(exe)) and not shutil.which(self.config.command):
            self.error = (f"command {self.config.command!r} is not on PATH, so server "
                          f"{self.config.name!r} cannot be started")
            return False
        try:
            self._proc = subprocess.Popen(
                [exe, *self.config.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=self._env(), cwd=self.config.cwd,
                bufsize=0,
            )
        except OSError as exc:
            self.error = f"could not start {self.config.name!r}: {type(exc).__name__}: {exc}"
            return False

        self._reader = threading.Thread(target=self._read_loop, name=f"mcp-{self.config.name}",
                                        daemon=True)
        self._reader.start()
        threading.Thread(target=self._drain_stderr, name=f"mcp-{self.config.name}-err",
                         daemon=True).start()

        try:
            result = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": "1"},
            })
        except MCPError as exc:
            self.error = f"{self.config.name!r} failed to initialise: {exc}"
            self.close()
            return False

        self.server_info = result.get("serverInfo") or {}
        self.protocol = str(result.get("protocolVersion") or "")
        # The handshake is asynchronous: the server may not consider the session
        # open until it sees this. Sending it before tools/list avoids the race
        # where list comes back empty on a strict implementation.
        self._notify("notifications/initialized", {})
        return True

    def _env(self) -> dict[str, str]:
        """The child's environment: enough to run, nothing that is a secret.

        The agent's own environment is not handed over wholesale — it holds the
        API key this process was configured with, and an MCP server has no
        business seeing it. A server that genuinely needs a variable gets it in
        its own `env` block, where the operator wrote it down on purpose.
        """
        keep = ("PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR",
                "COMSPEC", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG",
                "PYTHONIOENCODING", "PYTHONPATH", "NUMBER_OF_PROCESSORS",
                "APPDATA", "LOCALAPPDATA", "PROGRAMFILES", "PROGRAMDATA")
        env = {k: v for k, v in os.environ.items() if k in keep}
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.update(self.config.env)
        return env

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self, timeout: float = 5.0) -> None:
        """Stop the child. Idempotent, and never leaves it running."""
        self._closed = True
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()          # a server that ignores SIGTERM still goes
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream and not stream.closed:
                    stream.close()
            except OSError:
                pass

    # -- transport -----------------------------------------------------
    def _read_loop(self) -> None:
        """Feed parsed messages to the queue; end with None so waiters wake.

        A child that prints something that is not JSON-RPC (a banner, a Python
        traceback) is kept in `_logs` and skipped. Dying on it would mean one
        stray `print` in a third-party server disables the whole server, and the
        log is what makes the failure diagnosable.
        """
        proc = self._proc
        stream = proc.stdout if proc else None
        try:
            while stream is not None:
                line = stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except ValueError:
                    self._logs.append(text[:500])
                    continue
                if isinstance(msg, dict):
                    self._q.put(msg)
                else:
                    self._logs.append(text[:500])
        except (OSError, ValueError) as exc:
            self._logs.append(f"reader stopped: {type(exc).__name__}: {exc}")
        finally:
            self._q.put(None)

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            self._stderr = proc.stderr.read().decode("utf-8", "replace")[-4000:]
        except (OSError, ValueError):
            pass

    def _next(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _send(self, msg: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise MCPError(f"server {self.config.name!r} is not running")
        blob = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            proc.stdin.write(blob)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPError(f"{self.config.name!r} closed the pipe: {exc}") from exc

    def _notify(self, method: str, params: dict[str, Any]) -> bool:
        """Send a notification. Never raises: no response is coming either way.

        A notification is fire-and-forget by the protocol, so a send that fails
        because the child has just died is not an exception to unwind out of
        `start()` — it is a fact. Recording it means the next real request
        produces a diagnostic that includes it, instead of the startup path
        failing with a message about a connection the caller never opened.
        """
        try:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})
            return True
        except MCPError as exc:
            self._logs.append(f"notification {method} not delivered: {exc}")
            return False

    def _diagnostic(self) -> str:
        """Whatever the child said on the way down, for the error message."""
        bits = []
        if self._stderr.strip():
            bits.append(f"stderr: {self._stderr.strip()[:800]}")
        if self._logs:
            bits.append(f"unparsed output: {self._logs[-1][:200]}")
        if not self.alive and self._proc is not None:
            bits.append(f"exit code {self._proc.returncode}")
        return " | ".join(bits)

    def _request(self, method: str, params: dict[str, Any],
                 timeout: float | None = None) -> dict[str, Any]:
        """Send a request and wait for *its* response, discarding the rest."""
        rid = self._next()
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + (timeout or self.config.timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"{method}: no response within "
                               f"{timeout or self.config.timeout:.0f}s. {self._diagnostic()}")
            try:
                msg = self._q.get(timeout=remaining)
            except queue.Empty:
                raise MCPError(f"{method}: no response within "
                               f"{timeout or self.config.timeout:.0f}s. {self._diagnostic()}") from None
            if msg is None:
                raise MCPError(f"{method}: server closed the connection. {self._diagnostic()}")
            if "id" not in msg:                     # a notification; keep it aside
                self._notes.append(msg)
                continue
            if msg.get("id") != rid:                # a late answer to an abandoned call
                continue
            if "error" in msg:
                err = msg["error"]
                detail = err.get("message", err) if isinstance(err, dict) else err
                raise MCPError(f"{method}: {detail}")
            result = msg.get("result")
            return result if isinstance(result, dict) else {"value": result}

    # -- the two useful verbs ------------------------------------------
    def list_tools(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Ask what the server offers. Raises MCPError rather than returning []."""
        if not self.alive and not self.start():
            raise MCPError(self.error or f"server {self.config.name!r} is not running")
        result = self._request("tools/list", {}, timeout=timeout)
        tools = result.get("tools")
        if tools is None:
            tools = []
        if not isinstance(tools, list):
            raise MCPError(f"tools/list returned {type(tools).__name__}, not a list")
        self.tools = [t for t in tools if isinstance(t, dict) and t.get("name")]
        return self.tools

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None,
                  timeout: float | None = None) -> dict[str, Any]:
        if not self.alive and not self.start():
            raise MCPError(self.error or f"server {self.config.name!r} is not running")
        return self._request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout or self.config.call_timeout,
        )

    def __enter__(self) -> "MCPClient":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def render_result(result: dict[str, Any]) -> tuple[bool, str]:
    """Turn an MCP tools/call result into (ok, text).

    The `isError` flag is the server's own report of failure, so it is honoured
    rather than second-guessed. Content blocks other than text are named instead
    of dropped — silently returning an empty string for an image result would
    read to the agent as "the tool succeeded and produced nothing".
    """
    ok = not bool(result.get("isError"))
    blocks = result.get("content")
    if blocks is None:
        structured = result.get("structuredContent")
        if structured is not None:
            return ok, json.dumps(structured, ensure_ascii=False, indent=2)
        return ok, json.dumps(result, ensure_ascii=False) if result else ""
    if not isinstance(blocks, list):
        return ok, json.dumps(blocks, ensure_ascii=False)

    out: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            out.append(str(block))
            continue
        kind = block.get("type")
        if kind == "text":
            out.append(str(block.get("text", "")))
        elif kind == "resource":
            res = block.get("resource") or {}
            out.append(f"[resource {res.get('uri', '?')}]")
        else:
            out.append(f"[{kind or 'unknown'} content from the MCP server]")
    return ok, "\n".join(out)


# ----------------------------------------------------------------------
# importing a server's tools into the registry
# ----------------------------------------------------------------------
def tool_name(server: str, remote: str) -> str:
    """`mcp__<server>__<tool>` — the server is in the name so a tool cannot be
    mistaken for a local one, and so two servers offering `search` coexist."""
    return f"mcp__{server}__{remote}"


def spec_from_remote(server: MCPServerConfig, client: MCPClient,
                     remote: dict[str, Any]) -> Any:
    """Build a ToolSpec that calls through to the server.

    The parameters schema is passed through as the server declared it, through
    `normalise_parameters` so a sloppy schema cannot reach the validator. The
    runner closure captures the live client, so a call goes over the pipe that
    is already open instead of starting a second server.
    """
    from .tools.spec import ToolSpec, ToolState, normalise_parameters

    name = tool_name(server.name, str(remote["name"]))
    description = str(remote.get("description") or f"(no description) from {server.name}")
    schema = normalise_parameters(remote.get("inputSchema"))

    def runner(_name: str, args: dict[str, Any]) -> Any:
        # Imported at call time: `registry` imports `spec`, and this module is
        # imported by the agent, so a module-level import would close the cycle.
        from .tools.registry import ToolResult

        started = time.perf_counter()
        try:
            result = client.call_tool(str(remote["name"]), args)
        except MCPError as exc:
            return ToolResult(
                name, False, "",
                error=f"{server.name} refused: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        ok, text = render_result(result)
        return ToolResult(
            name, ok, text,
            error=None if ok else (text or "the server reported an error"),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    spec = ToolSpec(
        name=name,
        description=f"[{server.name}] {description}",
        parameters=schema,
        fn=lambda **kw: None,        # never called: `runner` takes precedence
        code="",
        source="mcp",
        effect_signature=server.scope,
        tags=["mcp", server.name],
        cost_hint="moderate",
        runner=runner,
    )
    spec.state = ToolState.DRAFT
    # Say what is *not* known. An empty verification dict would render as "not
    # verified yet", which invites the reader to assume a probe is coming. None
    # is: the implementation lives in another process, and this framework cannot
    # test it the way it tests forged code.
    spec.verification = {
        "verified": False,
        "reason": (f"imported from MCP server {server.name!r}; the implementation "
                   f"is not in this repository and was not probed"),
        "server": server.name,
        "remote_tool": remote.get("name"),
        "protocol": client.protocol,
        "server_info": client.server_info,
    }
    return spec


@dataclass
class ImportReport:
    """What happened when servers were brought in — for the log and the prompt."""

    servers: int = 0
    tools: int = 0
    imported: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"servers": self.servers, "tools": self.tools,
                "imported": list(self.imported), "problems": list(self.problems)}

    def summary(self) -> str:
        if not self.servers:
            return "no MCP servers configured"
        line = f"{self.servers} MCP server(s), {self.tools} tool(s) imported"
        if self.problems:
            line += f", {len(self.problems)} problem(s)"
        return line


class MCPHub:
    """All configured servers, their clients, and the tools they contributed.

    Started explicitly rather than on import: a config file describing a server
    the user rarely needs should not cost a subprocess on every agent boot.
    """

    def __init__(self, servers: list[MCPServerConfig] | None = None) -> None:
        self.servers = list(servers or [])
        self.clients: dict[str, MCPClient] = {}
        self.report = ImportReport()

    def add(self, cfg: MCPServerConfig) -> None:
        self.servers.append(cfg)

    def start(self, only: list[str] | None = None) -> ImportReport:
        """Start each server and collect its tools. Never raises for one server.

        A server that will not start costs its own tools and nothing else — the
        alternative, aborting the whole import, means one broken third-party
        package disables MCP entirely.
        """
        wanted = [s for s in self.servers if only is None or s.name in only]
        for cfg in wanted:
            client = MCPClient(cfg)
            self.clients[cfg.name] = client
            try:
                if not client.start():
                    self.report.problems.append(f"{cfg.name}: {client.error}")
                    continue
                tools = client.list_tools()
            except MCPError as exc:
                self.report.problems.append(f"{cfg.name}: {exc}")
                client.close()
                continue
            self.report.servers += 1
            self.report.tools += len(tools)
            for t in tools:
                self.report.imported.append(tool_name(cfg.name, str(t["name"])))
        return self.report

    def install(self, registry: Any, only: list[str] | None = None) -> ImportReport:
        """Start servers and register their tools in `registry`."""
        self.start(only=only)
        for cfg in self.servers:
            client = self.clients.get(cfg.name)
            if client is None or not client.alive:
                continue
            for remote in client.tools:
                spec = spec_from_remote(cfg, client, remote)
                registry.register(spec)
                # Promoted because an unreachable tool is indistinguishable from
                # a missing one: the registry hides DRAFT by default, so leaving
                # these in DRAFT would import a server's tools and offer none of
                # them. Trust is not the thing being granted here — the
                # `verification` block above states that plainly.
                registry.promote(spec.name)
        return self.report

    def close(self) -> None:
        for client in self.clients.values():
            client.close()
        self.clients.clear()

    def __enter__(self) -> "MCPHub":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = [
    "MCPClient", "MCPHub", "MCPError", "MCPServerConfig", "ImportReport",
    "PROTOCOL_VERSION", "IMPORTED_SCOPE", "servers_from_config",
    "spec_from_remote", "tool_name", "render_result",
]
