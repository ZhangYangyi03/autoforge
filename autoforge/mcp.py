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


def _seconds(value: Any, fallback: float, name: str, field_name: str) -> float:
    """A timeout written by someone else, read as a number or refused loudly.

    Imported configs are not always well-formed, and a string where a number
    belongs would otherwise surface much later as a `TypeError` inside a socket
    call — a traceback naming neither the server nor the field.
    """
    if value is None:
        return fallback
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise MCPError(f"server {name!r}: {field_name} {value!r} is not a number") from None
    if seconds <= 0:
        raise MCPError(f"server {name!r}: {field_name} must be positive, got {value!r}")
    return seconds


def _boolish(value: Any) -> bool:
    """`true`/`false` as YAML, TOML and several not-quite-JSON writers spell it.

    `bool("false")` is `True`, so a config that disables a server with the
    string "false" would silently enable it — the kind of bug that is invisible
    until someone wonders why a server they switched off is running.
    """
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off", ""}
    return bool(value)


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
@dataclass
class MCPServerConfig:
    """One server, as the operator described it.

    Two transports live here, and which one is in use is decided by the
    operator's own description rather than guessed: a `url` means the
    streamable-HTTP transport, a `command` means a child process speaking
    JSON-RPC over its own stdin/stdout. Both spellings are accepted because
    both are written by the tools this config is imported from.

    `command` is run directly, never through a shell: a shell would make the
    path a quoting question and would resolve `python` to whatever the platform
    prefers. The operator writes the executable they mean.
    """

    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    scope: str = IMPORTED_SCOPE
    cwd: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    call_timeout: float = DEFAULT_CALL_TIMEOUT
    enabled: bool = True
    #: Streamable HTTP: a server already running somewhere, reached over the
    #: network instead of started here.
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    #: The *name* of a variable holding a bearer token, which is how Codex
    #: writes it. A name rather than a value on purpose: the token stays in the
    #: environment, so importing someone's config cannot copy their secret into
    #: a second file that then has to be protected too.
    bearer_token_env_var: str = ""
    #: Tool-level gating, as Codex writes it. `enabled_tools` is an allowlist
    #: (empty means every tool the server offers); `disabled_tools` is a
    #: denylist applied afterwards. Both are honoured because a server imported
    #: from someone else's config must expose exactly the tools *they* chose:
    #: silently re-enabling a tool they switched off is the import changing the
    #: operator's security posture rather than reproducing it.
    enabled_tools: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)

    def allows(self, tool_name: str) -> bool:
        """Whether this config lets `tool_name` through.

        The short name is matched, not the namespaced one: the config was
        written against the server's own tool names, long before this framework
        prefixed them.
        """
        if self.enabled_tools and tool_name not in self.enabled_tools:
            return False
        return tool_name not in self.disabled_tools

    @property
    def transport(self) -> str:
        """`http` or `stdio`. Derived, so the two can never disagree."""
        return "http" if self.url else "stdio"

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "MCPServerConfig":
        if not isinstance(d, dict):
            raise MCPError(f"server {name!r}: expected an object, got {type(d).__name__}")
        cmd = d.get("command")
        url = str(d.get("url") or "")
        if not cmd and not url:
            # Checked here rather than at start time so the operator hears
            # about it while they are still looking at the file they wrote.
            raise MCPError(f"server {name!r}: needs either a 'command' to start "
                           f"or a 'url' to reach")
        args = d.get("args") or []
        if not isinstance(args, list):
            raise MCPError(f"server {name!r}: 'args' must be a list")
        env = d.get("env") or {}
        if not isinstance(env, dict):
            raise MCPError(f"server {name!r}: 'env' must be an object")
        headers = d.get("headers") or {}
        if not isinstance(headers, dict):
            raise MCPError(f"server {name!r}: 'headers' must be an object")
        # Every dialect spells its timeouts differently, and all of them mean
        # "how long to wait for a reply". Reading the aliases here keeps the
        # difference in one place instead of in every reader.
        timeout = d.get("timeout", d.get("connect_timeout",
                                        d.get("startup_timeout_sec", DEFAULT_TIMEOUT)))
        call_timeout = d.get("call_timeout", d.get("tool_timeout_sec",
                                                   DEFAULT_CALL_TIMEOUT))
        enabled_tools = d.get("enabled_tools") or []
        disabled_tools = d.get("disabled_tools") or []
        for field_name, value in (("enabled_tools", enabled_tools),
                                  ("disabled_tools", disabled_tools)):
            if not isinstance(value, list):
                raise MCPError(f"server {name!r}: {field_name!r} must be a list")
        # Cline and Claude's own UI write `"disabled": true` rather than an
        # `enabled` flag, and read `enabled` as its absence. Same statement,
        # opposite name — the alias is read here so neither dialect is silent.
        enabled = d.get("enabled")
        if enabled is None:
            enabled = not _boolish(d.get("disabled", False))
        return cls(
            name=name,
            command=str(cmd) if cmd else "",
            args=[str(a) for a in args],
            env={str(k): str(v) for k, v in env.items()},
            scope=str(d.get("scope") or IMPORTED_SCOPE),
            cwd=d.get("cwd"),
            timeout=_seconds(timeout, DEFAULT_TIMEOUT, name, "timeout"),
            call_timeout=_seconds(call_timeout, DEFAULT_CALL_TIMEOUT, name, "call_timeout"),
            enabled=_boolish(enabled),
            url=url,
            headers={str(k): str(v) for k, v in headers.items()},
            bearer_token_env_var=str(d.get("bearer_token_env_var") or ""),
            enabled_tools=[str(t) for t in enabled_tools],
            disabled_tools=[str(t) for t in disabled_tools],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "command": self.command, "args": list(self.args),
            "env": dict(self.env), "scope": self.scope, "cwd": self.cwd,
            "timeout": self.timeout, "call_timeout": self.call_timeout,
            "enabled": self.enabled, "url": self.url,
            "headers": dict(self.headers),
            "bearer_token_env_var": self.bearer_token_env_var,
            "enabled_tools": list(self.enabled_tools),
            "disabled_tools": list(self.disabled_tools),
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


class MCPHttpClient:
    """One MCP server reached over streamable HTTP.

    Same public surface as `MCPClient` — `start`, `alive`, `close`,
    `list_tools`, `call_tool`, and the `.tools`/`.error`/`.server_info`/
    `.protocol` attributes — so the hub does not care which of the two it holds.

    The transport is the streamable-HTTP shape: every message is its own POST
    carrying one JSON-RPC object, and the reply comes back either as a plain
    JSON body or as an SSE stream with the answer in a `data:` line. Both are
    read, because a server may choose per request and several do.

    There is no pipe to inspect, so `alive` means "this client has been started
    and not closed" rather than "the far end is definitely up". That is the
    honest reading for a transport that opens a connection per call: a server
    that has died is discovered on the next call, whose error names it, rather
    than by a liveness check that would itself be a network request.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._next_id = 0
        self._id_lock = threading.Lock()
        self._closed = False
        self._started = False
        self._session_id = ""
        self._notes: list[dict[str, Any]] = []
        self.error = ""
        self.server_info: dict[str, Any] = {}
        self.protocol: str = ""
        self.tools: list[dict[str, Any]] = []

    # -- lifecycle -----------------------------------------------------
    def start(self) -> bool:
        """Handshake. False (with `.error` set) if the server would not talk."""
        if self._closed:
            self.error = f"server {self.config.name!r} was closed; a closed client does not restart"
            return False
        if self._started:
            return True
        if not self.config.url:
            self.error = f"server {self.config.name!r} has no url to reach"
            return False
        try:
            result = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": "1"},
            })
        except MCPError as exc:
            self.error = f"{self.config.name!r} failed to initialise: {exc}"
            return False
        self.server_info = result.get("serverInfo") or {}
        self.protocol = str(result.get("protocolVersion") or "")
        self._started = True
        # A notification has no reply, so it is not part of the handshake's
        # success. Failing to deliver it is recorded and the server is still
        # used: a strict implementation that never sees it answers `tools/list`
        # with an error naming the cause, which is a better diagnostic than a
        # startup that failed for a reason the operator cannot see.
        try:
            self._notify("notifications/initialized", {})
        except MCPError as exc:
            self._notes.append({"notification": "initialized", "error": str(exc)})
        return True

    @property
    def alive(self) -> bool:
        return self._started and not self._closed

    def close(self, timeout: float = 5.0) -> None:
        """Stop using the session. Idempotent.

        The server is told the session is over, which lets it release whatever
        it held for it. A refusal is not an error worth raising: the session is
        going away regardless, and the operator cannot act on it.
        """
        self._closed = True
        self._started = False
        if not self._session_id:
            return
        sid, self._session_id = self._session_id, ""
        try:
            import requests
            requests.delete(self.config.url, headers=self._headers(session=sid),
                            timeout=timeout)
        except Exception:
            pass

    # -- transport -----------------------------------------------------
    def _headers(self, *, session: str | None = None) -> dict[str, str]:
        """The headers every message carries, plus whatever was configured.

        The bearer token is read from the environment at send time, not stored
        on this object, so a token that changes is picked up and a token that
        is absent produces a request without the header rather than a config
        file holding a secret.
        """
        headers = {
            "Content-Type": "application/json",
            # Both are acceptable, because the server chooses: a plain JSON
            # body for a quick answer, SSE when it wants to stream.
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol or PROTOCOL_VERSION,
        }
        headers.update(self.config.headers)
        token_var = self.config.bearer_token_env_var
        if token_var:
            token = os.environ.get(token_var, "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        sid = self._session_id if session is None else session
        if sid:
            headers["Mcp-Session-Id"] = sid
        return headers

    def _next(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _post(self, msg: dict[str, Any], timeout: float) -> Any:
        """One POST. Returns the parsed JSON-RPC reply, or None for an ack.

        Raises MCPError for everything that went wrong, because every caller
        here wants a sentence rather than an exception type they must map.
        """
        import requests

        try:
            resp = requests.post(self.config.url, json=msg,
                                 headers=self._headers(), timeout=timeout, stream=True)
        except requests.RequestException as exc:
            raise MCPError(f"server {self.config.name!r} is unreachable: {exc}") from exc
        try:
            # A server may hand out a session id on any reply; the initialize
            # reply is where it normally appears. Kept whenever it is offered.
            sid = resp.headers.get("Mcp-Session-Id")
            if sid:
                self._session_id = sid
            if resp.status_code >= 400:
                body = _peek(resp)
                raise MCPError(f"server {self.config.name!r} answered "
                               f"{resp.status_code}: {body}")
            return self._decode(resp, msg.get("id"))
        finally:
            resp.close()

    def _decode(self, resp: Any, wanted_id: Any) -> Any:
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype == "text/event-stream":
            return self._read_stream(resp, wanted_id)
        if resp.status_code == 202 or not resp.content:
            return None                     # an acknowledgement, nothing to read
        try:
            return resp.json()
        except ValueError as exc:
            raise MCPError(f"server {self.config.name!r} sent a body that is not "
                           f"JSON: {exc}") from None

    def _read_stream(self, resp: Any, wanted_id: Any) -> dict[str, Any] | None:
        """Read an SSE stream until the reply to `wanted_id` arrives.

        A stream may carry several messages before the one asked for — a
        notification, or a reply to an earlier call the caller gave up on — so
        the whole stream is walked rather than reading the first event. Events
        not addressed to `wanted_id` are kept in `_notes` and skipped, which is
        the same rule the stdio reader follows.
        """
        import requests

        data: list[str] = []
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                line = (raw or "").strip()
                if line.startswith("data:"):
                    data.append(line[5:].strip())
                    continue
                if line:
                    continue                # `event:`, `id:`, `retry:`, comments
                if not data:                # blank line with nothing buffered
                    continue
                payload, data = "\n".join(data), []
                try:
                    msg = json.loads(payload)
                except ValueError:
                    self._notes.append({"unparsed_event": payload[:500]})
                    continue
                if not isinstance(msg, dict):
                    continue
                if "id" not in msg:
                    self._notes.append(msg)
                    continue
                if msg.get("id") != wanted_id:
                    continue
                return msg
        except requests.RequestException as exc:
            raise MCPError(f"server {self.config.name!r} stopped streaming: {exc}") from exc
        return None

    def _notify(self, method: str, params: dict[str, Any]) -> bool:
        self._post({"jsonrpc": "2.0", "method": method, "params": params},
                   self.config.timeout)
        return True

    def _request(self, method: str, params: dict[str, Any],
                 timeout: float | None = None) -> dict[str, Any]:
        """Send a request and read *its* reply."""
        rid = self._next()
        msg = self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params},
                         timeout or self.config.timeout)
        if msg is None:
            raise MCPError(f"{method}: server {self.config.name!r} acknowledged "
                           f"without an answer")
        if not isinstance(msg, dict):
            if isinstance(msg, list):
                for item in msg:
                    if isinstance(item, dict) and item.get("id") == rid:
                        msg = item
                        break
                else:
                    raise MCPError(f"{method}: no reply for id {rid} in a batch")
            else:
                raise MCPError(f"{method}: unexpected reply {type(msg).__name__}")
        if "error" in msg:
            err = msg["error"]
            detail = err.get("message", err) if isinstance(err, dict) else err
            raise MCPError(f"{method}: {detail}")
        result = msg.get("result")
        return result if isinstance(result, dict) else {"value": result}

    # -- the two useful verbs ------------------------------------------
    def list_tools(self, timeout: float | None = None) -> list[dict[str, Any]]:
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

    def __enter__(self) -> "MCPHttpClient":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _peek(resp: Any, limit: int = 300) -> str:
    """A short, safe piece of an error body, for the sentence that reports it."""
    try:
        text = (resp.text or "").strip()
    except Exception:
        return "(body unreadable)"
    return text[:limit] if text else "(empty body)"


def client_for(config: MCPServerConfig) -> "MCPClient | MCPHttpClient":
    """The client that speaks this server's transport.

    A factory rather than a branch at each call site: which transport a config
    describes is the config's own property, and asking here means the answer
    cannot drift between the hub, the CLI and the tests.
    """
    return MCPHttpClient(config) if config.transport == "http" else MCPClient(config)


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
        self.clients: dict[str, MCPClient | MCPHttpClient] = {}
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
            client = client_for(cfg)
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
            # The config's tool gating is applied here, on the list the hub
            # goes on to use, rather than at call time. A tool the operator
            # switched off is then absent — not present and refusing — so the
            # report the agent reads and the registry it calls cannot disagree
            # about what this server offers.
            refused = [str(t["name"]) for t in tools if not cfg.allows(str(t["name"]))]
            if refused:
                tools = [t for t in tools if cfg.allows(str(t["name"]))]
                client.tools = tools
                self.report.problems.append(
                    f"{cfg.name}: {len(refused)} tool(s) not imported as configured: "
                    f"{', '.join(sorted(refused))}")
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
    "MCPClient", "MCPHttpClient", "MCPHub", "MCPError", "MCPServerConfig",
    "ImportReport", "PROTOCOL_VERSION", "IMPORTED_SCOPE", "client_for",
    "servers_from_config", "spec_from_remote", "tool_name", "render_result",
]
