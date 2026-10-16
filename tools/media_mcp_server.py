#!/usr/bin/env python3
"""A local MCP server that actually starts.

The server this replaces was reached over HTTP at `127.0.0.1:8000/mcp`. That is
a design that fails in a specific and silent way: a URL that nobody is listening
on does not report itself as missing, it reports itself as *unreachable*, and the
tools it offered simply stop appearing on the menu. Twenty-one tools were
whitelisted in the config for it and none of them could be called, which looks
like a broken agent rather than a stopped process.

So this one speaks the **stdio** transport instead. There is no port, no service
to remember to start, and no way for it to be half-up: the pipe is the tool's
lifetime. `autoforge/mcp.py` already supports both transports and picks between
them on `url` vs `command`; this is the `command` side.

Two groups of tools, and they are here for different reasons:

* **Media generation.** `generate_image` / `generate_video` reach the same
  gateway `autoforge.media` does, so the capability is available to any MCP
  client and not only to the built-in tool list. The HTTP logic is *imported*,
  not reimplemented -- a second copy of a client is how one of them ends up
  subtly wrong.

* **The local diagnostics.** These were agent-authored tools that got
  whitelisted into a config and then lost their host. They are cheap, they are
  the tools that would have caught the CJK width bug that took two sessions to
  find, and they are worth having somewhere permanent.

**Reads are confined.** `read_local_file` and its siblings resolve every path
against the roots named in `MEDIA_MCP_ROOT` (default: the working directory) and
refuse anything that escapes them. A server that will read any path it is handed
is a filesystem read primitive wearing a tool name, and the client cannot see
that: an imported tool's scope is `undeclared` by construction. Confining here is
the one place the check can actually be made.

`MEDIA_MCP_ROOT` takes more than one root, separated by the platform's path
separator (`;` on Windows, `:` elsewhere), because one is not enough in
practice: an agent that inspects its own state and its own source needs both the
store under `~/.autoforge` and the package under site-packages, and those are
not under a common parent.

The API key is read from this process's own environment. It is never taken from
the agent's environment -- `MCPClient._env` deliberately does not forward it --
so the operator writes the key into the server's own `env` block, where it was
put on purpose.

Usage:
    media_mcp_server.py            # stdio, per the MCP transport
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "autoforge-local"
SERVER_VERSION = "1.0.0"

#: How much text one read returns before it is truncated. A tool result rides
#: in the agent's prompt, so an unbounded read is a way to spend a whole
#: context window by accident.
READ_LIMIT = 200_000
GREP_LIMIT = 200

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# ----------------------------------------------------------------------
# the protocol
# ----------------------------------------------------------------------
def send(msg: dict[str, Any]) -> None:
    """One JSON-RPC message on stdout. Flushed, because the pipe is unbuffered.

    Nothing else may ever be written to stdout: the client treats a non-JSON
    line as a log line and skips it, so a stray `print` costs a diagnostic
    rather than the session -- but it is still the wrong stream. Logs go to
    stderr.
    """
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(text: str) -> None:
    """The diagnostic stream. The client keeps the tail of it for error messages."""
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def ok(rid: Any, blocks: list[dict[str, Any]], is_error: bool = False) -> None:
    send({"jsonrpc": "2.0", "id": rid,
          "result": {"content": blocks, "isError": is_error}})


def text(rid: Any, body: str, is_error: bool = False) -> None:
    ok(rid, [{"type": "text", "text": body}], is_error)


# ----------------------------------------------------------------------
# the sandbox for reads
# ----------------------------------------------------------------------
def _roots() -> list[Path]:
    """The directories reads are allowed inside.

    More than one, because a single root cannot cover an agent that inspects
    both its own store and its own installed source -- those live under
    `~/.autoforge` and site-packages, with no common parent worth allowing.
    """
    raw = os.environ.get("MEDIA_MCP_ROOT") or os.getcwd()
    parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
    return [Path(p).resolve() for p in parts] or [Path(os.getcwd()).resolve()]


def confined(path: str) -> Path:
    """A path inside one of the roots, or a refusal that lists them.

    Resolved before the check, so `..` and a symlink both land where they
    actually point rather than where they claim to.
    """
    if not path:
        raise ValueError("a path is required")
    roots = _roots()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    resolved = candidate.resolve()
    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved
    listed = ", ".join(str(r) for r in roots)
    raise ValueError(
        f"{path!r} is outside the readable roots ({listed}). Widen them by "
        f"setting MEDIA_MCP_ROOT in this server's env block if that is "
        f"intended.")


def read_text(path: str, limit: int = READ_LIMIT) -> str:
    resolved = confined(path)
    if not resolved.exists():
        raise ValueError(f"{resolved} does not exist")
    if resolved.is_dir():
        raise ValueError(f"{resolved} is a directory")
    data = resolved.read_bytes()[:limit]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", "replace")


def _numbered(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines)))
    return "\n".join(f"{start + i:>{width}} | {line}"
                     for i, line in enumerate(lines))


# ----------------------------------------------------------------------
# width, which is not len()
# ----------------------------------------------------------------------
def _cell_width(ch: str) -> int:
    """Display cells for one character.

    `len()` counts codepoints, and a terminal does not: a CJK glyph occupies
    two cells and a combining mark occupies none. Layout code that uses `len()`
    mis-measures every line containing either, which is the bug these three
    probes exist to make visible.
    """
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _probe_text(args: dict[str, Any]) -> str:
    value = args.get("text")
    if value is None:
        raise ValueError("'text' is required")
    return str(value)


# ----------------------------------------------------------------------
# media, imported rather than reimplemented
# ----------------------------------------------------------------------
_MEDIA: Any = None
_MEDIA_PROBLEM = ""


def _media() -> Any:
    """The shared client, built once. Returns None with a reason if unavailable."""
    global _MEDIA, _MEDIA_PROBLEM
    if _MEDIA is not None or _MEDIA_PROBLEM:
        return _MEDIA
    try:
        from autoforge.media import MediaClient, media_from_config
    except ImportError as exc:
        _MEDIA_PROBLEM = (f"autoforge.media is not importable ({exc}); put the "
                          f"package on PYTHONPATH in this server's env block")
        return None
    cfg: dict[str, Any] = {}
    raw = os.environ.get("AUTOFORGE_MEDIA_CONFIG")
    if raw:
        try:
            cfg = json.loads(Path(raw).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _MEDIA_PROBLEM = f"could not read AUTOFORGE_MEDIA_CONFIG: {exc}"
            return None
    else:
        # The operator's own config, which is where the endpoint already is.
        home = os.environ.get("AUTOFORGE_HOME") or os.path.join(
            os.path.expanduser("~"), ".autoforge")
        candidate = Path(home) / "config.json"
        if candidate.exists():
            try:
                cfg = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                _MEDIA_PROBLEM = f"could not read {candidate}: {exc}"
                return None
    client, problem = media_from_config(cfg)
    if client is None:
        _MEDIA_PROBLEM = problem or ("no media endpoint is configured; set "
                                     "media.base_url in the config")
        return None
    _MEDIA = client
    return _MEDIA


def _need_media() -> Any:
    client = _media()
    if client is None:
        raise ValueError(_MEDIA_PROBLEM)
    return client


# ----------------------------------------------------------------------
# the tools
# ----------------------------------------------------------------------
def t_generate_image(args: dict[str, Any]) -> str:
    made = _need_media().generate_image(
        str(args["prompt"]),
        model=str(args.get("model") or ""),
        size=str(args.get("size") or ""),
        save=args.get("save") or None)
    return (f"{made.report()}\n"
            f"Nothing here can tell whether it is the image you meant -- read "
            f"it back with a vision model if that matters.")


def t_generate_video(args: dict[str, Any]) -> str:
    made = _need_media().generate_video(
        str(args["prompt"]),
        model=str(args.get("model") or ""),
        size=str(args.get("size") or ""),
        save=args.get("save") or None)
    return f"{made.report()}\ntask {made.task_id}"


def t_video_status(args: dict[str, Any]) -> str:
    seen = _need_media().video_task(str(args["task_id"]))
    return json.dumps(seen, ensure_ascii=False, indent=2)


def t_media_models(args: dict[str, Any]) -> str:
    from autoforge.media import (
        DEFAULT_IMAGE_MODEL, DEFAULT_VIDEO_MODEL, IMAGE_SIZES, VIDEO_SIZES)
    client = _media()
    lines = [
        f"endpoint: {client.base_url if client else '(not configured)'}",
        f"default image model: {DEFAULT_IMAGE_MODEL}",
        f"default video model: {DEFAULT_VIDEO_MODEL}",
        f"image sizes: {', '.join(IMAGE_SIZES)}",
        f"video sizes: {', '.join(VIDEO_SIZES)}",
        "note: images spell a size '1024x1024' and video spells it '1280*720'; "
        "the separator is not interchangeable.",
    ]
    if _MEDIA_PROBLEM:
        lines.append(f"problem: {_MEDIA_PROBLEM}")
    return "\n".join(lines)


def t_read_local_file(args: dict[str, Any]) -> str:
    return read_text(str(args["path"]), int(args.get("max_bytes") or READ_LIMIT))


def t_read_file_lines(args: dict[str, Any]) -> str:
    start = max(1, int(args.get("start") or 1))
    count = int(args.get("count") or 200)
    lines = read_text(str(args["path"])).splitlines()
    chunk = lines[start - 1:start - 1 + count]
    if not chunk:
        return f"(no lines; the file has {len(lines)})"
    return "\n".join(chunk)


def t_read_source_lines(args: dict[str, Any]) -> str:
    start = max(1, int(args.get("start") or 1))
    count = int(args.get("count") or 200)
    lines = read_text(str(args["path"])).splitlines()
    chunk = lines[start - 1:start - 1 + count]
    if not chunk:
        return f"(no lines; the file has {len(lines)})"
    return _numbered(chunk, start)


def t_grep_lines_multi(args: dict[str, Any]) -> str:
    pattern = str(args["pattern"])
    paths = args.get("paths") or []
    if isinstance(paths, str):
        paths = [paths]
    flags = re.IGNORECASE if args.get("ignore_case") else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"the pattern does not compile: {exc}") from None
    limit = int(args.get("max_hits") or GREP_LIMIT)

    hits: list[str] = []
    for raw in paths:
        resolved = confined(str(raw))
        files = ([resolved] if resolved.is_file()
                 else sorted(p for p in resolved.rglob("*") if p.is_file()))
        for path in files:
            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(body.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{path}:{n}: {line[:200]}")
                    if len(hits) >= limit:
                        return "\n".join(hits) + f"\n(stopped at {limit} hits)"
    return "\n".join(hits) if hits else "(no matches)"


def t_netstat_ano(args: dict[str, Any]) -> str:
    exe = shutil.which("netstat")
    if not exe:
        raise ValueError("netstat is not on PATH")
    done = subprocess.run([exe, "-ano"], capture_output=True, text=True,
                          timeout=30, errors="replace")
    needle = str(args.get("filter") or "")
    lines = done.stdout.splitlines()
    if needle:
        lines = [ln for ln in lines if needle.lower() in ln.lower()]
    return "\n".join(lines[:400]) or "(nothing matched)"


def t_find_python_processes(args: dict[str, Any]) -> str:
    """Which python processes exist, so two agents on one db can be told apart."""
    if os.name == "nt":
        exe = shutil.which("wmic")
        if not exe:
            raise ValueError("wmic is not available on this host")
        done = subprocess.run(
            [exe, "process", "where", "name like '%python%'",
             "get", "ProcessId,CommandLine", "/format:list"],
            capture_output=True, text=True, timeout=30, errors="replace")
        return done.stdout.strip() or "(none found)"
    done = subprocess.run(["ps", "-eo", "pid,ppid,etime,cmd"],
                          capture_output=True, text=True, timeout=30,
                          errors="replace")
    rows = [ln for ln in done.stdout.splitlines() if "python" in ln.lower()]
    return "\n".join(rows) or "(none found)"


def t_db_schema_inspect(args: dict[str, Any]) -> str:
    import sqlite3

    resolved = confined(str(args["path"]))
    if not resolved.exists():
        raise ValueError(f"{resolved} does not exist")
    conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute(
            "select name from sqlite_master where type='table' order by name")]
        out = []
        for name in tables:
            try:
                count = conn.execute(
                    f'select count(*) from "{name}"').fetchone()[0]
            except sqlite3.Error:
                count = "?"
            out.append(f"{name}: {count}")
        return "\n".join(out) or "(no tables)"
    finally:
        conn.close()


def t_visible_len_probe(args: dict[str, Any]) -> str:
    body = _ANSI.sub("", _probe_text(args))
    cells = sum(_cell_width(ch) for ch in body)
    return (f"len()={len(body)}  display_cells={cells}  "
            f"difference={cells - len(body)}")


def t_wcwidth_probe(args: dict[str, Any]) -> str:
    body = _ANSI.sub("", _probe_text(args))
    rows = [f"{ch!r}  U+{ord(ch):04X}  "
            f"east_asian={unicodedata.east_asian_width(ch)}  "
            f"combining={bool(unicodedata.combining(ch))}  "
            f"cells={_cell_width(ch)}" for ch in body]
    return "\n".join(rows[:200]) or "(empty)"


def t_ansi_visible_len_probe(args: dict[str, Any]) -> str:
    raw = _probe_text(args)
    stripped = _ANSI.sub("", raw)
    return (f"raw len={len(raw)}  stripped len={len(stripped)}  "
            f"escape chars removed={len(raw) - len(stripped)}  "
            f"display_cells={sum(_cell_width(ch) for ch in stripped)}")


def t_media_status(args: dict[str, Any]) -> str:
    client = _media()
    if client is None:
        return f"media: not configured -- {_MEDIA_PROBLEM}"
    return client.report()


TOOLS: list[dict[str, Any]] = [
    {
        "name": "generate_image",
        "description": (
            "Generate an image from a text prompt and write it to a file. "
            "Returns the path and byte count. The file is written before this "
            "returns, so the path can be handed straight to a vision model to "
            "check -- which is the only way to know it is the image you meant."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "what to draw"},
                "model": {"type": "string", "description": "override the image model"},
                "size": {"type": "string", "description": "e.g. 1024x1024 (images use 'x')"},
                "save": {"type": "string", "description": "path to write to"},
            },
            "required": ["prompt"],
        },
        "handler": t_generate_image,
    },
    {
        "name": "generate_video",
        "description": (
            "Generate a short video from a text prompt. Slower by orders of "
            "magnitude than generate_image: it is a task on the far side that "
            "is polled to completion. Blocks until the task finishes or fails."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "what to render, including the motion"},
                "model": {"type": "string", "description": "override the video model"},
                "size": {"type": "string", "description": "e.g. 1280*720 (video uses '*')"},
                "save": {"type": "string", "description": "path to write to"},
            },
            "required": ["prompt"],
        },
        "handler": t_generate_video,
    },
    {
        "name": "video_status",
        "description": (
            "Read an existing video task by id without waiting for it. Use it "
            "on a task id from an earlier turn instead of generating again."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        "handler": t_video_status,
    },
    {
        "name": "media_models",
        "description": (
            "The configured endpoint, the default image and video models, and "
            "the accepted size spellings. Call it before guessing a model id."
        ),
        "inputSchema": {"type": "object", "properties": {}},
        "handler": t_media_models,
    },
    {
        "name": "media_status",
        "description": "Whether image and video generation are configured, and why not if not.",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": t_media_status,
    },
    {
        "name": "read_local_file",
        "description": (
            "Read a text file. Confined to the readable roots, which are the "
            "working directory unless MEDIA_MCP_ROOT says otherwise."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer"},
            },
            "required": ["path"],
        },
        "handler": t_read_local_file,
    },
    {
        "name": "read_file_lines",
        "description": "Read a line range from a file, unnumbered.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "description": "1-based"},
                "count": {"type": "integer"},
            },
            "required": ["path"],
        },
        "handler": t_read_file_lines,
    },
    {
        "name": "read_source_lines",
        "description": (
            "Read a line range from a file with line numbers, so a location can "
            "be cited without counting."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "description": "1-based"},
                "count": {"type": "integer"},
            },
            "required": ["path"],
        },
        "handler": t_read_source_lines,
    },
    {
        "name": "grep_lines_multi",
        "description": (
            "Search a regular expression across one or more files or "
            "directories and return path:line matches."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "ignore_case": {"type": "boolean"},
                "max_hits": {"type": "integer"},
            },
            "required": ["pattern", "paths"],
        },
        "handler": t_grep_lines_multi,
    },
    {
        "name": "netstat_ano",
        "description": "Listening and established sockets with owning pids. An optional filter narrows the output.",
        "inputSchema": {
            "type": "object",
            "properties": {"filter": {"type": "string"}},
        },
        "handler": t_netstat_ano,
    },
    {
        "name": "find_python_processes",
        "description": (
            "Python processes with their pids and command lines. Two agents on "
            "one store cannot otherwise be told apart."
        ),
        "inputSchema": {"type": "object", "properties": {}},
        "handler": t_find_python_processes,
    },
    {
        "name": "db_schema_inspect",
        "description": "Tables in a sqlite file and their row counts. Opened read-only.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "handler": t_db_schema_inspect,
    },
    {
        "name": "visible_len_probe",
        "description": (
            "Compare len() against terminal display cells for a string. Use it "
            "on any text whose layout looks wrong: a CJK glyph is two cells and "
            "a combining mark is none, and len() sees both as one."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "handler": t_visible_len_probe,
    },
    {
        "name": "wcwidth_probe",
        "description": "Per-character width breakdown: east-asian class, combining status, cell count.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "handler": t_wcwidth_probe,
    },
    {
        "name": "ansi_visible_len_probe",
        "description": "Display width of a string after ANSI escape sequences are removed.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "handler": t_ansi_visible_len_probe,
    },
]

_BY_NAME = {tool["name"]: tool for tool in TOOLS}


# ----------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------
def handle_call(rid: Any, name: str, args: dict[str, Any]) -> None:
    tool = _BY_NAME.get(name)
    if tool is None:
        # A tool result, not a protocol error: the server understood the call
        # and is reporting that it has no such tool. The distinction matters to
        # the client, which renders the two differently.
        text(rid, f"no such tool: {name}", is_error=True)
        return
    try:
        text(rid, tool["handler"](args))
    except Exception as exc:                      # noqa: BLE001
        # Everything a handler can raise is a tool failure, never a crash: a
        # server that dies takes every other tool it offers with it.
        log(f"{name} failed: {type(exc).__name__}: {exc}")
        text(rid, f"{type(exc).__name__}: {exc}", is_error=True)


def main() -> None:
    log(f"{SERVER_NAME} {SERVER_VERSION} ready; readable roots "
        f"{', '.join(str(r) for r in _roots())}")
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            log(f"ignoring a non-JSON line: {raw[:120]}")
            continue
        if not isinstance(msg, dict):
            continue

        method = msg.get("method")
        rid = msg.get("id")

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }})
        elif method == "notifications/initialized":
            pass                                   # no response is expected
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "tools": [{k: v for k, v in tool.items() if k != "handler"}
                          for tool in TOOLS]}})
        elif method == "tools/call":
            params = msg.get("params") or {}
            handle_call(rid, str(params.get("name") or ""),
                        params.get("arguments") or {})
        elif rid is not None:
            send({"jsonrpc": "2.0", "id": rid, "error": {
                "code": -32601, "message": f"method not found: {method}"}})


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
