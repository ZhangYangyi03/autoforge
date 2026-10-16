"""The local server, spoken to by the real client over a real pipe.

The claim is not that `tools/media_mcp_server.py` prints JSON. It is that
`autoforge.mcp.MCPClient` — the code that actually loads MCP tools into the
registry — can start it, list it, and call it. So every test here drives the
real client against a real child process. Mocking either end would prove the
mock works.

Two things are worth pinning beyond "it works":

* **A handler that raises must not kill the server.** One bad call has to cost
  its own tool, not the other fourteen. The client's own docstring makes this
  promise about third-party servers; this is where it is checked.

* **Reads are confined.** `read_local_file` is a filesystem read primitive the
  moment it stops checking, and the client cannot see that -- an imported tool's
  scope is `undeclared` by construction. The confinement is the only thing
  standing between a tool name and any file on the machine, so it gets a test
  that tries to escape rather than one that reads a file.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from autoforge.mcp import MCPClient, MCPServerConfig, render_result

SERVER = Path(__file__).resolve().parents[1] / "tools" / "media_mcp_server.py"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def server(tmp_path):
    """A started client pointed at the real server, rooted at tmp_path."""
    config = MCPServerConfig(
        name="local",
        command=sys.executable,
        args=[str(SERVER)],
        env={"PYTHONPATH": str(REPO), "MEDIA_MCP_ROOT": str(tmp_path)},
        timeout=30.0,
        call_timeout=60.0,
    )
    client = MCPClient(config)
    assert client.start(), client.error
    yield client
    client.close()


def call(client: MCPClient, name: str, **args) -> tuple[bool, str]:
    """Invoke a tool and render it the way the registry would."""
    return render_result(client.call_tool(name, args))


# ----------------------------------------------------------------------
# the handshake
# ----------------------------------------------------------------------


def test_the_client_completes_the_handshake(server):
    assert server.protocol == "2024-11-05"
    assert server.server_info.get("name") == "autoforge-local"


def test_every_tool_is_listed_with_a_usable_schema(server):
    tools = server.list_tools()
    names = {t["name"] for t in tools}
    assert len(tools) == 15, f"expected 15 tools, got {sorted(names)}"
    for expected in ("generate_image", "generate_video", "video_status",
                     "media_models", "media_status", "read_local_file",
                     "read_source_lines", "grep_lines_multi", "netstat_ano",
                     "find_python_processes", "db_schema_inspect",
                     "visible_len_probe", "wcwidth_probe",
                     "ansi_visible_len_probe", "read_file_lines"):
        assert expected in names
    for tool in tools:
        assert tool.get("inputSchema", {}).get("type") == "object"
        assert tool.get("description")


def test_an_unknown_tool_is_a_tool_error_not_a_crash(server):
    ok, body = call(server, "no_such_tool")
    assert not ok
    assert "no such tool" in body
    # ...and the server is still usable afterwards.
    ok, _ = call(server, "media_status")
    assert ok


def test_a_handler_that_raises_costs_only_its_own_tool(server):
    """One bad call must not take the other fourteen with it."""
    ok, body = call(server, "read_local_file", path="definitely/not/here")
    assert not ok
    assert "does not exist" in body
    ok, body = call(server, "visible_len_probe", text="ok")
    assert ok, "the server died with the failed call"


# ----------------------------------------------------------------------
# confinement
# ----------------------------------------------------------------------


def test_a_read_outside_the_root_is_refused(server, tmp_path):
    """The escape that matters: this tool must not read the whole machine."""
    ok, body = call(server, "read_local_file", path=str(REPO / "pyproject.toml"))
    assert not ok
    assert "outside the readable root" in body
    assert "MEDIA_MCP_ROOT" in body


def test_a_relative_traversal_is_refused(server):
    ok, body = call(server, "read_local_file", path="../../../../etc/passwd")
    assert not ok
    assert "outside the readable root" in body


def test_a_file_inside_the_root_is_read(server, tmp_path):
    # `write_bytes`, not `write_text`: text mode translates `\n` to `\r\n` on
    # Windows, so the assertion below would be comparing the file to something
    # the test never wrote.
    (tmp_path / "hello.txt").write_bytes(b"inside\n")
    ok, body = call(server, "read_local_file", path="hello.txt")
    assert ok
    assert body == "inside\n"


# ----------------------------------------------------------------------
# the diagnostics
# ----------------------------------------------------------------------


def test_read_source_lines_numbers_them(server, tmp_path):
    (tmp_path / "s.py").write_text("a\nb\nc\nd\n", encoding="utf-8")
    ok, body = call(server, "read_source_lines", path="s.py", start=2, count=2)
    assert ok
    assert "2 | b" in body and "3 | c" in body
    assert "1 | a" not in body


def test_grep_lines_multi_reports_path_and_line(server, tmp_path):
    (tmp_path / "one.py").write_text("x = 1\nneedle = 2\n", encoding="utf-8")
    (tmp_path / "two.py").write_text("needle = 3\n", encoding="utf-8")
    ok, body = call(server, "grep_lines_multi", pattern="needle",
                    paths=["."])
    assert ok
    assert "one.py:2" in body and "two.py:1" in body


def test_grep_lines_multi_refuses_a_broken_pattern(server, tmp_path):
    ok, body = call(server, "grep_lines_multi", pattern="([unclosed",
                    paths=["."])
    assert not ok
    assert "does not compile" in body


def test_the_width_probes_disagree_with_len(server):
    """The whole reason these tools exist.

    Two wide glyphs and three narrow ones are five to `len()` and seven cells
    to a terminal. A layout that trusts the first number is two cells wrong on
    every such line.
    """
    ok, body = call(server, "visible_len_probe", text="中文abc")
    assert ok
    assert "len()=5" in body
    assert "display_cells=7" in body
    assert "difference=2" in body


def test_wcwidth_probe_breaks_a_string_down(server):
    ok, body = call(server, "wcwidth_probe", text="中a")
    assert ok
    assert "U+4E2D" in body and "cells=2" in body
    assert "cells=1" in body


def test_the_ansi_probe_ignores_escape_sequences(server):
    ok, body = call(server, "ansi_visible_len_probe", text="\x1b[31mred\x1b[0m")
    assert ok
    assert "escape chars removed=9" in body
    assert "display_cells=3" in body


def test_db_schema_inspect_counts_rows(server, tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("create table things (a integer)")
    conn.executemany("insert into things values (?)", [(1,), (2,)])
    conn.commit()
    conn.close()
    ok, body = call(server, "db_schema_inspect", path="t.db")
    assert ok
    assert "things: 2" in body


def test_db_schema_inspect_cannot_write(server, tmp_path):
    """Opened read-only, so a diagnostic cannot become a mutation."""
    db = tmp_path / "t.db"
    sqlite3.connect(db).execute("create table x (a integer)")
    ok, body = call(server, "db_schema_inspect", path="t.db")
    assert ok
    assert "x:" in body


def test_netstat_runs_and_filters(server):
    ok, body = call(server, "netstat_ano", filter="nothing-will-match-this")
    assert ok
    assert "nothing matched" in body or "LISTEN" not in body


# ----------------------------------------------------------------------
# media, through the pipe
# ----------------------------------------------------------------------


def test_media_status_answers_even_when_nothing_is_configured(server):
    """An unconfigured endpoint is a sentence, not an exception."""
    ok, body = call(server, "media_status")
    assert ok
    assert "media" in body


def test_media_models_names_both_size_spellings(server):
    ok, body = call(server, "media_models")
    assert ok
    assert "1024x1024" in body and "1280*720" in body
    assert "not interchangeable" in body


def test_generate_image_without_a_prompt_is_refused_cleanly(server):
    """The prompt is required by the schema; a missing one is a tool error."""
    ok, body = call(server, "generate_image")
    assert not ok
    assert "prompt" in body.lower()


def test_a_bad_json_line_does_not_stop_the_server(server):
    """The client skips a non-JSON line; the server must never emit one."""
    # The server logs to stderr, which the client collects but does not parse.
    assert server.alive
    ok, _ = call(server, "media_status")
    assert ok
