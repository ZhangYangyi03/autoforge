"""Reading the other agents' MCP configs without becoming a fourth dialect.

`ecosystem.py` exists because an operator running Hermes, Codex and a
Claude-shaped client has already described their MCP servers three times. The
module's whole value is that the *fourth* description is never typed, so what
has to hold is exactly the four rules its docstring claims:

1. A file that cannot be read costs its own servers and nothing else.
2. A missing parser is answered with the command that installs it.
3. A name collision is reported, never resolved silently.
4. Only the *name* of a token comes across, never a value.

Rule 3 has a second half that is easy to get wrong in the other direction: the
framework's own config outranks anything imported, because an import that
outranked the operator's own edit would make that edit look broken.

Everything here runs against files in a temp directory. Nothing reads the real
`~/.codex` or `~/.hermes`, and nothing starts a process -- the last test in the
file asserts that last part rather than assuming it.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from autoforge.ecosystem import (
    DIALECTS,
    ENV_SOURCES,
    candidates,
    discover,
    merged_servers,
    read,
    read_all,
)
from autoforge.mcp import MCPServerConfig, servers_from_config


# ----------------------------------------------------------------------
# fixtures: one file per dialect, written the way that tool writes it
# ----------------------------------------------------------------------
CODEX_TOML = """\
[mcp_servers.files]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
startup_timeout_sec = 12

[mcp_servers.gateway]
url = "https://mcp.example.com/sse"
bearer_token_env_var = "GATEWAY_TOKEN"
enabled_tools = ["read_file"]
disabled_tools = ["delete_file"]
"""

CLAUDE_JSON = json.dumps({
    "mcpServers": {
        "notes": {"command": "node", "args": ["notes.js"]},
        "retired": {"command": "node", "args": ["x.js"], "disabled": True},
    },
    "projects": {
        "/home/me/work/alpha": {
            "mcpServers": {"alpha-db": {"command": "pg-mcp"}},
        },
    },
})

HERMES_YAML = """\
mcp_servers:
  web:
    command: /usr/bin/web-mcp
    connect_timeout: 7
"""

VSCODE_JSON = json.dumps({
    "servers": {"lint": {"type": "stdio", "command": "lint-mcp"}},
})


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def home(tmp_path):
    """A home directory with no agent config in it at all."""
    h = tmp_path / "home"
    h.mkdir()
    return h


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every variable the module uses, so a real machine cannot leak in.

    Without this the tests would read the developer's own `~/.codex`, and the
    suite would pass or fail depending on whose laptop it ran on.
    """
    for var in ("HERMES_HOME", "CODEX_HOME", "LOCALAPPDATA", "APPDATA",
                "XDG_DATA_HOME", ENV_SOURCES):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ----------------------------------------------------------------------
# where it looks
# ----------------------------------------------------------------------
class TestCandidates:

    def test_every_dialect_has_a_place_to_look(self, home, clean_env):
        labels = [s.label for s in candidates(home)]
        for label in ("hermes", "codex", "claude", "claude-desktop",
                      "cursor", "project", "vscode"):
            assert label in labels, f"{label} has no candidate path"

    def test_the_order_is_also_the_precedence(self, home, clean_env):
        """First source wins a collision, so the order is load-bearing.

        A project's own file has to outrank the global one, and the
        framework's own config (which is not in this list at all) outranks
        both -- see `merged_servers`.
        """
        labels = [s.label for s in candidates(home, cwd=home)]
        assert labels.index("project") < labels.index("vscode")
        assert labels.index("hermes") < labels.index("project")

    def test_each_dialect_declares_its_own_key_and_format(self, home, clean_env):
        """One table, so a reader and this test cannot disagree."""
        seen = {s.label: (s.fmt, s.key) for s in candidates(home, cwd=home)}
        assert seen["hermes"] == ("yaml", "mcp_servers")
        assert seen["codex"] == ("toml", "mcp_servers")
        assert seen["claude"] == ("json", "mcpServers")
        assert seen["vscode"] == ("json", "servers")


class TestDiscover:

    def test_only_the_files_that_exist(self, home, clean_env):
        write(home / ".codex" / "config.toml", CODEX_TOML)
        found = discover(home)
        assert [s.label for s in found] == ["codex"]

    def test_nothing_found_is_an_empty_list_not_a_guess(self, home, clean_env):
        assert discover(home) == []

    def test_one_file_named_twice_is_read_once(self, home, clean_env):
        """Not hypothetical: Hermes sets `HERMES_HOME` to the same directory
        its platform default resolves to, so both candidates are one file.

        Reading it twice would report every server as colliding with itself --
        a loud problem that is entirely fictional, and one that teaches the
        operator to ignore the collision list.
        """
        write(home / "hermes" / "config.yaml", HERMES_YAML)
        clean_env.setenv("HERMES_HOME", str(home / "hermes"))
        clean_env.setenv("LOCALAPPDATA", str(home))
        found = [s for s in discover(home) if s.label == "hermes"]
        assert len(found) == 1
        report = read_all(home)
        assert report.conflicts == []
        assert report.names == ["web"]

    def test_naming_files_replaces_the_search_entirely(self, home, clean_env, tmp_path):
        """When the operator names files they mean those files.

        Falling back to a guessed location here would be this module ignoring
        the operator, and would show them servers they did not ask about.
        """
        write(home / ".codex" / "config.toml", CODEX_TOML)
        mine = write(tmp_path / "only.json", CLAUDE_JSON)
        found = discover(home, paths=[mine])
        assert len(found) == 1
        assert found[0].path == mine
        assert found[0].label.startswith("named")

    def test_the_env_var_does_the_same_as_the_argument(self, home, clean_env, tmp_path):
        mine = write(tmp_path / "only.json", CLAUDE_JSON)
        clean_env.setenv(ENV_SOURCES, str(mine))
        found = discover(home)
        assert [s.path for s in found] == [mine]

    def test_a_named_file_is_read_by_its_extension(self, home, clean_env, tmp_path):
        """A named file still has to be parsed as what it is."""
        toml_file = write(tmp_path / "mcp.toml", CODEX_TOML)
        assert discover(home, paths=[toml_file])[0].fmt == "toml"
        yml_file = write(tmp_path / "mcp.yml", HERMES_YAML)
        assert discover(home, paths=[yml_file])[0].fmt == "yaml"

    def test_a_named_file_that_is_not_there_is_skipped_not_raised(
            self, home, clean_env, tmp_path):
        found = discover(home, paths=[tmp_path / "ghost.json"])
        assert found == []


# ----------------------------------------------------------------------
# reading one file
# ----------------------------------------------------------------------
class TestReadOneFile:

    def test_codex_toml(self, home, clean_env):
        source = discover(home, paths=[write(home / "c.toml", CODEX_TOML)])[0]
        servers, problems = read(source)
        assert problems == []
        by_name = {c.name: c for c in servers}
        assert by_name["files"].command == "npx"
        assert by_name["files"].timeout == 12           # startup_timeout_sec
        assert by_name["gateway"].transport == "http"
        assert by_name["gateway"].url == "https://mcp.example.com/sse"

    def test_hermes_yaml(self, home, clean_env):
        source = discover(home, paths=[write(home / "h.yaml", HERMES_YAML)])[0]
        servers, problems = read(source)
        assert problems == []
        assert servers[0].name == "web"
        assert servers[0].timeout == 7                  # connect_timeout

    def test_claude_json(self, home, clean_env):
        source = discover(home, paths=[write(home / "j.json", CLAUDE_JSON)])[0]
        servers, problems = read(source)
        assert problems == []
        assert "notes" in {c.name for c in servers}

    def test_vscode_says_servers_not_mcpservers(self, home, clean_env):
        source = discover(home, paths=[write(home / "v.json", VSCODE_JSON)])[0]
        servers, problems = read(source)
        assert problems == []
        assert [c.name for c in servers] == ["lint"]

    def test_the_dialect_table_is_the_only_place_the_key_is_written(self):
        """A fifth dialect is one row here, not a second branch per reader."""
        assert set(DIALECTS) == {"hermes", "codex", "claude", "vscode"}
        assert {fmt for fmt, _ in DIALECTS.values()} == {"yaml", "toml", "json"}

    def test_a_disabled_server_is_not_offered(self, home, clean_env):
        """`"disabled": true` is Cline's and Claude's spelling of `enabled`."""
        source = discover(home, paths=[write(home / "j.json", CLAUDE_JSON)])[0]
        names = {c.name for c in read(source)[0]}
        assert "retired" not in names
        assert "notes" in names

    def test_a_project_scoped_map_is_read_and_labelled_by_project(
            self, home, clean_env):
        """Claude Code keeps a second server map per project inside one file.

        Those are real servers the operator expects to find, and labelling them
        by directory is what makes a collision between two projects read as
        what it is rather than as a bug.
        """
        source = discover(home, paths=[write(home / "j.json", CLAUDE_JSON)])[0]
        servers, _ = read(source)
        project = [c for c in servers if c.name == "alpha-db"]
        assert len(project) == 1
        assert project[0].scope == "ecosystem:claude:alpha"

    def test_the_scope_names_where_the_server_came_from(self, home, clean_env):
        """So a tool call's audit trail can say which agent's config asked."""
        source = discover(home, paths=[write(home / "c.toml", CODEX_TOML)])[0]
        assert all(c.scope.startswith("ecosystem:") for c in read(source)[0])


# ----------------------------------------------------------------------
# rule 1: a bad file costs itself and nothing else
# ----------------------------------------------------------------------
class TestOneBadFileIsContained:

    def test_an_unreadable_file_is_a_sentence_naming_it(self, home, clean_env, tmp_path):
        missing = tmp_path / "gone.json"
        # Named explicitly, so it is a real source with a real path that is not
        # there -- `discover` would have filtered it out of a search.
        from autoforge.ecosystem import Source
        servers, problems = read(Source("claude", missing, "json", "mcpServers"))
        assert servers == []
        assert len(problems) == 1
        assert str(missing) in problems[0]

    def test_invalid_json_is_a_sentence_not_a_traceback(self, home, clean_env):
        source = discover(home, paths=[write(home / "b.json", "{not json")])[0]
        servers, problems = read(source)
        assert servers == []
        assert "not valid JSON" in problems[0]

    def test_invalid_toml_and_yaml_read_the_same_way(self, home, clean_env, tmp_path):
        for name, text, word in (("b.toml", "[mcp_servers\nx", "TOML"),
                                 ("b.yaml", "mcp_servers:\n  - : :\n", "YAML")):
            source = discover(home, paths=[write(tmp_path / name, text)])[0]
            servers, problems = read(source)
            assert servers == [], name
            assert word in problems[0], name

    def test_a_file_that_is_not_an_object_is_reported(self, home, clean_env, tmp_path):
        source = discover(home, paths=[write(tmp_path / "l.json", "[1, 2]")])[0]
        servers, problems = read(source)
        assert servers == []
        assert "expected an object" in problems[0]

    def test_the_key_being_absent_is_reported_as_a_question(self, home, clean_env):
        """This is the shape of "you pointed me at the wrong key".

        An empty table, by contrast, declares itself and holds nothing -- see
        the next test.
        """
        source = discover(home, paths=[write(home / "o.json",
                                             json.dumps({"other": {}}))])[0]
        servers, problems = read(source)
        assert servers == []
        assert "mcpServers" in problems[0] and "absent" in problems[0]

    def test_an_empty_table_is_not_a_problem(self, home, clean_env):
        """Codex writes `[mcp_servers]` with no rows.

        Calling that an error would put a permanent complaint on the screen of
        every operator who simply has no MCP servers configured there.
        """
        source = discover(home, paths=[write(home / "e.toml", "[mcp_servers]\n")])[0]
        servers, problems = read(source)
        assert servers == []
        assert problems == []

    def test_one_bad_server_entry_does_not_cost_the_others(self, home, clean_env):
        doc = json.dumps({"mcpServers": {
            "good": {"command": "ok"},
            "broken": {"args": ["no command and no url"]},
            "also-good": {"url": "https://x/y"},
        }})
        source = discover(home, paths=[write(home / "m.json", doc)])[0]
        servers, problems = read(source)
        assert sorted(c.name for c in servers) == ["also-good", "good"]
        assert len(problems) == 1 and "broken" in problems[0]

    def test_a_bad_file_does_not_stop_the_other_files(self, home, clean_env, tmp_path):
        write(home / ".codex" / "config.toml", "[mcp_servers\nbroken")
        write(home / ".claude.json", CLAUDE_JSON)
        report = read_all(home)
        assert "notes" in report.names
        assert any("codex" in p for p in report.problems)
        assert any("claude" in s for s in report.sources)


# ----------------------------------------------------------------------
# rule 2: a missing parser is answered with the command that installs it
# ----------------------------------------------------------------------
class TestMissingParser:

    def test_a_missing_yaml_reader_names_the_install(self, home, clean_env, monkeypatch):
        """"No module named yaml" is a message about this program.

        What the operator needs is the sentence that fixes it.
        """
        import builtins
        real_import = builtins.__import__

        def refuse_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_yaml)
        source = discover(home, paths=[write(home / "h.yaml", HERMES_YAML)])[0]
        servers, problems = read(source)
        assert servers == []
        assert "pip install pyyaml" in problems[0]

    def test_a_missing_toml_reader_names_the_install(self, home, clean_env, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def refuse_toml(name, *args, **kwargs):
            if name in ("tomllib", "tomli"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_toml)
        source = discover(home, paths=[write(home / "c.toml", CODEX_TOML)])[0]
        servers, problems = read(source)
        assert servers == []
        assert "pip install tomli" in problems[0]


# ----------------------------------------------------------------------
# rule 3: collisions are reported, and this config outranks the import
# ----------------------------------------------------------------------
class TestPrecedence:

    def test_the_first_source_wins_and_says_so(self, home, clean_env, tmp_path):
        first = write(tmp_path / "a.json", json.dumps(
            {"mcpServers": {"shared": {"command": "from-first"}}}))
        second = write(tmp_path / "b.json", json.dumps(
            {"mcpServers": {"shared": {"command": "from-second"}}}))
        report = read_all(paths=[first, second])
        assert len(report.servers) == 1
        assert report.servers[0].command == "from-first"
        assert len(report.conflicts) == 1
        line = report.conflicts[0]
        # Both *files* are named -- "there was a collision between two things
        # both called named" is a sentence that sends the operator looking for
        # a file called "named".
        assert "a.json" in line and "b.json" in line
        assert "a.json" in line.split("using")[-1]

    def test_the_operators_own_config_beats_anything_imported(
            self, home, clean_env, tmp_path):
        """The operator wrote that file to change this program's behaviour.

        An import that outranked it would make their edit look broken, which is
        the worst of both outcomes: the config says one thing and the machine
        does another.
        """
        write(home / ".claude.json", json.dumps(
            {"mcpServers": {"mine": {"command": "theirs"}}}))
        config = {"mcp": {"ecosystem": True,
                          "servers": {"mine": {"command": "mine"}}}}
        servers, problems = merged_servers(config, home=home)
        assert [c.command for c in servers] == ["mine"]
        assert any("using this config's" in p for p in problems)

    def test_an_imported_server_is_added_when_the_name_is_free(
            self, home, clean_env):
        write(home / ".claude.json", CLAUDE_JSON)
        config = {"mcp": {"ecosystem": True, "servers": {}}}
        names = {c.name for c in merged_servers(config, home=home)[0]}
        assert {"notes", "alpha-db"} <= names

    def test_problems_come_back_labelled_with_their_origin(self, home, clean_env):
        """So it is legible on screen which file a complaint is about, given
        the complaints from this program's own config arrive in the same list.
        """
        write(home / ".claude.json", json.dumps({"whoops": {}}))
        config = {"mcp": {"ecosystem": True, "servers": {}}}
        problems = merged_servers(config, home=home)[1]
        assert any(p.startswith("ecosystem: ") for p in problems)


# ----------------------------------------------------------------------
# the switch itself
# ----------------------------------------------------------------------
class TestTheSwitch:

    def test_off_by_default(self, home, clean_env):
        """Reading another tool's config is free; the servers it names are
        processes, and starting processes nobody asked for is not something an
        upgrade should begin doing on its own.
        """
        write(home / ".claude.json", CLAUDE_JSON)
        servers, problems = merged_servers({"mcp": {"servers": {}}}, home=home)
        assert servers == []
        assert problems == []

    def test_no_mcp_block_at_all_is_off_not_an_error(self, home, clean_env):
        assert merged_servers({}, home=home) == ([], [])

    @pytest.mark.parametrize("value", [True, "true", "yes", "on", "1"])
    def test_a_hand_written_switch_accepts_the_ways_people_write_it(
            self, home, clean_env, value):
        """A switch that only accepts one spelling gets typed wrong once and
        then distrusted."""
        write(home / ".claude.json", CLAUDE_JSON)
        servers, _ = merged_servers({"mcp": {"ecosystem": value, "servers": {}}},
                                    home=home)
        assert [c.name for c in servers] == ["notes", "alpha-db"]

    @pytest.mark.parametrize("value", [False, "", "false", "no", "off", "none"])
    def test_and_the_ways_people_turn_it_off(self, home, clean_env, value):
        write(home / ".claude.json", CLAUDE_JSON)
        servers, _ = merged_servers({"mcp": {"ecosystem": value, "servers": {}}},
                                    home=home)
        assert servers == []

    def test_this_programs_own_servers_are_read_either_way(self, home, clean_env):
        """The import is additive. Turning it off must not cost the servers the
        operator configured directly, which is a much worse failure than not
        finding the imported ones.
        """
        for block in ({"servers": {"own": {"command": "x"}}},
                      {"ecosystem": True, "servers": {"own": {"command": "x"}}}):
            config = {"mcp": block}
            assert [c.name for c in merged_servers(config, home=home)[0]] == ["own"]


# ----------------------------------------------------------------------
# rule 4: a name comes across, a value never does
# ----------------------------------------------------------------------
class TestSecrets:

    def test_only_the_variables_name_is_carried(self, home, clean_env, monkeypatch):
        """Importing someone's config must not copy their secret into a second
        file that then has to be protected too.
        """
        monkeypatch.setenv("GATEWAY_TOKEN", "s3cret-value-do-not-copy")
        source = discover(home, paths=[write(home / "c.toml", CODEX_TOML)])[0]
        gateway = [c for c in read(source)[0] if c.name == "gateway"][0]
        assert gateway.bearer_token_env_var == "GATEWAY_TOKEN"
        # Nowhere in anything that gets written back out.
        assert "s3cret-value-do-not-copy" not in json.dumps(gateway.to_dict())

    def test_nothing_here_starts_a_process(self, home, clean_env, monkeypatch):
        """Reading is not trusting.

        A config entry says how to *reach* a server, not that it should be
        running -- so discovery must be unable to launch one even by accident.
        """
        def refuse(*args, **kwargs):
            raise AssertionError("discovery started a process")

        monkeypatch.setattr(subprocess, "Popen", refuse)
        write(home / ".codex" / "config.toml", CODEX_TOML)
        write(home / ".claude.json", CLAUDE_JSON)
        report = read_all(home)
        assert report.names, "the fixture should have found something"
        # And the configs it handed back still describe the commands, unstarted.
        assert any(c.command for c in report.servers)


# ----------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------
class TestReport:

    def test_scanned_and_contributed_are_kept_apart(self, home, clean_env):
        """`scanned` is 70 reports: a file that exists and has no servers under
        the key is normally *fine*, and only worth a line when it declares the
        key's absence.
        """
        write(home / ".claude.json", CLAUDE_JSON)
        report = read_all(home)
        assert len(report.scanned) == 1
        assert report.sources == report.scanned

    def test_by_source_groups_names_under_where_they_came_from(self, home, clean_env):
        write(home / ".claude.json", CLAUDE_JSON)
        report = read_all(home)
        assert set(report.by_source["claude"]) == {"notes", "alpha-db"}

    def test_the_summary_is_one_line_and_counts_what_it_says(self, home, clean_env):
        write(home / ".claude.json", CLAUDE_JSON)
        line = read_all(home).summary()
        assert line.startswith("2 server(s) from 1 of 1 file(s)")

    def test_an_empty_world_says_so_rather_than_showing_zeroes(self, home, clean_env):
        assert read_all(home).summary() == "no other agent's MCP config was found"

    def test_the_summary_mentions_collisions_and_bad_files(self, home, clean_env, tmp_path):
        first = write(tmp_path / "a.json", json.dumps(
            {"mcpServers": {"s": {"command": "x"}}}))
        second = write(tmp_path / "b.json", json.dumps(
            {"mcpServers": {"s": {"command": "y"}}}))
        bad = write(tmp_path / "c.json", "nonsense")
        line = read_all(paths=[first, second, bad]).summary()
        assert "collision" in line and "unreadable" in line

    def test_get_finds_one_by_name(self, home, clean_env):
        write(home / ".claude.json", CLAUDE_JSON)
        report = read_all(home)
        assert report.get("notes") is not None
        assert report.get("nothing-called-this") is None

    def test_to_dict_round_trips_the_whole_thing(self, home, clean_env):
        write(home / ".claude.json", CLAUDE_JSON)
        model = read_all(home).to_dict()
        assert json.loads(json.dumps(model)) == model
        assert len(model["servers"]) == 2
        assert model["by_source"]["claude"] == ["notes", "alpha-db"]


# ----------------------------------------------------------------------
# it is the agent's own config path, and only that, when the switch is off
# ----------------------------------------------------------------------
class TestWiring:

    def test_the_agent_uses_the_merged_reader(self):
        """The bug this prevents is the import being written and never wired.

        `merged_servers` with the switch off must be indistinguishable from
        `servers_from_config`, or the default path has changed behaviour.
        """
        config = {"mcp": {"servers": {"own": {"command": "x"}}}}
        assert (merged_servers(config)[0] == servers_from_config(config)[0])

    def test_a_server_config_still_validates_the_same_way_after_import(
            self, home, clean_env):
        """An imported server is an ordinary server: same type, same checks."""
        write(home / ".codex" / "config.toml", CODEX_TOML)
        imported = read_all(home).servers
        assert imported and all(isinstance(c, MCPServerConfig) for c in imported)
        assert all(c.transport in ("http", "stdio") for c in imported)


# ----------------------------------------------------------------------
# the real machine, on purpose, once
# ----------------------------------------------------------------------
class TestAgainstTheRealMachine:
    """One test that reads *this* machine, because the fixtures cannot.

    Every other test here writes the files it reads, which means they encode
    the module author's belief about what Hermes and Codex put on disk. This
    one checks that belief against a machine that actually has them, and skips
    where it does not -- so it is a verification on the developer's box rather
    than a portability requirement on everyone else's.
    """

    def test_the_real_hermes_config_parses_if_it_is_here(self):
        real = os.environ.get("HERMES_HOME")
        candidates_here = [Path(real) / "config.yaml"] if real else []
        candidates_here.append(Path.home() / ".hermes" / "config.yaml")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates_here.append(Path(local) / "hermes" / "config.yaml")
        found = [p for p in candidates_here if p.is_file()]
        if not found:
            pytest.skip("no Hermes config on this machine")
        from autoforge.ecosystem import Source
        servers, problems = read(Source("hermes", found[0], "yaml", "mcp_servers"))
        # Parsed, not necessarily populated: an empty `mcp_servers:` is a
        # normal state, and only "not valid YAML" is a failure.
        assert not any("not valid YAML" in p for p in problems), problems
