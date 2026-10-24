"""The store must never create a directory that shadows this package.

This file exists because of a bug that hid for hours behind four plausible
wrong answers. The symptom was a console window flashing on the operator's
desktop every few seconds -- a `conhost.exe` per spawn, under Windows Terminal.

The flag was wrong at first, and fixing the flag did nothing, because the hook
that applies the flag lives in `autoforge/__init__.py` and `__init__.py` was
never being executed. `import autoforge` had stopped resolving to the package:
a stray directory named `autoforge` sat in the home directory, and the home
directory is `sys.path[0]` whenever a process runs with it as the cwd -- which
is how the agent runs. Python resolves that directory as a *namespace package*,
importing cleanly as an empty namespace with no `__init__.py`, so the hook went
missing silently. Every subprocess then started with no flag at all.

Where the stray directory came from is the test below: `_default_home()` chose
`~/autoforge` whenever `LOCALAPPDATA` was unset, `Path(...).parent.mkdir()`
created it, and the sandbox's own `env_allow` was dropping `LOCALAPPDATA` from
the environment it gives contained code. So a sandboxed job that built a store
manufactured the shadow, and the sandbox manufactured the conditions.

The three pins are separate on purpose: one for where state goes, one for the
environment the child is handed, one for the end-to-end consequence, so a
regression says which link broke.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from autoforge import store

_ROOT = str(Path(__file__).resolve().parents[1])


def test_default_home_never_collides_with_the_package_name(monkeypatch):
    """The invariant the whole bug turns on: a data dir named `autoforge`
    sitting in a directory Python imports from is a shadow, and `~` is such a
    directory. The fallback must therefore not be called `autoforge`."""
    for var in ("AUTOFORGE_HOME", "LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    home = store._default_home()
    leaf = os.path.basename(home.rstrip("\\/"))
    assert leaf != "autoforge", (
        "the home fallback would shadow the package when cwd is the home dir")
    assert leaf == ".autoforge"


def test_default_home_still_prefers_localappdata(monkeypatch):
    """The fix must not have moved anybody's existing ledger.

    `%LOCALAPPDATA%\\autoforge` is not on any import path, so it was never the
    problem and it stays where it was.
    """
    monkeypatch.delenv("AUTOFORGE_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", os.path.join("C:", os.sep, "Users", "someone", "AppData", "Local"))
    assert store._default_home() == os.path.join(
        os.environ["LOCALAPPDATA"], "autoforge")


def test_the_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("AUTOFORGE_HOME", os.path.join("D:", os.sep, "state"))
    assert store._default_home() == os.environ["AUTOFORGE_HOME"]


def test_env_allow_carries_the_vars_a_child_resolves_its_data_home_from():
    """Link two: the child has to be able to find its own data directory.

    It is not only that dropping `LOCALAPPDATA` produced a bad path -- it
    produced a *shadow*, and the child had no way to know. Both spellings are
    checked because only one of them is normally set.
    """
    from autoforge.forge.sandbox import Sandbox

    allow = Sandbox().env_allow
    assert "LOCALAPPDATA" in allow
    assert "APPDATA" in allow


@pytest.mark.skipif(os.name != "nt", reason="the shadow was measured on Windows")
def test_a_child_without_localappdata_does_not_shadow_the_package(tmp_path):
    """Link three: run it and look at the filesystem.

    A child is started the way the sandbox starts one -- trimmed environment,
    user profile redirected at a scratch directory -- and told to build a
    store. Afterwards the scratch directory must not contain a directory named
    `autoforge`; and, as the consequence that actually mattered, an interpreter
    whose cwd *is* that scratch directory must still import the real package
    and still get the quiet-Popen hook.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("LOCALAPPDATA", "XDG_DATA_HOME", "AUTOFORGE_HOME")}
    env["USERPROFILE"] = str(tmp_path)
    drive, tail = os.path.splitdrive(str(tmp_path))
    env["HOMEDRIVE"], env["HOMEPATH"] = drive, tail
    env["PYTHONPATH"] = _ROOT

    made = subprocess.run(
        [sys.executable, "-c",
         "from autoforge.store import ToolStore;"
         "print(ToolStore().db_path)"],
        cwd=str(tmp_path), env=env, capture_output=True,
        encoding="mbcs", errors="replace")
    assert made.returncode == 0, made.stderr[-600:]
    assert not (tmp_path / "autoforge").exists(), (
        "a directory named `autoforge` was created where Python imports from: "
        "this is the shadow that disabled the window hook")

    hook = subprocess.run(
        [sys.executable, "-c",
         "import subprocess, autoforge;"
         "print(autoforge.__file__);"
         "print(subprocess.Popen.__name__)"],
        cwd=str(tmp_path), env=env, capture_output=True,
        encoding="mbcs", errors="replace")
    assert hook.returncode == 0, hook.stderr[-600:]
    lines = [l for l in hook.stdout.splitlines() if l.strip()]
    assert lines[-1].strip() == "_QuietPopen", (
        "import from the home cwd lost the quiet-Popen hook: %r" % hook.stdout[-300:])


# ── sample_expect must survive the round trip ────────────────────────────────
#
# `sample_call` was persisted; `sample_expect` was not -- not in the DDL, the
# dataclass, `from_spec`, `to_spec`, the INSERT or the reader. A tool could
# therefore be forged with a perfectly good exam, verified against it, and then
# reloaded with the *call* but no *expectation*, so every later re-audit could
# only re-run it and never re-judge it. On this host that showed up as an
# execution check reporting "correctness is unprobed" for tools that had just
# passed it.
#
# These pins are separate on purpose: one for the declaration round-tripping,
# one for the migration that gives an existing store the column, one for the
# honest default on rows that predate it.

def _spec_with_sample():
    from autoforge.tools.spec import ToolSpec

    return ToolSpec(
        name="t",
        description="d",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        fn=lambda **_: "",
        code="def t(x):\n    return x\n",
        source="generated",
        sample_call={"x": "hello"},
        sample_expect="hello",
    )


def test_sample_expect_round_trips_through_the_store(tmp_path):
    from autoforge.store import ToolStore

    store = ToolStore(str(tmp_path / "s.db"))
    store.save_tool(_spec_with_sample())

    reloaded = ToolStore(str(tmp_path / "s.db")).load_all_tools()["t"]
    assert reloaded.sample_call == {"x": "hello"}
    assert reloaded.sample_expect == "hello", (
        "the expectation was lost in storage; a re-audit can re-run this tool "
        "but cannot judge it")


def test_an_existing_store_gains_the_column(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` does not alter a table that is already there.

    Built the way the running machine's store was: a database whose `tools`
    table predates the column.
    """
    import sqlite3

    from autoforge.store import ToolStore

    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE tools (name TEXT PRIMARY KEY, version INTEGER, "
                 "description TEXT, parameters TEXT, code TEXT, entry TEXT, "
                 "source TEXT, generator TEXT, probes TEXT, "
                 "effect_signature TEXT, sample_call TEXT, state TEXT, tags TEXT, "
                 "cost_hint TEXT, created_at REAL, verification TEXT, stats TEXT, "
                 "old_versions TEXT, updated_at REAL)")
    conn.commit()
    conn.close()

    # Opening it is what migrates it.
    ToolStore(str(path))

    cols = [r[1] for r in sqlite3.connect(str(path)).execute("pragma table_info(tools)")]
    assert "sample_expect" in cols


def test_a_row_written_before_the_column_reads_as_unprobed(tmp_path):
    """Absent must read as "un-probed", never as "no sample needed"."""
    from autoforge.store import ToolStore

    store = ToolStore(str(tmp_path / "s.db"))
    store.save_tool(_spec_with_sample())
    reloaded = ToolStore(str(tmp_path / "s.db")).load_all_tools()["t"]
    assert reloaded.sample_expect == "hello"

    # A spec that declared no expectation keeps the empty string, which is what
    # the verifier reads as "cannot judge".
    from autoforge.tools.spec import ToolSpec

    bare = ToolSpec(name="u", description="d", parameters={"type": "object"},
                    fn=lambda **_: "", code="def u():\n    return 1\n")
    store.save_tool(bare)
    again = ToolStore(str(tmp_path / "s.db")).load_all_tools()["u"]
    assert again.sample_expect == ""
