"""The lane: a shared file has one writer at a time, and the refusing is real.

Every test here exists because the advisory version failed on this machine. The
bus printed "declare your lane before editing a shared file", and two sessions
still wrote `forge/wsl_isolation.py` and `forge/sandbox.py` inside the same
minute -- one of them declaring its lane at 14:51:22, after the other had
created and committed those files. So what is asserted below is not "the lane is
recorded" (it always was) but "the second writer is stopped": a guard that never
refuses is indistinguishable from no guard.

Four bugs were caught while writing this file, and each one is of the shape
"looks right, misbehaves quietly":

  * `is_free` was inverted (`not any(...)`), so every path outside the temp
    directory was judged not worth guarding -- the guard was silent about
    exactly the files it exists to protect;
  * a lane claimed from a short-lived helper read back as takeable, because
    liveness is judged on the claiming process (right for a one-edit lane, a
    trap for a long one -- asserted in both directions below);
  * `claim(force=True)` could take a live lane: the advisory rule with an extra
    step. The flag is gone; breaking a live lane is `break_lane(why=...)`;
  * the guard's first test could not fail, because pytest's `tmp_path` lives
    inside the temp directory -- so "temp is free" sent every guarded-path test
    down the free path. Hence the explicit `scratch` fixture: which paths need
    no lane is policy, and it is passed in rather than assumed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from autoforge import lanes


@pytest.fixture()
def scratch(tmp_path):
    """The policy for this file: inside pytest's tmp, only `scratch/` is nobody's business.

    Not the whole temp directory -- `tmp_path` is inside it, so "temp is free"
    would make every guarded-path test here exercise the free path instead.
    """
    d = tmp_path / "scratch"
    d.mkdir(exist_ok=True)
    return (str(d),)


@pytest.fixture()
def store(tmp_path):
    return str(tmp_path / "lanes")


@pytest.fixture()
def target(tmp_path):
    """A path the policy guards: outside the declared scratch space."""
    d = tmp_path / "repo"
    d.mkdir(exist_ok=True)
    return str(d / "shared.py")


class TestWhichFilesNeedALane:
    def test_a_project_file_is_guarded(self, target, scratch):
        assert lanes.is_free(target, scratch) is False

    def test_the_declared_scratch_is_free(self, scratch, tmp_path):
        assert lanes.is_free(str(tmp_path / "scratch" / "x.py"), scratch) is True

    def test_claiming_a_free_path_writes_no_lane_file(self, store, scratch, tmp_path):
        free = str(tmp_path / "scratch" / "no_lane_needed.py")
        lanes.claim(free, session="A", root=store, scratch=scratch)
        assert not lanes.lane_file(free, store).exists()


class TestOneWriterAtATime:
    def test_the_second_session_is_refused(self, target, store, scratch):
        lanes.claim(target, session="A", name="worker-A", why="first",
                    root=store, scratch=scratch)
        with pytest.raises(lanes.LaneRefused) as exc:
            lanes.claim(target, session="B", name="worker-B", why="second",
                        root=store, scratch=scratch)
        # The refusal has to name who to talk to, or it is only "busy".
        assert "worker-A" in str(exc.value)
        assert exc.value.lane.name == "worker-A"

    def test_the_same_session_may_re_enter(self, target, store, scratch):
        lanes.claim(target, session="A", root=store, scratch=scratch)
        again = lanes.claim(target, session="A", root=store, scratch=scratch)
        assert again.state() == "live"

    def test_no_override_exists_on_claim(self):
        """`force=` is gone: it could only ever have meant "ignore a live lane"."""
        import inspect
        assert "force" not in inspect.signature(lanes.claim).parameters

    def test_breaking_a_live_lane_requires_a_reason(self, target, store, scratch):
        lanes.claim(target, session="A", name="worker-A", root=store, scratch=scratch)
        with pytest.raises(TypeError):
            lanes.break_lane(target, session="B")            # no `why`
        got = lanes.break_lane(target, session="B", name="worker-B",
                               why="A's process is wedged", root=store)
        assert got.session == "B"
        assert got.why == "A's process is wedged"
        assert got.renewed, "a takeover leaves a timestamp, or it leaves no trace"


class TestLanesDoNotOutliveTheirHolder:
    def test_an_expired_lane_is_takeable_without_ceremony(self, target, store, scratch):
        lanes.claim(target, session="A", ttl_s=0.01, root=store, scratch=scratch)
        time.sleep(0.05)
        assert lanes.read_lane(target, store).state() == "expired"
        assert lanes.claim(target, session="B", root=store,
                           scratch=scratch).session == "B"

    def test_a_dead_holder_releases_the_lane(self, target, store, scratch):
        """A lock whose owner was killed and which still blocks breaks the repo.

        Measured for real: the first claim made in this repository came from a
        short-lived helper subprocess and read back as takeable seconds later.
        Correct for a one-edit lane, wrong for a long one -- and either way it is
        the process table, not the timestamp, that decides.
        """
        lanes.claim(target, session="A", root=store, scratch=scratch)
        path = lanes.lane_file(target, store)
        d = json.loads(path.read_text(encoding="utf-8"))
        d.update({"session": "C", "pid": 999999, "expires_at": time.time() + 9999})
        path.write_text(json.dumps(d), encoding="utf-8")
        assert lanes.read_lane(target, store).state() == "holder-gone"
        assert lanes.claim(target, session="D", root=store,
                           scratch=scratch).session == "D"

    def test_a_lane_from_a_dead_subprocess_is_still_visible(self, target, store, scratch):
        """The other half of the same fact, asserted so it is a decision not a surprise.

        A lane must not be *silently dropped* when its holder dies -- it stays on
        record as takeable, because "who had this last" is what the next session
        needs in order to ask.
        """
        lanes.claim(target, session="ghost", root=store, scratch=scratch)
        path = lanes.lane_file(target, store)
        d = json.loads(path.read_text(encoding="utf-8"))
        d["pid"] = 999999
        path.write_text(json.dumps(d), encoding="utf-8")
        on_record = lanes.read_lane(target, store)
        assert on_record is not None and on_record.session == "ghost"

    def test_renewing_someone_elses_lane_is_refused(self, target, store, scratch):
        lanes.claim(target, session="A", root=store, scratch=scratch)
        with pytest.raises(lanes.LaneRefused):
            lanes.renew(target, session="B", root=store)

    def test_renewing_a_target_with_no_lane_says_so(self, target, store):
        """A different situation from refusal, and a different fix."""
        with pytest.raises(lanes.LaneMissing):
            lanes.renew(target, session="A", root=store)

    def test_release_only_affects_my_own_lane(self, target, store, scratch):
        lanes.claim(target, session="A", root=store, scratch=scratch)
        assert lanes.release(target, session="B", root=store) is False
        assert lanes.read_lane(target, store).state() == "live"
        assert lanes.release(target, session="A", root=store) is True
        assert lanes.read_lane(target, store).state() == "expired"


class TestTheHistorySurvives:
    def test_the_lane_file_is_never_deleted(self, target, store, scratch):
        """Shared disk, append-only: a takeover is visible, a deletion is not."""
        lanes.claim(target, session="A", root=store, scratch=scratch)
        lanes.release(target, session="A", root=store)
        assert lanes.lane_file(target, store).exists()
        assert lanes.read_lane(target, store).session == "A"

    def test_a_takeover_shows_in_the_report(self, target, store, scratch):
        lanes.claim(target, session="A", name="worker-A", ttl_s=0.01,
                    root=store, scratch=scratch)
        time.sleep(0.05)
        lanes.claim(target, session="B", name="worker-B", root=store,
                    scratch=scratch)
        text = lanes.report(store)
        assert "worker-B" in text and "live" in text

    def test_a_torn_lane_file_is_reported_as_absent_not_crashed(self, target, store, scratch):
        """A half-written lane must not take the session down with it."""
        path = lanes.lane_file(target, store)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"target": "x", "session"', encoding="utf-8")
        assert lanes.read_lane(target, store) is None
        assert lanes.claim(target, session="A", root=store,
                           scratch=scratch).session == "A"


class TestTheEditPathIsGuarded:
    """The guard must be on the write, not only in the library."""

    def test_the_self_edit_tool_refuses_over_a_live_lane(
            self, tmp_path, store, scratch, monkeypatch):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(lanes.__file__)))
        tool = os.path.join(repo, "scripts", "self_source_tentacle.py")
        if not os.path.exists(tool):
            pytest.skip("self_source_tentacle.py not in this checkout")
        victim = tmp_path / "repo" / "victim.py"
        victim.parent.mkdir(parents=True, exist_ok=True)
        victim.write_text("VALUE = 1\n", encoding="utf-8")

        # `store` is <tmp>/lanes and the module's own root is <bus_dir>/lanes, so
        # the bus dir has to be the parent. Pointing both at `store` would put
        # the lane one directory deeper than the tool looks, and the tool would
        # see no lane at all -- which is what "held by me / EDITED" was: not a
        # hole in the guard, a mismatch in where the two sides agree lanes live.
        monkeypatch.setenv("AUTOFORGE_BUS_DIR", str(tmp_path))
        # Two different sessions, deliberately. The first version of this test
        # set the editor to the same session that held the lane and the tool
        # correctly let it through as re-entry -- the walk-in was the test's
        # fault, not the guard's, and finding that out required running it.
        monkeypatch.setenv("AUTOFORGE_SESSION", "self-editing-session")
        # The fence is an absolute machine path by default; point it at this
        # test's repo. Otherwise the tool refuses for the wrong reason and the
        # test would pass while proving nothing about lanes.
        monkeypatch.setenv("AUTOFORGE_EDIT_ROOTS", str(victim.parent))
        lanes.claim(str(victim), session="peer-session", name="peer-session",
                    why="peer is editing", root=store, scratch=scratch)

        before = victim.read_text(encoding="utf-8")
        r = subprocess.run([sys.executable, tool, "--file", str(victim),
                            "--find", "VALUE = 1", "--replace", "VALUE = 2",
                            "--expect", "1", "--why", "collision test"],
                           capture_output=True, text=True, timeout=200,
                           errors="replace", cwd=repo)
        assert r.returncode == 6, r.stdout + r.stderr
        assert "REFUSED" in r.stdout
        assert "peer-session" in r.stdout
        assert victim.read_text(encoding="utf-8") == before, \
            "the file was written despite the refusal"

    def test_the_report_line_names_live_lanes(self, target, store, scratch):
        lanes.claim(target, session="A", name="worker-A", why="editing",
                    root=store, scratch=scratch)
        text = lanes.report(store)
        assert "live" in text and "worker-A" in text and "shared.py" in text


class TestTheEnvironmentCannotLieToIt:
    def test_own_session_id_honours_the_override(self, monkeypatch):
        monkeypatch.setenv("AUTOFORGE_SESSION", "session-from-env")
        assert lanes.own_session_id() == "session-from-env"

    def test_lane_root_follows_the_bus_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTOFORGE_BUS_DIR", str(tmp_path))
        assert lanes.lanes_dir() == tmp_path / "lanes"



    """Forging writes no .py to disk -- it writes a ROW, into a shared store.

    So the collision on this path is not two writers on a file, it is two
    sessions claiming the same tool NAME. `save_tool` is an INSERT OR REPLACE,
    which means the second writer does not fail loudly: it destroys the first
    session's freshly forged tool and leaves two ledger entries that both look
    like successes. Measured on the demand side: 20 clusters of one job under
    different tool names, 145 more retried under one name
    (docs/DUPLICATE_NEEDS.md). The lane is what makes the second writer stop.
    """

    def _spec(self, name, desc):
        from autoforge.tools.spec import ToolSpec, ToolState

        def _fn():
            return 1

        return ToolSpec(name=name, description=desc,
                        parameters={"type": "object", "properties": {}},
                        fn=_fn, code=f"def {name}():\n    return 1\n",
                        source="forged", state=ToolState.ACTIVE)

    def test_a_live_lane_on_the_name_refuses_the_save(self, store, monkeypatch):
        # The claim goes through the ENV, not through `root=store`: `save_tool`
        # resolves lanes from AUTOFORGE_BUS_DIR, and pointing the two sides at
        # different directories is exactly how this test passed while proving
        # nothing the first time it was written.
        from autoforge.store import ToolStore
        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        monkeypatch.setenv("AUTOFORGE_SESSION", "session-A")
        st = ToolStore(os.path.join(store, "t.db"))
        try:
            st.save_tool(self._spec("reader", "A's reader"))
            lanes.claim(lanes.resource("tool/reader"), session="session-B",
                        name="peer", why="forging same name")
            with pytest.raises(ValueError) as exc:
                st.save_tool(self._spec("reader", "B's reader"))
            assert "peer" in str(exc.value)
            # And the original survived -- a refusal that still replaced the row
            # would be the same silent loss with an exception attached.
            assert st.load_all_tools()["reader"].description == "A's reader"
        finally:
            st.close()

    def test_my_own_saves_are_never_blocked(self, store, monkeypatch):
        """Re-entry, or `evolve_tool` would be unable to save its own winner."""
        from autoforge.store import ToolStore
        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        monkeypatch.setenv("AUTOFORGE_SESSION", "session-A")
        st = ToolStore(os.path.join(store, "t.db"))
        try:
            st.save_tool(self._spec("reader", "v1"))
            st.save_tool(self._spec("reader", "v2"))
            assert st.load_all_tools()["reader"].description == "v2"
        finally:
            st.close()

    def test_the_lane_is_released_after_the_write(self, store, monkeypatch):
        """The lane marks the write in progress, not the name forever.

        A permanent claim would make the first forge of a name block every later
        save of it -- evolve included -- which is the opposite of the intent.
        """
        from autoforge.store import ToolStore
        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        monkeypatch.setenv("AUTOFORGE_SESSION", "session-A")
        st = ToolStore(os.path.join(store, "t.db"))
        try:
            st.save_tool(self._spec("reader", "v1"))
            again = ToolStore(os.path.join(store, "other.db"))
            try:
                again.save_tool(self._spec("reader", "v2"))
            finally:
                again.close()
        finally:
            st.close()

    def test_a_broken_lane_store_does_not_block_saving(self, store, monkeypatch):
        """A lane module that cannot load must not make the shelf read-only."""
        from autoforge.store import ToolStore
        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        st = ToolStore(os.path.join(store, "t.db"))
        try:
            st.save_tool(self._spec("reader", "v1"))
        finally:
            st.close()

    def test_overwrite_is_available_but_has_to_be_asked_for(self, store, monkeypatch):
        from autoforge.store import ToolStore
        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        monkeypatch.setenv("AUTOFORGE_SESSION", "session-A")
        st = ToolStore(os.path.join(store, "t.db"))
        try:
            st.save_tool(self._spec("reader", "v1"))
            lanes.claim(lanes.resource("tool/reader"), session="session-B",
                        name="peer", why="forging same name")
            st.save_tool(self._spec("reader", "forced"), allow_overwrite=True)
            assert st.load_all_tools()["reader"].description == "forced"
        finally:
            st.close()


class TestTheWriterStampOnAVersion:
    """A live lane covers a write in flight; it does NOT cover two sessions a
    few seconds apart. That gap was measured end to end: A saved a tool, released
    the lane, B saved the same name and replaced A's work, and nothing anywhere
    objected. Closing it needs a per-write writer stamp, so `tool_versions` grew a
    `session` column -- and a column added to a table that already exists is
    exactly the kind of change that gets believed without ever running.
    """

    def _mk(self, name, desc, body):
        from autoforge.tools.spec import ToolSpec, ToolState

        return ToolSpec(name=name, description=desc,
                        parameters={"type": "object", "properties": {}},
                        fn=lambda: 1,
                        code=f"# {body}\ndef {name}():\n    return 1\n",
                        source="forged", state=ToolState.ACTIVE)

    def test_the_column_is_added_to_a_store_that_predates_it(self, store, monkeypatch):
        """The migration path, on a store created without the column."""
        import sqlite3

        from autoforge.store import ToolStore

        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        db = os.path.join(store, "old.db")
        os.makedirs(store, exist_ok=True)
        raw = sqlite3.connect(db)
        raw.execute("CREATE TABLE tool_versions (tool TEXT NOT NULL, version INTEGER NOT"
                    " NULL, blob_sha TEXT NOT NULL, parent_version INTEGER, verification"
                    " TEXT NOT NULL DEFAULT '{}', archived_at REAL NOT NULL,"
                    " PRIMARY KEY (tool, version))")
        raw.execute("INSERT INTO tool_versions (tool, version, blob_sha,"
                    " parent_version, verification, archived_at)"
                    " VALUES (?,?,?,?,?,?)", ("old", 1, "sha", None, "{}", 1.0))
        raw.commit()
        raw.close()

        st = ToolStore(db)
        try:
            cols = [d[1] for d in st._conn.execute("PRAGMA table_info(tool_versions)")]
            assert "session" in cols
            # The pre-existing row survives, and reads as UNKNOWN rather than as
            # someone else's -- refusing on it would block every tool the agent
            # had before today.
            assert st._conn.execute("SELECT COUNT(*) FROM tool_versions").fetchone()[0] == 1
            assert st.tools_written_by_other_sessions() == []
        finally:
            st.close()

    def test_the_migration_is_idempotent(self, store, monkeypatch):
        from autoforge.store import ToolStore

        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        db = os.path.join(store, "twice.db")
        st = ToolStore(db)
        st.close()
        st = ToolStore(db)
        st.close()

    def test_a_foreign_write_seconds_ago_refuses_the_next_save(self, store, monkeypatch):
        """The measured collision, in-process: two sessions, no overlap in time."""
        import subprocess
        import sys

        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        db = os.path.join(store, "x.db")
        child = (
            "import os,sys\n"
            "sys.path.insert(0, __REPR__)\n"
            "os.environ['AUTOFORGE_BUS_DIR']=sys.argv[2]\n"
            "os.environ['AUTOFORGE_SESSION']=sys.argv[3]\n"
            "from autoforge.store import ToolStore\n"
            "from autoforge.tools.spec import ToolSpec, ToolState\n"
            "n=sys.argv[1]\n"
            "st=ToolStore(sys.argv[4])\n"
            "s=ToolSpec(name=n, description='peer', parameters={'type':'object',"
            "'properties':{}}, fn=lambda:1, code='# peer\\ndef %s():\\n    return 1\\n'%n,"
            " source='forged', state=ToolState.ACTIVE)\n"
            "try:\n"
            "    st.save_tool(s); print('SAVED')\n"
            "except ValueError as e:\n"
            "    print('REFUSED', e)\n"
        ).replace('__REPR__', repr(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        p = os.path.join(store, "child.py")
        os.makedirs(store, exist_ok=True)
        open(p, "w", encoding="utf-8").write(child)
        r = subprocess.run([sys.executable, p, "shared_name", store, "peer-session", db],
                           capture_output=True, text=True, timeout=120, errors="replace")
        assert "SAVED" in r.stdout, r.stdout + r.stderr

        from autoforge.store import ToolStore
        monkeypatch.setenv("AUTOFORGE_SESSION", "my-session")
        st = ToolStore(db)
        try:
            assert st.tools_written_by_other_sessions() == ["shared_name"]
            with pytest.raises(ValueError) as exc:
                st.save_tool(self._mk("shared_name", "mine", "mine"))
            assert "peer-session" not in str(exc.value)   # names the age, not a guess
            assert "second" not in str(exc.value)
        finally:
            st.close()

    def test_the_identical_body_is_not_refused(self, store, monkeypatch):
        """Nobody loses anything when the body is the same; refusing would only
        teach the caller to pass allow_overwrite out of habit."""
        import subprocess
        import sys

        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        db = os.path.join(store, "same.db")
        os.makedirs(store, exist_ok=True)
        from autoforge.store import ToolStore
        monkeypatch.setenv("AUTOFORGE_SESSION", "first")
        st = ToolStore(db)
        st.save_tool(self._mk("twin", "one", "same body"))
        st.close()
        monkeypatch.setenv("AUTOFORGE_SESSION", "second")
        st = ToolStore(db)
        try:
            st.save_tool(self._mk("twin", "two", "same body"))
        finally:
            st.close()

    def test_a_deliberate_replacement_is_allowed(self, store, monkeypatch):
        from autoforge.store import ToolStore

        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        db = os.path.join(store, "force.db")
        monkeypatch.setenv("AUTOFORGE_SESSION", "first")
        st = ToolStore(db)
        st.save_tool(self._mk("swap", "one", "v1"))
        st.close()
        monkeypatch.setenv("AUTOFORGE_SESSION", "second")
        st = ToolStore(db)
        try:
            st.save_tool(self._mk("swap", "two", "v2"), allow_overwrite=True)
            assert st.load_all_tools()["swap"].description == "two"
        finally:
            st.close()

    def test_an_unstamped_row_is_unknown_not_foreign(self, store, monkeypatch):
        """Every tool forged before today has session=''. Treating that as
        "someone else's" would make the whole existing shelf unsaveable."""
        from autoforge.store import ToolStore

        monkeypatch.setenv("AUTOFORGE_BUS_DIR", store)
        db = os.path.join(store, "legacy.db")
        st = ToolStore(db)
        st.save_tool(self._mk("legacy", "one", "v1"))
        st._conn.execute("UPDATE tool_versions SET session=''")
        st._conn.commit()
        st.close()
        monkeypatch.setenv("AUTOFORGE_SESSION", "someone-new")
        st = ToolStore(db)
        try:
            assert st.tools_written_by_other_sessions() == []
            st.save_tool(self._mk("legacy", "two", "v2"), allow_overwrite=False)
        finally:
            st.close()
