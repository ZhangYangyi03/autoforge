"""Skills: procedures that get written down, offered, and ranked by use.

The gap this file pins is not storage. Tools already persisted — run a forge,
restart, the tool is still there. What the agent had no way to keep was a
*procedure*: the order of steps, the flag that bit last time, the check that
made a deploy trustworthy. So every session it re-derived the same approach,
and could not tell a procedure it had run forty times from one it had never
tried.

Four properties carry the feature, and each one has a way of failing quietly:

  * the file is the truth — content on disk, editable by a human, surviving
    the agent that wrote it. A skill that only exists in a database is not a
    skill, it is a cache.
  * the load count is the retrieval signal — written by the act of loading,
    never reset by a re-scan, so it cannot drift from what happened.
  * bodies stay out of the prompt — the menu is an index. If bodies ship every
    turn then "progressive disclosure" is a word in a docstring, and the agent
    is paying for forty procedures to read one.
  * a malformed file is reported, not skipped — a silently skipped skill reads
    to the agent as "I have no procedure for this", which is the original bug
    wearing a new hat.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from autoforge.agent import ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import MockLLMClient
from autoforge.route.router import SkillRouter
from autoforge.skills import (
    SkillError, SkillLibrary, default_skill_dirs, parse_tags,
    render_skill_text, split_frontmatter, valid_name,
)
from autoforge.store import ToolStore


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture()
def home(tmp_path):
    d = tmp_path / "home"
    (d / "skills").mkdir(parents=True)
    return d


@pytest.fixture()
def store(tmp_path):
    s = ToolStore(str(tmp_path / "af.db"))
    yield s
    s.close()


@pytest.fixture()
def project(tmp_path):
    d = tmp_path / "proj" / "skills"
    d.mkdir(parents=True)
    return d


def _write_raw(directory, name, text):
    p = directory / (name + ".md")
    p.write_text(text, encoding="utf-8")
    return p


GOOD = """---
name: deploy-verify
description: prove a deploy took effect
when_to_use: after any deploy, before calling it done
tags: deploy, verify
---
1. hit /healthz
2. compare the commit sha
"""


def _lib(home, store=None, project=None):
    dirs = []
    if project is not None:
        dirs.append(("project", str(project)))
    dirs.append(("user", str(home / "skills")))
    return SkillLibrary(store=store, dirs=dirs)


def _agent(store=None):
    return ForgeAgent(MockLLMClient(), store=store, policy=FULL_FREEDOM)


# ======================================================================
# the format: strict enough that a broken file is loud
# ======================================================================
class TestFrontmatter:
    def test_a_well_formed_skill_parses(self):
        meta, body = split_frontmatter(GOOD)
        assert meta["name"] == "deploy-verify"
        assert meta["description"] == "prove a deploy took effect"
        assert meta["when_to_use"] == "after any deploy, before calling it done"
        assert body.startswith("1. hit /healthz")

    def test_missing_fence_is_an_error_not_a_guess(self):
        with pytest.raises(SkillError, match="does not start with"):
            split_frontmatter("# just a body\n")

    def test_unterminated_fence_is_an_error(self):
        with pytest.raises(SkillError, match="no closing"):
            split_frontmatter("---\nname: x\ndescription: y\n")

    def test_no_description_is_refused(self):
        # Without a description the skill cannot be routed, and a skill that
        # cannot be routed is invisible — so this must fail at parse time
        # rather than sit in the library looking healthy.
        with pytest.raises(SkillError, match="no description"):
            split_frontmatter("---\nname: x\n---\nbody\n")

    def test_unknown_keys_are_tolerated(self):
        meta, _ = split_frontmatter("---\ndescription: d\nauthor: someone\n---\nb\n")
        assert meta["author"] == "someone"

    def test_quoted_values_lose_their_quotes(self):
        meta, _ = split_frontmatter('---\ndescription: "a: b"\n---\nb\n')
        assert meta["description"] == "a: b"

    def test_a_header_line_without_a_colon_is_an_error(self):
        with pytest.raises(SkillError, match="not 'key: value'"):
            split_frontmatter("---\ndescription: d\noops\n---\nb\n")

    def test_tags_split_on_commas(self):
        assert parse_tags("a, b ,, c") == ["a", "b", "c"]
        assert parse_tags("") == []

    def test_render_round_trips_through_parse(self):
        text = render_skill_text("t", "d", "w", ["a", "b"], "body here")
        meta, body = split_frontmatter(text)
        assert (meta["name"], meta["description"], meta["when_to_use"]) == ("t", "d", "w")
        assert meta["tags"] == "a, b"
        assert body == "body here"

    def test_names_are_validated_not_sanitised(self):
        assert valid_name("deploy-verify_2")
        assert not valid_name("Deploy Verify")
        assert not valid_name("")
        assert not valid_name("-leading")
        assert not valid_name("x" * 65)


# ======================================================================
# the file is the truth
# ======================================================================
class TestFilesAreTheTruth:
    def test_a_hand_written_file_shows_up(self, home):
        _write_raw(home / "skills", "handwritten", GOOD.replace("deploy-verify", "handwritten"))
        lib = _lib(home)
        skills = lib.scan()
        assert [s.name for s in skills] == ["handwritten"]
        assert skills[0].source == "user"

    def test_the_name_falls_back_to_the_filename(self, home):
        _write_raw(home / "skills", "fromfile", "---\ndescription: d\n---\nb\n")
        assert [s.name for s in _lib(home).scan()] == ["fromfile"]

    def test_a_human_edit_is_picked_up_on_the_next_scan(self, home):
        p = _write_raw(home / "skills", "s", "---\ndescription: first\n---\nb\n")
        lib = _lib(home)
        lib.scan()
        p.write_text("---\ndescription: second\n---\nb\n", encoding="utf-8")
        assert lib.scan()[0].description == "second"

    def test_a_file_deleted_by_a_human_leaves_the_menu(self, home, store):
        p = _write_raw(home / "skills", "gone", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        os.unlink(p)
        assert lib.scan() == []
        assert store.skill_rows() == []

    def test_only_markdown_is_read(self, home):
        (home / "skills" / "notes.txt").write_text("not a skill", encoding="utf-8")
        assert _lib(home).scan() == []


# ======================================================================
# a scan that sees fewer skills is not proof that they were deleted
# ======================================================================
class TestScanIsNotEvidenceOfDeletion:
    """A scan may drop a row only when it has evidence the file is gone.

    `scan()` read "known but not seen this time" as "the file was deleted" and
    dropped the row -- every row, whenever the scan happened to look somewhere
    that did not hold them. Another `dirs`, a cwd without a `skills/`, or a
    `home` that is not the one the rows were written under all produce that,
    and the cost is every `loads` count in the table, which is the ranking
    input and cannot be recovered. Deleting a row is unrecoverable; keeping a
    stale one costs a ranking nudge the next real scan corrects.
    """

    def _elsewhere(self, tmp_path):
        d = tmp_path / "elsewhere"
        d.mkdir()
        return d

    def test_a_scan_of_somewhere_else_keeps_the_rows(self, home, store, tmp_path):
        _write_raw(home / "skills", "kept", "---\ndescription: d\n---\nb\n")
        _lib(home, store).scan()
        assert [r["name"] for r in store.skill_rows()] == ["kept"]
        # A second library over the same store, pointed at an empty directory.
        # It has never seen "kept"; it must not conclude that it is gone.
        SkillLibrary(store=store, dirs=[("user", str(self._elsewhere(tmp_path)))]).scan()
        assert [r["name"] for r in store.skill_rows()] == ["kept"]

    def test_the_load_counts_survive_it(self, home, store, tmp_path):
        _write_raw(home / "skills", "used", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        lib.load("used")
        lib.load("used")
        SkillLibrary(store=store, dirs=[("user", str(self._elsewhere(tmp_path)))]).scan()
        assert [r["loads"] for r in store.skill_rows()] == [2]

    def test_a_directory_that_is_absent_is_not_a_deletion_either(
            self, home, store, tmp_path):
        _write_raw(home / "skills", "kept", "---\ndescription: d\n---\nb\n")
        _lib(home, store).scan()
        absent = tmp_path / "no-such-dir"          # never created, never walked
        SkillLibrary(store=store, dirs=[("user", str(absent))]).scan()
        assert [r["name"] for r in store.skill_rows()] == ["kept"]

    def test_a_deletion_in_a_directory_this_scan_never_read_is_not_its_call(
            self, home, store, tmp_path):
        # The file really is gone, but not in a place this scan looked, so the
        # scan has no standing to say the skill left the library.
        p = _write_raw(home / "skills", "gone", "---\ndescription: d\n---\nb\n")
        _lib(home, store).scan()
        os.unlink(p)
        SkillLibrary(store=store, dirs=[("user", str(self._elsewhere(tmp_path)))]).scan()
        assert [r["name"] for r in store.skill_rows()] == ["gone"]

    def test_a_real_deletion_still_drops_the_row(self, home, store):
        # The guard must not make deletions invisible: here the file is gone
        # *and* the directory that held it was walked. That is evidence.
        p = _write_raw(home / "skills", "gone", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        os.unlink(p)
        assert lib.scan() == []
        assert store.skill_rows() == []


# ======================================================================
# a broken file must not take the library down, or hide
# ======================================================================
class TestBrokenFilesAreReported:
    def test_one_bad_file_does_not_cost_the_others(self, home):
        _write_raw(home / "skills", "good", "---\ndescription: d\n---\nb\n")
        _write_raw(home / "skills", "bad", "no frontmatter at all\n")
        lib = _lib(home)
        assert [s.name for s in lib.scan()] == ["good"]
        assert len(lib.errors) == 1
        assert "does not start with" in lib.errors[0]

    def test_the_error_names_the_file(self, home):
        _write_raw(home / "skills", "bad", "nope\n")
        lib = _lib(home)
        lib.scan()
        assert "bad.md" in lib.errors[0]

    def test_a_more_specific_directory_wins_and_says_so(self, home, project):
        _write_raw(project, "shared", "---\ndescription: from the project\n---\nb\n")
        _write_raw(home / "skills", "shared", "---\ndescription: from the user\n---\nb\n")
        lib = _lib(home, project=project)
        skills = lib.scan()
        assert [s.description for s in skills] == ["from the project"]
        # shadowing is reported rather than hidden: "why is my edit not taking
        # effect" should have an answer on the first look.
        assert lib.shadowed == [("shared", str(project / "shared.md"))]


# ======================================================================
# loads are the signal, and nothing else writes them
# ======================================================================
class TestLoadsAreTheSignal:
    def test_loading_counts(self, home, store):
        _write_raw(home / "skills", "s", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        assert lib.load("s").loads == 1
        assert lib.load("s").loads == 2

    def test_a_rescan_does_not_reset_the_count(self, home, store):
        _write_raw(home / "skills", "s", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        lib.load("s")
        lib.load("s")
        assert lib.scan()[0].loads == 2

    def test_the_count_survives_a_restart(self, home, store):
        _write_raw(home / "skills", "s", "---\ndescription: d\n---\nb\n")
        first = _lib(home, store)
        first.scan()
        for _ in range(4):
            first.load("s")
        second = _lib(home, store)          # a new process, same store
        assert second.scan()[0].loads == 4

    def test_scanning_alone_does_not_count_as_use(self, home, store):
        # Otherwise every unread skill would look used from the first boot and
        # the ranking would measure the library instead of the agent.
        _write_raw(home / "skills", "s", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        for _ in range(5):
            lib.scan()
        assert lib.all()[0].loads == 0
        assert store.report()["skills_never_loaded"] == 1

    def test_loading_an_unknown_skill_returns_none(self, home):
        lib = _lib(home)
        lib.scan()
        assert lib.load("nope") is None


# ======================================================================
# writing
# ======================================================================
class TestWriting:
    def test_write_creates_a_real_file(self, home, store):
        lib = _lib(home, store)
        lib.scan()
        skill = lib.write("new-proc", "does a thing", "when needed", "step one")
        assert os.path.exists(skill.path)
        assert split_frontmatter(open(skill.path, encoding="utf-8").read())[1] == "step one"

    def test_a_write_is_indexed_immediately(self, home, store):
        lib = _lib(home, store)
        lib.scan()
        lib.write("fresh", "d", "", "b")
        assert "fresh" in [s.name for s in lib.all()]
        assert [r["name"] for r in store.skill_rows()] == ["fresh"]

    def test_editing_keeps_the_load_count(self, home, store):
        _write_raw(home / "skills", "s", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        lib.load("s")
        lib.load("s")
        assert lib.write("s", "d2", "", "new body").loads == 2

    def test_an_edit_goes_in_place_rather_than_shadowing(self, home, store, project):
        # Writing to the user dir while a project skill of the same name exists
        # would shadow the file the agent just read: "I edited it and nothing
        # changed" points at everything except the cause.
        _write_raw(project, "s", "---\ndescription: project version\n---\nb\n")
        lib = _lib(home, store, project=project)
        lib.scan()
        skill = lib.write("s", "edited", "", "b2")
        assert skill.path == str(project / "s.md")
        assert not (home / "skills" / "s.md").exists()

    def test_a_bad_name_is_refused_with_the_reason(self, home):
        lib = _lib(home)
        lib.scan()
        with pytest.raises(SkillError, match="unusable skill name"):
            lib.write("Not A Name", "d", "", "b")

    def test_a_missing_description_is_refused(self, home):
        lib = _lib(home)
        lib.scan()
        with pytest.raises(SkillError, match="needs a description"):
            lib.write("ok-name", "   ", "", "b")

    def test_no_temp_file_is_left_behind(self, home):
        lib = _lib(home)
        lib.scan()
        lib.write("clean", "d", "", "b")
        assert [p.name for p in (home / "skills").iterdir()] == ["clean.md"]


# ======================================================================
# retiring archives, it does not destroy
# ======================================================================
class TestArchive:
    def test_archive_moves_the_file_and_drops_the_row(self, home, store):
        _write_raw(home / "skills", "s", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        dest = lib.archive("s")
        assert os.path.exists(dest)
        assert not (home / "skills" / "s.md").exists()
        assert store.skill_rows() == []
        assert lib.all() == []

    def test_a_retired_skill_is_not_re_indexed_by_the_next_scan(self, home):
        _write_raw(home / "skills", "s", "---\ndescription: d\n---\nb\n")
        lib = _lib(home)
        lib.scan()
        lib.archive("s")
        assert lib.scan() == []          # .archive is not a skills directory

    def test_archiving_twice_does_not_overwrite_the_first_copy(self, home):
        _write_raw(home / "skills", "s", "---\ndescription: first\n---\nb\n")
        lib = _lib(home)
        lib.scan()
        first = lib.archive("s")
        _write_raw(home / "skills", "s", "---\ndescription: second\n---\nb\n")
        lib.scan()
        second = lib.archive("s")
        assert first != second
        assert os.path.exists(first) and os.path.exists(second)

    def test_archiving_an_unknown_skill_says_so(self, home):
        lib = _lib(home)
        lib.scan()
        with pytest.raises(SkillError, match="no skill named"):
            lib.archive("nope")


# ======================================================================
# the menu is an index, never the library
# ======================================================================
class TestTheMenuIsAnIndex:
    def test_an_empty_library_says_how_to_fix_it(self, home):
        lib = _lib(home)
        lib.scan()
        assert "skill_write" in lib.menu()[0]

    def test_the_menu_lists_names_and_triggers_not_bodies(self, home, store):
        _write_raw(home / "skills", "s", "---\ndescription: d\nwhen_to_use: when x\n---\nSECRET BODY\n")
        lib = _lib(home, store)
        lib.scan()
        text = "\n".join(lib.menu())
        assert "s (" in text and "when x" in text
        assert "SECRET BODY" not in text

    def test_the_menu_shows_how_proven_each_skill_is(self, home, store):
        _write_raw(home / "skills", "used", "---\ndescription: d\n---\nb\n")
        _write_raw(home / "skills", "unused", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        lib.load("used")
        text = "\n".join(lib.menu())
        # A tier, not a count: "used 1x" would move this line — and every
        # message behind it in the prompt — on every single load.
        assert "used (used before)" in text
        assert "unused (never used)" in text

    def test_the_menu_text_does_not_move_inside_a_tier(self, home, store):
        """Only a change of kind is worth a change of prompt.

        The exact count is what made this line churn; the claim it was carrying
        survives at two transitions instead of at every load.
        """
        _write_raw(home / "skills", "one", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        lib.load("one")
        once = "\n".join(lib.menu())
        lib.load("one")
        lib.load("one")
        assert "\n".join(lib.menu()) == once

        for _ in range(3):                     # crosses into the top tier
            lib.load("one")
        assert "one (well proven)" in "\n".join(lib.menu())

    def test_the_menu_truncates_and_says_so(self, home, store):
        for i in range(40):
            _write_raw(home / "skills", f"s{i:02d}",
                       f"---\ndescription: {'d' * 120}\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        text = "\n".join(lib.menu(budget=500))
        assert "more -- skill_list reads them all" in text
        assert text.count("never used") < 40

    def test_the_menu_orders_by_use(self, home, store):
        _write_raw(home / "skills", "always", "---\ndescription: d\n---\nb\n")
        _write_raw(home / "skills", "never", "---\ndescription: d\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        lib.load("always")
        assert "always" in lib.menu()[1]


# ======================================================================
# routing: fit blended with evidence, and honest about which is which
# ======================================================================
class TestSkillRouting:
    def _lib_with_usage(self, home, store):
        _write_raw(home / "skills", "deploy-verify",
                   "---\ndescription: prove a deploy took effect\n"
                   "when_to_use: after any deploy\n---\nb\n")
        _write_raw(home / "skills", "deploy-notes",
                   "---\ndescription: prove a deploy took effect\n"
                   "when_to_use: after any deploy\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        for _ in range(3):
            lib.load("deploy-verify")
        return lib

    def test_use_breaks_a_tie_that_text_cannot(self, home, store):
        # Same words, same trigger: the only thing telling them apart is that
        # one has actually been loaded. Text similarity is blind to this — it
        # is exactly the "reads right, runs wrong" failure.
        lib = self._lib_with_usage(home, store)
        assert SkillRouter(lib).route("prove the deploy worked")[0] == "deploy-verify"

    def test_an_unloaded_skill_scores_zero_evidence_not_neutral(self, home, store):
        lib = self._lib_with_usage(home, store)
        cand = [c for c in SkillRouter(lib).rank("deploy") if c.name == "deploy-notes"][0]
        assert cand.breakdown["proven"] == 0.0
        assert cand.state == "never-used"

    def test_text_still_decides_when_nobody_has_used_anything(self, home, store):
        _write_raw(home / "skills", "unrelated", "---\ndescription: bake bread\n---\nb\n")
        _write_raw(home / "skills", "related", "---\ndescription: run the deploy\n---\nb\n")
        lib = _lib(home, store)
        lib.scan()
        assert SkillRouter(lib).route("deploy")[0] == "related"

    def test_a_used_but_irrelevant_skill_does_not_beat_a_relevant_one(self, home, store):
        lib = self._lib_with_usage(home, store)
        router = SkillRouter(lib)
        # "deploy-notes" is never used, but it is about the same thing as the
        # query — so evidence must not override relevance wholesale, or the
        # most-used skill would answer every question.
        assert "deploy-verify" in router.route("deploy", k=2)
        assert router.route("bake a cake", k=2) != ["deploy-verify"]

    def test_the_ranking_is_stable(self, home, store):
        lib = self._lib_with_usage(home, store)
        router = SkillRouter(lib)
        assert router.route("deploy") == router.route("deploy")


# ======================================================================
# the tools, through the registry (the gate included)
# ======================================================================
class TestTheTools:
    def _agent_with(self, home, store):
        a = _agent(store)
        a.skills = _lib(home, store)
        a.skills.scan()
        return a

    def test_the_five_tools_exist(self, store):
        a = _agent(store)
        for name in ("skill_list", "skill_view", "skill_write",
                     "skill_forget", "skill_errors"):
            assert a.registry.get(name) is not None

    def test_skill_tools_declare_what_they_touch(self, store):
        a = _agent(store)
        assert a.registry.get("skill_view").effect_signature == "read_only"
        assert a.registry.get("skill_write").effect_signature == "local_write"
        assert a.registry.get("skill_forget").effect_signature == "local_write"

    def test_write_then_list_then_view(self, home, store):
        a = self._agent_with(home, store)
        a.registry.call("skill_write", {"name": "proc", "description": "does a thing",
                                        "body": "STEP ONE", "when_to_use": "when x"})
        assert "proc" in a.registry.call("skill_list", {}).output
        assert "STEP ONE" in a.registry.call("skill_view", {"name": "proc"}).output

    def test_viewing_counts_the_load_and_traces_it(self, home, store):
        a = self._agent_with(home, store)
        a.registry.call("skill_write", {"name": "proc", "description": "d", "body": "b"})
        a.registry.call("skill_view", {"name": "proc"})
        # The durable record of a load is the counter in the skills table. It is
        # not also written to the ledger: a load happens often, and a row per
        # load would grow the ledger by the turn count — the same bloat that
        # keeping the memory block out of `_record` avoids.
        assert store.report()["skill_loads"] == 1
        assert store.report()["event_kinds"] == {"agent_init": 1}
        assert any(e["kind"] == "skill_view" for e in a.trace)
        assert any(e["kind"] == "skill_write" for e in a.trace)

    def test_viewing_something_that_is_not_there_suggests_the_closest(self, home, store):
        a = self._agent_with(home, store)
        a.registry.call("skill_write", {"name": "deploy-verify", "description": "d", "body": "b"})
        out = a.registry.call("skill_view", {"name": "deploy"}).output
        assert "No skill named" in out and "deploy-verify" in out

    def test_a_rejected_write_returns_the_reason_not_an_exception(self, home, store):
        a = self._agent_with(home, store)
        out = a.registry.call("skill_write", {"name": "Bad Name",
                                              "description": "d", "body": "b"}).output
        assert out.startswith("Not saved:")
        assert "lower-case" in out

    def test_errors_reads_the_skip_reasons(self, home, store):
        a = self._agent_with(home, store)
        _write_raw(home / "skills", "broken", "no fence\n")
        out = a.registry.call("skill_errors", {}).output
        assert "unusable file" in out and "broken.md" in out

    def test_errors_says_so_when_nothing_is_wrong(self, home, store):
        a = self._agent_with(home, store)
        assert "Nothing is being skipped" in a.registry.call("skill_errors", {}).output

    def test_forget_archives_rather_than_deletes(self, home, store):
        a = self._agent_with(home, store)
        a.registry.call("skill_write", {"name": "proc", "description": "d", "body": "b"})
        out = a.registry.call("skill_forget", {"name": "proc"}).output
        assert "archived" in out
        assert os.path.exists(os.path.join(str(home / "skills"), ".archive", "proc.md"))


# ======================================================================
# progressive disclosure, in the request the model actually receives
# ======================================================================
class TestItReachesTheModel:
    def test_the_menu_is_in_the_system_message(self, home, store):
        a = _agent(store)
        a.skills = _lib(home, store)
        a.skills.write("deploy-verify", "prove a deploy took effect",
                       "after any deploy", "STEP ONE")
        a.skills.scan()
        a.run("say hi")
        system = a.llm.calls[0][0][0]
        assert "deploy-verify" in system.content
        assert "after any deploy" in system.content

    def test_the_body_is_not_in_the_system_message(self, home, store):
        a = _agent(store)
        a.skills = _lib(home, store)
        a.skills.write("proc", "d", "when x", "SECRET PROCEDURE BODY")
        a.skills.scan()
        a.run("say hi")
        system = a.llm.calls[0][0][0]
        assert "SECRET PROCEDURE BODY" not in system.content

    def test_a_skill_written_mid_run_is_offered_on_the_next_request(self, home, store):
        a = _agent(store)
        a.skills = _lib(home, store)
        a.skills.scan()
        a.registry.call("skill_write", {"name": "learned", "description": "d",
                                        "body": "b", "when_to_use": "next time"})
        a.run("go")
        assert "learned" in a.llm.calls[0][0][0].content

    def test_the_self_report_counts_them(self, home, store):
        a = _agent(store)
        a.skills = _lib(home, store)
        a.skills.write("one", "d", "", "b")
        a.skills.scan()
        a.skills.load("one")
        a.run("go")
        text = a.llm.calls[0][0][0].content
        assert "1 procedure(s) on disk" in text
        # The load counter is deliberately NOT in the prompt: it moves every time
        # a body is loaded, and the prompt is the prefix of every request, so a
        # changing integer there re-bills the whole conversation uncached. The
        # count is still available from the library — see test_prompt_stability.
        assert a.skills.report()["loads_total"] == 1

    def test_no_store_still_offers_skills_from_disk(self, home):
        # Skills do not need the database to exist — the files are the truth,
        # and only the ranking degrades without a store.
        a = _agent(store=None)
        a.skills = _lib(home, store=None)
        a.skills.write("disk-only", "d", "when x", "b")
        a.skills.scan()
        a.run("go")
        assert "disk-only" in a.llm.calls[0][0][0].content


# ======================================================================
# defaults
# ======================================================================
class TestDefaults:
    def test_dirs_are_project_then_user(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_SKILLS_DIRS", raising=False)
        dirs = default_skill_dirs(cwd=str(tmp_path), home=str(tmp_path / "h"))
        assert [s for s, _ in dirs] == ["project", "user"]
        assert dirs[0][1].endswith("skills")

    def test_the_env_var_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTOFORGE_SKILLS_DIRS",
                           os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")]))
        dirs = default_skill_dirs(cwd="ignored", home="ignored")
        assert [d for _, d in dirs] == [str(tmp_path / "a"), str(tmp_path / "b")]

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        lib = SkillLibrary(dirs=[("user", str(tmp_path / "nope"))])
        assert lib.scan() == []
        assert lib.errors == []


# ======================================================================
# the scanner sees trees, not just the top level
# ======================================================================
PKG = """---
name: {name}
description: {desc}
when_to_use: {when}
tags: t
---
body of {name}
"""


def _pkg(directory, category, name):
    """Write a packaged skill: <category>/<name>/SKILL.md."""
    d = directory / category / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        PKG.format(name=name, desc=f"does {name}", when=f"when {name}"),
        encoding="utf-8",
    )
    return d


class TestScannerFindsPackages:
    """The library is a tree of categories, and the first scanner could not see it.

    The original scan globbed `*.md` one level deep, so it found every skill the
    agent had *written itself* (flat, at the top) and none of the skills that
    were *installed* (a package under a category). On this machine that was 0 of
    239 discovered, with no error — a silent wrong answer, which is why the
    counting assertion below is the point of the class, not a decoration.
    """

    def test_a_skill_inside_a_category_is_discovered(self, home):
        _pkg(home / "skills", "research", "arxiv")
        lib = _lib(home)
        names = [s.name for s in lib.scan()]
        assert names == ["arxiv"], "a packaged skill must be visible to the scan"

    def test_several_categories_and_depths_all_come_back(self, home):
        root = home / "skills"
        _pkg(root, "research", "arxiv")
        _pkg(root, "software-development", "tdd")
        _pkg(root, "mlops", "ollama")
        # already at the top level, so this one is a package with no category
        _pkg(root, "", "flat-package")
        names = sorted(s.name for s in _lib(home).scan())
        assert names == ["arxiv", "flat-package", "ollama", "tdd"]

    def test_the_agents_own_flat_skills_still_work(self, home):
        # `write` produces <name>.md at the top; that layout must not regress
        # while fixing the packaged one.
        lib = _lib(home)
        lib.write("mine", "does mine", "when mine", "b")
        assert "mine" in [s.name for s in lib.scan()]

    def test_flat_and_packaged_are_both_seen(self, home):
        root = home / "skills"
        _write_raw(root, "hand-written", PKG.format(name="hand-written", desc="d", when="w"))
        _pkg(root, "research", "installed")
        names = sorted(s.name for s in _lib(home).scan())
        assert names == ["hand-written", "installed"]

    def test_an_archived_skill_is_not_resurrected(self, home):
        # `.archive/` holds retired skills. They stay on disk for the record and
        # must not be indexed — otherwise deleting a skill does nothing.
        root = home / "skills"
        _pkg(root, ".archive", "retired")
        _pkg(root, "live", "current")
        names = [s.name for s in _lib(home).scan()]
        assert names == ["current"], "an archived package must stay archived"

    def test_a_quarantined_skill_is_not_indexed(self, home):
        # `.hub/quarantine/` is the same contract for a bad import.
        root = home / "skills"
        _pkg(root, ".hub/quarantine", "suspect")
        _pkg(root, "live", "current")
        assert [s.name for s in _lib(home).scan()] == ["current"]

    def test_a_support_dir_inside_a_skill_is_not_a_skill(self, home):
        # A skill package may keep a complete old copy under references/. That
        # is documentation data, reachable by file_path, not a skill.
        root = home / "skills"
        skill = _pkg(root, "live", "real")
        old = skill / "references" / "old-skill"
        old.mkdir(parents=True)
        (old / "SKILL.md").write_text(
            PKG.format(name="old-skill", desc="d", when="w"), encoding="utf-8")
        assert [s.name for s in _lib(home).scan()] == ["real"]

    def test_a_category_named_like_a_support_dir_survives(self, home):
        # `scripts/foo/SKILL.md` is a legitimate skill; `scripts` is only a
        # support area *inside* a package. The pruning must know the difference.
        root = home / "skills"
        _pkg(root, "scripts", "deploy")
        _pkg(root, "templates", "scaffold")
        names = sorted(s.name for s in _lib(home).scan())
        assert names == ["deploy", "scaffold"]

    def test_a_dependency_tree_is_not_walked(self, home):
        root = home / "skills"
        _pkg(root, "node_modules", "junk")
        _pkg(root, "__pycache__", "junk2")
        _pkg(root, ".venv/lib", "junk3")
        _pkg(root, "live", "real")
        assert [s.name for s in _lib(home).scan()] == ["real"]

    def test_a_partial_write_is_not_indexed(self, home):
        # An interrupted `write` can leave `<name>.md.tmp`; indexing it would
        # offer a skill whose body is half a file.
        root = home / "skills"
        (root / "half.md.tmp").write_text("---\nname: half\n", encoding="utf-8")
        _pkg(root, "live", "real")
        assert [s.name for s in _lib(home).scan()] == ["real"]

    def test_the_flat_file_wins_a_name_collision(self, home):
        # The agent's own edit outranks an installed skill of the same name, and
        # the shadowing is reported rather than silent.
        root = home / "skills"
        _write_raw(root, "dup",
                   PKG.format(name="dup", desc="agent version", when="w"))
        _pkg(root, "research", "dup")
        lib = _lib(home)
        found = {s.name: s for s in lib.scan()}
        assert found["dup"].description == "agent version"
        assert any("dup" in str(x) for x in lib.shadowed)

    def test_a_broken_package_is_loud_and_does_not_kill_the_scan(self, home):
        # One malformed file must surface in errors and leave its neighbours
        # discoverable — a scan that dies on the first bad file hides the rest.
        root = home / "skills"
        bad = root / "research" / "broken"
        bad.mkdir(parents=True)
        (bad / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")
        _pkg(root, "research", "fine")
        lib = _lib(home)
        names = [s.name for s in lib.scan()]
        assert names == ["fine"]
        assert lib.errors, "a malformed skill must be reported, not swallowed"
