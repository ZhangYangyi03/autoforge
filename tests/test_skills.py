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
        assert "used (used 1x)" in text
        assert "unused (never used)" in text

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
        assert "1 load(s) across them" in text

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
