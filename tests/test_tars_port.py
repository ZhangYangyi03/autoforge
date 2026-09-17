"""The tars port: borrowed code that has to keep working, and say what it is.

Every assertion here is about one of two things going wrong quietly.

The first is rot. `vendor/tars` is 22 files copied from somebody else's
repository; the adapter imports one of them *by path*, so nothing else in this
package would notice if it disappeared, got pruned by a packaging step, or
stopped parsing under a newer Python. A port that silently stops loading reads
as "I have no borrowed procedures", which is the failure mode this whole change
exists to prevent.

The second is a claim outliving its truth. The self report says how many
borrowed procedures are loadable by name, and the probe block says whether the
vendored tree is on disk. Both are printed into the system prompt every turn, so
if either can drift from the machine, the agent reasons from a description
instead of from the host -- which is the exact habit the ported gate is for.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoforge import tars_port  # noqa: E402
from autoforge.skills import TARS_SKILL_TAG, SkillLibrary, port_tars_skills  # noqa: E402
from autoforge.store import ToolStore  # noqa: E402

UPSTREAM = "https://github.com/intelligence-indeed/intelligence-indeed"


# -- the vendored tree is really there, and really theirs -------------------
def test_vendor_tree_is_present_and_licensed():
    d = tars_port.vendor_dir()
    assert d.is_dir(), "the vendored tars tree is gone; the adapter cannot load"
    assert (d / "LICENSE").is_file(), "a copy without its licence is not a copy"
    assert (d / "ORIGIN.txt").is_file(), "no provenance: where did these bytes come from"
    origin = (d / "ORIGIN.txt").read_text(encoding="utf-8")
    assert UPSTREAM in origin
    assert (d / "skill_system" / "static_skills.py").is_file()


def test_the_vendored_file_is_byte_identical_to_upstream_copy():
    """The point of a copy is that it was not edited on the way in.

    Checked by content, not by trust: the sha256 of the file on disk is compared
    against the sha256 recorded when it was copied. If a later session "tidies"
    the vendored file -- reindents it, renames a class -- this fails, and that is
    the intent: the fix is to change the adapter, not the borrowed code.
    """
    import hashlib
    p = tars_port.vendor_dir() / "skill_system" / "static_skills.py"
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    listed = tars_port.VENDOR_SHA256.get("skill_system/static_skills.py")
    assert listed, "no recorded hash for the vendored file"
    assert digest == listed, (
        "the vendored static_skills.py has been edited in place; the value of a "
        "copy is that it is unchanged -- change tars_port.py instead")


# -- and it imports, from the path, without the upstream package ------------
def test_the_skill_registry_loads_and_indexes_nine_procedures():
    ported = tars_port.PortedSkillSystem()
    names = ported.names()
    assert len(names) == 9, names
    assert "skill-chrome" in names
    index = ported.index()
    # Progressive disclosure: the index carries names and descriptions, and no
    # body. If a body leaks into the index, every turn pays for all of it.
    assert "skill-chrome" in index
    assert "## Operating rules" not in index


def test_a_body_is_readable_and_is_not_the_index():
    ported = tars_port.PortedSkillSystem()
    body = ported.body("skill-chrome")
    assert len(body) > 200
    assert "## Domain notes" in body


def test_gate_probe_text_is_empty_in_empty_out():
    """A gate with nothing to say must not inject a heading that says nothing."""
    assert tars_port.gate_probe_text("") == ""
    assert tars_port.gate_probe_text("   \n ") == ""
    text = tars_port.gate_probe_text("Chrome can print to PDF.", ["skill-chrome"])
    assert "Dynamic skill: gate-probe" in text
    assert "Loaded skills: skill-chrome" in text
    assert text.rstrip().endswith("Chrome can print to PDF.")


# -- the probe answers about THIS host, read-only, and stays stable ---------
def test_probe_is_cheap_read_only_and_mentions_the_vendor_tree():
    lines = tars_port.probe_host()
    joined = "\n".join(lines)
    assert "vendor dir" in joined
    assert "upstream skills indexed: 9" in joined
    for line in lines:
        assert not line.endswith("MISSING at"), line


def test_probe_output_is_stable_between_calls():
    """It goes into the system prompt, which is the request *prefix*.

    A line that changes between two calls -- an elapsed millisecond figure, a
    timestamp, a cached-then-not marker -- re-bills every character behind it on
    an otherwise identical request. The probe may change when the MACHINE
    changes; it may not change because it was called twice.
    """
    assert tars_port.probe_host() == tars_port.probe_host()


# -- the port into the library ---------------------------------------------
def test_porting_writes_nine_tagged_skills_and_carries_loads(tmp_path):
    store = ToolStore(os.path.join(str(tmp_path), "t.db"))
    lib = SkillLibrary(store, dirs=[("user", os.path.join(str(tmp_path), "skills"))])
    lib.scan()
    res = port_tars_skills(library=lib)
    assert res["ok"], res
    assert len(res["written"]) == 9
    lib.scan()
    ported = [s for s in lib.all() if TARS_SKILL_TAG in s.tags]
    assert len(ported) == 9
    for s in ported:
        assert s.name.startswith("tars-")
        assert UPSTREAM in s.body, "a copy that lost its source is a claim"
        assert s.when_to_use, "a skill nobody can route to is a file, not a skill"

    # Re-porting refreshes the text and must NOT reset the usage history: nine
    # procedures silently going back to "never loaded" is a loss nothing reports.
    first = lib.load("tars-skill-chrome")
    assert first is not None
    assert first.loads >= 1
    port_tars_skills(library=lib)
    lib.scan()
    again = lib.get("tars-skill-chrome")
    assert again is not None
    assert again.loads == first.loads
    store.close()


def test_porting_is_idempotent_in_count(tmp_path):
    lib = SkillLibrary(None, dirs=[("user", os.path.join(str(tmp_path), "skills"))])
    port_tars_skills(library=lib)
    port_tars_skills(library=lib)
    lib.scan()
    assert len([s for s in lib.all() if TARS_SKILL_TAG in s.tags]) == 9


# -- and the tools that expose it ------------------------------------------
def test_agent_exposes_the_three_tars_tools_and_declares_their_scope():
    from autoforge.agent import BUILTIN_SCOPES, ForgeAgent

    for name in ("tars_probe", "tars_skills", "tars_port_skills"):
        assert name in BUILTIN_SCOPES, f"{name} would be gated as undeclared"
    assert BUILTIN_SCOPES["tars_probe"] == "read_only"
    assert BUILTIN_SCOPES["tars_skills"] == "read_only"
    # The one that mints files under the skills directory is not a read, and
    # saying so is what keeps the autonomy gate honest about it.
    assert BUILTIN_SCOPES["tars_port_skills"] == "local_write"

    agent = ForgeAgent(llm=None, store=None)
    for name in ("tars_probe", "tars_skills", "tars_port_skills"):
        assert agent.registry.get(name) is not None, f"{name} was never registered"


def test_self_report_counts_what_is_loadable_not_what_exists_on_disk(tmp_path):
    """The number in the prompt has to be about this agent, not about a checkout."""
    from autoforge.agent import ForgeAgent

    agent = ForgeAgent(llm=None, store=None)
    blank = agent._self_report()
    assert "Borrowed:" not in blank, (
        "with no ported skills in the library, the report must not claim any")

    lib = SkillLibrary(None, dirs=[("user", os.path.join(str(tmp_path), "skills"))])
    port_tars_skills(library=lib)
    agent.skills = lib
    report = agent._self_report()
    assert "Borrowed: 9 procedure(s)" in report, report
