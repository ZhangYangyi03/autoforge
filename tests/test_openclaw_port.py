"""Porting another agent's skills: what is copied, what is refused, and why.

Asked for by the operator in these words: "openclaw 52 技能 + 41 扩展是产品化你直接拿去复制粘贴".
The honest version of that is narrower than the sentence suggests, and these
tests are where the narrowing is written down:

  * the SKILL.md files of an installed openclaw are read with a LENIENT parser
    (every one of them carries a nested `metadata:` block that this library's
    own strict frontmatter reader rejects -- fifty-two files at the door);
  * a skill that DECLARES another OS is skipped and reported, not copied;
  * "another OS" is decided by the declaration, not by the word "macos"
    appearing in prose -- the first draft threw away `xurl`, `prose`,
    `obsidian` and `healthcheck`, all of which run here;
  * re-running is idempotent and does not reset how often a skill was loaded.
"""
from __future__ import annotations

import os
import textwrap

import pytest

from autoforge.skills import (SkillLibrary, _lenient_frontmatter,
                              iter_openclaw_skills, openclaw_runs_here,
                              port_openclaw_skills)


def _pkg(tmp_path, files: dict[str, str]):
    """A stand-in openclaw package: {relative path: text}."""
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(tmp_path / "skills")


PLAIN = """\
    ---
    name: github
    description: GitHub operations via the gh CLI.
    metadata:
      {
        "openclaw": { "emoji": "x" }
      }
    ---

    # GitHub

    Use `gh pr view`.
    """

DARWIN = """\
    ---
    name: apple-notes
    description: Notes on macOS.
    metadata:
      { "openclaw": { "emoji": "n", "os": ["darwin"] } }
    ---

    Use `osascript` to add a note.
    """

MULTI_OS = """\
    ---
    name: tmux
    description: Control tmux sessions.
    metadata:
      { "openclaw": { "emoji": "t", "os": ["darwin", "linux"] } }
    ---

    tmux send-keys -t 0 'ls' Enter
    """

NO_FRONTMATTER = """\
    # Canvas Skill

    Display HTML on connected nodes.
    """

BLOCK_SCALAR = """\
    ---
    name: summarize
    description: |
      Summarize a URL or a transcript.
      Works offline.
    metadata: {}
    ---

    body
    """


# -- reading a foreign header -------------------------------------------
def test_a_nested_metadata_block_does_not_break_the_read():
    """The reason this parser exists: openclaw puts `metadata:` with nested
    braces in every skill, and the library's strict reader raises on it."""
    meta, body = _lenient_frontmatter(textwrap.dedent(PLAIN))
    assert meta["name"] == "github"
    assert meta["description"] == "GitHub operations via the gh CLI."
    assert "gh pr view" in body


def test_a_block_scalar_description_is_joined_not_dropped():
    meta, _ = _lenient_frontmatter(textwrap.dedent(BLOCK_SCALAR))
    assert meta["description"] == "Summarize a URL or a transcript. Works offline."


def test_a_file_without_frontmatter_yields_no_meta_and_its_own_body():
    meta, body = _lenient_frontmatter(textwrap.dedent(NO_FRONTMATTER))
    assert meta == {}
    assert body.lstrip().startswith("# Canvas Skill")


# -- who can actually run here ------------------------------------------
def test_a_declared_other_os_is_refused_and_the_reason_is_specific():
    ok, why = openclaw_runs_here(textwrap.dedent(DARWIN))
    assert not ok
    assert "darwin" in why


def test_a_multi_os_declaration_that_excludes_windows_is_refused():
    ok, why = openclaw_runs_here(textwrap.dedent(MULTI_OS))
    assert not ok and "linux" in why


def test_prose_mentioning_macos_is_not_grounds_for_refusal():
    """Measured against the real library: seven of openclaw's skills say
    "macos" somewhere and run here perfectly well. Narrowing the markers after
    that measurement is what recovered them."""
    text = ("---\nname: x\ndescription: y\n---\n\n"
            "Install with npm; on macOS you may also use brew.\n")
    ok, why = openclaw_runs_here(text)
    assert ok, why


def test_an_osascript_procedure_is_refused_even_without_a_declaration():
    text = "---\nname: x\ndescription: y\n---\n\nrun osascript -e 'beep'\n"
    ok, why = openclaw_runs_here(text)
    assert not ok and "osascript" in why


def test_silence_about_the_os_is_treated_as_usable():
    """Most of the 52 declare nothing; refusing them for silence would throw
    the library away to avoid a few wrong menu lines."""
    ok, _ = openclaw_runs_here(textwrap.dedent(PLAIN))
    assert ok


# -- the port itself ----------------------------------------------------
def test_the_port_writes_one_prefixed_skill_per_usable_file(tmp_path):
    root = _pkg(tmp_path, {"skills/github/SKILL.md": PLAIN,
                           "skills/apple-notes/SKILL.md": DARWIN})
    lib = SkillLibrary(None, dirs=[("user", str(tmp_path / "out"))])
    rep = port_openclaw_skills(roots=[root], library=lib)
    assert rep["ok"] and rep["count"] == 1
    written = [w["name"] for w in rep["written"]]
    assert written == ["openclaw-github"]
    assert [k["name"] for k in rep["skipped"]] == ["apple-notes"]
    assert "darwin" in rep["skipped"][0]["why"]
    body = lib.get("openclaw-github")
    assert body is not None and "gh pr view" in body.body


def test_the_body_arrives_unchanged_behind_a_provenance_line(tmp_path):
    root = _pkg(tmp_path, {"skills/github/SKILL.md": PLAIN})
    lib = SkillLibrary(None, dirs=[("user", str(tmp_path / "out"))])
    port_openclaw_skills(roots=[root], library=lib)
    skill = lib.get("openclaw-github")
    assert "gh pr view" in skill.body, "the body must be the copied text"
    assert "openclaw" in skill.body.split("-->")[0], "provenance line missing"
    assert "MIT" in skill.body.split("-->")[0], "licence missing"


def test_a_second_port_is_idempotent_and_does_not_reset_load_count(tmp_path):
    root = _pkg(tmp_path, {"skills/github/SKILL.md": PLAIN})
    lib = SkillLibrary(None, dirs=[("user", str(tmp_path / "out"))])
    port_openclaw_skills(roots=[root], library=lib)
    lib.load("openclaw-github")          # count one use
    lib.load("openclaw-github")
    before = lib.get("openclaw-github").loads
    port_openclaw_skills(roots=[root], library=lib)
    after = lib.get("openclaw-github").loads
    assert after == before, "a re-port must not look like a fresh skill"


def test_a_file_with_no_frontmatter_is_given_its_heading_as_description(tmp_path):
    """Unroutable means never offered, so a description is not optional."""
    root = _pkg(tmp_path, {"skills/canvas/SKILL.md": NO_FRONTMATTER})
    lib = SkillLibrary(None, dirs=[("user", str(tmp_path / "out"))])
    rep = port_openclaw_skills(roots=[root], library=lib)
    assert rep["count"] == 1
    assert lib.get("openclaw-canvas").description.strip() == "Canvas Skill"


def test_a_missing_package_is_reported_not_raised(tmp_path):
    rep = port_openclaw_skills(roots=[], library=SkillLibrary(
        None, dirs=[("user", str(tmp_path / "out"))]))
    # An empty root list falls through to discovery; force the empty case.
    rep2 = port_openclaw_skills(roots=[str(tmp_path / "nowhere")],
                                library=SkillLibrary(None, dirs=[("user", str(tmp_path / "out"))]))
    assert isinstance(rep, dict) and isinstance(rep2, dict)


def test_extension_packages_are_found_too(tmp_path):
    """41 extensions, four of which carry skills of their own: the feishu set
    and the ACP router are as real as the 52 core ones."""
    root = _pkg(tmp_path, {"extensions/feishu/skills/feishu-doc/SKILL.md": PLAIN,
                           "skills/github/SKILL.md": PLAIN})
    found = iter_openclaw_skills([str(tmp_path / "skills"),
                                  str(tmp_path / "extensions" / "feishu" / "skills")])
    names = sorted(n for _, n, _ in found)
    assert names == ["feishu-doc", "github"]
