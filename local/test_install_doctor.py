"""Tests for install_doctor.py and the `auto doctor` CLI hook.

The bug this closes: `pip install autoforge-agent==0.4.2` over the patched
0.4.2+local.1 build leaves a package that imports and half-works. pip's
uninstall deletes only the files listed in the RECORD it is replacing, so the
files the patched wheel *added* (`media.py`, `tools/exposure.py`) survive while
the files it *modified* are restored to stock. Nothing raises; the prefix-cache
saving is simply gone.

So the doctor has to do three things, and this suite pins each one:

1. **Name it.** Version drift plus stale files, without crying wolf on a
   healthy install. False positives are worse than no auditor: an auditor that
   is always wrong gets ignored, which is the same as having none.
2. **Stay in its lane.** certifi, requests and pip live in the same
   site-packages. They are not in this manifest and they are not this audit's
   business.
3. **Survive the thing it reports.** A doctor written as `autoforge/doctor.py`
   is deleted by the stock reinstall it exists to catch. The real one is a
   standalone script outside the package, and the CLI loads it by path.

Run:
    C:/autoforge/venv/Scripts/python.exe -m unittest test_install_doctor -v
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
DOCTOR_PATH = HERE / "install_doctor.py"

import importlib.util                                    # noqa: E402

_spec = importlib.util.spec_from_file_location("_doctor_under_test", DOCTOR_PATH)
doctor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doctor)

SITE_PKG = Path(r"C:\autoforge\venv\Lib\site-packages")
DIST = "autoforge_agent-0.4.2+local.1.dist-info"


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def fake_manifest(version: str = "0.4.2+local.1") -> dict:
    return {
        "format": 1,
        "distribution": "autoforge-agent",
        "version": version,
        "stock_version": version.split("+", 1)[0],
        "wheel": {"name": "x.whl", "sha256": "0" * 64, "bytes": 1,
                  "url": "https://example.invalid/x.whl"},
        "files": {},
    }


def build_tree(root: Path, *, patched: bool, debris: bool) -> Path:
    """A miniature site-packages: autoforge + certifi + optional debris.

    `patched=True` means every file on disk hashes the way the manifest says.
    `patched=False` is the half-patch: the modified files are reverted to their
    stock bytes and the added files are gone.
    """
    site = root / "site-packages"
    (site / "autoforge" / "tools").mkdir(parents=True)
    (site / "autoforge" / "core").mkdir(parents=True)

    added = {"autoforge/media.py": b"# added by the local wheel\n",
             "autoforge/tools/exposure.py": b"# added by the local wheel\n"}
    modified = {"autoforge/core/llm.py": b"# interrupt polling, patched\n",
                "autoforge/tools/registry.py": b"# exposure, patched\n",
                "autoforge/agent.py": b"# wiring, patched\n"}
    untouched = {"autoforge/__init__.py": b"# same in both builds\n",
                 "autoforge/core/message.py": b"# same in both builds\n"}
    dist_info = {f"{DIST}/METADATA": b"Metadata-Version: 2.1\nName: autoforge-agent\n"}

    # The manifest describes the patched build, whatever is on disk now.
    expect: dict[str, bytes] = {}
    expect.update(added)
    expect.update(modified)
    expect.update(untouched)
    expect.update(dist_info)

    on_disk: dict[str, bytes] = {}
    on_disk.update(untouched)
    on_disk.update(dist_info)
    if patched:
        on_disk.update(added)
    for rel, data in modified.items():
        on_disk[rel] = data if patched else b"# stock bytes, shorter\n"

    # pip-generated: present on disk, never hashed.
    pip_files = {f"{DIST}/RECORD": b"autoforge/__init__.py,,\n",
                 f"{DIST}/INSTALLER": b"pip\n",
                 f"{DIST}/direct_url.json": b"{}\n"}
    on_disk.update(pip_files)

    for rel, data in on_disk.items():
        p = site / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    # A different distribution sharing the tree. Not in the manifest, and
    # reporting it would make every audit fail forever.
    cert = site / "certifi"
    cert.mkdir()
    (cert / "__init__.py").write_bytes(b"# certifi\n")
    (site / "certifi-2026.7.22.dist-info").mkdir()
    (site / "certifi-2026.7.22.dist-info" / "RECORD").write_bytes(b"certifi/__init__.py,,\n")

    if debris:
        # Two kinds of debris, at the two depths pip leaves them. It stashes
        # the dist-info it is about to delete at the top level of
        # site-packages, and the package subdirectories inside the package.
        # The top-level one is only claimed when it is recognisably ours:
        # a bare `~ore` up there could belong to any distribution, and
        # reporting somebody else's cleanup is noise.
        top = site / "~utoforge_agent-0.4.2.dist-info"
        top.mkdir()
        (top / "leftover.py").write_bytes(b"# interrupted uninstall\n")
        inner = site / "autoforge" / "tools" / "~%pycache__"
        inner.mkdir(parents=True)
        (inner / "mod.cpython-313.pyc").write_bytes(b"\x00bytecode")

    man = fake_manifest()
    import hashlib
    for rel, data in expect.items():
        man["files"][rel] = {"sha256": hashlib.sha256(data).hexdigest(),
                             "bytes": len(data)}
    # Written next to the tree, not inside it: a file inside the tree would be
    # audited as part of the package.
    (root / "_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return site


def run_audit(site: Path, version: str) -> dict:
    man = json.loads((site.parent / "_manifest.json").read_text())
    with mock.patch.object(importlib.metadata, "version",
                           lambda _n, _v=version: _v):
        return doctor.audit(man, site)


class DoctorCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="doctor_test_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp,
                                                            ignore_errors=True))

    def site(self, **kw) -> Path:
        return build_tree(self.tmp, **kw)


# ----------------------------------------------------------------------
# the classifier
# ----------------------------------------------------------------------
class TestClassifiers(unittest.TestCase):
    def test_pip_generated_matches_by_suffix(self) -> None:
        self.assertTrue(doctor.is_pip_generated(
            "autoforge_agent-0.4.2+local.1.dist-info/RECORD"))
        self.assertTrue(doctor.is_pip_generated(
            "autoforge_agent-0.4.2+local.1.dist-info/INSTALLER"))
        self.assertTrue(doctor.is_pip_generated(
            "autoforge_agent-0.4.2+local.1.dist-info/direct_url.json"))
        self.assertTrue(doctor.is_pip_generated(
            "autoforge_agent-0.4.2+local.1.dist-info/WHEEL"))

    def test_pip_generated_does_not_swallow_real_files(self) -> None:
        self.assertFalse(doctor.is_pip_generated("autoforge/agent.py"))
        self.assertFalse(doctor.is_pip_generated("autoforce/INSTALLER.py"))
        # a package legitimately named record.py is not pip's RECORD
        self.assertFalse(doctor.is_pip_generated("autoforge/record.py"))
        self.assertFalse(doctor.is_pip_generated(
            "autoforge_agent-0.4.2+local.1.dist-info/METADATA"))
        self.assertFalse(doctor.is_pip_generated(
            "autoforge_agent-0.4.2+local.1.dist-info/entry_points.txt"))

    def test_owned_debris_masks_leading_characters(self) -> None:
        # pip renames autoforge -> ~utoforge, so the marker survives masking.
        self.assertTrue(doctor.is_owned_debris("~utoforge_agent-0.4.2.dist-info"))
        self.assertTrue(doctor.is_owned_debris("~utoforge_agent-0.4.2+local.1.dist-info"))
        self.assertTrue(doctor.is_owned_debris("~utoforge"))
        # ...and it masks an inconsistent number of them, so the marker has to
        # survive more than one character. This one is real, from this machine.
        self.assertTrue(doctor.is_owned_debris("~-toforge_agent-0.4.2.dist-info"))

    def test_owned_debris_rejects_other_distributions(self) -> None:
        self.assertFalse(doctor.is_owned_debris("certifi"))
        self.assertFalse(doctor.is_owned_debris("~ertifi"))
        self.assertFalse(doctor.is_owned_debris("requests"))
        self.assertFalse(doctor.is_owned_debris("autoforge"))       # live, not stashed
        self.assertFalse(doctor.is_owned_debris("pip-26.1.2.dist-info"))


# ----------------------------------------------------------------------
# state A: the install is the build it should be
# ----------------------------------------------------------------------
class TestHealthyInstall(DoctorCase):
    def test_clean_install_is_ok(self) -> None:
        site = self.site(patched=True, debris=False)
        rep = run_audit(site, "0.4.2+local.1")
        self.assertEqual(rep["verdict"], "ok", rep["problems"])
        self.assertEqual(rep["counts"]["stale"], 0)
        self.assertEqual(rep["counts"]["missing"], 0)
        self.assertEqual(rep["counts"]["unexpected"], 0)
        self.assertEqual(rep["counts"]["stash"], 0)
        self.assertEqual(rep["version_drift"], [])

    def test_other_distributions_are_not_reported(self) -> None:
        """certifi sits in the same site-packages and is none of our business."""
        site = self.site(patched=True, debris=False)
        rep = run_audit(site, "0.4.2+local.1")
        unexpected = {p["path"] for p in rep["problems"]["unexpected"]}
        self.assertFalse([p for p in unexpected if "certifi" in p], unexpected)
        self.assertFalse([p for p in unexpected if "dist-info" in p], unexpected)

    def test_pip_generated_files_do_not_read_as_stale(self) -> None:
        """pip rewrites RECORD on every install; hashing it fails forever."""
        site = self.site(patched=True, debris=False)
        rep = run_audit(site, "0.4.2+local.1")
        stale = {p["path"] for p in rep["problems"]["stale"]}
        self.assertFalse([p for p in stale if p.endswith("RECORD")], stale)
        self.assertFalse([p for p in stale if p.endswith("INSTALLER")], stale)
        self.assertFalse([p for p in stale if "direct_url" in p], stale)

    def test_stash_debris_alone_is_not_a_reinstall(self) -> None:
        """Debris from a killed pip is a cleanup, not a reason to reinstall."""
        site = self.site(patched=True, debris=True)
        rep = run_audit(site, "0.4.2+local.1")
        self.assertEqual(rep["counts"]["stash"], 2)
        self.assertEqual(rep["counts"]["unexpected"], 0)
        self.assertEqual(rep["counts"]["stale"], 0)
        self.assertEqual(rep["counts"]["missing"], 0)
        self.assertNotEqual(rep["verdict"], "ok")     # still reported
        # but the tree is otherwise the right build
        self.assertFalse(rep["version_drift"])

    def test_each_file_is_reported_once(self) -> None:
        """The walk runs once per distinct top-level, not once per manifest file.

        `expected` holds one entry per file, so iterating it and walking the
        top-level each time walks the package as many times as the manifest
        lists it. Seven manifest files under `autoforge/` means the same
        `~ore/leftover.py` counted seven times, and an audit that reports
        "83 pip stashes" for ten directories is an audit nobody trusts.
        """
        site = self.site(patched=True, debris=True)
        rep = run_audit(site, "0.4.2+local.1")
        seen: dict[str, int] = {}
        for item in rep["problems"]["stash"]:
            seen[item["path"]] = seen.get(item["path"], 0) + 1
        self.assertEqual(max(seen.values()), 1, seen)
        self.assertEqual(rep["counts"]["stash"], len(seen))


# ----------------------------------------------------------------------
# state B: the half-patch
# ----------------------------------------------------------------------
class TestHalfPatch(DoctorCase):
    def test_stock_wheel_names_the_half_patch(self) -> None:
        site = self.site(patched=False, debris=False)
        rep = run_audit(site, "0.4.2")
        self.assertEqual(rep["installed_version"], "0.4.2")
        self.assertEqual(len(rep["version_drift"]), 1)
        self.assertEqual(rep["version_drift"][0]["want"], "0.4.2+local.1")

    def test_modified_files_read_stale(self) -> None:
        site = self.site(patched=False, debris=False)
        rep = run_audit(site, "0.4.2")
        stale = {p["path"] for p in rep["problems"]["stale"]}
        self.assertIn("autoforge/core/llm.py", stale)
        self.assertIn("autoforge/tools/registry.py", stale)
        self.assertIn("autoforge/agent.py", stale)

    def test_untouched_files_do_not(self) -> None:
        site = self.site(patched=False, debris=False)
        rep = run_audit(site, "0.4.2")
        stale = {p["path"] for p in rep["problems"]["stale"]}
        self.assertNotIn("autoforge/__init__.py", stale)
        self.assertNotIn("autoforge/core/message.py", stale)

    def test_added_files_read_missing_when_gone(self) -> None:
        site = self.site(patched=False, debris=False)
        rep = run_audit(site, "0.4.2")
        missing = {p["path"] for p in rep["problems"]["missing"]}
        self.assertIn("autoforge/media.py", missing)
        self.assertIn("autoforge/tools/exposure.py", missing)

    def test_debris_reported_as_stash_not_unexpected(self) -> None:
        site = self.site(patched=False, debris=True)
        rep = run_audit(site, "0.4.2")
        stash = [p["path"] for p in rep["problems"]["stash"]]
        self.assertIn("~utoforge_agent-0.4.2.dist-info/", stash)
        # the deep `__pycache__` stash is recorded as a directory: all it
        # holds is bytecode, which is skipped, so the directory itself is the
        # only thing that can report it
        self.assertIn("autoforge/tools/~%pycache__/", stash)
        # ...and never twice: once as debris, once as "unexpected"
        unexpected = {p["path"] for p in rep["problems"]["unexpected"]}
        self.assertFalse([p for p in unexpected if "~" in p], unexpected)
        unexpected = {p["path"] for p in rep["problems"]["unexpected"]}
        self.assertFalse([p for p in unexpected if p.startswith("~")], unexpected)

    def test_verdict_is_drifted(self) -> None:
        site = self.site(patched=False, debris=True)
        rep = run_audit(site, "0.4.2")
        self.assertEqual(rep["verdict"], "drifted")


# ----------------------------------------------------------------------
# quarantine mechanics
# ----------------------------------------------------------------------
class TestQuarantine(DoctorCase):
    def test_move_never_nests(self) -> None:
        """shutil.move(src, dst) moves *into* dst when dst already exists."""
        q = self.tmp / "q"
        src = self.tmp / "~ore"
        src.mkdir()
        (src / "leftover.py").write_bytes(b"x")
        first = doctor.quarantine_move(src, q)
        self.assertEqual(first.parent, q)
        self.assertTrue((first / "leftover.py").is_file())

        src2 = self.tmp / "site" / "~ore"
        src2.parent.mkdir(parents=True)
        src2.mkdir()
        (src2 / "other.py").write_bytes(b"y")
        second = doctor.quarantine_move(src2, q)
        self.assertEqual(second.parent, q)
        self.assertNotEqual(first, second)
        # and it is a move, not a copy: the source is gone
        self.assertFalse(src2.exists())
        self.assertTrue((second / "other.py").is_file())

    def test_repeated_repair_is_idempotent(self) -> None:
        q = self.tmp / "q"
        for i in range(3):
            src = self.tmp / f"round{i}" / "autoforge" / "~ore"
            src.parent.mkdir(parents=True)
            src.mkdir()
            (src / "leftover.py").write_bytes(b"x")
            site = self.tmp / f"round{i}"
            with mock.patch.object(doctor, "QUARANTINE", q):
                moved = doctor.clear_stashes(site)
            self.assertEqual(moved, 1)
            self.assertFalse([e for e in site.glob("**/~*") if e.is_dir()])
        names = sorted(p.name for p in q.iterdir())
        self.assertEqual(names, ["~ore", "~ore.2", "~ore.3"])
        # every copy is intact, nothing was overwritten
        for p in q.iterdir():
            self.assertTrue((p / "leftover.py").is_file())

    def test_clear_stashes_reaches_any_depth(self) -> None:
        """Debris is not only at the top of the package tree.

        `autoforge/tools/~-re` and the `__pycache__` stashes
        (`autoforge/tools/~.pycache__`) both turn up two levels down. A
        one-level `iterdir()` leaves them exactly where they were, which is
        the difference between an audit that says "clean" and a clean tree.
        """
        site = self.tmp / "sp"
        for rel in ("autoforge/tools/~-re", "autoforge/~ore",
                    "autoforge/web/~.pycache__",
                    "autoforge/tools/~.pycache__"):
            d = site / rel
            d.mkdir(parents=True)
            (d / "leftover.py").write_bytes(b"x")
        with mock.patch.object(doctor, "QUARANTINE", self.tmp / "q"):
            moved = doctor.clear_stashes(site)
        self.assertEqual(moved, 4)
        self.assertFalse([e for e in site.glob("**/~*") if e.is_dir()])
        self.assertTrue((self.tmp / "q" / "~-re").is_dir())
        self.assertTrue((self.tmp / "q" / "~ore").is_dir())

    def test_clear_stashes_ignores_other_distributions(self) -> None:
        site = self.tmp / "sp"
        for name in ("certifi", "requests", "~ertifi", "pip"):
            (site / name).mkdir(parents=True)
            (site / name / "x.py").write_bytes(b"x")
        (site / "~utoforge_agent-0.4.2.dist-info").mkdir()
        (site / "~utoforge_agent-0.4.2.dist-info" / "y.py").write_bytes(b"y")
        with mock.patch.object(doctor, "QUARANTINE", self.tmp / "q"):
            moved = doctor.clear_stashes(site)
        self.assertEqual(moved, 1)
        left = sorted(p.name for p in site.iterdir())
        self.assertIn("~ertifi", left)          # not ours to move
        self.assertNotIn("~utoforge_agent-0.4.2.dist-info", left)
        self.assertIn("certifi", left)


# ----------------------------------------------------------------------
# fetching the wheel from the network
# ----------------------------------------------------------------------
class TestDownloadWheel(unittest.TestCase):
    """The wheel, when the local copy is gone.

    `github.com` and `api.github.com` do not fail the same way. The sandbox's
    proxy answers the CONNECT to github.com with 502, which takes
    `pip install https://github.com/...` with it -- so the recorded release
    URL, the one that works everywhere else, works nowhere on this machine.
    The API route works, and a public repo's assets need no token.
    """

    URL = ("https://github.com/ZhangYangyi03/autoforge/releases/download/"
           "local-0.4.2.1/autoforge_agent-0.4.2%2Blocal.1-py3-none-any.whl")

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dltest_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp,
                                                            ignore_errors=True))
        import hashlib
        self.man = fake_manifest()
        self.man["wheel"] = {
            "name": "w.whl", "bytes": 4, "url": self.URL,
            "sha256": hashlib.sha256(b"data").hexdigest(),
        }

    def test_repo_and_tag_parsed_out_of_the_wheel_url(self) -> None:
        self.assertEqual(doctor._repo_and_tag(self.URL),
                         ("ZhangYangyi03/autoforge", "local-0.4.2.1"))

    def test_unparseable_url_is_named_not_swallowed(self) -> None:
        with self.assertRaises(ValueError):
            doctor._repo_and_tag("https://example.invalid/w.whl")

    def test_direct_url_tried_first(self) -> None:
        dest = self.tmp / "w.whl"
        calls = []

        def fake_urlopen(req, timeout=0):
            calls.append(getattr(req, "full_url", req))
            raise OSError("nope")

        with mock.patch("urllib.request.urlopen", fake_urlopen), \
                mock.patch.object(doctor, "download_via_api",
                                  lambda m, d: (_ for _ in ()).throw(
                                      AssertionError("api reached too early"))):
            got = doctor.download_wheel(self.man, dest)
        self.assertIsNone(got)
        self.assertEqual(len(calls), 1)          # direct only, then it stopped
        self.assertIn("github.com", calls[0])
        self.assertFalse(dest.exists())

    def test_falls_back_to_the_api_when_the_direct_url_is_blocked(self) -> None:
        import contextlib, io
        dest = self.tmp / "w.whl"

        def fake_urlopen(req, timeout=0):
            url = getattr(req, "full_url", req)
            if "api.github.com" in url:
                # the asset itself
                return contextlib.contextmanager(
                    lambda: (yield io.BytesIO(b"data")))()
            raise OSError("Tunnel connection failed: 502 Bad Gateway")

        with mock.patch("urllib.request.urlopen", fake_urlopen), \
                mock.patch.object(doctor, "_api_get_json",
                                  lambda u: {"assets": [
                                      {"name": "w.whl",
                                       "url": "https://api.github.com/assets/1"}]}):
            got = doctor.download_wheel(self.man, dest)
        self.assertIsNotNone(got)
        self.assertEqual(dest.read_bytes(), b"data")

    def test_a_wheel_with_the_wrong_hash_is_not_the_build(self) -> None:
        import contextlib, io
        dest = self.tmp / "w.whl"

        def fake_urlopen(req, timeout=0):
            return contextlib.contextmanager(
                lambda: (yield io.BytesIO(b"WRONG")))()

        with mock.patch("urllib.request.urlopen", fake_urlopen), \
                mock.patch.object(doctor, "_api_get_json",
                                  lambda u: {"assets": [
                                      {"name": "w.whl", "url": "x"}]}):
            got = doctor.download_wheel(self.man, dest)
        self.assertIsNone(got)                    # rejected, not written through
        self.assertFalse(dest.exists())            # and not left on disk to reuse

    def test_api_request_carries_a_user_agent(self) -> None:
        """GitHub 403s an API request with no User-Agent, which urllib omits."""
        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["ua"] = req.get_header("User-agent")
            seen["accept"] = req.get_header("Accept")
            import contextlib
            return contextlib.contextmanager(lambda: (yield io.BytesIO(
                b'{"assets": []}')))()

        import io
        with mock.patch("urllib.request.urlopen", fake_urlopen):
            try:
                doctor._api_get_json("https://api.github.com/x")
            except Exception:
                pass
        self.assertTrue(seen["ua"], seen)
        self.assertNotIn("urllib", seen["ua"])


# ----------------------------------------------------------------------
# the CLI hook
# ----------------------------------------------------------------------
class TestCliHook(unittest.TestCase):
    """The auditor must be reachable without the audited package."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(SITE_PKG))
        from autoforge import cli as _cli
        cls.cli = _cli

    def test_doctor_is_a_real_command(self) -> None:
        p = self.cli.build_parser()
        choices = p._subparsers._group_actions[0].choices
        self.assertIn("doctor", choices)
        self.assertIn("doctor", self.cli.command_table())
        self.assertTrue(callable(self.cli.command_table()["doctor"]))

    def test_doctor_takes_repair_and_json(self) -> None:
        p = self.cli.build_parser()
        args = p.parse_args(["doctor", "--repair", "--as-json"])
        self.assertTrue(args.repair)
        self.assertTrue(args.as_json)

    def test_every_command_has_a_handler(self) -> None:
        """The seam `tick` fell through: a parser command with no handler."""
        p = self.cli.build_parser()
        choices = set(p._subparsers._group_actions[0].choices)
        self.assertEqual(choices, set(self.cli.command_table()))

    def test_half_patch_warning_fires_on_stock(self) -> None:
        self.cli._manifest = lambda: fake_manifest()
        with mock.patch.object(importlib.metadata, "version",
                               lambda _n: "0.4.2"):
            msg = self.cli.half_patch_warning()
        self.assertIn("0.4.2", msg or "")
        self.assertIn("stock reinstall", msg or "")
        self.assertIn("auto doctor --repair", msg or "")

    def test_half_patch_warning_names_a_foreign_build_honestly(self) -> None:
        self.cli._manifest = lambda: fake_manifest()
        with mock.patch.object(importlib.metadata, "version",
                               lambda _n: "0.4.3"):
            msg = self.cli.half_patch_warning()
        self.assertIn("different build", msg or "")
        self.assertNotIn("stock reinstall", msg or "")

    def test_no_warning_on_the_right_build(self) -> None:
        self.cli._manifest = lambda: fake_manifest()
        with mock.patch.object(importlib.metadata, "version",
                               lambda _n: "0.4.2+local.1"):
            self.assertIsNone(self.cli.half_patch_warning())

    def test_no_warning_without_a_manifest(self) -> None:
        self.cli._manifest = lambda: None
        with mock.patch.object(importlib.metadata, "version",
                               lambda _n: "0.4.2"):
            self.assertIsNone(self.cli.half_patch_warning())

    def test_no_warning_when_suppressed(self) -> None:
        self.cli._manifest = lambda: fake_manifest()
        with mock.patch.dict(os.environ, {"AUTOFORGE_NO_DOCTOR": "1"}), \
                mock.patch.object(importlib.metadata, "version",
                                  lambda _n: "0.4.2"):
            self.assertIsNone(self.cli.half_patch_warning())

    def test_warning_costs_one_metadata_lookup(self) -> None:
        """It runs on every CLI start; it must not touch the filesystem much."""
        self.cli._manifest = lambda: fake_manifest()
        calls = []
        real = importlib.metadata.version
        def counting(name):
            calls.append(name)
            return "0.4.2+local.1"
        with mock.patch.object(importlib.metadata, "version", counting):
            self.cli.half_patch_warning()
        self.assertEqual(calls, ["autoforge-agent"])
        importlib.metadata.version = real


if __name__ == "__main__":
    unittest.main(verbosity=2)
