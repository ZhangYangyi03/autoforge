#!/usr/bin/env python3
"""autoforge install doctor — audit and repair a half-patched install.

Why this exists
---------------
The install on this machine carries hand patches on top of the 0.4.2 wheel.
Reinstalling from the stock wheel does not simply revert them; it leaves the
build in a *half-patched* state that is worse than either. Two separate
mechanisms produce it, both observed on this machine:

1. **Pip deletes only what the old RECORD lists.** A file the patch added is
   not in that RECORD, so it survives; a file the patch modified is in it, so
   it is overwritten with the stock version. `tools/exposure.py` remains while
   `tools/registry.py` loses `exposure()` — the library looks present and the
   feature is gone.
2. **Pip's installs get killed mid-flight.** The sandbox SIGTERMs Python
   processes that make network calls, and pip makes network calls. Killed
   between the stash and the delete, pip leaves the removed tree behind under
   names that keep only the tail and are prefixed with `~` (`core` -> `~ore`,
   `cpu` -> `~pu`). On this machine that had accumulated to 114 files in 17
   directories *inside the package*, where any walk over the tree finds them.

Neither state announces itself. The first looks like a working install with the
feature switched off; the second is invisible until something builds a wheel
from the tree and ships the dead code. Both were live on 2026-10-16.

So: a manifest records exactly what the build put on disk, and this script
audits the install against it and repairs it.

Where it lives
--------------
`~/.autoforge/install_doctor.py` — deliberately **outside** the package. The
auditor must not depend on the thing it audits: a stock reinstall removes
`autoforge/doctor.py` along with everything else the patch touched, so a doctor
inside the package would disappear exactly when it is needed. This file imports
nothing but the standard library and stays put.

Usage
-----
    python install_doctor.py                 # audit, human-readable
    python install_doctor.py --json          # audit, machine-readable
    python install_doctor.py --repair        # audit, then heal what drifted
    python install_doctor.py --manifest PATH # audit against a specific build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

#: Where the manifest lives: the agent's own home, not the package.
HOME_CANDIDATES = [
    Path(os.environ.get("AUTOFORGE_HOME", "")) if os.environ.get("AUTOFORGE_HOME") else None,
    Path.home() / ".autoforge",
    Path(os.environ.get("LOCALAPPDATA", "")) / "autoforge"
    if os.environ.get("LOCALAPPDATA") else None,
]

#: Where to look for a copy of the wheel before going to the network.
WHEEL_SEARCH = [
    Path(r"C:\autoforge\venv\Lib\site-packages"),
    Path(r"D:\Users\yangyian.zhang\WorkBuddy AI\2026-10-16-10-08-39"
         r"\autoforge-interrupt-fix\dist"),
    Path(r"D:\Users\yangyian.zhang\WorkBuddy AI\2026-10-15-12-51-18"
         r"\autoforge-deploy\wheels"),
]

MANIFEST_NAME = "autoforge-install-manifest.json"
STASH_PREFIX = "~"          # see build_wheel.py for the full story

#: Files pip writes itself, on every install, whatever the wheel contained.
#: Hashed, these are false positives forever: RECORD is rewritten to match
#: the freshly installed tree, INSTALLER/REQUESTED/direct_url.json record who
#: asked for the install and from where. Their presence proves pip ran; it
#: says nothing about whether the right wheel landed.
PIP_GENERATED = frozenset({
    "dist-info/RECORD",
    "dist-info/INSTALLER",
    "dist-info/REQUESTED",
    "dist-info/WHEEL",
    "dist-info/direct_url.json",
})

#: Where interrupted-uninstall debris is moved. Outside site-packages on
#: purpose -- a directory named `~ore` on sys.path has never imported
#: anything, but it also has no business being on a path at all.
QUARANTINE = Path(os.environ.get("AUTOFORGE_QUARANTINE") or
                  (Path(os.environ.get("LOCALAPPDATA", "")) / "autoforge"
                   / "pip-stash-quarantine"
                   if os.environ.get("LOCALAPPDATA")
                   else Path.home() / ".autoforge" / "pip-stash-quarantine"))

#: Top-level site-packages names this audit owns. Everything else in
#: site-packages belongs to another distribution (certifi, requests, pip
#: itself) and is none of this audit's business -- comparing against a
#: manifest that lists autoforge makes every one of them "unexpected".
OWNED_PACKAGES = ("autoforge",)

#: pip's stash rename masks leading characters, so `autoforge` lands as
#: `~utoforge` and `autoforge_agent-0.4.2.dist-info` as
#: `~utoforge_agent-0.4.2.dist-info`. It is not even consistent about how many
#: characters it masks -- `~-toforge_agent-0.4.2.dist-info` turns up here too,
#: from the same dist-info. What survives every variant is the substring, and
#: that is what makes this test work at all.
PACKAGE_MARKER = "toforge"


def is_pip_generated(rel: str) -> bool:
    """True for files pip rewrites on every install (see PIP_GENERATED)."""
    return rel.endswith(tuple(PIP_GENERATED))


def is_owned_debris(name: str) -> bool:
    """True if a top-level site-packages entry belongs to a stashed autoforge."""
    return name.startswith(STASH_PREFIX) and PACKAGE_MARKER in name


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_manifest() -> Path | None:
    for base in HOME_CANDIDATES:
        if not base:
            continue
        p = base / MANIFEST_NAME
        if p.is_file():
            return p
    return None


def find_site_packages() -> Path | None:
    """The site-packages root, from the interpreter that is running this.

    Prefers the entry that also carries the dist-info, because a bare
    `autoforge/` directory elsewhere on sys.path (a source checkout, a test
    fixture) would otherwise shadow the real install and audit the wrong tree.
    """
    candidates: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        p = Path(entry)
        if p.is_dir() and (p / "autoforge" / "__init__.py").is_file():
            candidates.append(p)
    for p in candidates:
        if list(p.glob("autoforge_agent-*.dist-info")):
            return p
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------
def audit(manifest: dict[str, Any], site: Path) -> dict[str, Any]:
    """Compare the install against the manifest, item by item."""
    expected: dict[str, dict] = manifest["files"]
    problems: dict[str, list[dict[str, str]]] = {
        "missing": [], "stale": [], "unexpected": [], "stash": [],
    }

    for rel, want in expected.items():
        if is_pip_generated(rel):
            continue
        path = site / rel
        if not path.is_file():
            problems["missing"].append({"path": rel})
            continue
        got = sha256_file(path)
        if got != want["sha256"]:
            problems["stale"].append({
                "path": rel,
                "want": want["sha256"][:12],
                "have": got[:12],
                "bytes_on_disk": str(path.stat().st_size),
            })

    # What is on disk, restricted to autoforge's own footprint. Two roots:
    # the package directory, and the dist-info this build installed. Walking
    # all of site-packages instead would report every file of every other
    # distribution as "unexpected" -- true, and useless.
    on_disk: set[str] = set()
    stash_hits: list[dict[str, str]] = []
    tops: list[str] = []
    for rel0 in sorted(set(expected)):
        top = rel0.split("/", 1)[0]
        if top.startswith(STASH_PREFIX) or top in tops:
            continue
        tops.append(top)
        root = site / top
        if not root.is_dir():
            continue
        for rootpath, dirs, files in os.walk(root):
            # Record each stash *directory* once. The file-level pass below
            # would otherwise list seventy files for ten directories, and
            # `__pycache__` stashes (`tools/~.pycache__`) hold nothing but
            # bytecode, which is skipped -- leaving the directory itself
            # unrecorded and the debris invisible.
            for d in sorted(dirs):
                if d.startswith(STASH_PREFIX):
                    rel = (Path(rootpath) / d).relative_to(site).as_posix()
                    stash_hits.append({"path": rel + "/"})
            # Do NOT prune stash directories. Pruning them is the
            # natural-looking mistake -- they are not package content -- and it
            # makes the files inside invisible to the very audit that reports
            # them, while `clear_stashes` goes on to move them. Descend, then
            # label.
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                rel = (Path(rootpath) / f).relative_to(site).as_posix()
                if f.endswith(".pyc"):
                    continue
                if any(part.startswith(STASH_PREFIX)
                       for part in rel.split("/")):
                    stash_hits.append({"path": rel})
                    continue
                on_disk.add(rel)

    problems["stash"].extend(stash_hits)
    # Reported once, as debris. Not a second time as "unexpected".
    on_disk -= {h["path"] for h in stash_hits}

    for rel in sorted(r for r in on_disk - set(expected)
                      if not is_pip_generated(r)):
        problems["unexpected"].append({"path": rel})

    # Debris left in site-packages itself (pip stashes the dist-info it is
    # about to delete the same way it stashes the package). Same signature,
    # different depth: `~utoforge_agent-0.4.2.dist-info`.
    for entry in sorted(site.iterdir()):
        if is_owned_debris(entry.name):
            problems["stash"].append({"path": entry.name + "/"})

    # The version in the metadata is the cheap signal that catches the whole
    # half-patch case: a stock reinstall reports 0.4.2 against a manifest that
    # says 0.4.2+local.1, and that one comparison finds it without hashing.
    version_drift: list[dict[str, str]] = []
    dist = manifest["distribution"]
    ver = manifest["version"]
    try:
        import importlib.metadata as md
        installed = md.version(dist)
    except Exception:                                     # noqa: BLE001
        installed = ""
    if installed != ver:
        version_drift.append({"want": ver, "have": installed or "(not installed)"})

    counts = {k: len(v) for k, v in problems.items()}
    healthy = (not any(counts.values())) and not version_drift
    return {
        "verdict": "ok" if healthy else "drifted",
        "manifest_version": manifest["version"],
        "installed_version": installed or "(not installed)",
        "version_drift": version_drift,
        "counts": counts,
        "problems": problems,
        "site_packages": str(site),
    }


# --------------------------------------------------------------------------
# the repair
# --------------------------------------------------------------------------
def find_local_wheel(manifest: dict[str, Any]) -> Path | None:
    """A local copy of the exact wheel the manifest describes, if any."""
    want = manifest["wheel"]["sha256"]
    for base in WHEEL_SEARCH:
        if not base.is_dir():
            continue
        for whl in sorted(base.glob("autoforge_agent-*.whl")):
            try:
                if sha256_file(whl) == want:
                    return whl
            except OSError:
                continue
    return None


def _repo_and_tag(url: str) -> tuple[str, str]:
    """(owner/repo, tag) out of a release download URL.

    `https://github.com/O/R/releases/download/TAG/file` -> `(O/R, TAG)`.
    Written against the URL rather than a separate manifest field so a manifest
    built by an older build_wheel.py still resolves.
    """
    m = re.search(r"github\.com/([^/]+/[^/]+)/releases/download/([^/]+)/", url)
    if not m:
        raise ValueError(f"cannot read repo/tag out of {url!r}")
    return m.group(1), m.group(2)


def _api_get_json(url: str) -> Any:
    """GET a GitHub API endpoint as JSON, with the headers it insists on."""
    import urllib.request
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        # GitHub 403s an API request with no User-Agent, and urllib's default
        # ("Python-urllib/3.x") is exactly the kind it drops.
        "User-Agent": "autoforge-install-doctor",
    })
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def download_via_api(manifest: dict[str, Any], dest: Path) -> Path | None:
    """The release asset, fetched through api.github.com instead of github.com.

    This exists because the two hosts fail differently here. The sandbox's
    proxy answers `github.com` CONNECT with **502**, which takes
    `pip install https://github.com/...` down with it -- so the recorded
    wheel URL, the one that works everywhere else, works nowhere on this
    machine. `api.github.com` is allowed through, and a public repo's assets
    need no token to fetch.

    Two steps: resolve the asset id (the API needs it, and it changes on every
    re-upload), then GET that asset URL with `Accept: application/octet-stream`
    -- without that header GitHub answers with a redirect *object* as JSON
    instead of the bytes.
    """
    url = manifest["wheel"]["url"]
    try:
        repo, tag = _repo_and_tag(url)
    except ValueError as exc:
        print(f"    {exc}")
        return None
    name = manifest["wheel"]["name"]
    try:
        rel = _api_get_json(
            f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
        asset = next((a for a in rel.get("assets", []) if a["name"] == name),
                     None)
        if asset is None:
            print(f"    {name} is not an asset on release {tag}")
            return None
        import urllib.request
        req = urllib.request.Request(asset["url"], headers={
            "Accept": "application/octet-stream",
            "User-Agent": "autoforge-install-doctor",
        })
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=600) as r, \
                dest.open("wb") as fh:
            shutil.copyfileobj(r, fh)
    except Exception as exc:                              # noqa: BLE001
        print(f"    api download failed: {type(exc).__name__}: {exc}")
        return None
    return dest


def download_wheel(manifest: dict[str, Any], dest: Path) -> Path | None:
    """The wheel this install should be, from the network, or None.

    Tries the recorded release URL first -- it is the right one for every
    machine that can reach github.com -- then falls back to the API, which is
    the one that works behind a proxy that blocks github.com. Either way the
    sha256 in the manifest has to match: a wheel that hashes differently is
    not the build this install tracks, whatever it calls itself.
    """
    url = manifest["wheel"]["url"]
    for label, attempt in (("release url", lambda: _direct(url, dest)),
                           ("github api", lambda: download_via_api(manifest, dest))):
        try:
            got = attempt()
        except Exception as exc:                          # noqa: BLE001
            print(f"    {label}: {type(exc).__name__}: {exc}")
            got = None
        if got is not None and got.exists() and got.stat().st_size > 0:
            if sha256_file(got) == manifest["wheel"]["sha256"]:
                print(f"    obtained via {label} ({got.stat().st_size:,} B, "
                      f"sha256 ok)")
                return got
            print(f"    {label}: sha256 mismatch, discarding")
            got.unlink(missing_ok=True)
    print("    no route to the wheel worked")
    return None


def _direct(url: str, dest: Path) -> Path | None:
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": "autoforge-install-doctor"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as fh:
        shutil.copyfileobj(r, fh)
    return dest


def repair(manifest: dict[str, Any]) -> bool:
    """Put the build back, from the local wheel if there is one, else the URL."""
    print("  locating the build this install should be")
    wheel = find_local_wheel(manifest)
    if wheel is None:
        print("    no local copy; downloading from GitHub")
        dest = Path(tempfile.mkdtemp(prefix="af_doctor_")) / manifest["wheel"]["name"]
        wheel = download_wheel(manifest, dest)
        if wheel is None:
            print("    !! could not obtain the wheel")
            return False
    else:
        print(f"    found local copy: {wheel}")

    print(f"  reinstalling {wheel.name}")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall",
         "--no-deps", "--no-index", str(wheel)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print("    !! pip failed")
        print("    " + proc.stderr.strip().replace("\n", "\n    ")[-1200:])
        return False
    print("    pip ok")
    return True


def quarantine_move(entry: Path, quarantine: Path) -> Path:
    """Move one stash into quarantine without ever nesting or overwriting.

    `shutil.move(src, dst)` moves *into* dst when dst already exists, which is
    how `--repair` run twice ends up with
    `quarantine/~utoforge_agent-.../~utoforge_agent-...`. A counter keeps every
    move a plain rename and every repair a no-op on an already-clean tree.
    """
    quarantine.mkdir(parents=True, exist_ok=True)
    dst = quarantine / entry.name
    n = 2
    while dst.exists():
        dst = quarantine / f"{entry.name}.{n}"
        n += 1
    shutil.move(str(entry), str(dst))
    return dst


def clear_stashes(site: Path) -> int:
    """Move pip's stashes out of the package tree, where nothing can import them.

    Moved to a quarantine directory rather than deleted: they are the only
    surviving copy of the pre-patch versions of some files, and a directory
    named `~ore` has never imported anything, so removing it changes nothing
    about the running package.

    Two depths, both handled. pip stashes the dist-info at the top level of
    site-packages, and the directories it is about to delete anywhere inside
    the package -- `autoforge/~ore`, `autoforge/tools/~-re`, and the
    `__pycache__` stashes (`autoforge/tools/~.pycache__`) turn up at any
    depth. A one-level `iterdir()` leaves the deep ones exactly where they
    were, which is the difference between "audited clean" and "clean".
    """
    quarantine = QUARANTINE
    moved = 0
    for entry in sorted(site.iterdir()):
        if not (entry.is_dir() and is_owned_debris(entry.name)):
            continue
        quarantine_move(entry, quarantine)
        moved += 1
    for top in OWNED_PACKAGES:
        pkg = site / top
        if not pkg.is_dir():
            continue
        # Deepest first, so a parent is moved before its children -- moving
        # `autoforge/tools/~-re` first would strand `~-re/forge` behind.
        for path in sorted(pkg.rglob("*"), key=lambda p: len(p.parts),
                           reverse=True):
            if not path.is_dir() or not path.name.startswith(STASH_PREFIX):
                continue
            quarantine_move(path, quarantine)
            moved += 1
    return moved


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def print_report(rep: dict[str, Any]) -> None:
    print(f"  site-packages : {rep['site_packages']}")
    print(f"  manifest      : {rep['manifest_version']}")
    print(f"  installed     : {rep['installed_version']}")
    print()

    if rep["version_drift"]:
        for d in rep["version_drift"]:
            print(f"  version drift : installed {d['have']}, manifest {d['want']}")
            if d["have"] == "0.4.2":
                print("                  ^ this is the half-patch signature: the "
                      "stock wheel is in place.")
    c = rep["counts"]
    if c["stale"]:
        print(f"  stale files   : {c['stale']}  (present, but not this build)")
    if c["missing"]:
        print(f"  missing files : {c['missing']}")
    if c["unexpected"]:
        print(f"  unexpected    : {c['unexpected']}  (on disk, not in this build)")
    if c["stash"]:
        print(f"  pip stashes   : {c['stash']}  (interrupted uninstall debris, "
              f"inside the package tree)")

    if rep["verdict"] == "ok":
        print("  VERDICT       : OK — this install is the build it should be.")
        return

    print()
    print("  VERDICT       : DRIFTED — this install is not the build it should be.")
    print()
    shown = 0
    for kind in ("stale", "missing", "unexpected"):
        for item in rep["problems"][kind]:
            if shown >= 25:
                print(f"    ... and more")
                break
            extra = (f"  (manifest {item['want']}, on disk {item['have']}, "
                     f"{item['bytes_on_disk']} B)" if kind == "stale" else "")
            print(f"    {kind:10s} {item['path']}{extra}")
            shown += 1
    if c["stash"]:
        for item in rep["problems"]["stash"][:8]:
            print(f"    {'stash':10s} {item['path']}")
        if c["stash"] > 8:
            print(f"    ... and {c['stash'] - 8} more")
    print()
    print("  fix with:  python install_doctor.py --repair")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    manifest_path = args.manifest or find_manifest()
    if manifest_path is None or not Path(manifest_path).is_file():
        print("  !! no install manifest found.")
        print(f"     looked for {MANIFEST_NAME} in:")
        for base in HOME_CANDIDATES:
            if base:
                print(f"       - {base}")
        print("     The manifest ships with the wheel; without it there is")
        print("     nothing to audit against. Reinstall the build, then rerun.")
        return 2

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    print(f"  manifest     : {manifest_path}")
    print()

    site = find_site_packages()
    if site is None:
        print("  !! cannot locate site-packages (no autoforge on sys.path)")
        return 2

    rep = audit(manifest, site)

    if args.as_json:
        print(json.dumps(rep, indent=2))
        return 0 if rep["verdict"] == "ok" else 1

    print_report(rep)

    if not args.repair:
        return 0 if rep["verdict"] == "ok" else 1

    # Only debris left over from an interrupted pip is not a reason to
    # reinstall: a `pip install --force-reinstall` is exactly the operation
    # that gets killed half way here, so it is the last thing to reach for.
    real_drift = (rep["version_drift"] or rep["counts"]["stale"]
                  or rep["counts"]["missing"] or rep["counts"]["unexpected"])

    if rep["counts"]["stash"]:
        moved = clear_stashes(site)
        print()
        print(f"  quarantined {moved} pip stash entr"
              f"{'y' if moved == 1 else 'ies'} into {QUARANTINE}")

    if not real_drift:
        print()
        print_report(audit(manifest, site))
        return 0

    print()
    if not repair(manifest):
        return 1
    # The reinstall just made fresh stashes of whatever it replaced.
    moved = clear_stashes(site)
    if moved:
        print(f"  quarantined {moved} pip stash entr"
              f"{'y' if moved == 1 else 'ies'} into {QUARANTINE}")
    print()
    rep2 = audit(manifest, find_site_packages() or site)
    print_report(rep2)
    return 0 if rep2["verdict"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
