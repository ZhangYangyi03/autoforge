"""Install the schema-exposure backport into the running autoforge build.

Why this exists
---------------
The deployed `autoforge_agent-0.4.2` wheel predates upstream's exposure split
(`autoforge/tools/exposure.py`). Without it the loop serialises every visible
tool schema into every model call, forever: measured on this machine, 60
schemas / 26,872 characters per request, of which 45 tools had not been called
in a week.

Re-installing the wheel wipes this. Run this script to put it back.

What it does
------------
1. Verifies the target files are either the pristine 0.4.2 originals (by md5
   against `rollback-exposure/`) or already patched (by md5 against
   `patched/`). Anything else is refused rather than clobbered.
2. Copies the five files from `patched/` into site-packages.
3. Clears the affected `__pycache__` entries so the change takes effect.
4. Prints the resulting md5s.

Usage
-----
    C:/autoforge/venv/Scripts/python.exe install_exposure.py [--dry-run]

Rollback: `bash rollback-exposure.sh`
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = Path(r"C:\autoforge\venv\Lib\site-packages\autoforge")

#: target relative path -> (pristine 0.4.2 copy, patched copy)
#: `None` as the pristine copy means the file does not exist before the patch.
FILES: dict[str, tuple[str | None, str]] = {
    "tools/exposure.py": (None,                      "patched/tools/exposure.py"),
    "tools/registry.py": ("rollback-exposure/tools/registry.py", "patched/tools/registry.py"),
    "core/agent.py":     ("rollback-exposure/core/agent.py",     "patched/core/agent.py"),
    "store.py":          ("rollback-exposure/store.py",          "patched/store.py"),
    "agent.py":          ("rollback-exposure/agent.py",          "patched/agent.py"),
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, change nothing")
    ap.add_argument("--force", action="store_true",
                    help="install even if a target is neither pristine nor patched")
    args = ap.parse_args()

    if not SITE.is_dir():
        print(f"  !! no autoforge install at {SITE}")
        return 2

    # ---- 1. decide, before touching anything ---------------------------
    plan: list[tuple[str, Path, Path, str]] = []   # (label, target, source, state)
    problems: list[str] = []
    for rel, (pristine_rel, patched_rel) in FILES.items():
        target = SITE / rel
        source = HERE / patched_rel
        if not source.exists():
            problems.append(f"missing payload {patched_rel}")
            continue

        pristine = HERE / pristine_rel if pristine_rel else None
        if not target.exists():
            state = "new" if pristine is None else "MISSING"
            if pristine is not None:
                problems.append(f"{rel} does not exist but should")
        else:
            cur = md5(target)
            if md5(source) == cur:
                state = "already-patched"
            elif pristine is not None and pristine.exists() and md5(pristine) == cur:
                state = "pristine"
            else:
                state = "unknown"
                if not args.force:
                    problems.append(
                        f"{rel} is neither the pristine 0.4.2 file nor the "
                        f"patched one (md5 {cur[:12]}); refusing to clobber")
        plan.append((rel, target, source, state))

    if problems:
        print("  refusing to install:")
        for p in problems:
            print(f"    - {p}")
        print("\n  (pass --force to overwrite anyway)")
        return 1

    # ---- 2. apply ------------------------------------------------------
    print(f"  target: {SITE}")
    print(f"  mode  : {'DRY RUN' if args.dry_run else 'installing'}")
    print()
    for rel, target, source, state in plan:
        if args.dry_run:
            print(f"  would {state:16s} {rel}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"  {state:16s} {rel}  -> {md5(target)[:12]}")

    if args.dry_run:
        return 0

    # ---- 3. drop the stale bytecode ------------------------------------
    cleared = 0
    for pkg in ("", "core", "tools"):
        cache = SITE / pkg / "__pycache__"
        if not cache.is_dir():
            continue
        for name in os.listdir(cache):
            if name.split(".")[0] in {"agent", "registry", "store", "exposure"}:
                try:
                    os.remove(cache / name)
                    cleared += 1
                except OSError:
                    pass
    print(f"\n  cleared {cleared} stale .pyc file(s)")

    # ---- 4. prove it imports -------------------------------------------
    print("\n  import check:")
    try:
        sys.path.insert(0, str(SITE.parent))
        from autoforge.tools.exposure import HOT_FLOOR          # noqa: F401
        from autoforge.agent import ForgeAgent                  # noqa: F401
        from autoforge.tools.registry import ToolRegistry       # noqa: F401
        reg = ToolRegistry()
        assert hasattr(reg, "exposure") and hasattr(reg, "days_since_called")
        assert hasattr(reg, "last_called_source")
        print("    exposure.py, registry.exposure(), ForgeAgent -- all present")
    except Exception as exc:                                    # noqa: BLE001
        print(f"    FAILED: {exc!r}")
        return 1

    print("\n  done. Verify with:")
    print("    C:/autoforge/venv/Scripts/python.exe -m unittest "
          "test_exposure_backport -v")
    print("    C:/autoforge/venv/Scripts/python.exe redcheck_exposure.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
