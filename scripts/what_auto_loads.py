"""What does `auto` actually import? Run real CLI invocations, union the modules.

Static reachability says which modules *could* be imported. This says which ones
a real session actually loads. Difference = code that ships but never runs.
"""
import os
import subprocess
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DUMP = r'''
import sys, json, io, os
os.environ.setdefault("AUTOFORGE_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("AUTOFORGE_API_KEY", "sk-not-a-real-key")
os.environ.setdefault("AUTOFORGE_MODEL", "dummy-model")
os.environ.setdefault("AUTOFORGE_MAX_TOKENS", "3000")
from autoforge.cli import main
sys.stdin = io.StringIO("")
try:
    main()
except SystemExit:
    pass
except BaseException as e:
    print(f"[{type(e).__name__}: {e}]", file=sys.stderr)
mods = sorted(m for m in sys.modules if m == "autoforge" or m.startswith("autoforge."))
print("@@MODS@@" + json.dumps(mods))
'''

INVOCATIONS = {
    "auto --help": ["--help"],
    "auto modes": ["modes"],
    "auto config": ["config"],
    "auto list": ["list"],
    "auto chat (empty stdin)": ["chat"],
    "auto run <task>": ["run", "list the files here"],
    "auto tick --quiet": ["tick", "--quiet"],
    "auto forge <need>": ["forge", "normalise ISBNs"],
    "auto web (help only)": ["web", "--help"],
}

os.chdir(BASE)
per_invocation = {}
union = set()
for label, argv in INVOCATIONS.items():
    p = subprocess.run(
        [sys.executable, "-c", DUMP] + argv,
        capture_output=True, text=True, timeout=180, cwd=BASE,
        env={**os.environ, "PYTHONPATH": BASE},
    )
    mods = []
    for line in p.stdout.splitlines():
        if line.startswith("@@MODS@@"):
            mods = json.loads(line[len("@@MODS@@"):])
    per_invocation[label] = mods
    union |= set(mods)
    print(f"{label:<28} imported {len(mods):>3} autoforge modules   rc={p.returncode}"
          + (f"  stderr={p.stderr.strip().splitlines()[-1][:70]}" if p.returncode else ""))

# what exists on disk
disk = set()
for dirpath, dirnames, filenames in os.walk(os.path.join(BASE, "autoforge")):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        rel = os.path.relpath(os.path.join(dirpath, fn), BASE).replace(os.sep, ".")
        name = rel[:-3]
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        disk.add(name)

never = sorted(disk - union)
print()
print(f"modules on disk:      {len(disk)}")
print(f"modules auto loaded:  {len(union)}")
print(f"NEVER LOADED by any invocation: {len(never)}")
for m in never:
    print("   ", m)

print()
print("=== per-module: first invocation that loads it ===")
for m in sorted(union):
    first = next((k for k, v in per_invocation.items() if m in v), "?")
    print(f"   {m:<40} {first}")
