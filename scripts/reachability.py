"""Static reachability: which autoforge modules can `auto` ever import?

Builds the intra-package import graph with ast (no imports executed, so it works
for triton-gated modules too) and reports modules unreachable from the real
entry points.
"""
import ast
import os
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)  # repo root when run from scripts/
ROOT = os.path.join(BASE, "autoforge")
PKG = "autoforge"

# module name -> file
mods = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, os.path.dirname(ROOT)).replace(os.sep, "/")
        name = rel[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        mods[name] = full


def resolve(name, module):
    """Resolve an absolute/relative autoforge import to a real module name."""
    if name.startswith(PKG + ".") or name == PKG:
        return name if name in mods else None
    return None


def imports_of(path, module):
    """Set of autoforge modules this file imports (incl. `from . import x`)."""
    out = set()
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError as e:
        print(f"!! syntax error in {path}: {e}")
        return out
    pkg = module if not module.rpartition(".")[2].istitle() else module.rpartition(".")[0]
    # package the module lives in
    if os.path.basename(path) == "__init__.py":
        pkg = module
    else:
        pkg = module.rpartition(".")[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                r = resolve(a.name, module)
                if r:
                    out.add(r)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg
                for _ in range(node.level - 1):
                    base = base.rpartition(".")[0]
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            target = target.rstrip(".")
            if target in mods:
                out.add(target)
            for a in node.names:
                sub = f"{target}.{a.name}"
                if sub in mods:
                    out.add(sub)
    return out


graph = {m: imports_of(p, m) for m, p in mods.items()}

ROOTS = [
    "autoforge.cli",
    "autoforge.__main__",
    "autoforge",  # package __init__ re-exports
]

seen = set()
q = deque(r for r in ROOTS if r in mods)
seen.update(q)
while q:
    m = q.popleft()
    for dep in graph.get(m, ()):
        if dep not in seen:
            seen.add(dep)
            q.append(dep)

unreachable = sorted(m for m in mods if m not in seen)

print(f"modules on disk:            {len(mods)}")
print(f"reachable from auto's entry: {len(seen)}")
print(f"UNREACHABLE:                {len(unreachable)}")
print()
for m in unreachable:
    line = sum(1 for _ in open(mods[m], encoding="utf-8", errors="replace"))
    imported_by = sorted(k for k, v in graph.items() if m in v)
    print(f"{m:<42} {line:>5} lines   imported-by: {imported_by or 'NOTHING'}")

print()
print("--- per-module import counts inside the package (orphan detection) ---")
referenced = set()
for v in graph.values():
    referenced |= v
never_referenced = sorted(m for m in mods if m not in referenced and not m.endswith("__init__"))
print("referenced by no other module in the package:")
for m in never_referenced:
    print("   ", m)
