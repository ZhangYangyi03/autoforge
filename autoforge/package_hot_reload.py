"""Install edited source into a running agent, without restarting it.

Why this file exists: `amend_self` changes the *prompt string* in memory and
dies with the process. Editing agent.py persists but needs a restart. Neither
one is "hot": there was no path from "I changed the code" to "the running
process is now executing it". This is that path.

What makes it safe enough to run on a live agent:

  * the package is reloaded from a directory given to it, so an edit can be
    staged and tested in a COPY first and the live install only happens if the
    copy worked;
  * every module's `__dict__` is snapshotted before it is re-executed, because
    `importlib.reload` re-executes *into the live namespace* -- a module that
    raises halfway through leaves a half-old, half-new namespace behind. That
    is the failure mode this file is written around, and the reason a failed
    reload restores every snapshot instead of just reporting the error;
  * a byte-identity check refuses to install a tree that is not the tree that
    was verified;
  * the running object is found by walking the garbage collector, which is the
    only honest way to ask a process "which agent am I?" from inside a tool;
    its class is then swapped in place, so every existing reference -- the
    registry, the selfmod log, the run loop -- keeps pointing at it.

The class swap is the whole trick. `obj.__class__ = new_class` changes what
existing instances dispatch to; it works when both classes have plain instance
dicts (no __slots__ difference). A running dataclass whose fields changed
keeps its attributes and gains the new methods.

Known limit, stated plainly: the loop that is *already executing* resumes in
the old frame. The new code takes effect at the next dispatch through the
swapped object -- a tool call, a turn boundary -- not mid-statement.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
import time
import traceback


def package_base(start: str | None = None) -> str:
    """The directory that *contains* the `autoforge` package."""
    if start:
        return os.path.abspath(start)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def module_files(base: str, package: str = "autoforge") -> list[tuple[str, str]]:
    """(module name, path) for every .py in the package, shallow modules first."""
    root = os.path.join(base, package)
    out: list[tuple[str, str]] = []
    for dp, dn, fn in os.walk(root):
        if "__pycache__" in dp:
            continue
        for f in fn:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dp, f)
            rel = os.path.relpath(path, base)
            name = rel[:-3].replace(os.sep, ".")
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            out.append((name, path))
    out.sort(key=lambda item: (item[0].count("."), item[0]))
    return out


def snapshot(module):
    return dict(module.__dict__)


def restore(module, snap: dict) -> None:
    """Put a half-reloaded namespace back the way it was."""
    d = module.__dict__
    for key in list(d):
        if key not in snap:
            del d[key]
    d.update(snap)


def find_running_agent(package: str = "autoforge"):
    """The live object, found by asking the garbage collector for it.

    A tool's code is compiled in its own namespace, so it has no `self` to hand
    over. `gc.get_objects` reaches the real instance anyway. Candidates are
    collected and then ordered by what makes a good *target*, because more than
    one agent-shaped object can be live at once -- an evaluator holding a second
    agent, a child kept for inspection -- and swapping the wrong one reports
    success while changing nothing that runs.

    The order, and why:

      1. an instance of the class this package's agent module defines right now.
         A stale instance from an earlier generation is a worse choice than a
         current one: swapping it is work done to a dead object;
      2. among those, one that carries a `store`. The store is what persists a
         self-modification, so the object holding it is the one whose changes
         outlive the process -- the operator's own agent, not a probe;
      3. otherwise the first candidate seen.

    Deliberately not "the first object with a matching attribute": that is how
    this picks an object that merely looks like the agent.
    """
    agent_mod = sys.modules.get(package + ".agent")
    wanted = getattr(agent_mod, "ForgeAgent", None) if agent_mod else None

    candidates = []
    for obj in gc.get_objects():
        try:
            cls = type(obj)
            mod = getattr(cls, "__module__", "") or ""
            if not mod.startswith(package):
                continue
            score = sum(1 for a in ("registry", "policy", "system_prompt",
                                    "selfmod", "forge_config", "trace")
                        if hasattr(obj, a))
            if score >= 3:
                candidates.append(obj)
        except Exception:                                     # noqa: BLE001
            continue

    if wanted is not None:
        exact = [o for o in candidates if isinstance(o, wanted)]
        if exact:
            candidates = exact
    with_store = [o for o in candidates if getattr(o, "store", None) is not None]
    if with_store:
        candidates = with_store
    return candidates[0] if candidates else None


def install(base: str, *, target: str = "autoforge/agent.py", check_sha: str = "",
            swap: bool = True, package: str = "autoforge") -> dict:
    """Reload the package from `base`, all-or-nothing. Returns a report dict."""
    t0 = time.time()
    report: dict = {"base": base, "target": target, "ts": time.time(),
                    "reloaded": [], "failed": [], "rolled_back": False, "tools_added": [],
                    "swapped": False, "swap_error": None, "sha_before": None}

    target_path = os.path.join(base, target.replace("/", os.sep))
    if not os.path.exists(target_path):
        report["failed"].append([target, "no such file"])
        report["ok"] = False
        return report
    report["sha_before"] = sha256_file(target_path)
    if check_sha and report["sha_before"] != check_sha:
        report["failed"].append([target, "sha256 %s != expected %s"
                                 % (report["sha_before"], check_sha)])
        report["ok"] = False
        return report

    # Make sure the code under test is the code that gets imported: the staged
    # tree must win over whatever is already on sys.path.
    if sys.path and sys.path[0] != base:
        sys.path.insert(0, base)
    importlib.invalidate_caches()

    snaps: dict[str, dict] = {}
    modules = module_files(base, package)
    for name, _path in modules:
        mod = sys.modules.get(name)
        if mod is not None:
            snaps[name] = snapshot(mod)

    for name, _path in modules:
        if name.endswith("package_hot_reload"):
            # This file is the one running; re-executing it would re-enter the
            # driver mid-flight. Its own content is picked up next time.
            continue
        try:
            mod = sys.modules.get(name)
            if mod is None:
                importlib.import_module(name)
            else:
                importlib.reload(mod)
            report["reloaded"].append(name)
        except BaseException as e:                            # noqa: BLE001
            report["failed"].append([name, "".join(
                traceback.format_exception_only(type(e), e)).strip()])
            break

    if report["failed"]:
        for name, snap in snaps.items():
            mod = sys.modules.get(name)
            if mod is not None:
                restore(mod, snap)
        report["rolled_back"] = True
        report["ok"] = False
        report["elapsed_s"] = time.time() - t0
        return report

    if swap:
        victim = find_running_agent()
        report["running_object"] = None if victim is None else type(victim).__name__
        if victim is None:
            report["swap_error"] = "no running agent found in this process"
        else:
            agent_mod = sys.modules.get(package + ".agent")
            new_cls = getattr(agent_mod, "ForgeAgent", None)
            if new_cls is None:
                report["swap_error"] = "reloaded %s.agent has no ForgeAgent" % package
            elif type(victim) is new_cls:
                report["swapped"] = True
                report["swap_error"] = "already current"
            else:
                old = type(victim).__name__
                try:
                    victim.__class__ = new_cls
                    report["swapped"] = True
                    report["swap_error"] = "%s -> %s" % (old, new_cls.__name__)
                    # A method added by the edit only becomes a *tool* if the
                    # registration pass runs again: the schemas the model sees
                    # come from the registry, not from the class. Without this
                    # the new code is present and unreachable.
                    reg = getattr(victim, "_register_meta_tools", None)
                    if callable(reg):
                        before = set(victim.registry.names()) if getattr(victim, "registry", None) else set()
                        reg()
                        after = set(victim.registry.names())
                        report["tools_added"] = sorted(after - before)
                        report["tool_count"] = len(after)
                except Exception as e:                        # noqa: BLE001
                    report["swap_error"] = "%s: %s" % (type(e).__name__, e)

    # The install is a self-modification like any other, so it goes where the
    # others go: the agent's own ledger, written by the store that holds it.
    victim = find_running_agent(package)
    store = getattr(victim, "store", None) if victim is not None else None
    if store is not None:
        try:
            store.log_event("hot_reload", {
                "base": base, "target": target, "sha256": report["sha_before"],
                "modules": len(report["reloaded"]), "failed": report["failed"],
                "rolled_back": report["rolled_back"], "swapped": report["swapped"],
                "swap_error": report["swap_error"],
                "tools_added": report.get("tools_added", []),
            })
            report["ledger"] = "recorded"
        except Exception as e:                                # noqa: BLE001
            report["ledger"] = "not recorded: %s: %s" % (type(e).__name__, e)

    report["ok"] = not report["failed"]
    report["elapsed_s"] = time.time() - t0
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="reload the agent package in this process")
    ap.add_argument("--base", default="")
    ap.add_argument("--target", default="autoforge/agent.py")
    ap.add_argument("--check-sha", default="")
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--report", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    rep = install(package_base(a.base or None), target=a.target,
                  check_sha=a.check_sha, swap=not a.no_swap)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
    if not a.quiet:
        print(json.dumps({k: v for k, v in rep.items() if k != "reloaded"},
                         ensure_ascii=False)[:900])
        print("  reloaded %d module(s) in %.2fs" % (len(rep["reloaded"]), rep.get("elapsed_s", 0)))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
