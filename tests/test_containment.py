"""Containment and declaration, against the real kernel — not a mock.

Every assertion here was first a hand-run with a printed number, and the
numbers are what the test keeps. A job object that is only described in a
docstring is a claim; these are the five measurements that make it a fact on
this host:

  * a 600 MB allocation under a 256 MB cap raises MemoryError inside the child
    (so the tool can still apologise) instead of thrashing the machine;
  * a 4th fork under ActiveProcessLimit=3 is refused by the kernel
    (WinError 1816), not killed afterwards by a counter in Python;
  * a busy loop under a 3 s CPU cap dies with the kernel's own verdict,
    STATUS_JOB_TIME_LIMIT, and the error names that limit;
  * a deliberately detached grandchild is gone after the run, where the
    uncontained path leaves it running — that difference is the whole point
    of KILL_ON_JOB_CLOSE and it is asserted as a difference;
  * a normal tool is unaffected: 0.1 s, correct answer.

The manifest tests are the AgenticOS ordering — declare, admit, then run —
with the one honest negative that matters: `network: False` is recorded and
NOT enforced, and the report says so in those words.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from autoforge.forge.manifest import (CEILING, CapabilityManifest,
                                      ManifestRefused, apply_declaration,
                                      intent_for, reconcile)
from autoforge.forge.sandbox import Sandbox

pytestmark = pytest.mark.skipif(
    os.name != "nt", reason="job objects are the Windows containment backend")

DETACH = (
    "import subprocess, sys\n"
    "def f():\n"
    "    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],\n"
    "                         creationflags=0x00000008 | 0x00000200)\n"
    "    return p.pid\n"
)


def _alive(pid) -> bool:
    # encoding="mbcs": tasklist speaks the ANSI code page, and this host's is
    # CP936 -- its header contains bytes that are not valid UTF-8, so the
    # default text mode raised in the reader thread and left stdout as None
    # (which then blew up on the `in`). The PID we look for is ASCII, so
    # replacing the undecodable header bytes loses nothing.
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                         capture_output=True, encoding="mbcs",
                         errors="replace").stdout or ""
    return str(pid) in out


def _kill(pid) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)


def test_default_sandbox_is_contained():
    box = Sandbox()
    assert box.contain is True
    assert "job object" in box._containment_line()


def test_memory_cap_reaches_the_child():
    box = Sandbox(timeout=10.0, memory_mb=256)
    r = box.run("def f():\n    return len(bytearray(600 * 1024 * 1024))", "f")
    assert not r.ok
    assert "MemoryError" in (r.error or "")


def test_process_cap_is_enforced_by_the_kernel():
    box = Sandbox(timeout=15.0, max_processes=3)
    r = box.run(
        "import subprocess, sys\n"
        "def f():\n"
        "    return len([subprocess.Popen([sys.executable, '-c',"
        " 'import time; time.sleep(2)']) for _ in range(6)])", "f")
    assert not r.ok
    assert "1816" in (r.error or "") or "\u914d\u989d" in (r.error or "")


def test_cpu_cap_names_itself():
    box = Sandbox(timeout=10.0, cpu_seconds=3.0)
    r = box.run("def f():\n    while True: pass", "f")
    assert not r.ok and r.timed_out
    assert "CPU-time limit" in (r.error or ""), r.error


def test_detached_grandchild_outlives_the_uncontained_run_and_not_the_contained_one():
    """The A/B. One of these two is the reason the module exists."""
    plain = Sandbox(timeout=10.0, contain=False)
    r = plain.run(DETACH, "f")
    assert r.ok, r.error
    time.sleep(1.5)
    try:
        assert _alive(r.output), "the uncontained path was expected to leak this process"
    finally:
        _kill(r.output)

    box = Sandbox(timeout=10.0)
    r2 = box.run(DETACH, "f")
    assert r2.ok, r2.error
    time.sleep(1.5)
    assert not _alive(r2.output), "kill-on-close did not reach the detached grandchild"


def test_normal_tool_is_unaffected():
    box = Sandbox(timeout=10.0)
    r = box.run("def f(a=1):\n    return {'v': a * 2}", "f", {"a": 21})
    assert r.ok and r.output == {"v": 42}
    # 2, not 1. The runner is started with CREATE_NO_WINDOW, so Windows gives it
    # a console -- invisible -- and that console is a hidden conhost.exe inside
    # the job this number is read from. The tool launched nothing itself; the
    # second process is the platform's. The number was 1 while the runner was
    # started DETACHED_PROCESS, which is the flag that leaked windows from the
    # snippet's own children, so the 1 was the cheaper of two wrong readings.
    assert r.accounting.get("processes_launched") == 2


def test_the_process_count_moves_with_the_tool_not_the_platform():
    """The counterpart to the floor above: the number still detects a leak.

    Pinned as its own measurement because changing 1 to 2 in a single test would
    otherwise look like an assertion loosened to make a red test green. It is a
    calibrated floor: the constant part is the runner and its hidden console, and
    a tool that starts one child on top of that moves the count again.
    """
    box = Sandbox(timeout=15.0)
    r = box.run(
        "import subprocess, sys\n"
        "def f():\n"
        "    subprocess.Popen([sys.executable, '-c',"
        " 'import time; time.sleep(1)']).wait()\n"
        "    return 'done'\n", "f")
    assert r.ok, r.error
    assert r.accounting.get("processes_launched", 0) >= 3


def test_accounting_travels_on_the_result():
    box = Sandbox(timeout=10.0, memory_mb=256)
    r = box.run("def f(n=1000000):\n    return sum(i * i for i in range(n))", "f")
    assert r.ok
    assert r.accounting.get("peak_job_bytes", 0) > 0
    assert r.accounting.get("cpu_ms", 0) > 0


# -- the declaration ------------------------------------------------------

def test_manifest_sets_the_sandbox_fields():
    m = CapabilityManifest(intent="count rows in a csv", memory_mb=64,
                           max_processes=2, cpu_seconds=5.0, wall_s=10.0)
    m.admit()
    box = apply_declaration(Sandbox(), m)
    assert (box.memory_mb, box.max_processes) == (64, 2)
    assert box.timeout == 10.0


def test_manifest_over_the_ceiling_is_refused_before_anything_runs():
    m = CapabilityManifest(intent="mine", memory_mb=CEILING["memory_mb"] + 1)
    with pytest.raises(ManifestRefused) as exc:
        m.admit()
    assert "before any code ran" in str(exc.value)


def test_manifest_without_an_intent_is_refused():
    with pytest.raises(ManifestRefused):
        CapabilityManifest(intent="   ").admit()


def test_general_purpose_run_python_is_not_capped_by_a_tool_budget():
    m = intent_for(name="run_python")
    assert m.general_purpose and m.memory_mb >= 512


def test_network_field_is_declared_and_not_enforced():
    line = CapabilityManifest(intent="x", network=False).network_line()
    assert "NOT enforced" in line
    assert "socket" in line


def test_reconcile_reports_under_declaration_on_the_axis_that_moved():
    """One axis at a time, because the tight one masks the other.

    The first version of this test declared 8 MB *and* 0.5 s. The child died of
    MemoryError in 15 ms and never touched the CPU budget, so the CPU finding
    it was asserting never happened. Which is itself the useful fact: a
    declaration can be too small in two different ways and only one of them
    shows up as "used more than declared".

    The work is ~0.25 s (n=2_000_000, measured), not the ~0.8 s it started at,
    and that is a deliberate move off a knife edge rather than a loosened
    assertion. `JOB_OBJECT_LIMIT_JOB_TIME` is checked when the kernel
    reschedules, so it lands *late* -- 2.4 s of overshoot for a 3 s budget,
    measured on this host and written down in containment.py. A run whose CPU
    cost sits under that overshoot is therefore a coin flip between "finished"
    and "killed by the CPU limit", and at 0.8 s against a 0.1 s budget this test
    was exactly that: it passed alone and failed in file order, three runs out
    of four. The process the hidden console adds (conhost.exe, see
    test_normal_tool_is_unaffected) was enough to tip it. 0.25 s still
    under-reports the declaration by 2.5x, which is all the assertion needs.
    """
    m = CapabilityManifest(intent="tiny", memory_mb=256, cpu_seconds=0.1, wall_s=30.0)
    box = apply_declaration(Sandbox(), m)
    r = box.run("def f(n=2000000):\n    return sum(i * i for i in range(n))", "f")
    out = reconcile(m, r, r.accounting)
    assert r.ok, r.error
    assert out["under_declared"], "a 0.1s declaration with 0.25s of work must be reported"
    assert any("cpu" in f for f in out["under_declared"])


def test_reconcile_reports_a_declaration_that_was_too_small_to_survive():
    m = CapabilityManifest(intent="tiny", memory_mb=8, cpu_seconds=5.0, wall_s=10.0)
    box = apply_declaration(Sandbox(), m)
    r = box.run("def f():\n    return len(bytearray(600 * 1024 * 1024))", "f")
    out = reconcile(m, r, r.accounting)
    assert any("whole declared 8 MB" in f for f in out["under_declared"]), out
