"""The Linux boundary: that it refuses, and that it is reachable from a run.

The important tests here do not need a running distro, and that is on purpose.
The two failures that matter most are the ones you can only see without one:
that a run which asked for isolation and could not get it is reported as a
REFUSAL rather than quietly handed to the Windows path, and that the path
translation for the code being copied is right -- a wrong one shows up as
"file not found" from inside a VM, which reads as a broken tool.
"""
from __future__ import annotations

import os

import pytest

from autoforge.forge import wsl_isolation as wi
from autoforge.forge.sandbox import Sandbox


class TestPathTranslation:
    def test_windows_paths_become_wsl_mounts(self):
        assert wi._to_wsl_path(r"D:\Users\china\a.py") == "/mnt/d/Users/china/a.py"
        assert wi._to_wsl_path("C:\\Users\\china\\b.py") == "/mnt/c/Users/china/b.py"

    def test_spaces_survive_because_the_caller_quotes(self):
        # Not cosmetic: the guest command quotes this path, and a translator
        # that mangled a space would fail on a username with one.
        assert " " in wi._to_wsl_path(r"C:\Users\first last\t")


class TestRefusalIsNotDowngrade:
    def test_a_missing_distro_is_a_refusal(self):
        """The whole point of asking for isolation.

        Measured behaviour before this test existed: nothing wired the Linux
        side into any execution path at all, so "the probe proves the kernel
        refuses things" and "a forged tool is refused things" were different
        claims and only the first was true.
        """
        sb = Sandbox(timeout=5.0, isolate=True, distro="NoSuchDistroAtAll")
        result = sb.run("def main():\n    return 1\n", "main")
        assert result.ok is False
        assert "refusing to run" in (result.error or "")
        # And in particular it did NOT run on the Windows path and report
        # success, which is the failure mode a refusal is here to prevent.
        assert result.output is None

    def test_isolation_wins_over_a_custom_runner(self):
        """`isolate` is the stronger promise, so it is checked first."""
        called = []
        sb = Sandbox(timeout=5.0, isolate=True, distro="NoSuchDistroAtAll",
                     runner=lambda *a, **k: called.append(1))
        sb.run("def main():\n    return 1\n", "main")
        assert called == []


class TestTheBoundaryIsStatedWhereItIsUsed:
    def test_the_sandbox_has_the_switch(self):
        assert Sandbox().isolate is False          # opt-in, needs a running VM
        assert Sandbox().distro == "Ubuntu"

    def test_runner_for_reads_limits_from_the_sandbox(self):
        sb = Sandbox(timeout=7.0, memory_mb=256, cpu_seconds=11, max_processes=3)
        seen = {}

        def fake(code, entry, args, **kw):
            seen.update(kw)
            raise wi.WslUnavailable("stop here")

        real = wi._run_isolated
        wi._run_isolated = fake
        try:
            with pytest.raises(wi.WslUnavailable):
                wi.runner_for(sb)("x", "main")
        finally:
            wi._run_isolated = real
        assert seen["memory_mb"] == 256
        assert seen["cpu_seconds"] == 11
        assert seen["procs"] == 3
        assert seen["timeout"] == 21.0             # wall clock allows for VM startup
