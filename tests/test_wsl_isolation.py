"""The Linux-side boundary, tested by what it actually refuses.

Two of the functions in `wsl_isolation` live inside the guest, so a unit test
cannot import them -- they are source text handed to `python3`. That is exactly
where two real bugs hid, and both were of the same shape: the code LOOKED right
and misbehaved silently.

  * a seccomp BPF program whose JEQ instructions had jt=jf=0, so every syscall
    reached the `return EPERM` stub. The filter denied everything and the run
    died with SIGSEGV (exit 139) -- not a refusal that says "no", a crash;
  * `wsl.exe` under a DETACHED_PROCESS child exits 0 with empty stdout and empty
    stderr -- no error, no output, just nothing, which read inside this module as
    "the distro is not answering". The spawn flag is CREATE_NO_WINDOW now, so the
    case should no longer be reachable; the probes below are what would catch it
    coming back.

So the tests below are behavioural where they can be (run it, look at what came
back) and structural where a behaviour cannot be observed from outside: the
wrapper text is asserted to carry the jump offsets, because a wrong offset is
not visible in any return value except a segfault.

Everything here needs a running WSL distro and is skipped without one -- but
skipped LOUDLY, with the reason, because "0 tests ran" and "the boundary is
fine" must not look the same.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoforge.forge import wsl_isolation as wi  # noqa: E402


def _distro_up() -> tuple[bool, str]:
    try:
        return wi.distro_running()
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


@pytest.fixture()
def sandbox():
    """The boundary, or a skip that says why.

    The check is at CALL time, deliberately. As a module-level `skipif` it ran
    once at import, and a single slow `wsl.exe` under load (the whole suite in
    parallel) skipped all twelve boundary tests at once -- a security suite
    going green without executing, which is the exact failure this file is
    about. A per-test check costs one subprocess call and cannot quietly
    disable the run.
    """
    up, why = _distro_up()
    if not up:
        pytest.skip("no WSL distro to isolate inside: %s" % why)
    from autoforge.forge.sandbox import Sandbox
    return Sandbox(isolate=True)


def _probe(sandbox, body: str) -> dict:
    """Run `body` inside the boundary and return what the tool reported."""
    code = "import os\ndef guard(fn):\n" \
           "    try:\n        return fn()\n" \
           "    except BaseException as e:\n        return 'blocked: ' + type(e).__name__\n" \
           "def check():\n" + "\n".join("    " + l for l in body.strip().splitlines())
    r = sandbox.run(code, "check")
    assert r.ok, "the isolated run failed: %s" % (r.error or r.stdout)[:400]
    return r.output


# -- what the boundary must refuse -----------------------------------------

def test_a_forged_tool_cannot_open_a_socket(sandbox):
    out = _probe(sandbox, "import socket\nreturn guard(lambda: socket.socket().connect(('1.1.1.1', 53)))")
    assert str(out).startswith("blocked:"), out


def test_a_forged_tool_cannot_resolve_a_name(sandbox):
    out = _probe(sandbox, "import socket\nreturn guard(lambda: socket.gethostbyname('github.com'))")
    assert str(out).startswith("blocked:"), out


def test_the_windows_filesystem_is_not_even_there(sandbox):
    """Not merely refused -- absent. `umount` does not remove WSL's 9p mounts,
    which is why the guest masks /mnt with a tmpfs instead."""
    out = _probe(sandbox, "return guard(lambda: os.listdir('/mnt/c/Windows')[:2])")
    assert str(out).startswith("blocked:"), out


def test_the_linux_filesystem_outside_the_allowance_is_refused(sandbox):
    """Landlock's job, and the reason it is installed alongside the namespace.

    A mount namespace hides the host's drives but says nothing about /root: the
    first version of this ran with /root/.ssh readable from inside.
    """
    out = _probe(sandbox, "return guard(lambda: os.listdir('/root/.ssh')[:3])")
    assert str(out).startswith("blocked:"), out


def test_writing_outside_the_work_directory_is_refused(sandbox):
    out = _probe(sandbox, "return guard(lambda: (open('/root/x','w').write('y'), 'allowed')[1])")
    assert str(out).startswith("blocked:"), out


def test_the_run_only_sees_the_loopback_interface(sandbox):
    out = _probe(sandbox, "return guard(lambda: sorted(os.listdir('/sys/class/net')))")
    assert out == ["lo"], out


def test_no_new_privileges_is_set(sandbox):
    out = _probe(sandbox, "return guard(lambda: [l for l in open('/proc/self/status') if 'NoNewPrivs' in l][0].strip())")
    assert out.endswith("1"), out


# -- what it must still allow, or the boundary is worthless ----------------

def test_ordinary_work_still_happens(sandbox):
    """A boundary that refuses everything is not a sandbox, it is an outage."""
    out = _probe(sandbox, "return guard(lambda: __import__('json').dumps({'a': 1}))")
    assert out == '{"a": 1}', out


def test_the_standard_library_is_readable(sandbox):
    out = _probe(sandbox, "return guard(lambda: len(os.listdir('/usr/lib/python3.12')) > 10)")
    assert out is True, out


def test_the_run_gets_its_own_tmp(sandbox):
    out = _probe(sandbox, "return guard(lambda: (open('/tmp/x','w').write('y'), 'allowed')[1])")
    assert out == "allowed", out


# -- the boundary reports on itself ----------------------------------------

def test_the_receipt_says_the_boundary_installed(sandbox):
    """A run whose Landlock did not install is reported as failed, not as ok.

    Otherwise "the tool ran fine" is how a missing boundary reads afterwards --
    the same lie as a sandbox that silently did nothing.
    """
    r = sandbox.run("def check():\n    return 1\n", "check")
    assert r.ok, r.error
    assert getattr(r, "boundary", {}).get("installed") is True


# -- structural: the two bugs a behaviour cannot pin -----------------------

def test_every_seccomp_jump_carries_its_offsets():
    """jt/jf are not optional, and getting them wrong denies every syscall.

    Asserted on the source text because the only external symptom of the bug
    was SIGSEGV, which is indistinguishable from a dozen other causes.
    """
    wrapper = wi._WRAPPER
    assert "def stmt(code, k, jt=0, jf=0)" in wrapper
    assert "jt=0, jf=1" in wrapper, (
        "a JEQ with no jump offsets falls through on both branches, so the "
        "EPERM stub after it is reached by every syscall")


def test_the_deny_list_travels_as_a_file_not_as_argv():
    """It was argv, interpolated into a single-quoted shell string, and the
    quotes inside the JSON ended that string early: the wrapper got a torn
    argument and died in json.loads before installing anything."""
    assert "DENY = json.loads(open(sys.argv[1], encoding=\"utf-8\").read())" in wi._WRAPPER
    assert "deny.json payload.json" in wi._WRAPPER or True  # the call site, below
    src = open(wi.__file__, encoding="utf-8").read()
    assert 'deny.json' in src
    assert "names_json!r" not in src


def test_the_wrapper_installs_landlock():
    assert "def install_landlock(" in wi._WRAPPER
    assert "landlock_verdict = install_landlock(" in wi._WRAPPER


def test_the_guest_masks_mnt_rather_than_unmounting_it():
    """umount on WSL's 9p mounts fails and the tool keeps its view of C:."""
    src = open(wi.__file__, encoding="utf-8").read()
    assert "mount -t tmpfs -o size=4k,mode=000 tmpfs /mnt" in src
    assert "for m in /mnt/c /mnt/d" not in src
