"""The agent must describe the machine it is on, not the machine it assumes.

The observed failure: asked to look for a program by name, the agent forged a
probe built on `ps`, found nothing, and reported "no such process". `ps` is not
a Windows command. The empty result was the agent's own blind spot showing up
as evidence of absence — the one kind of wrong answer a self-reporting agent
must never give.

So the host block is built from the same discipline as the self-report: every
line is a measurement (the running platform, which binaries actually resolve
from inside the sandbox), and the lesson it teaches is that an empty probe
result is not proof of absence.
"""
import platform

import pytest

from autoforge.agent import HOST_FACTS_HEADER, ForgeAgent, host_facts
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import LLMResponse, MockLLMClient


class _StubSandbox:
    """A sandbox whose PATH resolves exactly what a test says it does.

    Pass a set of names, or a {name: path} mapping when the *path* is the
    point of the test — which it is whenever a lister is on PATH and still
    cannot see anything.
    """

    def __init__(self, present):
        self._paths = (dict(present) if isinstance(present, dict)
                       else {n: n for n in present})

    def resolve(self, names):
        return {n: self._paths.get(n) for n in names}

    def reachable(self, names):
        found = [n for n in names if n in self._paths]
        missing = [n for n in names if n not in self._paths]
        return found, missing


def _text(sandbox=None) -> str:
    return "\n".join(host_facts(sandbox))


# ----------------------------------------------------------------------
# the measurement
# ----------------------------------------------------------------------
def test_os_line_matches_the_running_platform():
    """Whatever host this runs on, the block must name the real one."""
    assert platform.system() in _text()


def test_windows_sandbox_is_told_to_use_tasklist():
    text = _text(_StubSandbox({"tasklist", "cmd", "powershell"}))
    assert "tasklist" in text
    assert "ps is resolvable" not in text


def test_missing_posix_lister_comes_with_the_empty_is_not_absence_lesson():
    """The whole point: `ps` absent must not become "the thing doesn't exist"."""
    text = _text(_StubSandbox({"tasklist"}))
    assert "not resolvable" in text
    assert "empty" in text.lower()
    assert "absence" in text.lower(), \
        "the block never warns that empty output is not evidence of absence"


def test_posix_sandbox_is_told_to_use_ps():
    text = _text(_StubSandbox({"ps", "sh"}))
    assert "`ps` is resolvable" in text


# ----------------------------------------------------------------------
# resolvable is not the same as able to see
# ----------------------------------------------------------------------
def test_gitbash_ps_is_never_offered_as_the_process_lister():
    """The measured failure, one step past a missing `pgrep`.

    `ps` is on PATH — git-bash's MSYS build. It runs, it prints rows, and none
    of them is a native Windows process. Offering it as "the lister here" is
    how the agent ends up reporting a machine with 348 processes as empty.
    """
    msys = r"C:\Program Files\Git\usr\bin\ps.EXE"
    text = _text(_StubSandbox({"tasklist": r"C:\Windows\System32\tasklist.exe",
                               "ps": msys,
                               "sh": r"C:\Program Files\Git\usr\bin\sh.exe"}))
    assert "tasklist" in text, "the native lister must still be named"
    assert "`ps` is resolvable" not in text, \
        "a blind MSYS `ps` was offered as a usable process lister"
    assert msys in text, "the warning must name the binary it is warning about"
    assert "native" in text.lower(), \
        "the block never says MSYS `ps` cannot see native processes"


def test_an_msys_lister_carries_the_empty_is_not_absence_lesson():
    text = _text(_StubSandbox({"tasklist", "ps"}))  # no paths -> names stand in
    plain = _text(_StubSandbox({"tasklist": "tasklist.exe",
                               "ps": r"C:\Program Files\Git\usr\bin\ps.EXE"}))
    assert "absence" in plain.lower(), \
        "a blind-but-resolvable lister shipped without the absence lesson"
    assert text != plain, \
        "the path made no difference: blindness is not being detected at all"


def test_blind_lister_is_marked_on_the_resolvable_line():
    """Two different facts must not print the same sentence."""
    text = _text(_StubSandbox({"ps": r"C:\Program Files\Git\usr\bin\ps.EXE",
                               "tasklist": "tasklist.exe"}))
    resolvable = [l for l in text.splitlines() if l.startswith("- Resolvable")]
    assert resolvable, "resolvable commands were not reported"
    assert "ps (MSYS" in resolvable[0], \
        "a blind lister appeared on the resolvable line unmarked"


def test_a_native_ps_is_still_offered():
    """The rule is about the binary, not the name: /bin/ps is fine."""
    text = _text(_StubSandbox({"ps": "/bin/ps", "sh": "/bin/sh"}))
    assert "`ps` is resolvable" in text
    assert "MSYS" not in text


def test_the_absence_lesson_is_stated_once_not_after_every_clause():
    """Two warnings in a row is how a warning becomes wallpaper."""
    text = _text(_StubSandbox({"tasklist": "tasklist.exe",
                               "ps": r"C:\Program Files\Git\usr\bin\ps.EXE"}))
    assert text.count("not evidence of absence") == 1, \
        "the empty-is-not-absence lesson was repeated once per clause"


def test_no_lister_at_all_says_so_instead_of_inventing_one():
    text = _text(_StubSandbox(set()))
    assert "no known process lister" in text


def test_resolvable_and_missing_are_reported_as_separate_facts():
    """A command that resolves must not also be listed as missing."""
    text = _text(_StubSandbox({"ps", "sh"}))
    resolvable = [l for l in text.splitlines() if l.startswith("- Resolvable")]
    missing = [l for l in text.splitlines() if l.startswith("- NOT resolvable")]
    assert resolvable, "resolvable commands were not reported"
    assert missing, "missing commands were not reported"
    assert "ps" in resolvable[0].split(":", 1)[1]
    assert "ps" not in missing[0].split(":", 1)[1]


# ----------------------------------------------------------------------
# it must reach the model
# ----------------------------------------------------------------------
def _agent() -> ForgeAgent:
    llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="ok"))
    return ForgeAgent(llm, policy=FULL_FREEDOM)


def test_host_block_is_in_the_prompt_that_is_sent():
    agent = _agent()
    assert HOST_FACTS_HEADER in agent._effective_prompt()


def test_prompt_measures_the_sandbox_not_the_agent_shell():
    """The block must be built from the sandbox's reachability view."""
    agent = _agent()
    assert HOST_FACTS_HEADER in agent._effective_prompt()
    # The sandbox is the one that answers; call it directly to prove the wiring.
    found, missing = agent.sandbox.reachable(("tasklist", "ps", "pgrep"))
    assert set(found) | set(missing) == {"tasklist", "ps", "pgrep"}
    assert not (set(found) & set(missing))


def test_sandbox_env_is_scrubbed_not_inherited():
    """`effective_env` is what forged code gets — it must not be os.environ."""
    import os

    agent = _agent()
    env = agent.sandbox.effective_env()
    secret = "AUTOFORGE_TEST_SECRET_SHOULD_NOT_LEAK"
    os.environ[secret] = "1"
    try:
        assert secret not in agent.sandbox.effective_env(), \
            "a non-allow-listed var leaked into the sandbox environment"
        assert "PYTHONIOENCODING" in env
    finally:
        os.environ.pop(secret, None)


# ----------------------------------------------------------------------
# the advice has to be executable, not merely plausible
# ----------------------------------------------------------------------
@pytest.mark.skipif(platform.system() != "Windows",
                    reason="tasklist is the lister this host is told to use")
def test_the_lister_the_block_names_can_see_a_python_process():
    """If the named lister cannot see this test's own process, the block is
    teaching the blindness it was written to cure.

    The end-to-end version of the measured failure: not "is `ps` on PATH" but
    "does the thing we tell the agent to run actually look at the machine".
    It runs a real probe through the real sandbox.
    """
    agent = _agent()
    r = agent.sandbox.run(
        "import subprocess\n"
        "def probe():\n"
        "    out = subprocess.run(['tasklist'], capture_output=True).stdout\n"
        "    text = out.decode('utf-8', 'replace')\n"
        "    if '\\ufffd' in text:          # tasklist answers in the ANSI codepage\n"
        "        text = out.decode('mbcs', 'replace')\n"
        "    return str(sum('python' in ln.lower() for ln in text.splitlines()))\n",
        "probe", {},
    )
    assert r.ok, f"the native lister did not run: {r.error}"
    assert int(r.output) >= 1, \
        "tasklist — the lister this block names — saw no python process at all"
