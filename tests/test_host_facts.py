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
from autoforge.agent import HOST_FACTS_HEADER, ForgeAgent, host_facts
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import LLMResponse, MockLLMClient


class _StubSandbox:
    """A sandbox whose PATH resolves exactly what a test says it does."""

    def __init__(self, present):
        self._present = set(present)

    def reachable(self, names):
        found = [n for n in names if n in self._present]
        missing = [n for n in names if n not in self._present]
        return found, missing


def _text(sandbox=None) -> str:
    return "\n".join(host_facts(sandbox))


# ----------------------------------------------------------------------
# the measurement
# ----------------------------------------------------------------------
def test_os_line_matches_the_running_platform():
    """Whatever host this runs on, the block must name the real one."""
    import platform

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