"""The prompt prefix must not churn, or every turn re-bills the history.

A prefix cache pays only for the characters *before* the first difference. The
system prompt is the first thing in every request, so it is the prefix; the
messages follow it. One changing integer inside it therefore does not cost one
integer — it costs everything behind it, every turn, at the uncached rate.

That is not hypothetical here. The measured failure, before this file existed:
the self-report block carried the ledger's event count, its per-kind breakdown,
and the skill load counter — all of which move on essentially every turn — and a
single `log_event` call moved the block at character 5,125 of a 6,457-character
prompt, invalidating that tail plus the entire conversation after it.

What may still change the prompt, on purpose:

  * a new tool, because the tool list rides in the same request and has changed
    anyway — so the prompt agreeing with it costs nothing extra
  * a new skill on disk, because the skills menu is in the prompt too
  * anything about the machine, because that block is a measurement

Everything else is a bug in this file's terms, and the tests below say so.
"""
from __future__ import annotations

import pytest

from autoforge.agent import ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import MockLLMClient
from autoforge.store import ToolStore
from autoforge.tools.spec import ToolSpec


@pytest.fixture()
def store(tmp_path):
    s = ToolStore(str(tmp_path / "af.db"))
    yield s
    s.close()


def prompts_sent(agent: ForgeAgent) -> list[str]:
    """Every system prompt this agent has put on the wire, in order."""
    return [call[0][0].content for call in agent.llm.calls if call[0]]


# ======================================================================
# stable across the things that happen every turn
# ======================================================================
def test_two_runs_send_the_same_prompt(store):
    """The chat loop calls run() per message. The prefix has to survive it.

    This is the whole point: a conversation is many runs, and each one used to
    recompose a prompt whose ledger counts had moved — so the cache was dead
    from the first turn of every message.
    """
    a = ForgeAgent(MockLLMClient(), store=store, policy=FULL_FREEDOM)
    a.run("first")
    a.run("second")

    sent = prompts_sent(a)
    assert len(sent) >= 2
    assert sent[0] == sent[1]
    assert "Memory: sqlite at" in sent[0]      # not vacuous
    assert "MEASURED SELF-REPORT" in sent[0]


def test_writing_to_the_ledger_does_not_move_the_prompt(store):
    """The ledger is written on every run; the prompt must not notice.

    79.4% of the prompt survived one `log_event` before this was fixed, which
    means the other 20.6% — and the whole history behind it — was re-sent
    uncached on the strength of one integer.
    """
    a = ForgeAgent(MockLLMClient(), store=store, policy=FULL_FREEDOM)
    before = a._effective_prompt()

    store.log_event("run", {"task": "hello"})
    store.log_event("forge", {"name": "thing"})
    after = a._effective_prompt()

    assert before == after


def test_a_repeated_load_does_not_move_the_prompt(store, tmp_path):
    """The menu claims how proven a skill is, not how many times it ran.

    A counter would move the menu — and with it the whole conversation behind
    it — on every single load. A tier moves it once, on the load that changes
    what the menu can honestly claim, which is the first one.
    """
    from autoforge.skills import SkillLibrary

    a = ForgeAgent(MockLLMClient(), store=store, policy=FULL_FREEDOM)
    a.skills = SkillLibrary(store=store, dirs=[("user", str(tmp_path / "skills"))])
    a.skills.write("one", "d", "when", "body")
    a.skills.scan()

    never = a._effective_prompt()
    a.skills.load("one")
    used = a._effective_prompt()
    assert used != never                # never used -> used before: real news
    assert "one (used before)" in used

    a.skills.load("one")
    assert a._effective_prompt() == used    # the second load is not news


# ======================================================================
# and unstable for the things that genuinely changed
# ======================================================================
def test_a_forged_tool_still_moves_the_prompt(store):
    """Stability is not the goal; not lying is.

    A forge writes a row to the ledger, so the ledger's tool count is one of
    the few numbers in the prompt that *should* move: it is the agent saying it
    grew. That happens once per tool, not once per turn, which is the whole
    difference between a cache break that buys something and one that does not.
    """
    a = ForgeAgent(MockLLMClient(), store=store, policy=FULL_FREEDOM)
    before = a._effective_prompt()

    store.save_tool(ToolSpec(name="thing", description="a thing",
                             parameters={}, fn=lambda **_: "ok"))
    after = a._effective_prompt()

    assert after != before
    assert "1 tool(s) on the ledger" in after
