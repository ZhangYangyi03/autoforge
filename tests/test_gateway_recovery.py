"""A gateway outage must not strand a run; bad credentials must not spin."""
import pytest

from autoforge.core.agent import Agent
from autoforge.core.llm import LLMError, LLMTransientError, LLMResponse, MockLLMClient
from autoforge.tools.registry import ToolRegistry


def test_gateway_outage_recovers_without_losing_task(monkeypatch):
    monkeypatch.setenv("AUTOFORGE_GATEWAY_WAIT_S", "2")
    monkeypatch.setenv("AUTOFORGE_GATEWAY_RETRY_S", "0.01")
    calls = []
    def handler(messages, tools, **kwargs):
        calls.append(messages[-1].content)
        if len(calls) < 3:
            raise LLMTransientError("503")
        return LLMResponse(content="recovered")
    agent = Agent(MockLLMClient(handler=handler), ToolRegistry(), allow_self_terminate=False)
    assert agent.run("do work").content == "recovered"
    assert calls == ["do work"] * 3


def test_permanent_failure_is_not_retried(monkeypatch):
    calls = []
    def handler(messages, tools, **kwargs):
        calls.append(1)
        raise LLMError("401")
    agent = Agent(MockLLMClient(handler=handler), ToolRegistry(), allow_self_terminate=False)
    with pytest.raises(LLMError):
        agent.run("do work")
    assert len(calls) == 1


def test_outage_budget_expires(monkeypatch):
    monkeypatch.setenv("AUTOFORGE_GATEWAY_WAIT_S", "0")
    def handler(messages, tools, **kwargs):
        raise LLMTransientError("503")
    agent = Agent(MockLLMClient(handler=handler), ToolRegistry(), allow_self_terminate=False)
    with pytest.raises(LLMTransientError):
        agent.run("do work")
