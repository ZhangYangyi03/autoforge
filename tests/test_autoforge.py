"""Test suite for autoforge. Runs offline: no API keys, no network."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.core.agent import Agent
from autoforge.core.llm import LLMResponse, MockLLMClient, tool_call
from autoforge.core.message import Message, ToolCall
from autoforge.forge.generator import GeneratedTool, TemplateGenerator, extract_json
from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.route.router import BehaviourRouter
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec, ToolState, TriggerProbe


# -- fixtures ---------------------------------------------------------
def add_tool(**kw) -> ToolSpec:
    def fn(a: float = 0, b: float = 0) -> str:
        return str(a + b)
    defaults = dict(
        name="add", description="Add two numbers together.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        },
        fn=fn,
    )
    defaults.update(kw)
    return ToolSpec(**defaults)


def sample_generated() -> GeneratedTool:
    code = (
        "def reverse_text(text: str = '') -> str:\n"
        "    if not isinstance(text, str):\n"
        "        raise ValueError('need a string')\n"
        "    return text[::-1]\n"
    )
    return GeneratedTool(
        name="reverse_text",
        description="Reverse a piece of text.",
        code=code, entry="reverse_text",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        probes=[
            TriggerProbe(
                query="Reverse this string for me",
                expect="call",
                negative_query="What is 2 plus 2?",
            )
        ],
        tags=["text"],
    )


def reverse_model(messages, tools, **kw) -> LLMResponse:
    """Calls reverse_text only when the user asks to reverse."""
    last = next((m.content for m in reversed(messages) if m.role == "user"), "")
    names = [t["function"]["name"] for t in (tools or [])]
    if "reverse" in last.lower() and "reverse_text" in names:
        return LLMResponse(tool_calls=[tool_call("reverse_text", {"text": last})])
    return LLMResponse(content="direct answer")


# ======================================================================
# message + llm plumbing
# ======================================================================
class TestMessage:
    def test_tool_call_roundtrip(self):
        tc = ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})
        api = tc.to_api()
        assert api["function"]["name"] == "add"
        back = ToolCall.from_api(api)
        assert back.arguments == {"a": 1, "b": 2}

    def test_tool_call_tolerates_bad_arguments(self):
        back = ToolCall.from_api(
            {"id": "x", "function": {"name": "f", "arguments": "{not json"}}
        )
        assert "_raw" in back.arguments  # degrades, does not raise

    def test_message_shapes(self):
        assert Message.user("hi").to_api()["role"] == "user"
        assert Message.assistant("").to_api()["content"] == ""


class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('text\n```json\n{"a": 1}\n```\nmore') == {"a": 1}

    def test_prose_wrapped(self):
        assert extract_json('Sure! Here you go: {"a": 1} Hope that helps.') == {"a": 1}

    def test_no_json(self):
        assert extract_json("no json here at all") is None

    def test_empty(self):
        assert extract_json("") is None


# ======================================================================
# registry: lifecycle + ledger
# ======================================================================
class TestRegistry:
    def test_register_and_call(self):
        reg = ToolRegistry()
        reg.register(add_tool())
        r = reg.call("add", {"a": 2, "b": 3})
        assert r.ok and r.output == "5"
        assert reg.get("add").stats.calls == 1

    def test_unknown_tool(self):
        r = ToolRegistry().call("nope", {})
        assert not r.ok and "unknown" in r.error

    def test_draft_not_exposed(self):
        reg = ToolRegistry()
        reg.register(add_tool())
        assert reg.schemas() == []

    def test_active_exposed(self):
        reg = ToolRegistry()
        reg.register(add_tool())
        reg.promote("add")
        assert [s["function"]["name"] for s in reg.schemas()] == ["add"]

    def test_failure_recorded(self):
        reg = ToolRegistry()

        def boom(**_) -> str:
            raise ValueError("nope")

        reg.register(add_tool(name="boom", fn=boom))
        r = reg.call("boom", {})
        assert not r.ok and "ValueError" in r.error
        assert reg.get("boom").stats.failures == 1

    def test_auto_quarantine_on_consecutive_failures(self):
        reg = ToolRegistry(min_calls_for_judgement=2, quarantine_consecutive_failures=2)

        def boom(**_) -> str:
            raise RuntimeError("down")

        reg.register(add_tool(name="flaky", fn=boom))
        reg.promote("flaky")
        for _ in range(2):
            reg.call("flaky", {})
        assert reg.get("flaky").state == ToolState.QUARANTINED
        assert [e["kind"] for e in reg.events()].count("auto_quarantine") == 1

    def test_auto_quarantine_on_low_success_rate(self):
        reg = ToolRegistry(min_calls_for_judgement=4, quarantine_success_rate=0.6)

        def flip(ok: bool = True, **_) -> str:
            if not ok:
                raise RuntimeError("x")
            return "ok"

        reg.register(add_tool(name="t", fn=flip))
        reg.promote("t")
        for ok in (True, False, False, False):
            reg.call("t", {"ok": ok})
        assert reg.get("t").state == ToolState.QUARANTINED

    def test_healthy_tool_not_quarantined(self):
        reg = ToolRegistry(min_calls_for_judgement=3, quarantine_success_rate=0.6)
        reg.register(add_tool())
        reg.promote("add")
        for _ in range(5):
            reg.call("add", {"a": 1, "b": 1})
        assert reg.get("add").state == ToolState.ACTIVE

    def test_quarantined_hidden_but_forceable(self):
        reg = ToolRegistry(min_calls_for_judgement=1, quarantine_consecutive_failures=1)

        def boom(**_) -> str:
            raise RuntimeError("x")

        reg.register(add_tool(name="q", fn=boom))
        reg.promote("q")
        reg.call("q", {})
        assert reg.get("q").state == ToolState.QUARANTINED
        assert reg.schemas() == []
        assert not reg.call("q", {}).ok
        assert reg.call("q", {}).quarantined
        # force still runs it — quarantine is a trust signal, not a wall
        reg.unregister("q")
        reg.register(add_tool(name="q2"))
        reg.promote("q2")
        reg.quarantine("q2")
        assert reg.call("q2", {"a": 1, "b": 1}, force=True).ok

    def test_rehab_resets_ledger(self):
        reg = ToolRegistry(min_calls_for_judgement=1, quarantine_consecutive_failures=1)

        def boom(**_) -> str:
            raise RuntimeError("x")

        reg.register(add_tool(name="q", fn=boom))
        reg.promote("q")
        reg.call("q", {})
        assert reg.get("q").state == ToolState.QUARANTINED
        reg.rehab("q")
        assert reg.get("q").state == ToolState.PROBATION
        assert reg.get("q").stats.calls == 0

    def test_report_shape(self):
        reg = ToolRegistry()
        reg.register(add_tool())
        reg.promote("add")
        rep = reg.report()
        assert rep["total"] == 1 and rep["by_state"]["active"] == 1

    def test_replace_guard(self):
        reg = ToolRegistry()
        reg.register(add_tool())
        with pytest.raises(ValueError):
            reg.register(add_tool(), replace=False)


# ======================================================================
# sandbox: real out-of-process execution
# ======================================================================
class TestSandbox:
    def test_runs_clean_code(self):
        sb = Sandbox(timeout=8)
        r = sb.run("def f(x=0):\n    return x*2\n", "f", {"x": 21})
        assert r.ok and r.output == 42

    def test_catches_exception(self):
        sb = Sandbox(timeout=8)
        r = sb.run("def f():\n    raise ValueError('bad input')\n", "f", {})
        assert not r.ok and "ValueError" in r.error

    def test_timeout_is_reaped(self):
        sb = Sandbox(timeout=2)
        r = sb.run("def f():\n    while True:\n        pass\n", "f", {})
        assert not r.ok and r.timed_out

    def test_missing_entry(self):
        sb = Sandbox(timeout=8)
        r = sb.run("x = 1\n", "f", {})
        assert not r.ok and "not found" in r.error

    def test_stdout_does_not_corrupt_protocol(self):
        sb = Sandbox(timeout=8)
        code = "def f():\n    print('noise' * 100)\n    return 'clean'\n"
        r = sb.run(code, "f", {})
        assert r.ok and r.output == "clean"
        assert "noise" in r.stdout

    def test_os_exit_cannot_kill_host(self):
        sb = Sandbox(timeout=8)
        r = sb.run("import os\ndef f():\n    os._exit(3)\n", "f", {})
        assert not r.ok  # died, but the parent survived

    def test_non_ascii_args_survive_the_round_trip(self):
        # Regression: `-I` makes PYTHONIOENCODING inert, so the child used the
        # locale codec (cp936 here) to decode a UTF-8 payload. Any non-ASCII
        # argument mangled into a bogus escape -> "bad payload: Invalid \escape".
        # A path under 项目_开发 is the everyday case that triggers it.
        sb = Sandbox(timeout=8)
        r = sb.run("def f(s):\n    return s\n", "f", {"s": "项目_开发/路径"})
        assert r.ok and r.output == "项目_开发/路径"

    def test_non_ascii_result_survives(self):
        sb = Sandbox(timeout=8)
        r = sb.run("def f():\n    return '中文结果'\n", "f", {})
        assert r.ok and r.output == "中文结果"

    def test_forged_code_reaches_the_host_filesystem(self, tmp_path):
        # The load-bearing claim in AUTONOMOUS_SYSTEM and my_capabilities: the
        # sandbox bounds blast radius, it does not take away file access. A
        # forged tool is shell access with a timeout. If this ever fails, both
        # of those texts are lying.
        f = tmp_path / "outside.txt"
        f.write_text("reached", encoding="utf-8")
        sb = Sandbox(timeout=8)
        r = sb.run(
            "def f(path):\n    with open(path, encoding='utf-8') as fh:\n"
            "        return fh.read()\n",
            "f", {"path": str(f)},
        )
        assert r.ok and r.output == "reached"

    def test_restrict_builtins_removes_file_access(self, tmp_path):
        # Negative control for the claim above: the flag must actually narrow,
        # or "restrict_builtins=True" is decoration.
        f = tmp_path / "outside.txt"
        f.write_text("reached", encoding="utf-8")
        sb = Sandbox(timeout=8, restrict_builtins=True)
        r = sb.run(
            "def f(path):\n    with open(path, encoding='utf-8') as fh:\n"
            "        return fh.read()\n",
            "f", {"path": str(f)},
        )
        assert not r.ok and "open" in (r.error or "")


# ======================================================================
# self-knowledge: the agent must not deny reach it actually has
# ======================================================================
class TestSystemPromptReach:
    """A capability the agent has and denies is worse than one it lacks.

    An agent that reads "forge_tool -- create a new tool" and concludes "I
    have no file tools" leaves the whole machinery unused. The prompt has to
    say what forged code can reach, and point at the runtime check.
    """

    def test_prompt_states_the_filesystem_and_socket_reach(self):
        from autoforge.agent import AUTONOMOUS_SYSTEM
        low = AUTONOMOUS_SYSTEM.lower()
        assert "filesystem" in low
        assert "socket" in low or "network" in low

    def test_prompt_names_the_false_denial_it_forbids(self):
        from autoforge.agent import AUTONOMOUS_SYSTEM
        assert "have no file tools" in AUTONOMOUS_SYSTEM.lower()

    def test_prompt_advertises_my_capabilities_as_the_check(self):
        from autoforge.agent import AUTONOMOUS_SYSTEM
        assert "my_capabilities" in AUTONOMOUS_SYSTEM
        assert "check my_capabilities" in AUTONOMOUS_SYSTEM.lower()


# ======================================================================
# forge pipeline: the three-stage loop
# ======================================================================
class TestForgePipeline:
    def _pipeline(self, gen=None, **cfg):
        llm = MockLLMClient(handler=reverse_model)
        sandbox = Sandbox(timeout=8)
        verifier = ToolVerifier(llm, sandbox=sandbox)
        registry = ToolRegistry()
        pipeline = ForgePipeline(
            gen or TemplateGenerator(recipes={"reverse": sample_generated()}),
            verifier, registry, sandbox=sandbox,
            config=ForgeConfig(promote_on_pass=True, **cfg),
        )
        return pipeline, registry

    def test_forge_success(self):
        pipeline, registry = self._pipeline()
        res = pipeline.forge("I need to reverse text")
        assert res.ok and res.rounds == 1
        assert res.spec.state == ToolState.ACTIVE
        assert registry.get("reverse_text") is not None

    def test_forged_tool_is_actually_callable(self):
        pipeline, registry = self._pipeline()
        pipeline.forge("I need to reverse text")
        r = registry.call("reverse_text", {"text": "abc"})
        assert r.ok and r.output == "cba"

    def test_verification_record_attached(self):
        pipeline, _ = self._pipeline()
        res = pipeline.forge("I need to reverse text")
        checks = res.spec.verification["checks"]
        assert any(c["name"] == "execution" for c in checks)
        assert any(c["name"] == "trigger" for c in checks)
        assert any(c["name"] == "negative" for c in checks)

    def test_broken_tool_never_goes_active(self):
        broken = sample_generated()
        broken.code = "def reverse_text(text=''):\n    raise RuntimeError('boom')\n"
        pipeline, registry = self._pipeline(
            gen=TemplateGenerator(recipes={"reverse": broken}), max_rounds=2
        )
        res = pipeline.forge("I need to reverse text")
        assert not res.ok
        assert registry.get("reverse_text") is None  # never registered

    def test_retry_feeds_failure_back(self):
        """A generator that fails once then succeeds should be retried."""
        calls = {"n": 0}
        good = sample_generated()
        broken = sample_generated()
        broken.code = "def reverse_text(text=''):\n    raise RuntimeError('boom')\n"

        class FlakyGen:
            def generate(self, need, context=""):
                calls["n"] += 1
                return broken if calls["n"] == 1 else good

        llm = MockLLMClient(handler=reverse_model)
        sandbox = Sandbox(timeout=8)
        pipeline = ForgePipeline(
            FlakyGen(), ToolVerifier(llm, sandbox=sandbox), ToolRegistry(),
            sandbox=sandbox, config=ForgeConfig(promote_on_pass=True, max_rounds=3),
        )
        res = pipeline.forge("I need to reverse text")
        assert res.ok and res.rounds == 2

    def test_retry_feedback_is_not_duplicated(self):
        """The repair prompt must carry the failure once, not twice.

        Round 2 used to get the feedback twice: once spliced into the need by
        _repair_prompt, once prefixed by forge() itself. On a token-starved
        local model that doubling is not free -- it is roughly half the budget
        spent saying the same thing twice.
        """
        prompts: list[str] = []
        good = sample_generated()
        broken = sample_generated()
        broken.code = "def reverse_text(text=''):\n    raise RuntimeError('boom')\n"

        class RecordingGen:
            def generate(self, need, context=""):
                prompts.append(need)
                return broken if len(prompts) == 1 else good

        llm = MockLLMClient(handler=reverse_model)
        sandbox = Sandbox(timeout=8)
        pipeline = ForgePipeline(
            RecordingGen(), ToolVerifier(llm, sandbox=sandbox), ToolRegistry(),
            sandbox=sandbox, config=ForgeConfig(promote_on_pass=True, max_rounds=3),
        )
        res = pipeline.forge("I need to reverse text")
        assert res.ok and res.rounds == 2

        assert len(prompts) == 2
        retry = prompts[1]
        assert retry.count("Failed checks:") == 1, "failure detail sent twice"
        assert "Previous attempt failed" not in retry, "two competing banners"
        assert "Fix the root cause" in retry, "the model is told what to do"
        assert prompts[0].count("Failed checks:") == 0, "round 1 gets no feedback"

    def test_exception_is_logged_with_a_traceback(self):
        """A swallowed traceback is a debugging dead end; keep it in the log.

        The generic except used to reduce every internal failure to
        "TypeName: message", which is unactionable when the fault is in the
        pipeline rather than in the generated tool.
        """
        class BoomGen:
            def generate(self, need, context=""):
                raise RuntimeError("kaboom")

        llm = MockLLMClient(handler=reverse_model)
        sandbox = Sandbox(timeout=8)
        pipeline = ForgePipeline(
            BoomGen(), ToolVerifier(llm, sandbox=sandbox), ToolRegistry(),
            sandbox=sandbox, config=ForgeConfig(promote_on_pass=True, max_rounds=1),
        )
        res = pipeline.forge("I need to reverse text")

        assert not res.ok
        assert res.attempts[0].error == "RuntimeError: kaboom"
        errs = [e for e in pipeline.log if e["kind"] == "forge_error"]
        assert len(errs) == 1
        tb = errs[0]["traceback"]
        assert "Traceback (most recent call last)" in tb
        assert "kaboom" in tb

    def test_event_log_records_attempts(self):
        pipeline, _ = self._pipeline()
        pipeline.forge("I need to reverse text")
        kinds = [e["kind"] for e in pipeline.log]
        assert "forge_attempt" in kinds and "forge_done" in kinds


# ======================================================================
# verifier: does it run, does it survive, does it fire, does it stay quiet
# ======================================================================
class TestVerifier:
    def test_trigger_detects_a_never_firing_tool(self):
        """The Constraint Tax case: correct tool, agent never calls it."""
        llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="I'll answer directly."))
        v = ToolVerifier(llm, sandbox=Sandbox(timeout=8))
        spec = ToolSpec(
            name="ghost", description="Does nothing useful.",
            parameters={"type": "object", "properties": {}},
            fn=lambda **_: "", source="generated",
            probes=[TriggerProbe(query="please use the ghost tool")],
        )
        report = v.verify(spec)
        assert not report.passed
        assert any(c.name == "trigger" and not c.passed for c in report.checks)

    def test_negative_check_catches_over_firing(self):
        """A tool that fires on everything is worse than one that never fires."""
        always = MockLLMClient(handler=lambda m, t, **k: LLMResponse(
            tool_calls=[tool_call("greedy")]
        ))
        v = ToolVerifier(always, sandbox=Sandbox(timeout=8), run_execution_check=False,
                         run_trigger_check=False)
        spec = ToolSpec(
            name="greedy", description="Fires always.",
            parameters={"type": "object", "properties": {}}, fn=lambda **_: "",
            probes=[TriggerProbe(query="x", negative_query="unrelated question")],
        )
        report = v.verify(spec)
        assert not report.passed
        assert any(c.name == "negative" and not c.passed for c in report.checks)

    def test_execution_check_catches_bad_code(self):
        v = ToolVerifier(MockLLMClient(), sandbox=Sandbox(timeout=8),
                         run_trigger_check=False, run_negative_check=False)
        spec = ToolSpec(
            name="broken", description="x",
            parameters={"type": "object", "properties": {}},
            code="def broken():\n    raise KeyError('nope')\n", fn=lambda **_: "",
        )
        report = v.verify(spec)
        assert not report.passed
        assert "KeyError" in report.checks[0].detail

    def test_bare_type_name_in_parameters_no_longer_crashes_the_battery(self):
        """The live defect, end to end.

        qwen2.5:7b emitted {"properties": {"isbn": "string"}} -- a bare type
        name where a schema belongs. With no explicit sample_args the verifier
        fell into _infer_args, which did schema.get("type") on a str and raised
        AttributeError three frames away from the cause, costing a whole round.
        """
        v = ToolVerifier(MockLLMClient(), sandbox=Sandbox(timeout=8),
                         run_trigger_check=False, run_negative_check=False)
        spec = ToolSpec(
            name="normalize_isbn", description="x",
            parameters={"type": "object", "properties": {"isbn": "string"}},
            code="def normalize_isbn(isbn=''):\n    return isbn\n", fn=lambda **_: "",
        )
        report = v.verify(spec)          # sample_args=None -> _infer_args runs

        assert any(c.name == "execution" and c.passed for c in report.checks)
        assert spec.parameters["properties"]["isbn"] == {"type": "string"}


# ======================================================================
# router: behaviour over text
# ======================================================================
class TestRouter:
    def _pair(self):
        reg = ToolRegistry(min_calls_for_judgement=2)

        def good(**_) -> str:
            return "ok"

        def bad(**_) -> str:
            raise RuntimeError("broken")

        reg.register(add_tool(name="parse_json_safe",
                              description="Safely parse a JSON document",
                              fn=bad))
        reg.register(add_tool(name="parse_json_strict",
                              description="Parse JSON text strictly",
                              fn=good))
        reg.promote("parse_json_safe")
        reg.promote("parse_json_strict")
        for _ in range(3):
            reg.call("parse_json_safe", {})
            reg.call("parse_json_strict", {})
        return reg

    def test_behaviour_beats_text_similarity(self):
        reg = self._pair()
        router = BehaviourRouter(
            reg, include_states={ToolState.ACTIVE, ToolState.QUARANTINED}
        )
        ranked = router.rank("parse this JSON document safely")
        assert ranked[0].name == "parse_json_strict"

    def test_quarantined_scored_low(self):
        reg = self._pair()
        router = BehaviourRouter(
            reg, include_states={ToolState.ACTIVE, ToolState.QUARANTINED}
        )
        by_name = {c.name: c for c in router.rank("parse json")}
        assert by_name["parse_json_safe"].score < by_name["parse_json_strict"].score

    def test_production_router_hides_quarantined(self):
        reg = self._pair()
        picks = BehaviourRouter(reg).route("parse json", k=5)
        assert "parse_json_safe" not in picks

    def test_trust_ordering(self):
        reg = ToolRegistry()
        reg.register(add_tool(name="draft_t"))
        reg.register(add_tool(name="active_t"))
        reg.promote("active_t")
        router = BehaviourRouter(
            reg, include_states={ToolState.DRAFT, ToolState.ACTIVE}
        )
        ranked = {c.name: c for c in router.rank("add two numbers")}
        assert ranked["active_t"].score > ranked["draft_t"].score


# ======================================================================
# end-to-end: the agent loop drives the forge
# ======================================================================
class TestEndToEnd:
    def test_agent_uses_an_existing_tool(self):
        reg = ToolRegistry()
        reg.register(add_tool())
        reg.promote("add")
        llm = MockLLMClient(script=[
            LLMResponse(tool_calls=[tool_call("add", {"a": 2, "b": 40})]),
            LLMResponse(content="The answer is 42."),
        ])
        res = Agent(llm, reg).run("What is 2 + 40?")
        assert res.content == "The answer is 42."
        assert res.used_tools and res.tool_calls == ["add"]

    def test_agent_answers_without_tools(self):
        reg = ToolRegistry()
        llm = MockLLMClient(script=[LLMResponse(content="Peru.")])
        res = Agent(llm, reg).run("Capital of Peru?")
        assert not res.used_tools


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-o", "addopts=", "-p", "no:cacheprovider"]))
