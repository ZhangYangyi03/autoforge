"""Reach must be measured, and the agent must not deny what it has.

The failure this guards against is specific and was observed: asked whether it
could touch the machine, the agent reasoned from its tool list ("I have no
read_file or run_shell"), concluded it was walled off, and then produced a
fluent architectural story — "no shared filesystem, no shared process space, no
network channel" — to explain a limitation it did not have. Forged code is a
subprocess of the agent process: it has the whole host filesystem and outbound
network. Every clause of that story was false.

Prose in a prompt did not fix it, because prose competes with the model's prior
that an LLM has no hands. So these tests hold two things in place:

  * `reach()` reports the sandbox truthfully, and `probe=True` measures it with
    a real round-trip rather than asserting it
  * the claim is carried in the tool *descriptions* — which ride in every
    request — not only in a paragraph the model may skim past

A scan is not proof, but a description that goes missing is a regression, and
these tests make that regression loud.
"""
from __future__ import annotations

import os
import tempfile

from autoforge.agent import ForgeAgent
from autoforge.core.llm import MockLLMClient
from autoforge.forge.sandbox import Sandbox


def _agent() -> ForgeAgent:
    return ForgeAgent(MockLLMClient())


# ======================================================================
# reach() tells the truth about the sandbox
# ======================================================================
class TestReachIsMeasuredNotAsserted:
    def test_reach_says_host_and_denies_isolation(self):
        facts = Sandbox().reach()
        assert facts["isolated_from_host"] is False
        assert "subprocess" in facts["host"]
        assert "read-write" in facts["filesystem"]
        assert "outbound" in facts["network"]

    def test_bounds_are_reported_but_not_as_capability_limits(self):
        facts = Sandbox().reach()
        # These three bound blast radius. They must not read as "cannot reach".
        assert "temp dir" in facts["cwd"]
        assert "allow-listed" in facts["env"]
        assert isinstance(facts["timeout_s"], (int, float))
        assert facts["restrict_builtins"] is False

    def test_probe_measures_a_real_host_round_trip(self):
        facts = Sandbox().reach(probe=True)
        pr = facts["probe"]
        assert "error" not in pr, pr
        fs = pr["host_filesystem"]
        assert fs["ok"] is True, fs
        # Written to the host temp dir, i.e. deliberately outside the per-call cwd
        assert "Temp" in fs["detail"] or "tmp" in fs["detail"].lower()
        # A truthful network field: a bool plus a detail, whatever the network does
        net = pr["network"]
        assert isinstance(net["ok"], bool)
        assert net["detail"]

    def test_the_probe_writes_outside_the_sandbox_cwd(self):
        # Writing inside the fresh cwd would prove nothing about escaping it.
        # Independently confirm: a file in the host temp dir is visible to the
        # sandbox even though that is not the sandbox's cwd.
        marker = os.path.join(tempfile.gettempdir(), ".autoforge_reach_independent")
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("outside-cwd")
        try:
            r = Sandbox().run(
                "def probe(p):\n"
                "    import os\n"
                "    return {'cwd': os.getcwd(), 'read': open(p, encoding='utf-8').read()}\n",
                "probe", {"p": marker},
            )
            assert r.ok, r.error
            assert r.output["read"] == "outside-cwd"
            # The sandbox cwd lives *inside* the temp dir, so the marker at the
            # temp dir's root is genuinely outside it — that is what makes the
            # read a reach claim rather than a cwd claim.
            assert tempfile.gettempdir() in r.output["cwd"]
        finally:
            if os.path.exists(marker):
                os.remove(marker)

    def test_the_sandbox_can_write_to_the_host_outside_the_project(self):
        target = os.path.join(tempfile.gettempdir(), ".autoforge_reach_write")
        try:
            r = Sandbox().run(
                "def probe(p):\n"
                "    open(p, 'w', encoding='utf-8').write('forged')\n"
                "    return True\n",
                "probe", {"p": target},
            )
            assert r.ok, r.error
            assert os.path.exists(target)                     # the host sees it
            with open(target, encoding="utf-8") as fh:
                assert fh.read() == "forged"
        finally:
            if os.path.exists(target):
                os.remove(target)


# ======================================================================
# the agent's report is the measurement, not a paragraph
# ======================================================================
class TestCapabilitiesReport:
    def test_report_describes_measured_reach(self):
        out = _agent().registry.call("my_capabilities", {}).output
        assert "read-write" in out
        assert "subprocess" in out
        assert "network" in out
        # The exact denial that started this must not be reachable from the tool
        assert "I have no file tools" in out and "false" in out

    def test_report_can_prove_it_with_a_live_round_trip(self):
        out = _agent().registry.call("my_capabilities", {"probe": True}).output
        assert "Live self-test" in out
        assert "host filesystem: OK" in out

    def test_the_claim_agrees_with_what_the_sandbox_does(self):
        # The point of measuring: if reach() claims read-write, a forged tool
        # must actually be able to write. Cross-check the claim against reality.
        a = _agent()
        out = a.registry.call("my_capabilities", {}).output
        assert "read-write" in out
        target = os.path.join(tempfile.gettempdir(), ".autoforge_claim_check")
        try:
            r = a.sandbox.run(
                "def probe(p):\n"
                "    open(p, 'w', encoding='utf-8').write('ok')\n"
                "    return True\n",
                "probe", {"p": target},
            )
            assert r.ok and os.path.exists(target), "the report overclaimed"
        finally:
            if os.path.exists(target):
                os.remove(target)


# ======================================================================
# the claim rides in every request, via the tool schemas
# ======================================================================
class TestSchemasCarryTheClaim:
    def test_forge_tool_description_states_host_reach(self):
        a = _agent()
        desc = a.registry.schemas()
        forge = next(s for s in desc
                     if s["function"]["name"] == "forge_tool")["function"]["description"]
        assert "THIS host" in forge
        assert "filesystem" in forge
        assert "run_shell" in forge          # names the tool it is often mistaken for

    def test_capabilities_description_offers_the_probe(self):
        a = _agent()
        cap = next(s for s in a.registry.schemas()
                   if s["function"]["name"] == "my_capabilities")["function"]
        assert "probe" in cap["parameters"]["properties"]
        assert "THIS host" in cap["description"]

    def test_system_prompt_still_states_reach(self):
        # Belt and braces: the prompt paragraph must survive alongside the
        # schemas, because a forged-tool denial can start from either.
        sp = _agent().system_prompt
        assert "real filesystem" in sp
        assert "forge it" in sp
