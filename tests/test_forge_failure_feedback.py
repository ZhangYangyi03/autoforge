"""A failed forge round must say what failed; a bad envelope must not crash.

Two defects from the same transcript (``evidence/run_20260912_153837.jsonl``
and ``..._163001.jsonl``):

  * the sandbox read the child's *last* stdout line and assumed it was an
    object, so a stray bare JSON scalar produced
    ``AttributeError: 'str' object has no attribute 'get'`` instead of a result;
  * a round that finished cleanly (``finish_reason='stop'``) yet failed to
    parse got no retry advice at all -- only the truncated (``'length'``) case
    did -- so the retry repeated the same escaping mistake and all three rounds
    failed the same way.
"""
from __future__ import annotations

import pytest

from autoforge.core.llm import LLMResponse, MockLLMClient
from autoforge.forge.generator import LLMToolGenerator
from autoforge.forge.sandbox import Sandbox

ENVELOPE = b'{"ok": true, "output": "9780306406157", "stdout": ""}'


class _Proc:
    """Stands in for the sandbox child's `Popen`.

    The sandbox no longer uses `subprocess.run`: `run` hides the pid, and the
    pid is what lets a long child be killed the moment the operator speaks. So
    the stand-in carries the surface the polled wait actually touches --
    `communicate` to drain the pipes, `poll` for the exit status.
    """

    def __init__(self, stdout: bytes) -> None:
        self.stdout = stdout
        self.stderr = b""
        self.returncode = 0
        self.pid = -1

    def communicate(self, input=None):        # noqa: A002 - matches Popen
        return self.stdout, self.stderr

    def poll(self):
        return self.returncode


def _sandbox_emitting(monkeypatch, stdout: bytes) -> Sandbox:
    monkeypatch.setattr(
        "autoforge.forge.sandbox.subprocess.Popen", lambda *a, **kw: _Proc(stdout)
    )
    return Sandbox(timeout=5.0)


# -- the stdout envelope -------------------------------------------------

def test_envelope_is_found_when_a_bare_scalar_follows_it(monkeypatch):
    sb = _sandbox_emitting(monkeypatch, ENVELOPE + b'\n"late"\n')
    res = sb.run("def f(): return 1", "f")
    assert res.ok is True
    assert res.output == "9780306406157"


def test_extra_prints_after_the_envelope_do_not_hide_it(monkeypatch):
    sb = _sandbox_emitting(monkeypatch, ENVELOPE + b"\nsome log line\n")
    assert sb.run("def f(): return 1", "f").ok is True


def test_a_bare_scalar_with_no_envelope_reports_clearly(monkeypatch):
    sb = _sandbox_emitting(monkeypatch, b'"9780306406157"\n')
    res = sb.run("def f(): return 1", "f")
    assert res.ok is False
    err = res.error or ""
    assert "no result envelope" in err
    assert "has no attribute" not in err


def test_no_output_at_all_is_still_reported(monkeypatch):
    sb = _sandbox_emitting(monkeypatch, b"")
    res = sb.run("def f(): return 1", "f")
    assert res.ok is False
    assert "no output" in (res.error or "")


# -- the retry advice covers both defects --------------------------------

def _raising_generator(content: str, finish: str) -> LLMToolGenerator:
    llm = MockLLMClient(
        script=[LLMResponse(content=content, raw={"choices": [{"finish_reason": finish}]})]
    )
    return LLMToolGenerator(llm)


def test_complete_but_malformed_reply_gets_escaping_advice():
    with pytest.raises(ValueError) as ei:
        _raising_generator("not json at all", "stop").generate("need")
    msg = str(ei.value)
    assert "complete but not valid JSON" in msg
    assert "escape every inner double quote" in msg


def test_truncated_reply_keeps_the_shorter_denser_advice():
    with pytest.raises(ValueError) as ei:
        _raising_generator('{"name": "x"', "length").generate("need")
    msg = str(ei.value)
    assert "cut off mid-JSON" in msg
    assert "complete but not valid JSON" not in msg
