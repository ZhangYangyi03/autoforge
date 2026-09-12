"""End-to-end self-forging run that emits *checkable* evidence.

`demo_live.py` prints what happened, which means you have to trust the
scrollback. This emits a hash-chained record and then attacks its own claim:

  1. REPLAY    re-execute the sealed tool from the record alone and compare
               against the recorded outputs. A record that cannot be replayed
               is a log, not evidence.
  2. TAMPER    flip one byte in the record and show the chain breaks.
  3. BOUNDARY  push a tool that "passes" by doing nothing through the same
               robustness oracle and show it rejected. This demonstrates the
               claim's scope instead of asserting it.

Run against the gateway (default):
    python examples/demo_evidence.py
Run against a local model (reproducible, no key, no network):
    AUTOFORGE_BASE_URL=http://127.0.0.1:11434/v1 AUTOFORGE_MODEL=qwen2.5:7b \
        python examples/demo_evidence.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.core.llm import OpenAICompatClient
from autoforge.forge.generator import LLMToolGenerator
from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec

BAR = "=" * 66
PROXIES = {"http": "socks5://127.0.0.1:9674", "https": "socks5://127.0.0.1:9674"}


def rule(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def canon(obj: object) -> str:
    """Stable serialisation — the chain is only meaningful if this is total."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


class EvidenceChain:
    """Append-only record where each entry commits to the one before it.

    Reading the file in order, any edit to entry N invalidates N and every
    entry after it, because `hash` is taken over (prev_hash + body).
    """

    GENESIS = "0" * 64

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict] = []
        self._fh = path.open("w", encoding="utf-8")

    @property
    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else self.GENESIS

    def record(self, kind: str, payload: dict) -> dict:
        seq = len(self.entries)
        prev = self.head
        body = {"seq": seq, "t": round(time.time(), 3), "kind": kind, "payload": payload}
        digest = hashlib.sha256((prev + canon(body)).encode("utf-8")).hexdigest()
        entry = {**body, "prev": prev, "hash": digest}
        self.entries.append(entry)
        self._fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()
        return entry

    def close(self) -> None:
        self._fh.close()


def verify_chain(path: Path) -> tuple[bool, int, int | None]:
    """Recompute the chain from scratch. Returns (ok, entries, first_bad_seq)."""
    prev = EvidenceChain.GENESIS
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        body = {k: entry[k] for k in ("seq", "t", "kind", "payload")}
        expect = hashlib.sha256((prev + canon(body)).encode("utf-8")).hexdigest()
        if entry.get("prev") != prev or entry.get("hash") != expect:
            return False, n, entry.get("seq")
        prev = entry["hash"]
        n += 1
    return True, n, None


def cheat_spec(name: str, params: dict) -> ToolSpec:
    """A tool that satisfies 'does not crash' and nothing else."""
    code = f"def {name}(**kwargs):\n    return 'nope'\n"
    return ToolSpec(
        name=name,
        description="validator that always answers the same thing",
        parameters=params,
        fn=lambda **_kw: "nope",
        code=code,
        source="generated",
    )


def main() -> int:
    base = os.environ.get("AUTOFORGE_BASE_URL", "https://aiping.cn/api/v1")
    model = os.environ.get("AUTOFORGE_MODEL", "DeepSeek-V4.1-Flash")
    local = any(h in base for h in ("127.0.0.1", "localhost"))
    key = os.environ.get("AUTOFORGE_API_KEY") or os.environ.get("AIPING_API_KEY") or ""
    if not key:
        if not local:
            print("AIPING_API_KEY not set (or set AUTOFORGE_BASE_URL/AUTOFORGE_MODEL)")
            return 1
        key = "ollama"
    use_proxy = os.environ.get("AUTOFORGE_PROXY", "0" if local else "1") == "1"

    out_dir = Path("evidence")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    chain_path = out_dir / f"run_{stamp}.jsonl"
    chain = EvidenceChain(chain_path)

    llm = OpenAICompatClient(
        model=model, base_url=base, api_key=key, timeout=600,
        proxies=PROXIES if use_proxy else None,
    )
    print(f"model   : {model}")
    print(f"base_url: {base}")
    print(f"proxy   : {use_proxy}")
    print(f"evidence: {chain_path}")

    registry = ToolRegistry()
    sandbox = Sandbox(timeout=30.0)
    # On CPU a 7B model needs minutes per call, and the adversarial gate alone
    # fires ~5 of them. AUTOFORGE_FAST drops the LLM-driven checks and keeps the
    # deterministic ones -- disclosed below and in the record, never silently.
    fast = os.environ.get("AUTOFORGE_FAST", "") == "1"
    max_tokens = int(os.environ.get("AUTOFORGE_MAX_TOKENS", "3000"))
    max_rounds = int(os.environ.get("AUTOFORGE_MAX_ROUNDS", "2"))
    verifier = ToolVerifier(
        llm, sandbox=sandbox,
        run_adversarial_check=not fast,
        run_trigger_check=not fast,
        run_negative_check=not fast,
    )
    pipeline = ForgePipeline(
        LLMToolGenerator(llm, max_tokens=max_tokens), verifier, registry,
        sandbox=sandbox,
        config=ForgeConfig(promote_on_pass=True, max_rounds=max_rounds),
    )
    print(f"profile : {'FAST (no LLM-driven checks)' if fast else 'full'}"
          f"  max_tokens={max_tokens}  max_rounds={max_rounds}")

    # -- 1. forge -------------------------------------------------------
    rule("STAGE 1 - a real model forges a real tool")
    need = (
        "I keep needing to validate and normalise ISBN-10 and ISBN-13 book "
        "identifiers: strip hyphens, check the check digit, and convert a "
        "valid ISBN-10 to its ISBN-13 form. I need this repeatedly."
    )
    chain.record("need", {"text": need})
    chain.record("profile", {
        "fast": fast, "max_tokens": max_tokens, "max_rounds": max_rounds,
        "llm_checks": "skipped" if fast else "run",
    })
    print(f"need: {need}\n")

    t0 = time.time()
    result = pipeline.forge(need)
    elapsed = round(time.time() - t0, 1)
    chain.record("forge_result", {
        "ok": result.ok, "rounds": result.rounds, "seconds": elapsed,
        "errors": [a.error for a in result.attempts],
    })
    print(f"\nforge ok : {result.ok}   rounds: {result.rounds}   {elapsed}s")

    if not result.ok:
        for a in result.attempts:
            print(f"  round {a.round}: {a.error}")
            if a.report:
                for c in a.report.checks:
                    print(f"    [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        chain.close()
        print(f"\nevidence written to {chain_path} (forge failed)")
        return 2

    spec = result.spec
    print(f"tool     : {spec.name}  [{spec.state.value}]")
    print(f"effect   : {spec.effect_signature}")
    print(f"spec hash: {spec.hash[:16]}")

    chain.record("sealed", {
        "name": spec.name, "entry": spec.name, "code": spec.code,
        "parameters": spec.parameters, "spec_hash": spec.hash,
        "effect_signature": spec.effect_signature,
        "invariances": list(spec.invariances),
    })
    for c in spec.verification.get("checks", []):
        chain.record("check", {"name": c["name"], "passed": c["passed"],
                               "detail": str(c["detail"])[:400]})
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']:<11} {str(c['detail'])[:70]}")

    # -- 2. call it for real --------------------------------------------
    rule("STAGE 2 - call the sealed tool for real, out-of-process")
    props = list((spec.parameters.get("properties") or {}).keys())
    arg = props[0] if props else "text"
    samples = ["ISBN 0-306-40615-2", "978-0-306-40615-7", "0306406152", "not-an-isbn"]
    recorded: list[dict] = []
    for s in samples:
        r = registry.call(spec.name, {arg: s})
        rec = {"args": {arg: s}, "ok": r.ok,
               "output": r.output if r.ok else None, "error": r.error}
        recorded.append(rec)
        chain.record("call", rec)
        print(f"  {s:<20} -> ok={r.ok}  {str(r.output)[:52]!r}")

    # -- 3. verify the chain itself -------------------------------------
    rule("STAGE 3 - verify the evidence chain")
    ok, n, bad = verify_chain(chain_path)
    chain.record("chain_verified", {"entries": n, "ok": ok})
    print(f"  entries: {n}   chain intact: {ok}")
    head = chain.head
    print(f"  head   : {head[:32]}...")

    # -- 4. replay from the record alone --------------------------------
    rule("STAGE 4 - REPLAY: re-execute the sealed tool from the record")
    blob = chain.entries[[e["kind"] for e in chain.entries].index("sealed")]
    sealed_code = blob["payload"]["code"]
    replay_agree = 0
    for rec in recorded:
        fresh = sandbox.run(sealed_code, blob["payload"]["entry"], rec["args"])
        same = fresh.ok == rec["ok"] and canon(fresh.output) == canon(rec["output"])
        replay_agree += same
        mark = "AGREE" if same else "DISAGREE"
        print(f"  {rec['args'][arg]:<20} recorded={str(rec['output'])[:26]!r:<28} "
              f"replayed={str(fresh.output)[:26]!r:<28} {mark}")
    chain.record("replay", {"agree": replay_agree, "total": len(recorded)})
    print(f"\n  replayed {replay_agree}/{len(recorded)} identically")

    # -- 5. tamper ------------------------------------------------------
    rule("STAGE 5 - TAMPER: edit one byte, expect the chain to break")
    lines = chain_path.read_text(encoding="utf-8").splitlines()
    victim = 2
    original = lines[victim]
    obj = json.loads(original)
    obj["payload"]["detail"] = "tampered: this check passed"
    lines[victim] = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    tampered = out_dir / f"run_{stamp}_tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")

    good_before, _, _ = verify_chain(chain_path)
    good_after, n_ok, first_bad = verify_chain(tampered)
    print(f"  original file : intact={good_before}")
    print(f"  tampered file : intact={good_after}  ({n_ok} entries verified before "
          f"breaking at seq={first_bad})")
    chain.record("tamper_test", {"detected": not good_after,
                                 "first_bad_seq": first_bad})

    # -- 6. boundary ----------------------------------------------------
    rule("STAGE 6 - BOUNDARY: does 'does not crash' still pass?")
    cheat = cheat_spec(spec.name + "_cheat", spec.parameters)
    real_check = verifier.check_robustness(spec)
    cheat_check = verifier.check_robustness(cheat)
    print(f"  real forged tool : passed={real_check.passed}  {real_check.detail[:78]}")
    print(f"  constant emitter : passed={cheat_check.passed}  {cheat_check.detail[:78]}")
    chain.record("boundary", {
        "real_passed": real_check.passed,
        "cheat_passed": cheat_check.passed,
        "cheat_detail": cheat_check.detail[:400],
    })

    # -- verdict --------------------------------------------------------
    chain_head = chain.head
    chain.record("verdict", {"replay_agree": replay_agree,
                             "total": len(recorded),
                             "tamper_detected": not good_after,
                             "cheat_rejected": not cheat_check.passed})
    chain.close()

    rule("VERDICT")
    verdicts = [
        ("forged by a real model", result.ok),
        ("verified before promotion", all(c["passed"] for c in spec.verification["checks"])),
        ("replay reproduces every output", replay_agree == len(recorded)),
        ("tampering is detected", not good_after),
        ("a no-op tool is rejected", not cheat_check.passed),
    ]
    for label, passed in verdicts:
        print(f"  [{'OK' if passed else 'XX'}] {label}")
    all_ok = all(p for _, p in verdicts)

    final = chain_path.read_text(encoding="utf-8")
    print(f"\n  record : {chain_path}  ({len(final.splitlines())} entries)")
    print(f"  file sha256 : {hashlib.sha256(final.encode('utf-8')).hexdigest()[:32]}...")
    print(f"  head       : {chain_head[:32]}...")
    print(f"\n{BAR}\n{'EVIDENCE OK' if all_ok else 'EVIDENCE INCOMPLETE'}\n{BAR}")
    return 0 if all_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
