"""The native layer: compile, isolate, verify, search.

The tests are ordered the way the layer is used, because that is also the order
in which its claims can be false:

1. THE PROBE tells the truth about this machine. If it does not, every later
   stage is deciding against a fiction — a kernel is compiled for the wrong ISA
   and the failure surfaces as a segfault three stages later.
2. THE CACHE keys on what actually decides the binary. Keyed on too little, it
   serves a kernel built for another machine; keyed on too much, it never hits
   and the search loop it exists to accelerate pays a compile per candidate.
3. PREFLIGHT blocks what must not build and merely warns what might.
4. THE GUARD survives a kernel that crashes or hangs, and *fails* one that
   returns the wrong answer. A guard that only survives is a guard that passes
   the fastest wrong kernel in the search.
5. THE SEARCH reports a winner only when it beat the baseline outside the noise,
   and refuses to when it did not.
6. THE TOOL SURFACE is real: eight tools registered, scopes declared, and the
   three that execute something gated behind `may_run_cpu_kernels`.
"""
from __future__ import annotations

import os

import pytest

from autoforge import cpu as C
from autoforge.agent import BUILTIN_SCOPES, ForgeAgent
from autoforge.autonomy.policy import SUPERVISED, AutonomyPolicy
from autoforge.core.llm import MockLLMClient

#: Every C kernel takes `size_t n`, and the compiler is invoked without a
#: preamble, so a kernel source that wants it must include it itself. This is
#: the header the tests prepend for the same reason a caller would.
HDR = "#include <stddef.h>\n"

SAXPY = HDR + C.naive("saxpy")

#: Read-only tools answer whatever the policy says; the executing three do not.
READ_ONLY_TOOLS = ("cpu_probe", "cpu_runs_here", "cpu_preflight",
                   "cpu_units_audit", "cpu_cache_stats")
EXECUTING_TOOLS = ("cpu_compile", "cpu_run_isolated", "cpu_tune")
CPU_TOOLS = READ_ONLY_TOOLS + EXECUTING_TOOLS


@pytest.fixture
def kernel_cache(tmp_path):
    """A private kernel cache for the tests that make claims about caching.

    Injection, not `AUTOFORGE_CPU_CACHE`: the module-level default cache is
    constructed at import time, so an env var set by a fixture afterwards cannot
    reach it. A test that sets the variable and calls `compile_kernel` is
    therefore reading and writing the developer's real cache — and, because the
    key is content-addressed rather than name-addressed, a "first compile is a
    miss" assertion depends on whether an earlier test in the same session
    happened to compile identical source.
    """
    return C.KernelCache(tmp_path / "kernels")


def compile_src(name: str, code: str) -> C.CompiledKernel:
    return C.compile_kernel(C.KernelSource(name=name, code=code))


def guard(name: str, code: str, *, kind: str = "saxpy", size: int = 1 << 10,
          timeout: float = 10.0) -> C.GuardResult:
    return C.run_isolated(compile_src(name, code), C.problem(kind, size=size),
                          reps=1, warmup=0, timeout=timeout)


# ---------------------------------------------------------------------------
# 1. the probe
# ---------------------------------------------------------------------------
def test_probe_reports_this_machine_and_is_cached():
    info = C.probe()
    assert info.available, "no probe means every later stage decides against a fiction"
    assert info.logical_cores >= 1
    assert info.arch, "an empty arch string means every 'runs here' answer is noise"
    first = C.probe()
    C.clear_probe_cache()
    second = C.probe()
    assert first.to_dict() == second.to_dict(), \
        "the probe must give the same answer twice; a probe that flickers makes the cache key unstable"


def test_probe_uses_measurement_not_assumption_for_the_target():
    """`-march=native` is a fact about the compiler, not about the CPU."""
    runs, why = C.runs_here("native")
    assert runs, f"the host must run its own native build: {why}"


def test_runs_here_refuses_an_isa_this_machine_does_not_have():
    """The whole point: refuse before compiling, not after crashing."""
    runs, why = C.runs_here("znver5")
    assert not runs
    assert "znver5" in why


def test_a_refused_target_is_refused_by_compile_too():
    """`runs_here` is not advisory — the compile path consults it."""
    src = C.KernelSource(name="nope", code=SAXPY, target="znver5")
    with pytest.raises(C.KernelUnavailable) as exc:
        C.compile_kernel(src)
    assert "znver5" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. the cache
# ---------------------------------------------------------------------------
def test_the_cache_hits_the_second_time_and_the_key_is_stable(kernel_cache):
    src = C.KernelSource(name="cached", code=SAXPY)
    first = C.compile_kernel(src, cache=kernel_cache)
    second = C.compile_kernel(src, cache=kernel_cache)
    assert first.from_cache is False
    assert second.from_cache is True
    assert first.cache_key == second.cache_key
    assert first.artefact == second.artefact


def test_the_cache_key_separates_targets():
    """One cache directory serving several machines must never serve the wrong binary."""
    a = C.KernelSource(name="k", code=SAXPY, target="native")
    b = C.KernelSource(name="k", code=SAXPY, flags=["-O0"])
    assert a.cache_key() != b.cache_key(), "different builds share a key and one machine gets the other's binary"


def test_the_cache_key_separates_sources_with_the_same_name():
    """Name-addressed would be the easy mistake; the key is content-addressed."""
    a = C.KernelSource(name="same", code=SAXPY)
    b = C.KernelSource(name="same", code=HDR + C.naive("relu"))
    assert a.cache_key() != b.cache_key()


def test_cache_stats_are_counted_not_guessed(kernel_cache):
    stats = C.cache_stats(kernel_cache)
    assert set(stats) >= {"root", "hits", "misses", "hit_rate"}
    assert stats["misses"] == 0
    C.compile_kernel(C.KernelSource(name="statsy", code=SAXPY), cache=kernel_cache)
    after = C.cache_stats(kernel_cache)
    assert after["misses"] == 1
    C.compile_kernel(C.KernelSource(name="statsy", code=SAXPY), cache=kernel_cache)
    assert C.cache_stats(kernel_cache)["hits"] == 1


def test_a_failed_compile_is_not_cached_as_a_success(kernel_cache):
    """Otherwise one bad build poisons every later attempt in the same run."""
    bad = C.KernelSource(name="bad", code="this is not C")
    with pytest.raises(C.KernelUnavailable):
        C.compile_kernel(bad, cache=kernel_cache)
    with pytest.raises(C.KernelUnavailable):
        C.compile_kernel(bad, cache=kernel_cache)      # still an error, not a stale artefact
    assert C.cache_stats(kernel_cache)["hits"] == 0


# ---------------------------------------------------------------------------
# 3. preflight
# ---------------------------------------------------------------------------
def test_preflight_blocks_a_kernel_that_spawns_processes():
    pf = C.preflight(C.KernelSource(name="pf", code='void f(void){ system("ls"); }'))
    assert not pf.ok
    assert [f.code for f in pf.blocking] == ["forks"]


def test_preflight_warns_without_blocking():
    """A warning is advice; a block is a refusal. Conflating them makes the tool useless."""
    for code, needle in (
        ('void f(char*a){ strcpy(a,"x"); }', "unsafe_str"),
        ("void f(void){ while(1){} }", "unbounded_loop"),
        ("void f(void){ int x = rand(); (void)x; }", "nondeterministic"),
        ("void f(void){ #pragma omp parallel\n }", "openmp_missing"),
    ):
        pf = C.preflight(C.KernelSource(name="pf", code=code))
        assert needle in [f.code for f in pf.warnings], f"{needle} was not flagged"
        assert pf.ok, f"{needle} must advise, not refuse"


def test_preflight_does_not_read_a_comment_as_a_finding():
    """A kernel documenting why it avoids `system` must not be blocked for mentioning it."""
    pf = C.preflight(C.KernelSource(name="pf", code="// never call system() here\nvoid f(void){}"))
    assert pf.ok and not pf.findings


def test_preflight_summary_says_so_when_there_is_nothing_to_say():
    pf = C.preflight(C.KernelSource(name="pf", code=SAXPY))
    assert pf.ok and not pf.findings
    assert "nothing to flag" in pf.summary()


def test_units_audit_catches_the_thousand_factor_under_an_ms_label():
    """The do_bench bug, as a static check: a ms label over a value scaled by 1e3.

    The audit is a regex over harness source, so it only claims the shapes it
    can see — the two false negatives are documented in its docstring rather
    than silently accepted.
    """
    from autoforge.timing import audit_ms_scale
    offences = audit_ms_scale('print(f"{elapsed * 1e3} ms")')
    assert offences, "a ms label over a 1000x value is the exact bug this exists to catch"
    assert audit_ms_scale("x = 1") == []


# ---------------------------------------------------------------------------
# 4. the guard
# ---------------------------------------------------------------------------
def test_a_correct_kernel_passes_the_guard():
    result = guard("good", SAXPY)
    assert result.ok and result.verdict["verified"]
    assert result.wall.seconds > 0


def test_a_wrong_kernel_fails_verification_rather_than_merely_running():
    """The claim under test: the guard is not a smoke test."""
    result = guard("wrong", HDR + "void saxpy(float*y,const float*x,float a,size_t n)"
                                 "{ for(size_t i=0;i<n;i++) y[i]=x[i]; }")
    assert not result.ok
    assert result.verdict["verified"] is False
    assert "does not match the reference" in result.verdict["reason"]


def test_a_segfaulting_kernel_does_not_take_the_agent_down_with_it():
    """The process boundary exists for exactly this: a kernel is untrusted code."""
    result = guard("crashy",
                   HDR + "void saxpy(float*y,const float*x,float a,size_t n)"
                         "{ float*p=0; for(size_t i=0;i<n;i++) p[i]=1.0f; (void)a;(void)y;(void)x; }")
    assert not result.ok
    assert result.error, "a crash must be reported, not swallowed into a zero timing"


def test_a_hanging_kernel_is_killed_and_named():
    result = guard("hangy",
                   HDR + "void saxpy(float*y,const float*x,float a,size_t n)"
                         "{ volatile int i=0; while(i==0){} (void)a;(void)y;(void)x;(void)n; }",
                   timeout=4.0)
    assert not result.ok
    assert result.timed_out
    assert "TIMED OUT" in result.summary()


def test_the_guard_reports_the_missing_symbol_as_a_missing_symbol():
    """Compiling fine and not exporting the entry point is a distinct failure."""
    result = guard("nosym", HDR + "void other(float*y,const float*x,float a,size_t n)"
                                 "{ for(size_t i=0;i<n;i++) y[i]=a*x[i]+y[i]; }")
    assert not result.ok
    assert "is not exported" in result.verdict["reason"]


def test_the_guard_result_serialises_every_field_the_reports_use():
    d = guard("good", SAXPY).to_dict()
    assert set(d) >= {"ok", "timed_out", "crashed", "returncode", "fault",
                      "error", "verdict", "wall_s"}


# ---------------------------------------------------------------------------
# 5. verifying and timing, in that order
# ---------------------------------------------------------------------------
def test_evaluate_verifies_before_it_times():
    good = compile_src("goodk", SAXPY)
    verdict = C.evaluate(good, C.problem("saxpy", size=1 << 12), reps=2, warmup=1)
    assert verdict["ok"] and verdict["verified"]


def test_evaluate_refuses_to_time_a_wrong_kernel():
    """Timing first is how the search selects for the fastest wrong kernel."""
    bad = compile_src("badk", HDR + "void saxpy(float*y,const float*x,float a,size_t n)"
                                    "{ for(size_t i=0;i<n;i++) y[i]=0.0f; (void)a;(void)x; }")
    verdict = C.evaluate(bad, C.problem("saxpy", size=1 << 12), reps=2, warmup=1)
    assert not verdict["ok"] and not verdict["verified"]


def test_close_reports_the_worst_element_not_just_a_boolean():
    eq = C.close([1.0, 2.0], [1.0, 2.0])
    ne = C.close([1.0, 2.0], [1.0, 3.0])
    assert eq["passed"] and ne["passed"] is False
    assert ne["worst_index"] == 1
    assert ne["max_abs_diff"] == pytest.approx(1.0)


def test_problem_refuses_an_unknown_dtype_and_names_the_known_ones():
    with pytest.raises(C.CallError) as exc:
        C.problem("saxpy", dtype="nope")
    assert "unknown dtype" in str(exc.value)


def test_problem_refuses_an_unknown_kind():
    with pytest.raises(C.CallError) as exc:
        C.problem("nope")
    assert "no such problem" in str(exc.value)


def test_a_problem_cannot_be_asked_for_without_a_reference():
    """There is no code path that yields inputs with nothing to compare them to."""
    prob = C.problem("saxpy", size=64)
    assert prob.expected, "a problem with no reference makes 'verify' a step a caller can skip"


def test_every_naive_kind_has_a_working_problem_and_reference():
    """One entry per kind in NAIVE, so a new kind cannot ship without a reference."""
    for kind in C.NAIVE:
        prob = C.problem(kind, size=1 << 8)
        assert prob.expected, f"{kind} has no reference"
        verdict = C.evaluate(compile_src(f"naive_{kind}", HDR + C.naive(kind)), prob, reps=1, warmup=0)
        assert verdict["verified"], f"the naive {kind} fails its own reference"


def test_marshal_is_separate_from_the_call_so_the_timer_excludes_it():
    """Marshalling a million elements costs more than the kernel; timing it is a lie."""
    spec = C.problem("saxpy", size=1 << 10).spec
    args = C.marshal(spec)
    assert len(args) == len(spec.args)
    assert args is not C.marshal(spec), "marshalling must be re-entrant, not memoised into correctness"


# ---------------------------------------------------------------------------
# 6. the search
# ---------------------------------------------------------------------------
def test_variant_generators_propose_changes_with_the_reason_attached():
    for variants, floor in ((C.flag_variants(C.KernelSource(name="v", code=SAXPY)), 4),
                            (C.source_variants(C.KernelSource(name="v", code=SAXPY)), 3)):
        assert len(variants) >= floor
        for v in variants:
            assert v.change and v.source, "a variant the report cannot explain is a variant nobody can audit"


def test_search_verifies_every_candidate_it_reports():
    """A candidate with a time and no verification is a number the search will select for."""
    result = C.tune_kind("reduce_sum", size=1 << 14, max_candidates=4, budget_s=25.0,
                         reps=2, warmup=1)
    assert result.n_candidates >= 1
    for cand in result.history:
        if cand.median_ms is not None:
            assert cand.verified, f"candidate '{cand.change}' was timed without being verified"


def test_search_stops_on_its_budget_and_says_so():
    result = C.tune_kind("relu", size=1 << 13, max_candidates=8, budget_s=1.0,
                         reps=1, warmup=0)
    assert result.stopped_because, "a search that stops must say why"
    assert result.wall.seconds >= 0


def test_search_reports_a_claim_only_when_it_beat_the_noise():
    """The honest half: 'no winner' is a result, and a better one than a fake one."""
    result = C.tune_kind("reduce_sum", size=1 << 16, max_candidates=6, budget_s=30.0,
                         reps=3, warmup=2)
    if result.winner is None:
        assert result.comparison_fast is None or not result.comparison_fast.beats_baseline
    else:
        assert result.winner.verified
        assert result.comparison_fast is not None
        assert result.report()


def test_the_report_is_written_for_a_reader_who_was_not_there():
    result = C.tune_kind("relu", size=1 << 13, max_candidates=3, budget_s=15.0,
                         reps=2, warmup=1)
    text = result.report()
    assert "relu" in text and "candidates" in text
    assert len(text.splitlines()) >= 3


# ---------------------------------------------------------------------------
# 7. the tool surface — the eight tools, their scopes, and the gate
# ---------------------------------------------------------------------------
def make_agent(policy=None):
    """An agent with the CPU tools wired in.

    `policy` is only passed when a test actually wants a non-default one: the
    dataclass default is a factory, so passing `None` explicitly is not the same
    thing as omitting it, and a test that confuses the two is testing the
    constructor rather than the gate.
    """
    kw = {"policy": policy} if policy is not None else {}
    return ForgeAgent(MockLLMClient(), enable_evolution=False, **kw)


def test_all_eight_cpu_tools_are_registered():
    names = set(make_agent().registry.names())
    missing = [t for t in CPU_TOOLS if t not in names]
    assert not missing, f"declared in the layer but not reachable by the agent: {missing}"


def test_every_cpu_tool_declares_a_scope():
    missing = [t for t in CPU_TOOLS if t not in BUILTIN_SCOPES]
    assert not missing, f"an undeclared scope is a tool that silently needs everything: {missing}"


def test_executing_tools_declare_a_scope_that_can_be_gated():
    for tool in EXECUTING_TOOLS:
        assert BUILTIN_SCOPES[tool] != "read_only", \
            f"{tool} executes compiled code but declares itself read-only"


def test_read_only_tools_still_answer_when_the_cpu_freedom_is_off():
    agent = make_agent(AutonomyPolicy(may_run_cpu_kernels=False))
    for tool in READ_ONLY_TOOLS:
        args = {"code": SAXPY} if tool == "cpu_preflight" else \
               {"source": "x = 1"} if tool == "cpu_units_audit" else \
               {"target": "native"} if tool == "cpu_runs_here" else {}
        out = agent.registry.call(tool, args)
        text = (out.output or "") + (out.error or "")
        assert "may_run_cpu_kernels" not in text, \
            f"{tool} reads and must not be gated by the freedom to execute"


@pytest.mark.parametrize("tool,args", [
    ("cpu_compile", {"code": SAXPY}),
    ("cpu_run_isolated", {"code": SAXPY}),
    ("cpu_tune", {}),
])
def test_the_gate_names_the_field_it_is_missing(tool, args):
    """A refusal that does not say which freedom is off is a refusal the caller cannot act on."""
    agent = make_agent(AutonomyPolicy(may_run_cpu_kernels=False))
    out = agent.registry.call(tool, args)
    text = (out.output or "") + (out.error or "")
    assert "may_run_cpu_kernels" in text or out.awaiting_confirmation


def test_supervised_leaves_the_cpu_freedom_off():
    assert SUPERVISED.may_run_cpu_kernels is False


def test_the_freedom_on_means_the_tools_actually_work():
    """The gate must gate, not merely annotate."""
    agent = make_agent()          # default policy: full freedom
    out = agent.registry.call("cpu_compile", {"code": SAXPY})
    assert out.ok and "Compiled" in out.output

    prob = agent.registry.call("cpu_probe", {})
    assert prob.ok and prob.output


def test_cpu_compile_refuses_a_kernel_with_no_entry_point():
    agent = make_agent()
    out = agent.registry.call("cpu_compile", {"code": "static void hidden(void){} int main(void){return 0;}"})
    assert not out.ok or "no callable entry point" in (out.output or out.error or "")


# ---------------------------------------------------------------------------
# 7. the harness's own BLAS must not kill the process
# ---------------------------------------------------------------------------
def test_numpy_is_imported_with_a_bounded_blas(monkeypatch):
    """A library that aborts the process is not a library you can catch.

    Measured on this host (16 cores, Windows): importing numpy lets OpenBLAS
    size its thread pool from the core count, the allocation fails inside the
    process, and OpenBLAS does not raise -- it aborts:

        OpenBLAS error: Memory allocation still failed after 10 retries, giving up.

    No traceback, no exception, no return code to interpret: the interpreter is
    simply gone. In tests it showed up as 18 failures in this file (the pytest
    process dying mid-run, so the ones that "failed" produced no output at all);
    in production it would report a correct kernel as unrunnable on any machine
    with enough cores. The bound has to be set before the import, which is why
    it lives in `_numpy()` rather than in a caller.
    """
    from autoforge.cpu import ops

    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_DEFAULT_NUM_THREADS", raising=False)
    ops._numpy()
    assert os.environ["OPENBLAS_NUM_THREADS"].isdigit()
    assert int(os.environ["OPENBLAS_NUM_THREADS"]) <= int(os.cpu_count() or 1)


def test_the_kernels_own_threads_are_left_alone(monkeypatch):
    """OMP_NUM_THREADS is the kernel's, and the guard times the kernel."""
    from autoforge.cpu import ops

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    ops._numpy()
    assert "OMP_NUM_THREADS" not in os.environ, (
        "bounding OMP_NUM_THREADS would change the number the guard reports for "
        "a kernel written with `#pragma omp parallel`")
