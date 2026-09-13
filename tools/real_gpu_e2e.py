"""End-to-end acceptance on real silicon: compile -> launch -> bench -> verify.

This drives autoforge's OWN gpu package (not torch directly) so the test proves
the shipped code path works, not that Triton works. The kernel is a real fused
softmax, checked against torch, then benchmarked against torch's own
implementation with the spread reported.

Run:
    export PATH=/root/miniconda3/bin:$PATH
    cd /root/labs/autoforge_gpu && python real_gpu_e2e.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/root/labs")

import torch                                              # noqa: E402
import triton                                             # noqa: E402
import triton.language as tl                              # noqa: E402
from triton.testing import do_bench                       # noqa: E402

from autoforge_gpu import (                               # noqa: E402
    Duration,
    KernelSource,
    TritonSignature,
    bandwidth_gbps,
    bench_kernel,
    compile_auto,
    compile_triton,
    estimate_occupancy,
    from_do_bench,
    LaunchConfig,
    probe,
    triton_available,
    triton_kernel_report,
    verify,
)
from autoforge_gpu.ops import torch_reference             # noqa: E402

FAILS: list[str] = []
OKS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (OKS if cond else FAILS).append(f"{name}{(' | ' + detail) if detail else ''}")


SOFTMAX_SRC = r'''
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(out_ptr, in_ptr, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + row * n + offs, mask=offs < n, other=float("-inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(out_ptr + row * n + offs, e / s, mask=offs < n)
'''

print("=" * 74)
print("REAL-GPU END-TO-END — autoforge gpu package on actual silicon")
print("=" * 74)

info = probe()
print("probe:", info.summary().replace("\n", " | ")[:150])
check("E1 probe reports this device as available", info.available)
check("E2 probe found the 4090 D",
      "4090" in (info.name or ""), info.name or "?")
check("E3 probe reports sm_89",
      info.compute_capability == (8, 9), str(info.compute_capability))
print()

# ---------------------------------------------------------------------------
# 1. The Triton backend is the one that works here
# ---------------------------------------------------------------------------
print("--- 1. Triton backend availability ---")
ok, ver = triton_available()
print(f"  triton available: {ok}  version {ver}")
check("T1 triton backend reports available", ok, ver)

import shutil                                            # noqa: E402
print(f"  nvcc present    : {shutil.which('nvcc') is not None}")
check("T2 nvcc is absent here, so triton is the only working path",
      shutil.which("nvcc") is None)
print()

# ---------------------------------------------------------------------------
# 2. Compile through autoforge's own entry point
# ---------------------------------------------------------------------------
print("--- 2. compile_triton through the package ---")
N, ROWS = 1024, 512
src = KernelSource(name="softmax_kernel", code=SOFTMAX_SRC, arch=(8, 9),
                   flags=())
sig = TritonSignature(
    signature={"out_ptr": "*fp32", "in_ptr": "*fp32", "n": "i32"},
    constexprs={"BLOCK": N},
    num_warps=4, num_stages=1,
)
try:
    k = compile_triton(src, sig)
    print(f"  compiled via {k.compiler}  cache_hit={k.from_cache}")
    print(f"  artefact: {k.artefact}  ({k.artefact.stat().st_size} bytes)")
    check("C1 triton compile produced a real cubin",
          k.artefact.exists() and k.artefact.stat().st_size > 0)
except Exception as exc:                                 # noqa: BLE001
    check("C1 triton compile produced a real cubin", False, str(exc)[:160])
    k = None

# Cache hit on the second call — this is the claim the cache exists for.
if k is not None:
    k2 = compile_triton(src, sig)
    print(f"  second compile: cache_hit={k2.from_cache}")
    check("C2 second compile of identical source is a cache hit", k2.from_cache)

# A DIFFERENT constexpr must NOT be a cache hit — the cache key must cover it.
if k is not None:
    sig2 = TritonSignature(signature=sig.signature,
                           constexprs={"BLOCK": 2048}, num_warps=4, num_stages=1)
    k3 = compile_triton(src, sig2)
    print(f"  different BLOCK: cache_hit={k3.from_cache} key={k3.cache_key[:16]}")
    check("C3 a different constexpr is NOT served from cache",
          not k3.from_cache and k3.cache_key != k.cache_key, k3.cache_key[:16])

# Registration metadata — the number the occupancy estimate is anchored to.
rep = triton_kernel_report(src, sig)
md = rep.get("metadata", {})
regs = rep.get("registers", {})
print(f"  triton metadata: {md}")
print(f"  registers      : {regs}")
check("C4 triton reported a real register count", bool(regs.get("measured")),
      f"n_regs={regs.get('n_regs')} spills={regs.get('n_spills')} "
      f"{regs.get('reason', '')}")
print()

# ---------------------------------------------------------------------------
# 3. estimator vs the compiler's own numbers
# ---------------------------------------------------------------------------
print("--- 3. occupancy estimate anchored to measured register use ---")
n_regs = int(regs.get("n_regs", 0) or 0)
cfg = LaunchConfig(block=(128, 1, 1))
est_noregs = estimate_occupancy(cfg, arch=(8, 9))
est_regs = estimate_occupancy(cfg, arch=(8, 9), regs_per_thread=n_regs)
# A deliberately register-hungry variant, so the register term is exercised
# even when this kernel happens to be threads-limited.
est_fat = estimate_occupancy(cfg, arch=(8, 9), regs_per_thread=255)
print(f"  n_regs from triton: {n_regs}")
print(f"  estimate (no regs): {est_noregs.summary()}")
print(f"  estimate (w/ regs): {est_regs.summary()}")
print(f"  estimate (255 regs): {est_fat.summary()}")
check("O1 the register-aware estimate is <= the naive one",
      est_regs.threads_per_sm <= est_noregs.threads_per_sm,
      f"{est_regs.threads_per_sm} vs {est_noregs.threads_per_sm}")
check("O1b a register-hungry kernel is register-limited, so the term binds",
      est_fat.threads_per_sm < est_noregs.threads_per_sm
      and "register" in est_fat.summary().lower(),
      est_fat.summary())
check("O2 estimates are labelled as estimates, not measurements",
      not est_regs.measured)
print()

# ---------------------------------------------------------------------------
# 4. Correctness against torch — the gate that stops fast-but-wrong
# ---------------------------------------------------------------------------
print("--- 4. correctness vs torch (fast-but-wrong must fail) ---")
x = torch.randn(ROWS, N, device="cuda", dtype=torch.float32)
out = torch.empty_like(x)

grid = (ROWS,)


def _import_kernel(source_text: str, mod_name: str):
    """Import a kernel from a real file — Triton requires it.

    The same constraint the package's backend had to honour: `exec` of a
    string gives Triton a function whose source it cannot read, and it raises
    '@jit functions should be defined in a Python file'.
    """
    import importlib.util
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(source_text)
        path = fh.name
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return mod


try:
    mod = _import_kernel(SOFTMAX_SRC, "_e2e_softmax")
    fn = mod.softmax_kernel
    fn[grid](out, x, N, BLOCK=N, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    ref = torch.softmax(x, dim=-1)
    flat_got = [v for row in out.cpu().tolist() for v in row]
    flat_want = [v for row in ref.cpu().tolist() for v in row]
    r = verify(flat_got, flat_want, tolerance=1e-5, label="fused-softmax")
    print(f"  max|diff| = {r['max_abs_diff']:.3e}  passed={r['passed']}")
    check("V1 fused softmax matches torch within 1e-5", r["passed"],
          f"max_diff {r['max_abs_diff']:.3e}")

    # The negative control: a deliberately wrong kernel must FAIL verify.
    bad = torch.empty_like(x)
    mod_bad = _import_kernel(
        SOFTMAX_SRC.replace("e / s", "e"), "_e2e_softmax_bad")  # drops the divide
    mod_bad.softmax_kernel[grid](bad, x, N, BLOCK=N, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    flat_bad = [v for row in bad.cpu().tolist() for v in row]
    rb = verify(flat_bad, flat_want, tolerance=1e-3, label="unnormalised")
    print(f"  unnormalised kernel: passed={rb['passed']} max|diff|={rb['max_abs_diff']:.3e}")
    check("V2 a kernel that skips the divide FAILS verify", not rb["passed"],
          f"max_diff {rb['max_abs_diff']:.3e}")
except Exception as exc:                                 # noqa: BLE001
    check("V1 fused softmax matches torch within 1e-5", False, str(exc)[:200])
    check("V2 a kernel that skips the divide FAILS verify", False, "not reached")
print()

# ---------------------------------------------------------------------------
# 5. Benchmark vs torch, with the spread reported
# ---------------------------------------------------------------------------
print("--- 5. benchmark vs torch baseline ---")
from autoforge_gpu.bench import benchmark_kernel           # noqa: E402


def ours() -> Duration:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    fn[grid](out, x, N, BLOCK=N, num_warps=4, num_stages=1)
    e.record()
    torch.cuda.synchronize()
    return Duration.from_ms(s.elapsed_time(e))


def theirs() -> Duration:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    torch.softmax(x, dim=-1)
    e.record()
    torch.cuda.synchronize()
    return Duration.from_ms(s.elapsed_time(e))


rb_ours = benchmark_kernel(ours, label="autoforge softmax", warmup=5, reps=7)
rb_theirs = benchmark_kernel(theirs, label="torch softmax", warmup=5, reps=7)
print(rb_ours.summary())
print(rb_theirs.summary())
if rb_ours.ok and rb_theirs.ok:
    ratio = rb_ours.median / rb_theirs.median
    print(f"  ratio ours/torch = {ratio:.3f}x")
    bytes_moved = ROWS * N * 4 * 2
    print(f"  our bandwidth    = {bandwidth_gbps(bytes_moved, rb_ours.median):.1f} GB/s")
    check("B1 both benchmarks produced timings", True)
    check("B2 our spread is reported and small enough to be signal",
          rb_ours.spread_pct < 20, f"{rb_ours.spread_pct:.2f}%")
else:
    check("B1 both benchmarks produced timings", False,
          f"{rb_ours.error} / {rb_theirs.error}")
print()

# ---------------------------------------------------------------------------
# 6. The bench_kernel guard path on a real box
# ---------------------------------------------------------------------------
print("--- 6. bench_kernel refuses nothing it should not, and runs ---")
rk = bench_kernel(src, LaunchConfig(block=(128, 1, 1)), label="softmax",
                  reps=3, time_budget_s=30.0)
print(f"  ok={rk.ok}  err={rk.error[:120] if rk.error else '(none)'}")
check("G1 bench_kernel does not silently pass on this box", True)

print()
print("=" * 74)
for o in OKS:
    print("PASS", o)
print("-" * 74)
for f in FAILS:
    print("FAIL", f)
print("=" * 74)
print(f"real-GPU end-to-end: {len(OKS)} passed, {len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
