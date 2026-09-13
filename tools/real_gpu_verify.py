"""The real-GPU half of the GPU layer's verification.

Every claim the local (CPU) tests can only assert structurally gets settled
here against actual silicon. The headline is units.py's contract — that
`do_bench` returns MILLISECONDS — proven by wall clock rather than by reading
documentation.

Run on the box:
    export PATH=/root/miniconda3/bin:$PATH
    cd /root/labs/autoforge_gpu && python real_gpu_verify.py
"""
from __future__ import annotations

import time

import torch
import triton
from triton.testing import do_bench

FAILS: list[str] = []
OKS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (OKS if cond else FAILS).append(f"{name}{(' | ' + detail) if detail else ''}")


print("=" * 74)
print("REAL GPU VERIFICATION — autoforge gpu layer")
print("=" * 74)

dev = torch.cuda.get_device_properties(0)
print(f"device  : {dev.name}  sm_{dev.major}{dev.minor}  {dev.multi_processor_count} SMs")
print(f"torch   : {torch.__version__}   triton: {triton.__version__}")
print()

# ---------------------------------------------------------------------------
# A. THE CLAIM: do_bench returns milliseconds. Settle it by wall clock.
# ---------------------------------------------------------------------------
print("--- A. do_bench unit settlement (wall clock, not documentation) ---")

import sys
sys.path.insert(0, "/root/labs")
import shutil
from autoforge_gpu.units import Duration, audit_ms_scale, from_do_bench   # type: ignore
from autoforge_gpu.bench import (                                        # type: ignore
    SPREAD_SIGNAL_PCT,
    bandwidth_gbps,
    verify,
)
from autoforge_gpu.ops import bench_kernel                              # type: ignore
from autoforge_gpu import probe as gpu_probe                            # type: ignore

dev = torch.cuda.get_device_properties(0)
print(f"device  : {dev.name}  sm_{dev.major}{dev.minor}  {dev.multi_processor_count} SMs")
print(f"torch   : {torch.__version__}   triton: {triton.__version__}")
print()

# ---------------------------------------------------------------------------
# A. THE CLAIM: do_bench returns milliseconds. Settle it by wall clock.
#
# Method: time a real kernel BOTH ways over the same work. `do_bench` gives one
# number; perf_counter around a synced loop gives the true elapsed time. If
# do_bench's number matches wall-in-ms and is 1000x off wall-in-s, the unit is
# settled — by measurement, not by documentation.
# ---------------------------------------------------------------------------
print("--- A. do_bench unit settlement (wall clock, not documentation) ---")

x = torch.randn(2 ** 24, device="cuda")
y = torch.randn(2 ** 24, device="cuda")

REPS = 50
# do_bench's own figure (median over many reps, as it uses internally).
claim = do_bench(lambda: x + y, warmup=10, rep=REPS)

# True wall clock for the same number of launches, synced.
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(REPS):
    x + y
torch.cuda.synchronize()
wall_total = time.perf_counter() - t0
wall_one = wall_total / REPS

as_ms = claim / 1000.0
err_ms = abs(as_ms - wall_one) / wall_one
err_s = abs(claim - wall_one) / wall_one

print(f"  do_bench returned  : {claim:.4f}")
print(f"  wall clock / launch: {wall_one * 1000:.4f} ms  ({wall_one:.6f} s)")
print(f"  if interpreted ms  : {claim:.4f} ms -> rel err {err_ms * 100:.1f}%")
print(f"  if interpreted s   : {claim:.4f} s  -> rel err {err_s * 100:.1f}%")

unit = "ms" if err_ms < err_s else "s"
check("A1 do_bench unit is MILLISECONDS (wall-clock settled)", unit == "ms",
      f"interpreted as {unit}; ms-err {err_ms * 100:.1f}% vs s-err {err_s * 100:.1f}%")
check("A2 do_bench agrees with the wall clock within 20%",
      err_ms < 0.20, f"ms-interpretation error {err_ms * 100:.1f}%")
check("A3 the seconds interpretation is off by ~1000x",
      err_s / max(err_ms, 1e-9) > 100, f"ratio {err_s / max(err_ms, 1e-9):.0f}")

# The bug itself: what the wrong line would have printed, versus the right one.
wrong = claim * 1e3
print(f"  correct print      : {claim:.4f} ms")
print(f"  the bug would print: {wrong:.1f} ms  <- {wrong / claim:.0f}x too large")
check("A4 the 1e3 bug inflates exactly 1000x", abs(wrong / claim - 1000) < 1,
      f"ratio {wrong / claim:.0f}")

d = from_do_bench(claim)
print(f"  from_do_bench      : {d.format('ms')}  == {d.format('us')}")
check("A5 from_do_bench preserves the ms reading",
      abs(d.ms - claim) < 1e-6, f"{d.ms:.4f} vs {claim:.4f}")

# ---------------------------------------------------------------------------
# B. verify() against a real torch reference — the correctness gate
# ---------------------------------------------------------------------------
print("\n--- B. verify() against a torch reference ---")
n = 4096
a = torch.randn(n, device="cuda", dtype=torch.float32)
b = torch.randn(n, device="cuda", dtype=torch.float32)
ref = (a + b).cpu().tolist()
same = (a + b).cpu().tolist()

r_same = verify(same, ref, tolerance=1e-4, label="identical")
print(f"  identical   : passed={r_same['passed']} max_diff={r_same['max_abs_diff']:.3e}")
check("B1 identical output passes", r_same["passed"])

# A fast-but-wrong kernel: drop the second operand (this is the exact failure
# mode verify() exists to catch).
wrong_out = a.cpu().tolist()
r_wrong = verify(wrong_out, ref, tolerance=1e-1, label="dropped-operand")
print(f"  wrong kernel: passed={r_wrong['passed']} max_diff={r_wrong['max_abs_diff']:.3e}")
check("B2 a dropped-operand kernel FAILS even at loose tolerance",
      not r_wrong["passed"], f"max_diff {r_wrong['max_abs_diff']:.3e}")
check("B3 max_diff==0 is reported as 'did the kernel do any work?'",
      "did the kernel do any work" in r_same["reason"])

# fp16 accumulation reality — the value the skill says is correct, not a bug.
ah = torch.randn(n, device="cuda", dtype=torch.float16)
bh = torch.randn(n, device="cuda", dtype=torch.float16)
refh = (ah.float() + bh.float()).cpu().tolist()
outh = (ah + bh).float().cpu().tolist()
r_h = verify(outh, refh, tolerance=1e-2, label="fp16")
print(f"  fp16        : max_diff={r_h['max_abs_diff']:.3e} (accumulation, not a bug)")
check("B4 fp16 diff is small but non-zero", 0 < r_h["max_abs_diff"] < 1e-2,
      f"{r_h['max_abs_diff']:.3e}")

# ---------------------------------------------------------------------------
# C. bandwidth cross-check — the arithmetic that catches the unit bug
# ---------------------------------------------------------------------------
print("\n--- C. bandwidth sanity vs the card ---")

N = 2 ** 24
x = torch.randn(N, device="cuda")
y = torch.randn(N, device="cuda")
ms = do_bench(lambda: x + y, warmup=10, rep=50)
t = from_do_bench(ms)
moved = 3 * N * 4          # read x, read y, write out, fp32
bw = bandwidth_gbps(moved, t)
peak = 1008.0              # 4090 D HBM, GB/s
print(f"  vector add N=2^24 : {t.ms:.4f} ms  {bw:.1f} GB/s  ({bw / peak * 100:.0f}% of ~{peak:.0f})")
check("C1 memory-bound op reaches >70% of peak (real silicon)",
      bw > 0.70 * peak, f"{bw:.1f} GB/s")

# The same number under the bug: what the wrong line implies.
bw_bug = bandwidth_gbps(moved, Duration.from_ms(ms * 1e3))
print(f"  under the 1e3 bug : {bw_bug:.4f} GB/s  <- absurd, and that is the tell")
check("C2 the bug's implied bandwidth is absurd (<2 GB/s)",
      bw_bug < 2.0, f"{bw_bug:.4f} GB/s")

# ---------------------------------------------------------------------------
# D. the units lint has teeth on real code shapes
# ---------------------------------------------------------------------------
print("\n--- D. audit_ms_scale on real code ---")
bad = 'ms = do_bench(fn)\nprint(f"{ms * 1e3:.4f} ms")'
good = 'ms = do_bench(fn)\nprint(from_do_bench(ms).format("ms"))'
print(f"  bad  -> {len(audit_ms_scale(bad))} offence(s)")
print(f"  good -> {len(audit_ms_scale(good))} offence(s)")
check("D1 lint fires on the real bug shape", len(audit_ms_scale(bad)) == 1)
check("D2 lint is quiet on the fixed shape", len(audit_ms_scale(good)) == 0)

# ---------------------------------------------------------------------------
# E. spread discipline on a real kernel
# ---------------------------------------------------------------------------
print("\n--- E. spread over repetitions (is a difference signal?) ---")


def three_runs():
    return [do_bench(lambda: x + y, warmup=10, rep=50) for _ in range(3)]


runs = three_runs()
sp = (max(runs) - min(runs)) / sorted(runs)[1] * 100
print(f"  3 runs: {[round(r, 4) for r in runs]}  spread {sp:.2f}%")
check("E1 repeated do_bench spread is small (memory-bound, stable)",
      sp < 5.0, f"{sp:.2f}%")

# ---------------------------------------------------------------------------
# F. compile path reality on a box with no nvcc
# ---------------------------------------------------------------------------
print("\n--- F. compile path reality (no nvcc here) ---")
has_nvcc = shutil.which("nvcc") is not None
print(f"  nvcc on PATH: {has_nvcc}")
check("F1 this box has no nvcc, so the nvcc compile path cannot be used here",
      not has_nvcc, "confirms the triton backend is required on rented boxes")

print()
print("=" * 74)
for o in OKS:
    print("PASS", o)
print("-" * 74)
for f in FAILS:
    print("FAIL", f)
print("=" * 74)
print(f"real-GPU verification: {len(OKS)} passed, {len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
