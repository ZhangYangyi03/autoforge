
"""Run every test file separately, with a per-file timeout, to a result file.

Whole-suite runs exceed the agent's per-call cap (>600s), and a single file
that hangs takes the whole run with it. Per-file means a hang costs one
line, and the file it happened in is named.
"""
import json, os, subprocess, sys, time

REPO = r"D:\Users\china\Desktop\项目_开发\autoforge"
PY = sys.executable
OUT = os.path.join(REPO, "_suite_results.jsonl")
tests = sorted(f for f in os.listdir(os.path.join(REPO, "tests"))
               if f.startswith("test_") and f.endswith(".py"))
done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        try: done.add(json.loads(line)["file"])
        except Exception: pass
with open(OUT, "a", encoding="utf-8") as fh:
    for f in tests:
        if f in done:
            continue
        t0 = time.time()
        rec = {"file": f}
        try:
            r = subprocess.run([PY, "-m", "pytest", "tests/" + f, "-q",
                                "--no-header", "-p", "no:cacheprovider", "-rf"],
                               cwd=REPO, capture_output=True, timeout=150)
            out = ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", "replace")
            summary = [l for l in out.splitlines() if l.strip()][-1][:200] if out.strip() else ""
            rec.update(seconds=round(time.time() - t0, 1), summary=summary)
        except subprocess.TimeoutExpired:
            rec.update(seconds=round(time.time() - t0, 1), summary="TIMEOUT >150s")
        fh.write(json.dumps(rec) + "\n"); fh.flush()
