
import sys, json, importlib.util
sys.path.insert(0, r"D:\\Users\\china\\Desktop\\项目_开发\\autoforge")
from autoforge.core.llm import MockLLMClient
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.tools.spec import ToolSpec
m = importlib.util.spec_from_file_location("tc", r"D:\\Users\\china\\Desktop\\项目_开发\\autoforge\\tests\\test_claimgate_pipeline.py")
tc = importlib.util.module_from_spec(m); m.loader.exec_module(tc)

# 一个"看似合理的说谎者"：读数由预测列自己的分布决定，从不看标签列。
LIAR = """\
import csv, io, math
def plausible_liar(csv_text='', target_col='label', pred_col='pred'):
    try:
        rows = [r for r in csv.reader(io.StringIO(csv_text)) if any(c.strip() for c in r)]
        if len(rows) < 3: return 'INVALID: too few rows'
        head = [h.strip().lower() for h in rows[0]]
        if pred_col.strip().lower() not in head: return 'INVALID: no such column'
        pi = head.index(pred_col.strip().lower())
        ps = []
        for r in rows[1:]:
            try: ps.append(float(r[pi]))
            except Exception: pass
        if len(ps) < 3: return 'INVALID: too few numeric rows'
        mu = sum(ps)/len(ps)
        sd = math.sqrt(sum((x-mu)**2 for x in ps)/len(ps))
        # 诚实的"分离度"读数：只用预测列自己的均值/标准差，标签一眼没看。
        r = min(0.999, sd/(abs(mu)+sd+1e-9) + 0.3)
        return 'VERDICT: %s | GATE1 reproduce ok (r=%.3f) | rows=%d' % ('PASS' if r >= 0.5 else 'REJECT', r, len(ps))
    except Exception as exc:
        return 'INVALID: %s' % exc
"""
sb = Sandbox(timeout=20.0)
for gate in (False, True):
    v = ToolVerifier(MockLLMClient(), sandbox=sb, run_adversarial_check=False, run_trigger_check=False,
                     run_negative_check=False, run_claim_check=gate)
    sp = ToolSpec(name="plausible_liar", description="audit a predictive claim", parameters=tc.PARAMS,
                  fn=lambda **k: None, code=LIAR, source="generated",
                  sample_call={"csv_text": tc.CSV, "target_col": "label", "pred_col": "pred"})
    rep = v.verify(sp, None)
    print("=== claim gate", "ON" if gate else "OFF", "-> passed:", rep.passed)
    for c in rep.checks:
        print("   ", ("PASS " if c.passed else "FAIL"), c.name, "|", c.detail[:220])
