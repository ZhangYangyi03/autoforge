
import sys, json, importlib.util
sys.path.insert(0, r"D:\\Users\\china\\Desktop\\项目_开发\\autoforge")
from autoforge.core.llm import MockLLMClient
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.tools.spec import ToolSpec
m = importlib.util.spec_from_file_location("tc", r"D:\\Users\\china\\Desktop\\项目_开发\\autoforge\\tests\\test_claimgate_pipeline.py")
tc = importlib.util.module_from_spec(m); m.loader.exec_module(tc)
sb = Sandbox(timeout=20.0)
import inspect
v = ToolVerifier(MockLLMClient(), sandbox=sb, run_adversarial_check=False, run_trigger_check=False,
                 run_negative_check=False, run_claim_check=False)
sp = ToolSpec(name="broken_audit", description="audit a predictive claim", parameters=tc.PARAMS,
              fn=lambda **k: None, code=tc.BROKEN, source="generated",
              sample_call={"csv_text": tc.CSV, "target_col": "label", "pred_col": "pred"})
rep = v.verify(sp, None)
for c in rep.checks:
    print(("PASS " if c.passed else "FAIL"), c.name, "|", c.detail[:900])
