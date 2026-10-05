
import sys, json
sys.path.insert(0, r"D:\\Users\\china\\Desktop\\项目_开发\\autoforge")
from autoforge.core.llm import MockLLMClient
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.tools.spec import ToolSpec
import importlib.util
spec_mod = importlib.util.spec_from_file_location("tc", r"D:\\Users\\china\\Desktop\\项目_开发\\autoforge\\tests\\test_claimgate_pipeline.py")
tc = importlib.util.module_from_spec(spec_mod); spec_mod.loader.exec_module(tc)
sb = Sandbox(timeout=20.0)
v = ToolVerifier(MockLLMClient(), sandbox=sb, run_adversarial_check=False, run_trigger_check=False, run_negative_check=False)
sp = ToolSpec(name="honest_audit", description="audit a predictive claim", parameters=tc.PARAMS,
              fn=lambda **k: None, code=tc.HONEST, source="generated",
              sample_call={"csv_text": tc.CSV, "target_col": "label", "pred_col": "pred"})
rep = v.verify(sp, None)
for c in rep.checks:
    print(("PASS " if c.passed else "FAIL"), c.name, "|", c.detail[:300])
    if c.name=="execution": print("   evidence:", json.dumps(c.evidence, ensure_ascii=False)[:400])
