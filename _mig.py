
import sys
sys.path.insert(0, r"D:\\Users\\china\\Desktop\\项目_开发\\autoforge")
from autoforge.store import ToolStore
st = ToolStore()
recs = st.load_all_tools()
print("tools loaded:", len(recs))
r = recs.get("audit_predictive_claim_csv")
print("sample_call on the audit tool:", (r.sample_call if r else "no row"))
