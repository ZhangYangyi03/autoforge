
import sys, json, sqlite3, time
sys.path.insert(0, r"D:\\Users\\china\\Desktop\\项目_开发\\autoforge")
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.claimgate import check_claim, find_kind, find_rail
from autoforge.tools.spec import ToolSpec

db = r"C:\\Users\\china\\AppData\\Local\\autoforge\\autoforge.db"
rows = sqlite3.connect(db).execute(
    "select name, description, parameters, code, verification from tools where state in ('active','probation','draft')").fetchall()
print("tools on the ledger:", len(rows))
sb = Sandbox(timeout=20.0)
audited, skipped, failed = [], 0, []
for name, desc, params, code, ver in rows:
    try:
        params = json.loads(params) if isinstance(params, str) else params
    except Exception:
        skipped += 1; continue
    spec = ToolSpec(name=name, description=desc or "", parameters=params or {},
                    fn=lambda **k: None, code=code or "", source="generated")
    if not code:
        skipped += 1; continue
    kind = find_kind(spec); rail, vals = find_rail(spec)
    if kind == "none":
        skipped += 1; continue
    audited.append({"name": name, "kind": kind, "rail": rail,
                    "has_sample": bool(spec.parameters), "state": None})
print("with a rail (auditable):", len(audited), "skipped:", skipped)
for a in audited[:40]:
    print(f"  {a['kind']:8s} rail={a['rail']:20s} {a['name']}")
