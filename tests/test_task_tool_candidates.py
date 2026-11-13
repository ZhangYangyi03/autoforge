from types import SimpleNamespace
from autoforge.agent import ForgeAgent

class Store:
    def load_all_tools(self):
        return {
            "git_repo_status": SimpleNamespace(description="Run git status branch remote", probes=[]),
            "mail": SimpleNamespace(description="send email notification", probes=[]),
        }

class Registry:
    def get(self, name):
        return object()

def test_task_candidates_surface_relevant_tools_without_auto_execution():
    agent = object.__new__(ForgeAgent)
    agent.store = Store()
    agent.registry = Registry()
    assert agent._task_tool_candidates("git repository status branch") == ["git_repo_status"]
    assert agent._task_tool_candidates("weather in Beijing") == []


def test_shared_search_runs_through_market(monkeypatch):
    import json
    import urllib.request
    agent = object.__new__(ForgeAgent)
    specs = []
    agent._add = specs.append
    agent._tool_shared_web_search()
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): return json.dumps({"ok": True, "output": "source " + "x" * 5000}).encode()
    def fake(req, timeout):
        assert req.full_url.endswith("/resources/tool:exa_web_search_sse_fixed/invoke")
        assert json.loads(req.data)["arguments"]["tool"] == "web_search_exa"
        return Response()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert len(specs[0].fn("Nvidia", "reliable sources")) == 5007


def test_market_discovery_exposes_and_invokes_remote_contract(monkeypatch):
    import json
    import urllib.request
    from types import SimpleNamespace
    agent = object.__new__(ForgeAgent)
    specs = {}
    agent.registry = SimpleNamespace(get=lambda name: specs.get(name))
    agent._add = lambda spec: specs.__setitem__(spec.name, spec)
    agent._record = lambda *a, **k: None
    class Response:
        def __init__(self, obj): self.obj = obj
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, n): return json.dumps(self.obj).encode()
    def fake(req, timeout):
        url = req if isinstance(req, str) else req.full_url
        if "/search/fast?" in url:
            return Response({"results": [{"resource_id":"tool:remote", "name":"remote", "description":"test"}]})
        if url.endswith("/invoke"):
            assert json.loads(req.data) == {"arguments":{"text":"hello"}}
            return Response({"ok":True,"output":"hello from market"})
        return Response({"state":"probation","contract":{"parameters":{"type":"object","properties":{"text":{"type":"string"}}}}})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    found = agent._shared_market_task_candidates("process text")
    assert found == ["shared_market_remote"]
    assert specs[found[0]].fn(text="hello") == "hello from market"


def test_market_discovery_reuses_local_tool_instead_of_clone(monkeypatch):
    import json
    import urllib.request
    from types import SimpleNamespace
    agent = object.__new__(ForgeAgent)
    specs = {"local_tool": object()}
    agent.registry = SimpleNamespace(get=lambda name: specs.get(name))
    agent._add = lambda spec: specs.__setitem__(spec.name, spec)
    agent._record = lambda *a, **k: None
    class Response:
        def __init__(self, obj): self.obj = obj
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, n): return json.dumps(self.obj).encode()
    def fake(req, timeout):
        url = req if isinstance(req, str) else req.full_url
        if "/search/fast?" in url:
            return Response({"results":[{"resource_id":"tool:local_tool","name":"local_tool","description":"x"}]})
        return Response({"state":"probation","contract":{"parameters":{"type":"object","properties":{}}}})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert agent._shared_market_task_candidates("local_tool task") == ["local_tool"]
    assert "shared_market_local_tool" not in specs
