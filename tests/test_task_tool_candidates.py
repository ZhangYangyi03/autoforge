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
    hits = agent._task_tool_candidates("git repository status branch")
    assert hits and hits[0] == "git_repo_status"
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
        if "/search?" in url:
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
        if "/search?" in url:
            return Response({"results":[{"resource_id":"tool:local_tool","name":"local_tool","description":"x"}]})
        return Response({"state":"probation","contract":{"parameters":{"type":"object","properties":{}}}})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert agent._shared_market_task_candidates("local_tool task") == ["local_tool"]
    assert "shared_market_local_tool" not in specs


# -- the two real tasks that measured 0% reuse on 2026-11-13 ---------------
def test_colloquial_git_push_task_hits_the_existing_push_tool():
    """The operator's own words: 传一下刚才做好的最新版autoforge去github...

    The old term-overlap injector returned nothing because 传/推 never appear
    in a tool description and 刚才/最新版/是不是好了 dilute the query.
    """
    from autoforge.store import ToolStore
    from pathlib import Path
    db = Path.home() / "AppData" / "Local" / "autoforge" / "autoforge.db"
    agent = object.__new__(ForgeAgent)
    agent.store = ToolStore(str(db))
    agent.registry = SimpleNamespace(get=lambda name: object())
    hits = agent._task_tool_candidates(
        "传一下刚才做好的最新版autoforge去github，拿这个实践一下工具复用命中率是不是好了")
    assert hits, "colloquial git-push task must surface a reusable tool"
    assert "git_add_commit_push" in hits[:3], hits


def test_research_word_no_longer_fires_on_latest_version():
    """'最新版' is not a news/research request; it must not expose web search."""
    from autoforge.agent import ForgeAgent
    a = object.__new__(ForgeAgent)
    assert a._research_tool_candidates("传一下刚才做好的最新版autoforge去github") == []
    assert a._research_tool_candidates("查一下今天的新闻") == ["shared_web_search"]


def test_search_views_include_a_canonical_rewrite():
    views = ForgeAgent._task_search_views(
        object.__new__(ForgeAgent),
        "传一下刚才做好的最新版autoforge去github，拿这个实践一下工具复用命中率是不是好了")
    assert len(views) >= 2
    assert any("commit" in v and "push" in v for v in views), views
