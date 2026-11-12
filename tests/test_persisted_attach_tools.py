"""Regression: stores attached after construction must load saved tools."""
from autoforge.agent import ForgeAgent


class _Registry:
    last_called_source = None
    controls = None


class _Controls:
    def bind(self, conn):
        self.conn = conn


class _Store:
    _conn = object()


def test_attach_store_loads_persisted_tools():
    agent = object.__new__(ForgeAgent)
    agent.controls = _Controls()
    agent.registry = _Registry()
    loaded = []
    agent._load_persisted_tools = lambda: loaded.append(agent.store)
    store = _Store()
    agent.attach_store(store)
    assert loaded == [store]
    assert agent.registry.last_called_source == agent._last_called_at
