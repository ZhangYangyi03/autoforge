"""Make every test hermetic about configuration.

Two leaks to close:

1. The developer's real `~/.autoforge/config.json`. Without this, tests that
   assert "a remote endpoint with no key must fail" start passing or failing
   depending on whether the person running them happens to have run
   `auto setup` — the classic works-on-my-machine failure.
2. Ambient `AUTOFORGE_*` / `AIPING_API_KEY` in the environment, for the same
   reason in the other direction.
"""
from __future__ import annotations

import pytest

_AMBIENT = ("AUTOFORGE_BASE_URL", "AUTOFORGE_MODEL", "AUTOFORGE_API_KEY",
            "AIPING_API_KEY", "AUTOFORGE_FAST", "AUTOFORGE_MAX_TOKENS",
            "AUTOFORGE_CONFIG")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point the config layer at a throwaway file and clear ambient overrides."""
    monkeypatch.setenv("AUTOFORGE_CONFIG", str(tmp_path / "config.json"))
    for name in _AMBIENT:
        if name != "AUTOFORGE_CONFIG":
            monkeypatch.delenv(name, raising=False)
    yield tmp_path / "config.json"
