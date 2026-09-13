"""Make every test hermetic about configuration.

Three leaks to close:

1. The developer's real `~/.autoforge/config.json`. Without this, tests that
   assert "a remote endpoint with no key must fail" start passing or failing
   depending on whether the person running them happens to have run
   `auto setup` — the classic works-on-my-machine failure.
2. Ambient `AUTOFORGE_*` / `AIPING_API_KEY` in the environment, for the same
   reason in the other direction.
3. The skill directories. These default to `<cwd>/skills` and
   `<home>/skills`, so a test run from the repo root would read — and, through
   skill_write, write — the project's own skills. A test that quietly edits the
   repository it is testing is worse than one that fails.
"""
from __future__ import annotations

import pytest

_AMBIENT = ("AUTOFORGE_BASE_URL", "AUTOFORGE_MODEL", "AUTOFORGE_API_KEY",
            "AIPING_API_KEY", "AUTOFORGE_FAST", "AUTOFORGE_MAX_TOKENS",
            "AUTOFORGE_CONFIG", "AUTOFORGE_HOME", "AUTOFORGE_SKILLS_DIRS")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point the config layer at a throwaway file and clear ambient overrides."""
    monkeypatch.setenv("AUTOFORGE_CONFIG", str(tmp_path / "config.json"))
    for name in _AMBIENT:
        if name != "AUTOFORGE_CONFIG":
            monkeypatch.delenv(name, raising=False)
    # Skills and state go under tmp_path, so a run cannot see the real ones.
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AUTOFORGE_SKILLS_DIRS",
                       str(tmp_path / "home" / "skills"))
    yield tmp_path / "config.json"
