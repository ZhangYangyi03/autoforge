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
4. The autonomy presets. `FULL_FREEDOM` and `SUPERVISED` are module-level
   singletons, and `set_autonomy` writes an amendment onto the policy object
   the agent was handed. A test that builds its agent with the singleton and
   then amends it has tightened the preset for every test that runs after it in
   the same process — the failure surfaces as an innocent test of the preset
   going red in the full suite while passing on its own.
"""
from __future__ import annotations

import pytest

_AMBIENT = ("AUTOFORGE_BASE_URL", "AUTOFORGE_MODEL", "AUTOFORGE_API_KEY",
            "AIPING_API_KEY", "AUTOFORGE_FAST", "AUTOFORGE_MAX_TOKENS",
            "AUTOFORGE_CONFIG", "AUTOFORGE_HOME", "AUTOFORGE_SKILLS_DIRS")


@pytest.fixture(autouse=True)
def isolated_policy_presets():
    """Put the autonomy presets back the way they were after every test."""
    from dataclasses import fields

    from autoforge.autonomy.policy import FULL_FREEDOM, SUPERVISED

    presets = (FULL_FREEDOM, SUPERVISED)
    before = [{f.name: getattr(p, f.name) for f in fields(p)} for p in presets]
    yield
    for preset, snapshot in zip(presets, before):
        for name, value in snapshot.items():
            setattr(preset, name, value)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point the config layer at a throwaway file and clear ambient overrides."""
    monkeypatch.setenv("AUTOFORGE_CONFIG", str(tmp_path / "config.json"))
    for name in _AMBIENT:
        if name != "AUTOFORGE_CONFIG":
            monkeypatch.delenv(name, raising=False)
    # Peer shelves. Set to EMPTY rather than deleted: present-but-empty is the
    # documented way to say "no peers", and deleting it would let the lookup
    # fall through to ~/.autoforge/peers.json or to HKCU\Environment, where
    # `setx` puts it. That is not a hypothetical -- after the operator exported a
    # real peer with setx, three tests in test_market_prelookup.py failed on the
    # machine and would have passed anywhere else. A test whose result depends
    # on whose machine it runs on is worse than a failing one.
    monkeypatch.setenv("AUTOFORGE_PEER_MARKETS", "")
    monkeypatch.setattr("autoforge.agent.ForgeAgent._peer_market_from_registry",
                        staticmethod(lambda: ""), raising=False)

    # Skills and state go under tmp_path, so a run cannot see the real ones.
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AUTOFORGE_SKILLS_DIRS",
                       str(tmp_path / "home" / "skills"))
    yield tmp_path / "config.json"
