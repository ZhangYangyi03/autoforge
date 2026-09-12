"""`auto setup` and the config file it writes.

The contract worth testing is the precedence chain — flag > env > file > default
— and that the wizard is safe to invoke without a terminal. The prompts
themselves are exercised through a scripted answer queue so the exact sequence
(bare Enter must keep the stored key, not store the mask) stays pinned.
"""
from __future__ import annotations

import argparse
import json

import pytest

from autoforge import cli, configfile, setup_wizard


def _args(**kw) -> argparse.Namespace:
    base = dict(model=None, base_url=None, api_key=None, fast=False,
                max_tokens=None, no_proxy=False, proxy=None)
    base.update(kw)
    return argparse.Namespace(**base)


class _Queue:
    """Answers `input` and `getpass` from one script, recording every prompt."""

    def __init__(self, *values: str) -> None:
        self.values = list(values)
        self.prompts: list[str] = []

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self.values:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return self.values.pop(0)

    def exhausted(self) -> bool:
        return not self.values


@pytest.fixture
def interactive(monkeypatch):
    """Drive the wizard as if on a terminal.

    `probe` stubs the connection test so no test touches the network; pass
    `probe=(False, "reason")` to model an endpoint that will not answer.
    """
    def _install(*answers: str, probe=(True, "stubbed")):
        queue = _Queue(*answers)
        monkeypatch.setattr(setup_wizard, "_tty", lambda: True)
        monkeypatch.setattr("builtins.input", queue)
        monkeypatch.setattr(setup_wizard.getpass, "getpass", queue)
        monkeypatch.setattr(setup_wizard, "_probe", lambda *a, **k: probe)
        return queue
    return _install


# -- config file --------------------------------------------------------
def test_round_trip(isolated_config):
    configfile.save({"base_url": "http://x/v1", "model": "m"}, isolated_config)
    assert configfile.load(isolated_config) == {"base_url": "http://x/v1", "model": "m"}


def test_save_merges_and_keeps_unknown_keys(isolated_config):
    """A hand-edited file must survive a wizard run instead of being clobbered."""
    isolated_config.write_text(json.dumps({"note": "mine", "model": "old"}), encoding="utf-8")
    configfile.save({"model": "new"}, isolated_config)
    data = configfile.load(isolated_config)
    assert data == {"note": "mine", "model": "new"}


def test_none_values_are_not_written(isolated_config):
    configfile.save({"model": "m", "api_key": None}, isolated_config)
    assert "api_key" not in configfile.load(isolated_config)


def test_missing_and_corrupt_files_are_empty(isolated_config):
    assert configfile.load(isolated_config) == {}
    isolated_config.write_text("{ this is not json", encoding="utf-8")
    assert configfile.load(isolated_config) == {}


def test_config_path_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFORGE_CONFIG", str(tmp_path / "elsewhere.json"))
    assert configfile.config_path() == tmp_path / "elsewhere.json"


def test_mask_never_reveals_the_key():
    assert configfile.mask("") == "(unset)"
    assert configfile.mask("short") == "*****"
    masked = configfile.mask("QC-5aflongsecretvalue000000")
    assert "longsecret" not in masked and "secretvalue" not in masked
    assert masked.startswith("QC-5")
    assert "0000 (len 27)" in masked            # 4-char tail plus length, nothing more


# -- precedence ---------------------------------------------------------
def test_config_file_is_used_when_env_is_absent():
    configfile.save({"base_url": "http://127.0.0.1:11434/v1", "model": "from-file",
                     "api_key": "file-key", "max_tokens": 4096})
    cfg, src = cli._resolve(_args())
    assert cfg["model"] == "from-file" and cfg["key"] == "file-key"
    assert cfg["max_tokens"] == 4096
    assert src["model"].startswith("config")


def test_env_beats_file(monkeypatch):
    configfile.save({"model": "from-file", "api_key": "file-key"})
    monkeypatch.setenv("AUTOFORGE_MODEL", "from-env")
    cfg, src = cli._resolve(_args())
    assert cfg["model"] == "from-env"
    assert src["model"] == "env AUTOFORGE_MODEL"


def test_flag_beats_env_and_file(monkeypatch):
    configfile.save({"model": "from-file", "api_key": "file-key",
                     "base_url": "https://aiping.cn/api/v1"})
    monkeypatch.setenv("AUTOFORGE_MODEL", "from-env")
    cfg, src = cli._resolve(_args(model="from-flag"))
    assert cfg["model"] == "from-flag" and src["model"] == "flag"


def test_stored_proxy_setting_is_respected():
    """A saved proxy choice must apply to a remote endpoint, not be re-derived."""
    configfile.save({"base_url": "https://aiping.cn/api/v1", "api_key": "k",
                     "proxy": False})
    assert cli._resolve(_args())[0]["proxy"] is False


def test_local_endpoint_never_uses_the_proxy():
    configfile.save({"base_url": "http://127.0.0.1:11434/v1", "proxy": True})
    assert cli._resolve(_args())[0]["proxy"] is False


def test_missing_key_still_refuses_strictly_but_reports_when_not(monkeypatch):
    monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://example.invalid/v1")
    with pytest.raises(SystemExit, match="auto setup"):
        cli._resolve(_args())
    cfg, src = cli._resolve(_args(), strict=False)
    assert cfg["key"] == "" and "unset" in src["api_key"]


# -- the wizard ---------------------------------------------------------
def test_non_interactive_never_prompts(monkeypatch, isolated_config):
    """`auto setup` in a pipe must write and exit, not hang on stdin."""
    monkeypatch.setattr(setup_wizard, "_tty", lambda: False)
    monkeypatch.setattr("builtins.input",
                        lambda *a: pytest.fail("prompted without a tty"))
    rc = setup_wizard.run(_args(base_url="https://aiping.cn/api/v1", api_key="sk-x"))
    assert rc == 0
    data = configfile.load(isolated_config)
    assert data["base_url"] == "https://aiping.cn/api/v1" and data["api_key"] == "sk-x"


def test_non_interactive_honours_the_no_proxy_flag(monkeypatch, isolated_config):
    """A flag has nowhere to be overridden without a prompt, so it must win."""
    monkeypatch.setattr(setup_wizard, "_tty", lambda: False)
    setup_wizard.run(_args(base_url="https://aiping.cn/api/v1", api_key="sk-x",
                           no_proxy=True))
    assert configfile.load(isolated_config)["proxy"] is False


def test_wizard_writes_what_was_typed(interactive, isolated_config):
    queue = interactive("1", "", "", "sk-typed", "2048", "n")
    assert setup_wizard.run(_args()) == 0
    data = configfile.load(isolated_config)
    assert data["base_url"] == "https://aiping.cn/api/v1"
    assert data["api_key"] == "sk-typed"
    assert data["max_tokens"] == 2048
    assert data["proxy"] is False
    assert queue.exhausted()


def test_bare_enter_keeps_the_stored_key(interactive, isolated_config):
    """The regression that matters: the prompt shows a mask, and Enter must not
    store that mask as the key."""
    configfile.save({"api_key": "sk-original", "base_url": "https://aiping.cn/api/v1"})
    queue = interactive("1", "", "", "", "3000", "")   # empty getpass = keep
    assert setup_wizard.run(_args()) == 0
    assert configfile.load(isolated_config)["api_key"] == "sk-original"
    assert queue.exhausted()


def test_rerunning_offers_the_stored_values(interactive, isolated_config):
    configfile.save({"base_url": "https://my.gateway/v1", "model": "my-model",
                     "api_key": "sk-1"})
    queue = interactive("3", "", "", "", "3000", "")
    assert setup_wizard.run(_args()) == 0
    data = configfile.load(isolated_config)
    assert data["base_url"] == "https://my.gateway/v1"
    assert data["model"] == "my-model"


def test_unverified_settings_are_saved_when_the_user_insists(interactive, isolated_config):
    queue = interactive("1", "", "", "sk-typed", "3000", "", "y",
                        probe=(False, "HTTP 503"))
    assert setup_wizard.run(_args()) == 0
    assert configfile.load(isolated_config)["api_key"] == "sk-typed"
    assert queue.exhausted()


def test_a_failing_probe_can_abort_without_writing(interactive, isolated_config):
    queue = interactive("1", "", "", "sk-typed", "3000", "", "n",
                        probe=(False, "HTTP 503"))
    assert setup_wizard.run(_args()) == 1
    assert configfile.load(isolated_config) == {}
    assert queue.exhausted()


def test_a_blank_max_tokens_does_not_crash(interactive, isolated_config):
    queue = interactive("1", "", "", "sk-typed", "not-a-number", "")
    assert setup_wizard.run(_args()) == 0
    assert isinstance(configfile.load(isolated_config)["max_tokens"], int)
    assert queue.exhausted()


# -- cli surface --------------------------------------------------------
def test_parser_exposes_setup_and_config():
    names = cli.build_parser()._subparsers._group_actions[0].choices
    assert {"chat", "forge", "list", "setup", "config"} <= set(names)


def test_setup_flags_reach_the_wizard(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.setup_wizard, "run", lambda a: seen.update(vars(a)) or 0)
    assert cli.main(["setup", "--api-key", "sk-z", "--model", "m1"]) == 0
    assert seen["api_key"] == "sk-z" and seen["model"] == "m1"


def test_config_command_shows_sources(capsys):
    configfile.save({"base_url": "https://aiping.cn/api/v1", "model": "m",
                     "api_key": "sk-secretvalue1234"})
    assert cli.main(["config"]) == 0
    out = capsys.readouterr().out
    assert "api_key" in out
    assert "sk-secretvalue1234" not in out       # masked
    assert str(configfile.config_path()) in out  # tells you which file


def test_config_command_survives_having_no_key(capsys):
    assert cli.main(["config"]) == 0
    assert "auto setup" in capsys.readouterr().out
