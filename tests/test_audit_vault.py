"""Vault, policy engine and audit — measured, not described.

The three modules ported from `_vendor/agentbox` share one claim: a decision
should be refusable *and* readable afterwards. Each test here is a claim that
could be false, and the ones that matter are the negative ones — a wrong
passphrase opening the vault, a tampered ciphertext decrypting, an undeclared
credential resolving, an unlisted host reaching the network. A test that only
checks the happy path would pass just as well against a module that does
nothing.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from autoforge import audit, chaining
from autoforge.forge import capability_policy as cp
from autoforge.vault import (Vault, VaultError, VaultLocked, VaultUnavailable,
                             get_secret, migrate_from_config)

PASS = "correct horse battery staple"


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("AUTOFORGE_VAULT_PASSPHRASE", PASS)
    monkeypatch.delenv("AUTOFORGE_VAULT_KEYFILE", raising=False)
    monkeypatch.delenv("AUTOFORGE_CONFIG", raising=False)
    return Vault()


def test_round_trip_and_rotation(vault):
    vault.add("api_key", "QC-secret-1234567890")
    assert vault.get("api_key") == "QC-secret-1234567890"
    vault.add("api_key", "QC-rotated")
    assert vault.get("api_key") == "QC-rotated"
    assert vault.names() == ["api_key"]


def test_plaintext_never_reaches_the_file(vault):
    vault.add("api_key", "QC-secret-1234567890")
    on_disk = vault.path.read_text(encoding="utf-8")
    assert "QC-secret-1234567890" not in on_disk
    assert json.loads(on_disk)["kdf"]["iterations"] >= 100_000


def test_wrong_passphrase_does_not_open(vault, monkeypatch):
    vault.add("api_key", "QC-secret")
    monkeypatch.setenv("AUTOFORGE_VAULT_PASSPHRASE", "not the passphrase")
    with pytest.raises(VaultLocked):
        Vault().get("api_key")


def test_tampered_ciphertext_does_not_open(vault):
    import base64
    vault.add("api_key", "QC-secret")
    data = json.loads(vault.path.read_text(encoding="utf-8"))
    raw = bytearray(base64.b64decode(data["credentials"][0]["ciphertext"]))
    raw[3] ^= 1                                     # one flipped bit
    data["credentials"][0]["ciphertext"] = base64.b64encode(bytes(raw)).decode()
    vault.path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(VaultLocked):
        Vault().get("api_key")


def test_ciphertext_moved_to_another_name_does_not_open(vault):
    """AAD binding: the name is part of what is authenticated."""
    vault.add("api_key", "QC-secret")
    vault.add("other", "unrelated")
    data = json.loads(vault.path.read_text(encoding="utf-8"))
    a = next(e for e in data["credentials"] if e["name"] == "api_key")
    b = next(e for e in data["credentials"] if e["name"] == "other")
    b["ciphertext"], b["nonce"] = a["ciphertext"], a["nonce"]
    vault.path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(VaultLocked):
        Vault().get("other")


def test_no_passphrase_source_refuses_rather_than_guessing(tmp_path, monkeypatch):
    """Also pins the ordering: no passphrase is reported as no passphrase.

    Reporting a missing *file* here would send the reader to look at a vault that
    is supposed to be missing.
    """
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.delenv("AUTOFORGE_VAULT_PASSPHRASE", raising=False)
    monkeypatch.setenv("AUTOFORGE_VAULT_KEYFILE", str(tmp_path / "absent.key"))
    with pytest.raises(VaultUnavailable):
        Vault().key()


def test_a_missing_vault_is_not_an_empty_password_vault(tmp_path, monkeypatch):
    """Regression: deriving against a salt that was never saved.

    `key()` used to invent a salt when the file was absent. The derive succeeded,
    and the salt that produced the key existed only in memory -- so the first
    write put a *different* salt on disk and the same process could no longer
    decrypt what it had just written. A vault that cannot read itself is worse
    than no vault, because the key is already gone from the config.
    """
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("AUTOFORGE_VAULT_PASSPHRASE", PASS)
    monkeypatch.delenv("AUTOFORGE_VAULT_KEYFILE", raising=False)
    with pytest.raises((VaultLocked, VaultUnavailable)):
        Vault().key()


def test_the_same_process_can_read_back_what_it_wrote(tmp_path, monkeypatch):
    """The other half of the same bug: one instance, add then get."""
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("AUTOFORGE_VAULT_PASSPHRASE", PASS)
    monkeypatch.delenv("AUTOFORGE_VAULT_KEYFILE", raising=False)
    v = Vault()
    v.add("api_key", "QC-first-write")
    assert v.get("api_key") == "QC-first-write"


def test_missing_credential_is_an_error_not_an_empty_string(vault):
    with pytest.raises(VaultError):
        vault.get("nope")


def test_secret_falls_back_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("AUTOFORGE_VAULT_PASSPHRASE", PASS)
    value, source = get_secret("api_key", fallback="plain")
    assert (value, source) == ("plain", "fallback")
    Vault().add("api_key", "from-vault")
    assert get_secret("api_key", fallback="plain") == ("from-vault", "vault")


def test_the_vault_outranks_a_stale_plaintext_copy(tmp_path, monkeypatch):
    """Vault first, config second. The reverse loses to a stale plaintext key."""
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("AUTOFORGE_VAULT_PASSPHRASE", PASS)
    Vault().add("api_key", "rotated-in-vault")
    assert get_secret("api_key", fallback="stale-plaintext")[0] == "rotated-in-vault"


def test_migration_moves_the_key_and_leaves_the_round_trip_provable(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("AUTOFORGE_VAULT_PASSPHRASE", PASS)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"api_key": "QC-plain", "model": "m"}), encoding="utf-8")
    report = migrate_from_config(cfg)
    assert report["ok"] and report["verified"] and report["moved"] == ["api_key"]
    after = json.loads(cfg.read_text(encoding="utf-8"))
    assert "api_key" not in after and after["api_key_vault"] == "api_key"
    assert after["model"] == "m"
    assert Vault().get("api_key") == "QC-plain"
    assert Path(report["backup"]).exists()
    assert "QC-plain" in Path(report["backup"]).read_text(encoding="utf-8")


# -- policy --------------------------------------------------------------

def engine() -> cp.PolicyEngine:
    return cp.PolicyEngine(cp.Permissions(
        filesystem_read=["D:/repos/mine", "~/data"],
        filesystem_write=["D:/repos/mine"],
        filesystem_deny=["D:/repos/mine/secret"],
        network_allow=["api.aiping.cn", "*.example.com"],
        network_deny=["evil.example.com"],
        tools_allow=["market_*", "read_file"],
        tools_deny=["market_delete*"],
        credentials=["api_key"],
        alert_on=["policy_denied"],
    ))


def test_filesystem_denies_by_default():
    e = engine()
    assert e.check_filesystem("read", r"D:\repos\mine\a.py").allowed
    assert not e.check_filesystem("read", r"D:\repos\mine\secret\x").allowed
    assert not e.check_filesystem("read", r"D:\elsewhere\x").allowed
    assert e.check_filesystem("write", "D:/repos/mine/new.py").allowed
    assert not e.check_filesystem("write", "D:/elsewhere/new.py").allowed


def test_separators_and_tilde_are_normalised():
    assert cp.path_matches(r"D:\repos\mine\a.py", "D:/repos/mine")
    assert cp.path_matches("~/data/x", "~/data")
    assert not cp.path_matches(r"D:\repos\other", "D:/repos/mine")


def test_deny_beats_allow():
    e = engine()
    assert not e.check_network("evil.example.com").allowed      # denied + wildcard-allowed
    assert not e.check_tool("market_delete_all").allowed        # denied + `market_*`


def test_network_allow_list_is_closed():
    e = engine()
    assert e.check_network("api.aiping.cn:443").allowed         # port stripped
    assert e.check_network("x.example.com").allowed
    assert not e.check_network("google.com").allowed
    assert not cp.PolicyEngine(cp.Permissions()).check_network("anything").allowed


def test_credentials_are_enumeration_only():
    e = engine()
    assert e.check_credential("api_key").allowed
    assert not e.check_credential("openai_key").allowed
    assert not e.check_credential("api_key_2").allowed          # no prefix matching


def test_every_decision_carries_a_rule():
    e = engine()
    for d in (e.check_filesystem("read", r"D:\x"), e.check_network("h"),
              e.check_tool("t"), e.check_credential("c")):
        assert d.rule and d.reason


def test_from_manifest_does_not_invent_a_grant():
    class M:
        network = True          # wants the network...
        writes = False
        general_purpose = False
    e = cp.from_manifest(M(), roots=["D:/work"])
    # ...but the resource manifest cannot name *whom*, so nothing is allowed
    assert not e.check_network("anything.com").allowed
    assert e.check_filesystem("read", "D:/work/x").allowed
    assert not e.check_filesystem("write", "D:/work/x").allowed


def test_enforcement_table_separates_the_hard_from_the_soft():
    table = cp.enforcement()
    assert "kernel" in table["memory_cpu_processes"]
    assert "gated" in table["network"]


# -- audit ---------------------------------------------------------------

@pytest.fixture()
def conn(tmp_path):
    """A real store, not a bare sqlite file.

    `chaining.ensure_schema` only *alters* an existing `forge_events` -- it is
    the migration path, not the create path -- so a test that calls it against
    an empty database fails with "no such table". The table is created by
    `store.ToolStore`, which is what the agent itself uses, so these audit rows
    land in the same schema they land in in production.
    """
    from autoforge.store import ToolStore
    st = ToolStore(str(tmp_path / "chain.db"))
    yield st._conn
    st.close()


def test_decisions_land_on_the_existing_chain(conn):
    e = engine()
    for op, path in [("read", r"D:\repos\mine\ok.py"), ("read", r"D:\nope")]:
        audit.record_decision(conn, e.check_filesystem(op, path), resource=path)
    audit.record_decision(conn, e.check_network("google.com"), resource="google.com")
    verdict = chaining.verify_chain(conn)
    assert verdict.get("ok"), verdict
    s = audit.summary(conn)
    assert s["entries"] == 3 and s["denied"] == 2 and s["chained"] == 3


def test_denial_is_attributable_to_a_rule(conn):
    e = engine()
    for i in range(3):
        audit.record_decision(conn, e.check_network("google.com"),
                              resource="google.com", extra="attempt %d" % i)
    audit.record_decision(conn, e.check_tool("shell"), resource="shell")
    top = audit.by_rule(conn)
    assert top[0]["rule"] == "policy: default_deny" and top[0]["count"] == 4
    assert {r["rule"] for r in top} == {"policy: default_deny"}


def test_allowed_entries_are_not_reported_as_denials(conn):
    e = engine()
    audit.record_decision(conn, e.check_tool("read_file"), resource="read_file")
    assert audit.denials(conn) == []
    assert audit.summary(conn)["allowed"] == 1


def test_limit_breach_reads_like_a_refusal(conn):
    audit.record_limit_breach(conn, "memory", "600MB > 256MB", agent="t")
    rows = audit.denials(conn)
    assert rows and rows[0]["event_type"] == "limit_breached"
    assert rows[0]["rule"] == "limit: memory"


def test_export_uses_the_vendored_field_names(conn, tmp_path):
    e = engine()
    audit.record_decision(conn, e.check_network("google.com"), resource="google.com",
                          agent="autoforge")
    out = audit.export_ndjson(conn, tmp_path / "audit.ndjson")
    assert out["entries"] == 1
    row = json.loads(Path(out["path"]).read_text(encoding="utf-8").splitlines()[0])
    assert set(row) >= {"id", "timestamp", "agent", "event_type", "allowed",
                        "resource", "rule", "reason", "extra"}
    assert row["event_type"] == "network_deny" and row["allowed"] is False


def test_the_dimension_comes_from_the_caller_not_from_the_rule(conn):
    """`policy: default_deny` is the shared rule for all four dimensions.

    Guessing from it filed a network denial under "policy", which is a row
    nobody goes looking for when they ask "what is being denied outbound".
    """
    e = engine()
    audit.record_decision(conn, e.check_network("google.com"), resource="google.com")
    audit.record_decision(conn, e.check_tool("shell"), resource="shell")
    audit.record_decision(conn, e.check_credential("openai_key"), resource="openai_key")
    audit.record_decision(conn, e.check_filesystem("read", r"D:\nope"), resource=r"D:\nope")
    types = [r["event_type"] for r in audit.denials(conn)]
    assert types == ["network_deny", "tool_deny", "credential_deny", "filesystem_deny"]


def test_event_type_vocabulary_matches_the_go_source():
    assert "filesystem_deny" in audit.EVENT_TYPES
    assert "credential_allow" in audit.EVENT_TYPES
    assert "limit_breached" in audit.EVENT_TYPES


def test_the_credential_resolver_prefers_the_vault(tmp_path, monkeypatch):
    """flag > env > vault > config, and the source is recorded either way.

    This is the whole point of the module reaching production: until the
    resolver consults the vault, moving the key into it changes nothing except
    where the copy nobody reads lives.
    """
    import argparse

    from autoforge import cli

    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("AUTOFORGE_VAULT_PASSPHRASE", PASS)
    monkeypatch.delenv("AUTOFORGE_API_KEY", raising=False)
    monkeypatch.delenv("AIPING_API_KEY", raising=False)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"api_key": "QC-plaintext", "model": "M",
                               "base_url": "https://aiping.cn/api/v1"}),
                   encoding="utf-8")
    monkeypatch.setenv("AUTOFORGE_CONFIG", str(cfg))

    ns = argparse.Namespace(base_url=None, model=None, api_key=None,
                            max_tokens=None, proxy=None, fast=False,
                            policy=None, mode=None, no_proxy=False)
    resolved, src = cli._resolve(ns, strict=False)
    assert (src["api_key"], resolved["key"]) == ("config", "QC-plaintext")

    assert migrate_from_config(cfg)["ok"]
    resolved, src = cli._resolve(ns, strict=False)
    assert (src["api_key"], resolved["key"]) == ("vault", "QC-plaintext")

    # A rotation in the vault outranks the stale plaintext the migration left in
    # the backup-era config: that is the failure a vault is supposed to prevent,
    # so the resolver must not prefer the file it was easier to read.
    Vault().add("api_key", "QC-rotated")
    resolved, src = cli._resolve(ns, strict=False)
    assert src["api_key"] == "vault" and resolved["key"] == "QC-rotated"

    # A flag still wins: the escape hatch has to stay on top.
    ns.api_key = "QC-flag"
    resolved, src = cli._resolve(ns, strict=False)
    assert (src["api_key"], resolved["key"]) == ("flag", "QC-flag")
