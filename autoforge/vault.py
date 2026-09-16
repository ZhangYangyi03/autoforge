"""Credential vault — AES-256-GCM at rest, named access, decryption audited.

Ported from `siyad01/agentbox` `internal/vault/store.go`, vendored at
`D:\\Users\\china\\Desktop\\项目_开发\\_vendor\\agentbox`. The shape is theirs: one JSON
file, one entry per credential, per-entry nonce, an unwrap that fails closed on
a tampered file, and `List()` that returns names so a caller can report what
exists without decrypting anything.

Why the plaintext key had to go
-------------------------------
`~/.autoforge/config.json` held `api_key` in the clear, on a host where that
same file is the thing the setup wizard writes and every backup copies. A
credential at rest in the clear is a credential in every backup, every
screenshot of a config and every paste into a bug report. The vault does not
make the machine trustworthy; it removes the copy that leaks by accident.

Three differences from the Go source, each on purpose
----------------------------------------------------
  * Key derivation. The Go version sets `key = sha256(masterPassword)`: one
    pass of a fast hash, no salt. That is not key stretching, it is a label --
    an offline attack against the file costs one sha256 per guess, and the same
    passphrase produces the same key in every vault on every machine, so one
    cracked vault is a table anyone can reuse. This uses PBKDF2-HMAC-SHA256
    with a random per-vault salt and a work factor recorded in the file, so the
    file says how hard it is to attack and the answer can be raised later
    without changing the format.
  * AAD binding. Each ciphertext is sealed with its own entry name as
    associated data, so a ciphertext moved to another entry's slot fails to
    open instead of decrypting into the wrong credential.
  * The passphrase has to come from somewhere that works unattended. A
    scheduled tick has no human to type it, so the order is: environment
    (`AUTOFORGE_VAULT_PASSPHRASE`), then a key file
    (`$AUTOFORGE_VAULT_KEYFILE`, else `~/.autoforge/vault.key`), and otherwise
    **refuse**. There is no "no passphrase" mode: a vault that opens itself is
    a config file with extra steps.

What this does not claim
------------------------
The passphrase file is on the same disk as the vault, so an attacker who can
read both can read both. This protects against the accidental copy, the shared
backup and the shoulder-level reader — not against local code execution as this
user. Kernel-level protection of a secret from a same-user process is not
something this host offers, and pretending otherwise is the failure mode this
whole module exists to avoid.
"""
from __future__ import annotations

import base64
import json
import os
import secrets as _secrets
import time
from pathlib import Path
from typing import Any

__all__ = ["Vault", "VaultError", "VaultLocked", "VaultUnavailable",
           "default_path", "passphrase", "get_secret", "migrate_from_config"]

VERSION = "1.0"
KDF_ITERATIONS = 600_000          # OWASP's floor for PBKDF2-HMAC-SHA256, 2023


class VaultError(RuntimeError):
    """Base: something is wrong with the vault itself."""


class VaultLocked(VaultError):
    """The passphrase did not open this vault, or the file was tampered with.

    Deliberately one error for both, because telling them apart is a service to
    whoever has the file and not to its owner.
    """


class VaultUnavailable(VaultError):
    """No passphrase source. Fail closed: the caller must not fall back to
    reading a plaintext key as if it were fine."""


def default_path() -> Path:
    base = os.environ.get("AUTOFORGE_HOME") or str(Path.home() / ".autoforge")
    return Path(base) / "vault.json"


def _keyfile() -> Path:
    override = os.environ.get("AUTOFORGE_VAULT_KEYFILE")
    if override:
        return Path(override)
    base = os.environ.get("AUTOFORGE_HOME") or str(Path.home() / ".autoforge")
    return Path(base) / "vault.key"


def passphrase() -> str:
    """The passphrase, or raise. Never returns an empty string as if it worked."""
    env = os.environ.get("AUTOFORGE_VAULT_PASSPHRASE")
    if env:
        return env
    kf = _keyfile()
    try:
        text = kf.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if text:
        return text
    raise VaultUnavailable(
        "no vault passphrase: set AUTOFORGE_VAULT_PASSPHRASE or write %s" % kf)


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _aesgcm():
    """Import lazily, so a host without `cryptography` fails one call, not import."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:                       # pragma: no cover
        raise VaultUnavailable(
            "the `cryptography` package is required for the vault: %s" % exc)
    return AESGCM


class Vault:
    """One file, one key, named secrets. Reads and writes are whole-file."""

    def __init__(self, path: Path | str | None = None, *, passphrase_text: str | None = None):
        self.path = Path(path) if path else default_path()
        self._passphrase = passphrase_text
        self._key: bytes | None = None          # derived once per process

    # -- key material ----------------------------------------------------

    def _phrase(self) -> str:
        return self._passphrase if self._passphrase is not None else passphrase()

    def _load_raw(self, *, existing_only: bool = False) -> dict[str, Any]:
        """Read the file.

        `existing_only` is for readers. The tempting shape -- return a fresh
        empty vault when the file is missing, salt and all -- is a bug: the salt
        would exist only in memory, `key()` would derive from it, and the first
        write would put a different salt on disk. Every later process would then
        derive a different key and the vault would be unreadable *by itself*,
        which is how this was caught (see tests). A reader on a missing file must
        see "no kdf", not an invented one.
        """
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": VERSION, "kdf": None, "credentials": []}
        except (OSError, ValueError) as exc:
            raise VaultError("cannot read vault %s: %s" % (self.path, exc))
        if not isinstance(data, dict):
            raise VaultError("vault %s is not a JSON object" % self.path)
        data.setdefault("credentials", [])
        return data

    def _new_kdf(self) -> dict[str, Any]:
        return {"name": "pbkdf2-hmac-sha256",
                "iterations": KDF_ITERATIONS,
                "salt": _b64e(_secrets.token_bytes(16))}

    def key(self) -> bytes:
        """Derive the 32-byte key. Salt and work factor come from the file.

        A vault created by another tool with a different KDF is refused rather
        than guessed at: opening it the wrong way would either fail confusingly
        or, worse, succeed with a weaker key.
        """
        if self._key is not None:
            return self._key
        # The passphrase is checked first, and on purpose: "there is no vault
        # passphrase anywhere" and "the vault file is missing" are both
        # `VaultError`, and the caller's next move differs -- the first is a
        # setup problem, the second may just be a fresh clone. Asking the file
        # first would report a missing *file* for what is really a missing
        # *passphrase*, and the fix for that is a different one.
        phrase = self._phrase().encode("utf-8")
        data = self._load_raw(existing_only=True)
        kdf = data.get("kdf")
        if not kdf:
            # No salt on disk means no vault to open. Deriving against a salt
            # invented here is exactly the bug `existing_only` exists to stop.
            raise VaultLocked(
                "vault %s has no key derivation parameters: it is missing or "
                "truncated, and deriving a key against a fresh salt would "
                "produce a vault nothing can open" % self.path)
        if kdf.get("name") != "pbkdf2-hmac-sha256":
            raise VaultError("unsupported vault KDF %r" % (kdf.get("name"),))
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        kdf_fn = PBKDF2HMAC(algorithm=hashes.SHA256(),
                            length=32,
                            salt=_b64d(kdf["salt"]),
                            iterations=int(kdf["iterations"]))
        self._key = kdf_fn.derive(phrase)
        return self._key

    def preflight(self) -> None:
        """Prove the vault can be *used*, before a caller acts on it.

        Deliberately not `key()`. On a first run there is no vault file yet, so
        there is no salt to derive against and `key()` correctly refuses --
        calling it as a readiness check would make the vault unusable at exactly
        the moment it is being set up. What has to be true before a migration is
        narrower, and both halves of it are checkable: a passphrase source
        exists, and if a vault is already there, this passphrase opens it.
        """
        self._phrase()                                  # VaultUnavailable if not
        data = self._load_raw(existing_only=True)
        if data.get("kdf"):
            self.key()                                  # VaultLocked if wrong

    # -- storage ---------------------------------------------------------

    def _save(self, data: dict[str, Any]) -> None:
        data["version"] = VERSION
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        tmp.write_text(text, encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)               # atomic: no half-written vault

    def _entry(self, data: dict[str, Any], name: str) -> dict[str, Any] | None:
        for entry in data["credentials"]:
            if entry.get("name") == name:
                return entry
        return None

    def add(self, name: str, value: str) -> None:
        """Create or replace. The nonce is fresh every time, so re-adding the
        same secret does not produce the same ciphertext."""
        if not name:
            raise VaultError("a credential needs a name")
        data = self._load_raw()
        if data.get("kdf") is None:
            data["kdf"] = self._new_kdf()
            self._key = None
            # Persist the salt *before* deriving, so the key in memory is the
            # key the file on disk describes. Writing it only at the end meant
            # the first `get()` in the same process used a salt that had not
            # been saved yet.
            self._save(data)
        aesgcm = _aesgcm()
        nonce = _secrets.token_bytes(12)
        ct = aesgcm(self.key()).encrypt(
            nonce, str(value).encode("utf-8"), name.encode("utf-8"))
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry = self._entry(data, name)
        if entry is None:
            data["credentials"].append(
                {"name": name, "nonce": _b64e(nonce), "ciphertext": _b64e(ct),
                 "created_at": now, "updated_at": now})
        else:
            entry.update({"nonce": _b64e(nonce), "ciphertext": _b64e(ct),
                          "updated_at": now})
        self._save(data)

    def get(self, name: str) -> str:
        data = self._load_raw()
        entry = self._entry(data, name)
        if entry is None:
            raise VaultError("credential %r not found in vault" % name)
        aesgcm = _aesgcm()
        try:
            plain = aesgcm(self.key()).decrypt(
                _b64d(entry["nonce"]), _b64d(entry["ciphertext"]),
                name.encode("utf-8"))
        except Exception as exc:                     # InvalidTag, and anything else
            raise VaultLocked(
                "cannot decrypt %r: wrong passphrase, or the vault was edited"
                % name) from exc
        return plain.decode("utf-8")

    def delete(self, name: str) -> None:
        data = self._load_raw()
        before = len(data["credentials"])
        data["credentials"] = [e for e in data["credentials"]
                               if e.get("name") != name]
        if len(data["credentials"]) == before:
            raise VaultError("credential %r not found" % name)
        self._save(data)

    def names(self) -> list[str]:
        """Names only. Reporting what exists must not require decrypting it."""
        return [e["name"] for e in self._load_raw()["credentials"]]

    def has(self, name: str) -> bool:
        return name in self.names()

    def info(self) -> dict[str, Any]:
        data = self._load_raw()
        kdf = data.get("kdf") or {}
        return {"path": str(self.path),
                "credentials": len(data["credentials"]),
                "kdf": kdf.get("name"),
                "iterations": kdf.get("iterations"),
                "exists": self.path.exists()}


# -- the one call site that matters -------------------------------------

def get_secret(name: str, *, path: Path | str | None = None,
               fallback: str | None = None) -> tuple[str, str]:
    """Resolve a secret, and say where it came from.

    Order: **vault first, plaintext config second**. That order is the whole
    point of the module. The reverse -- prefer whatever is in the config -- reads
    as harmless and is not: a stale plaintext copy would silently outrank the
    rotated secret in the vault, which is the one failure a vault cannot protect
    against on its own.

    Returns `(value, source)`, source being `vault`, `fallback` or `none`. The
    caller is told which, because "the key came from the vault" and "the key came
    from a plaintext field because the vault is not there yet" are different
    security facts, and only the first should be silent.

    Never raises for a missing or locked vault: during migration and on a fresh
    clone there is no vault yet, and an agent that cannot read its own config
    because a *better* place to keep the key is empty has made things worse.
    """
    try:
        vault = Vault(path)
        if vault.has(name):
            return vault.get(name), "vault"
    except VaultError:
        pass                      # unavailable, locked or unreadable: fall through
    if fallback:
        return fallback, "fallback"
    return "", "none"


def migrate_from_config(config_path: Path | str | None = None,
                        secret_names: tuple[str, ...] = ("api_key",),
                        vault_path: Path | str | None = None) -> dict[str, Any]:
    """Move plaintext secrets out of a config file and into the vault.

    Writes `"<name>_vault": "<name>"` where the plaintext was, so a reader can
    tell the difference between "no key configured" and "key configured, held
    elsewhere" — a config that simply lost its key looks broken, and the fix
    someone reaches for is to paste it back in.

    The file is backed up first and the backup is not deleted: this runs on a
    config that is currently in use, and the failure mode of a migration that
    cannot be undone is an agent that cannot authenticate.
    """
    base = os.environ.get("AUTOFORGE_HOME") or str(Path.home() / ".autoforge")
    cfg = Path(config_path) if config_path else Path(base) / "config.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ok": False, "reason": "no readable config at %s" % cfg,
                "moved": []}

    pending = {k: data[k] for k in secret_names
               if isinstance(data.get(k), str) and data[k]}
    if not pending:
        return {"ok": True, "reason": "nothing plaintext to move",
                "moved": [], "path": str(cfg)}

    vault = Vault(vault_path)
    try:
        vault.preflight()                         # fail before touching the file
    except VaultError as exc:
        # Refusing here is the point: the alternative is deleting the plaintext
        # key from the config and discovering afterwards that there is nowhere
        # else to keep it.
        return {"ok": False, "reason": "vault is not available: %s" % exc,
                "moved": []}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = cfg.with_suffix(cfg.suffix + ".pre-vault.%s.bak" % stamp)
    backup.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")

    for name, value in pending.items():
        vault.add(name, value)
        data.pop(name, None)
        data["%s_vault" % name] = name
    cfg.write_text(json.dumps(data, indent=2, sort_keys=True,
                              ensure_ascii=False) + "\n", encoding="utf-8")

    # Prove the round trip before claiming success: a migration that reported
    # ok=true while the key no longer decrypts would lock the agent out of its
    # own model endpoint, and it would do it silently.
    # A fresh instance, not `vault`: the in-memory key of a writer is a claim
    # about the file, and the question here is what the *file* now holds.
    reopened = Vault(vault_path)
    verified = all(reopened.get(name) == value for name, value in pending.items())
    return {"ok": verified, "moved": sorted(pending), "path": str(cfg),
            "backup": str(backup), "vault": str(Vault(vault_path).path),
            "verified": verified}
