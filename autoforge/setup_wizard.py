"""`auto setup` — the one-time wizard.

Same shape as `hermes setup`: pick a provider, answer a handful of questions,
and the answers are written where every later invocation finds them. Prompts
show the current value and accept it on a bare Enter, so a second run is just
Enter-Enter-Enter.

Safe to run with no terminal: without a tty it takes whatever the flags and
environment already say, saves that, and never blocks on a prompt. `auto setup`
in a pipe or a CI job is therefore a no-op or a scripted configure, never a hang.
"""
from __future__ import annotations

import getpass
import sys
from pathlib import Path

import requests

from . import configfile
from .core.llm import OpenAICompatClient
from .core.message import Message

# label, base_url, model, needs_key
PRESETS: list[tuple[str, str, str, bool]] = [
    ("aiping.cn gateway (hosted, needs an API key)",
     "https://aiping.cn/api/v1", "DeepSeek-V4.1-Flash", True),
    ("Ollama on this machine (local, no key)",
     "http://127.0.0.1:11434/v1", "qwen2.5:7b", False),
    ("Something else (any OpenAI-compatible endpoint)", "", "", False),
]

PROXIES = {"http": "socks5://127.0.0.1:9674", "https": "socks5://127.0.0.1:9674"}

_G, _D, _Y, _C, _R = "\033[32m", "\033[2m", "\033[33m", "\033[36m", "\033[0m"


def _tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_R}" if sys.stdout.isatty() else text


def _ask(prompt: str, current: str = "", *, secret: bool = False) -> str:
    """Prompt with an optional current value; a bare Enter keeps it.

    A secret prompt never echoes and never falls back to `current`, because the
    only thing we could echo is the mask — and storing the mask as the key is
    exactly the bug this avoids. Secrets return "" on a bare Enter and the
    caller decides what that means.
    """
    shown = "" if secret or not current else f" {_c(_D, '[' + current + ']')}"
    try:
        if secret:
            answer = getpass.getpass(f"  {prompt}{shown}: ")
        else:
            answer = input(f"  {prompt}{shown}: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return "" if secret else current
    answer = answer.strip()
    return answer if secret else (answer or current)


def _ask_choice(current: int = 0) -> int:
    print("  provider:")
    for i, (label, *_rest) in enumerate(PRESETS, 1):
        mark = "*" if i - 1 == current else " "
        print(f"   {mark} {i}) {label}")
    raw = _ask("choose 1-3", str(current + 1)).strip()
    try:
        n = int(raw)
    except ValueError:
        return current
    return n - 1 if 1 <= n <= len(PRESETS) else current


def _ask_yes(prompt: str, current: bool) -> bool:
    d = "Y/n" if current else "y/N"
    raw = _ask(f"{prompt} [{d}]").strip().lower()
    if not raw:
        return current
    return raw.startswith("y")


def _probe(base: str, model: str, key: str, use_proxy: bool, max_tokens: int) -> tuple[bool, str]:
    """One tiny round-trip. Returns (ok, human-readable detail).

    The message must be a ``Message``: the client serialises with ``m.to_api()``,
    so a raw dict reaches ``AttributeError`` and every endpoint -- including a
    perfectly good one -- gets reported as unreachable. That is the worst
    possible failure here, because the wizard is the first thing anyone runs and
    its verdict is what tells them whether the rest will work.
    """
    client = OpenAICompatClient(
        model=model, base_url=base, api_key=key or "none", timeout=60.0,
        proxies=PROXIES if use_proxy else None, default_max_tokens=max_tokens,
    )
    try:
        out = client.chat([Message.user("Reply with the single word: ready")])
    except requests.HTTPError as exc:
        # A key is provider-specific. Switching provider in the wizard offers the
        # stored key on a bare Enter, so the commonest failure here is a key that
        # belongs to the endpoint you just moved away from -- and a bare "401"
        # gives the user nothing to act on.
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            return False, (f"{status}: the endpoint rejected the key. Keys are "
                           f"provider-specific -- an existing key is reused when "
                           f"you switch provider, so enter this provider's key.")
        return False, f"HTTPError: {exc}"
    except Exception as exc:                                          # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    # A 200 carrying no content is not a working endpoint. A reasoning model can
    # spend the entire budget on reasoning_content and return an empty answer;
    # calling that "ok" would send the user away with settings that never
    # produce text, which is exactly what the probe exists to catch.
    text = (out.content or "").strip()
    if not text:
        return False, "the endpoint answered but sent no content — " + out.describe_shortfall()
    return True, f"model replied {text.replace(chr(10), ' ')[:60]!r}"


def run(args) -> int:
    """Entry point for `auto setup`. Returns a process exit code."""
    existing = configfile.load()

    # A default is offered, never imposed: whatever is already configured is
    # what the prompts show, so re-running the wizard is how you change one
    # field without retyping the rest.
    base = args.base_url or existing.get("base_url") or PRESETS[0][1]
    model = args.model or existing.get("model") or PRESETS[0][2]
    key = args.api_key or existing.get("api_key") or ""
    max_tokens = args.max_tokens or int(existing.get("max_tokens") or 3000)
    proxy_default = bool(existing.get("proxy", not any(
        h in base for h in ("127.0.0.1", "localhost"))))
    # An explicit flag must survive into a non-interactive run too, where there
    # is no prompt to override the default it computed.
    if getattr(args, "no_proxy", False) or getattr(args, "proxy", None) == "0":
        proxy_default = False
    elif getattr(args, "proxy", None) == "1":
        proxy_default = True

    if not _tty():
        return _save_quietly(base, model, key, max_tokens, proxy_default)

    print(_c(_C, "\nautoforge setup"))
    print(_c(_D, "answers are written to " + str(configfile.config_path())
                 + " — Enter keeps the shown value\n"))

    idx = 0
    for i, (_label, pbase, _pmodel, _needs) in enumerate(PRESETS):
        if base == pbase:
            idx = i
    choice = _ask_choice(idx)
    label, pbase, pmodel, needs_key = PRESETS[choice]
    if pbase:
        base = pbase
        model = model or pmodel
        print(_c(_D, f"  -> {label}"))

    base = _ask("base URL", base)
    model = _ask("model", model)

    local = any(h in base for h in ("127.0.0.1", "localhost"))
    if needs_key or not local:
        if key:
            print(_c(_D, f"  stored key: {configfile.mask(key)}"))
            keep = " (Enter keeps it)" if key else ""
        else:
            keep = ""
        key = _ask(f"API key{keep}", secret=True) or key
    if not key and not local:
        print(_c(_Y, "\n  no key given; saving anyway — `auto` will ask again "
                      "until one is set"))

    try:
        max_tokens = int(_ask("max_tokens (generator output cap)", str(max_tokens)))
    except ValueError:
        pass

    use_proxy = _ask_yes("route through the socks5 proxy at 127.0.0.1:9674?",
                         proxy_default)
    if local:
        use_proxy = False

    print(_c(_D, "\n  testing the endpoint (one short request) ..."))
    ok, detail = _probe(base, model, key, use_proxy, max_tokens)
    print(("  " + _c(_G, "ok  ") if ok else "  " + _c(_Y, "fail ")) + detail)
    if not ok and not _ask_yes("save these settings anyway?", True):
        print("  nothing written")
        return 1

    path = configfile.save({"base_url": base, "model": model, "api_key": key,
                            "max_tokens": max_tokens, "proxy": use_proxy})
    _report(path, base, model, key, max_tokens, use_proxy, ok)
    return 0


def _save_quietly(base, model, key, max_tokens, use_proxy) -> int:
    """No terminal: take what we have, write it, and say exactly what was taken.

    There is no prompt to read here, so the flags are the only way to set a value
    explicitly. Print the effective set — a silent write of values the caller
    never chose is how a config file ends up lying about what it holds.
    """
    if any(h in base for h in ("127.0.0.1", "localhost")):
        use_proxy = False
    path = configfile.save({"base_url": base, "model": model, "api_key": key,
                            "max_tokens": max_tokens, "proxy": use_proxy})
    print(f"wrote {path}")
    print(_c(_D, "  no terminal: prompts skipped; values kept as they were. "
                 "Set them with flags:"))
    print(_c(_D, "  --base-url / --model / --api-key / --max-tokens / --no-proxy"))
    print(f"  base_url {base}   model {model}   max_tokens {max_tokens}   "
          f"proxy {use_proxy}   api_key {configfile.mask(key)}")
    return 0


def _report(path: Path, base, model, key, max_tokens, use_proxy, tested) -> None:
    print(_c(_G, f"\n  saved to {path}"))
    print(f"    base_url    {base}")
    print(f"    model       {model}")
    print(f"    api_key     {configfile.mask(key)}")
    print(f"    max_tokens  {max_tokens}")
    print(f"    proxy       {use_proxy}")
    if not tested:
        print(_c(_Y, "    (the endpoint did not answer — settings saved unverified)"))
    print(_c(_D, "\n  `auto` now works in this and every new terminal. "
                 "Flags still win over the file when you need a one-off.\n"))
