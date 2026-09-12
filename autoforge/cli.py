"""Command-line entry point: `autoforge`, installed as `auto`.

    auto                    talk to the agent (bare invocation, on a terminal)
    auto setup              one-time wizard: provider, key, model -> config file
    auto config             show the effective settings and where each came from
    auto forge "<need>"     forge one tool, one shot, and stop
    auto list               show what the last forge produced

Global flags (also settable via environment, and persisted by `auto setup`):
    --model NAME        AUTOFORGE_MODEL       default DeepSeek-V4.1-Flash
    --base-url URL      AUTOFORGE_BASE_URL    default https://aiping.cn/api/v1
    --api-key KEY       AUTOFORGE_API_KEY     falls back to AIPING_API_KEY
    --fast              AUTOFORGE_FAST=1      drop the LLM-driven checks
    --max-tokens N      AUTOFORGE_MAX_TOKENS  default 3000
    --no-proxy          AUTOFORGE_PROXY=0     for local endpoints

Resolution order is flag > environment > ~/.autoforge/config.json > default.
The file is what `auto setup` writes, and it exists so that configuring this
once is enough: an environment variable set after a terminal was opened is
invisible to that terminal, which makes "it worked a minute ago" a real and
confusing failure. Run `auto config` to see which layer supplied what.

Everything the agent does is decided by its own policy; the flags here only
configure the model and how hard the verifier looks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import configfile, setup_wizard
from .agent import ForgeAgent
from .core.llm import OpenAICompatClient
from .forge.generator import LLMToolGenerator
from .forge.pipeline import ForgeConfig
from .forge.sandbox import Sandbox
from .forge.verifier import ToolVerifier
from .tools.registry import ToolRegistry

PROXIES = {"http": "socks5://127.0.0.1:9674", "https": "socks5://127.0.0.1:9674"}

_C = "\033[36m"
_D = "\033[2m"
_G = "\033[32m"
_Y = "\033[33m"
_R = "\033[0m"


def _tty() -> bool:
    return sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_R}" if _tty() else text


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
def _resolve(args: argparse.Namespace, *, strict: bool = True) -> tuple[dict, dict[str, str]]:
    """Merge flag > env > config file > default, recording where each value came from.

    The source map is not decoration: when a value is wrong, the first useful
    question is always "where did it come from", and the answer is
    unfalsifiable from the value alone. `auto config` prints it.

    `strict=False` reports a missing key instead of refusing to run, which is
    what `auto config` needs — the whole point there is to show you what is.
    """
    saved = configfile.load()
    src: dict[str, str] = {}

    def pick(field: str, flag, *env_names: str, default=None):
        if flag not in (None, ""):
            src[field] = "flag"
            return flag
        for name in env_names:
            value = os.environ.get(name)
            if value not in (None, ""):
                src[field] = f"env {name}"
                return value
        value = saved.get(field)
        if value not in (None, ""):
            src[field] = f"config {configfile.config_path()}"
            return value
        src[field] = "default"
        return default

    base = str(pick("base_url", args.base_url, "AUTOFORGE_BASE_URL",
                    default="https://aiping.cn/api/v1"))
    model = str(pick("model", args.model, "AUTOFORGE_MODEL",
                     default="DeepSeek-V4.1-Flash"))
    local = any(h in base for h in ("127.0.0.1", "localhost"))
    key = str(pick("api_key", args.api_key, "AUTOFORGE_API_KEY", "AIPING_API_KEY",
                   default=""))
    if not key:
        if not local:
            if strict:
                raise SystemExit(
                    "no API key: run `auto setup` once, or set AIPING_API_KEY, or\n"
                    "pass --api-key, or point --base-url at a local server\n"
                    "(e.g. http://127.0.0.1:11434/v1)"
                )
            src["api_key"] = "unset — run `auto setup`"
        else:
            key = "ollama"
            src["api_key"] = "not needed (local endpoint)"

    if args.no_proxy or args.proxy == "0":
        use_proxy = False
        src["proxy"] = "flag"
    elif args.proxy == "1":
        use_proxy = True
        src["proxy"] = "flag"
    elif saved.get("proxy") is not None and not local:
        use_proxy = bool(saved["proxy"])
        src["proxy"] = f"config {configfile.config_path()}"
    else:
        use_proxy = not local
        src["proxy"] = "default (off for local endpoints)"

    fast = bool(args.fast or os.environ.get("AUTOFORGE_FAST", "") == "1"
                or saved.get("fast") is True)
    src["fast"] = "flag" if args.fast else ("env AUTOFORGE_FAST" if
                                            os.environ.get("AUTOFORGE_FAST") == "1"
                                            else ("config" if saved.get("fast") else "default"))
    if args.max_tokens:
        max_tokens, src["max_tokens"] = args.max_tokens, "flag"
    else:
        max_tokens = int(pick("max_tokens", None, "AUTOFORGE_MAX_TOKENS", default=3000))
    return ({"base": base, "model": model, "key": key, "proxy": use_proxy,
             "fast": fast, "max_tokens": max_tokens}, src)


def _config(args: argparse.Namespace) -> dict:
    return _resolve(args)[0]


def _build(cfg: dict, *, meta_cognition: bool = True) -> ForgeAgent:
    llm = OpenAICompatClient(
        model=cfg["model"], base_url=cfg["base"], api_key=cfg["key"], timeout=600,
        proxies=PROXIES if cfg["proxy"] else None,
    )
    sandbox = Sandbox(timeout=30.0)
    verifier = ToolVerifier(
        llm, sandbox=sandbox,
        run_adversarial_check=not cfg["fast"],
        run_trigger_check=not cfg["fast"],
        run_negative_check=not cfg["fast"],
    )
    agent = ForgeAgent(
        llm=llm,
        registry=ToolRegistry(),
        sandbox=sandbox,
        generator=LLMToolGenerator(llm, max_tokens=cfg["max_tokens"]),
        forge_config=ForgeConfig(promote_on_pass=True, max_rounds=2),
        enable_meta_cognition=meta_cognition,
    )
    # __post_init__ built its own verifier; make both points honour --fast so
    # the agent can never disagree with itself about whether a tool passed.
    agent.verifier = verifier
    agent.pipeline.verifier = verifier
    return agent


def _describe(cfg: dict, agent: ForgeAgent) -> None:
    print(_c(_D, f"model {cfg['model']}  |  {cfg['base']}  |  "
                 f"{'FAST' if cfg['fast'] else 'full'} checks  |  "
                 f"proxy={cfg['proxy']}"))


# ----------------------------------------------------------------------
# trace rendering (shared by chat and forge)
# ----------------------------------------------------------------------
def _show_trace(agent: ForgeAgent, from_idx: int) -> None:
    for ev in agent.trace[from_idx:]:
        kind = ev.get("kind", "")
        if kind == "call":
            print(_c(_D, f"  -> {ev.get('tool')}"))
        elif kind == "forge_attempt":
            print(f"{_c(_Y, '  forging')} {str(ev.get('need', ''))[:64]}...")
        elif kind == "forge_done":
            ok = ev.get("ok")
            name = ev.get("name") or ev.get("tool") or "?"
            print(f"  {_c(_G, 'sealed') if ok else _c(_Y, 'rejected')} {name}")
        elif kind == "auto_quarantine":
            print(f"{_c(_Y, '  quarantined')} {ev.get('name', '?')} (failed review)")


def _print_checks(result) -> None:
    for a in result.attempts:
        if a.error:
            print(f"  round {a.round}: {_c(_Y, 'error')} {a.error}")
        if a.report:
            for c in a.report.checks:
                tag = _c(_G, "PASS") if c.passed else _c(_Y, "FAIL")
                print(f"    [{tag}] {c.name}: {str(c.detail)[:96]}")


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
HELP_BODY = """commands:
  /help      this list            /tools   the tool library + health
  /report    policy and amendments /trace   the agent's decision log
  /reset     forget the conversation (keeps forged tools)
  /quit      exit"""

BANNER = (f"{_c(_C, 'autoforge')} — an agent that writes, verifies and keeps its own tools.\n"
          f"Type a need in plain language. {_c(_D, '/help for commands, /quit to leave.')}")


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = _config(args)
    agent = _build(cfg)
    _describe(cfg, agent)
    print(BANNER)
    history: list = []

    while True:
        try:
            line = input(f"\n{_c(_C, 'you>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        cmd = line.split()[0].lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/help":
            print(HELP_BODY)
            continue
        if cmd == "/tools":
            rep = agent.registry.report()
            if not rep["tools"]:
                print(_c(_D, "(no tools yet — ask for something you need)"))
            for t in rep["tools"]:
                print(f"  {t.get('name'):<24} {_c(_D, str(t.get('state')))}")
            continue
        if cmd == "/report":
            print(json.dumps(agent.report(), indent=2, default=str))
            continue
        if cmd == "/trace":
            for ev in agent.trace:
                print(f"  {str(ev.get('kind')):<14} {str(ev)[:110]}")
            continue
        if cmd == "/reset":
            history = []
            print(_c(_D, "conversation cleared; forged tools kept"))
            continue

        mark = len(agent.trace)
        try:
            result = agent.run(line, history=history)
        except KeyboardInterrupt:
            print(f"\n{_c(_D, 'interrupted')}")
            continue
        except Exception as exc:                                   # noqa: BLE001
            print(f"{_c(_Y, 'error:')} {type(exc).__name__}: {exc}")
            continue

        print()
        _show_trace(agent, mark)
        if result.content:
            print(f"\n{_c(_C, 'agent>')} {result.content}")
        if result.self_terminated:
            print(_c(_D, "(the agent decided the task was done)"))
        history = result.messages

    print(_c(_D, f"bye — {len(agent.registry.names())} tool(s) this session"))
    return 0


def cmd_forge(args: argparse.Namespace) -> int:
    cfg = _config(args)
    agent = _build(cfg, meta_cognition=False)
    _describe(cfg, agent)
    need = " ".join(args.need)
    print(f"\n{_c(_D, 'need:')} {need}\n")

    mark = len(agent.trace)
    result = agent.pipeline.forge(need)
    _show_trace(agent, mark)
    _print_checks(result)

    if not result.ok:
        print(f"\n{_c(_Y, 'FORGE FAILED')} after {result.rounds} round(s)")
        return 2

    spec = result.spec
    print(f"\n{_c(_G, 'FORGED')} {spec.name}  [{spec.state.value}]  "
          f"hash={spec.hash[:16]}  effect={spec.effect_signature}")
    print(_c(_D, "code:"))
    for ln in spec.code.splitlines():
        print(f"  {ln}")

    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps({
            "name": spec.name, "code": spec.code, "parameters": spec.parameters,
            "effect_signature": spec.effect_signature, "hash": spec.hash,
            "invariances": list(spec.invariances),
            "verification": spec.verification,
        }, indent=2, default=str), encoding="utf-8")
        print(f"\n{_c(_D, f'written to {out}')}")

    if args.call:
        payload = {}
        for pair in args.call:
            if "=" not in pair:
                raise SystemExit(f"--call expects key=value, got {pair!r}")
            k, v = pair.split("=", 1)
            payload[k] = v
        r = agent.registry.call(spec.name, payload)
        print(f"\ncall {payload} -> ok={r.ok}  {r.output if r.ok else r.error}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Inspect JSON artifacts written by `forge --out`."""
    path = Path(args.path)

    def _enumerate(directory: Path) -> int:
        found = sorted(directory.glob("*.json"))
        if not found:
            print(_c(_D, "no artifacts — forge one with "
                         "`autoforge forge \"<need>\" --out tool.json`"))
            return 0
        for f in found:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                print(f"  {_c(_Y, '(unreadable)'):<24} {_c(_D, str(f))}")
                continue
            print(f"  {str(data.get('name', '?')):<24} {_c(_D, str(f))}")
        return 0

    if path.is_dir():
        return _enumerate(path)
    if path.exists():
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")),
                         indent=2, default=str))
        return 0
    return _enumerate(path)


def cmd_setup(args: argparse.Namespace) -> int:
    """One-time wizard. Everything it writes is picked up by later runs."""
    return setup_wizard.run(args)


def cmd_config(args: argparse.Namespace) -> int:
    """Print the effective settings and, for each, which layer supplied it."""
    cfg, src = _resolve(args, strict=False)
    path = configfile.config_path()
    print(f"\n  {_c(_D, 'config file')}  {path}"
          f"{'' if path.exists() else _c(_D, '  (not created yet)')}\n")
    rows = [("base_url", cfg["base"], src.get("base_url", "")),
            ("model", cfg["model"], src.get("model", "")),
            ("api_key", configfile.mask(cfg["key"]), src.get("api_key", "")),
            ("max_tokens", cfg["max_tokens"], src.get("max_tokens", "")),
            ("proxy", cfg["proxy"], src.get("proxy", "")),
            ("fast", cfg["fast"], src.get("fast", ""))]
    for name, value, origin in rows:
        print(f"  {name:<11}{str(value):<26}{_c(_D, origin)}")
    origin = src.get("api_key", "")
    if not cfg["key"]:
        print(_c(_Y, "\n  no key configured — run `auto setup` to store one\n"))
    elif origin.startswith("env"):
        stored = configfile.load().get("api_key")
        note = ("shadows the stored key for this shell only"
                if stored else "nothing is stored in the file yet")
        print(_c(_D, f"\n  {origin} {note}\n"))
    else:
        print()
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoforge",
        description="An agent that writes, verifies and keeps its own tools.",
    )
    p.add_argument("--model", help="model name (AUTOFORGE_MODEL)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (AUTOFORGE_BASE_URL)")
    p.add_argument("--api-key", help="API key (AUTOFORGE_API_KEY / AIPING_API_KEY)")
    p.add_argument("--fast", action="store_true", help="drop the LLM-driven checks")
    p.add_argument("--max-tokens", type=int, help="generator output cap (AUTOFORGE_MAX_TOKENS)")
    p.add_argument("--no-proxy", action="store_true", help="do not use the socks proxy")
    p.add_argument("--proxy", choices=["0", "1"], help="force proxy on/off")
    p.add_argument("--version", action="version", version="autoforge 0.4.0")

    sub = p.add_subparsers(dest="command")
    sub.add_parser("chat", help="talk to the agent in a REPL")

    f = sub.add_parser("forge", help="forge one tool from a need, then stop")
    f.add_argument("need", nargs="+", help="the recurring need, in plain language")
    f.add_argument("--out", help="write the sealed tool as JSON to this path")
    f.add_argument("--call", action="append", metavar="K=V",
                   help="call the forged tool with these args (repeatable)")

    li = sub.add_parser("list", help="inspect artifacts written by `forge --out`")
    li.add_argument("path", nargs="?", default="autoforge_tools",
                    help="artifact file, or a directory of them")

    su = sub.add_parser("setup", help="one-time wizard: provider, key, model -> config file")
    # Duplicated so `auto setup --api-key K` works as well as
    # `auto --api-key K setup`; both land on the same dest.
    su.add_argument("--base-url", help="OpenAI-compatible base URL")
    su.add_argument("--api-key", help="API key to store")
    su.add_argument("--model", help="model name")
    su.add_argument("--max-tokens", type=int, help="generator output cap")
    su.add_argument("--proxy", choices=["0", "1"], help="force the socks proxy on/off")
    su.add_argument("--no-proxy", action="store_true", help="do not use the socks proxy")

    cf = sub.add_parser("config", help="show effective settings and where each came from")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        # `auto` on its own summons the agent. Fall back to help when there is
        # no terminal to talk on (pipes, cron, CI).
        if not sys.stdin.isatty():
            parser.print_help()
            return 0
        args.command = "chat"
        args.need = None
        args.out = None
        args.call = None
        args.path = None
    return {"chat": cmd_chat, "forge": cmd_forge, "list": cmd_list,
            "setup": cmd_setup, "config": cmd_config}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
