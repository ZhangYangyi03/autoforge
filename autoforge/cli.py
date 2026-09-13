"""Command-line entry point: `autoforge`, installed as `auto`.

    auto                    talk to the agent (bare invocation, on a terminal)
    auto web                serve the harness UI — one command, no build step
    auto run "<task>"       run one task through a chosen mode, headless
    auto modes              list the runtime modes (standard / minimal)
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
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import configfile, setup_wizard
from .agent import ForgeAgent
from .autonomy.policy import (CONFIRM_REQUIRED, FULL_FREEDOM, SUPERVISED,
                             AutonomyPolicy)
from .core.llm import OpenAICompatClient
from .core.steering import Steering
from .forge.generator import LLMToolGenerator
from .forge.pipeline import ForgeConfig
from .forge.sandbox import Sandbox
from .forge.verifier import ToolVerifier
from .store import ToolStore
from .tools.registry import ToolRegistry

MODES = ("standard", "minimal")

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
            # Just "config": the path is printed once in the header, and repeating
            # it on every row buries the one thing this column is for — telling
            # env apart from file apart from default at a glance.
            src[field] = "config"
            return value
        src[field] = "default"
        return default

    base = str(pick("base_url", args.base_url, "AUTOFORGE_BASE_URL",
                    default="https://aiping.cn/api/v1"))
    model = str(pick("model", args.model, "AUTOFORGE_MODEL",
                     default="DeepSeek-V4.1-Flash"))
    local = any(h in base for h in ("127.0.0.1", "localhost"))
    # `AIPING_API_KEY` is provider-specific: it may only stand in as the key when
    # the endpoint actually is aiping. Treating it as a generic fallback is how
    # a DeepSeek run got handed the aiping key and came back 401 — the base_url
    # had changed but the credential had not, and nothing in the output said so.
    key_env = ["AUTOFORGE_API_KEY"]
    if "aiping" in base:
        key_env.append("AIPING_API_KEY")
    key = str(pick("api_key", args.api_key, *key_env, default=""))
    if not key:
        if not local:
            if strict:
                # Name the variable that actually works here: suggesting
                # AIPING_API_KEY on a non-aiping endpoint sends the reader down
                # a path the resolver has deliberately closed.
                env_hint = ("AIPING_API_KEY or AUTOFORGE_API_KEY" if "aiping" in base
                            else "AUTOFORGE_API_KEY")
                raise SystemExit(
                    f"no API key: run `auto setup` once, or set {env_hint}, or\n"
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
        src["proxy"] = "config"
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

    policy_name, policy_src = _resolve_policy(args, saved, src)
    return ({"base": base, "model": model, "key": key, "proxy": use_proxy,
             "fast": fast, "max_tokens": max_tokens, "policy": policy_name}, src)


POLICIES = {"full": FULL_FREEDOM, "supervised": SUPERVISED}


def _policy_for(name: str) -> AutonomyPolicy:
    """A *copy* of the named preset.

    selfmod.amend applies changes with setattr on the live object, so handing
    out the module-level singleton would let one session's set_autonomy leak
    into every agent built afterwards — including children. Each agent gets its
    own instance.
    """
    preset = POLICIES.get(name)
    if preset is None:
        raise SystemExit(f"unknown policy {name!r} (have: {', '.join(POLICIES)})")
    return replace(preset)


def _resolve_policy(args: argparse.Namespace, saved: dict, src: dict) -> tuple[str, str]:
    """Which autonomy policy to build the agent with.

    A preset nobody can select is the same as no preset at all, so the name is
    resolved here and threaded into everything downstream.
    """
    flag = getattr(args, "policy", None)
    if flag:
        src["policy"] = "flag"
        return flag, "flag"
    env = os.environ.get("AUTOFORGE_POLICY", "").strip().lower()
    if env:
        if env not in POLICIES:
            raise SystemExit(f"unknown AUTOFORGE_POLICY={env!r} (have: {', '.join(POLICIES)})")
        src["policy"] = "env AUTOFORGE_POLICY"
        return env, "env"
    cfg = str(saved.get("policy") or "").strip().lower()
    if cfg:
        if cfg not in POLICIES:
            raise SystemExit(f"unknown policy={cfg!r} in config (have: {', '.join(POLICIES)})")
        src["policy"] = "config"
        return cfg, "config"
    src["policy"] = "default"
    return "full", "default"


def _config(args: argparse.Namespace) -> dict:
    return _resolve(args)[0]


class _TerminalConfirmer:
    """Ask the operator, on the terminal, before a gated tool runs.

    Answers the gate's three-way question: True (yes), False (no), None (nobody
    to ask). Silence is never a yes — an empty line, a Ctrl-C, a closed stdin,
    or a worker thread with a browser on the other end all answer None, and the
    gate refuses on None.

    The prompt names the switch, the tool and the arguments: an approval prompt
    that does not say what is being approved trains its reader to say yes
    without looking, which is worse than having no prompt at all.
    """

    def __init__(self, *, stream: Any = None) -> None:
        self._stream = stream            # injectable so a test can read it back

    def _usable(self) -> bool:
        # A worker thread has no terminal of its own even when the process does.
        # The web harness runs agents in threads, so prompting there would block
        # a request on input nobody can see.
        if threading.current_thread() is not threading.main_thread():
            return False
        try:
            return sys.stdin.isatty()
        except Exception:                                    # noqa: BLE001
            return False

    def __call__(self, tool: str, arguments: dict,
                 freedoms: list[str]) -> bool | None:
        if not self._usable():
            return None
        out = self._stream or sys.stdout
        print(_c(_Y, "\n  -- confirmation gate --"), file=out)
        print(f"  {tool!r} needs {', '.join(freedoms)}, which your policy has off.",
              file=out)
        if arguments:
            shown = json.dumps(arguments, ensure_ascii=False, default=str)
            print(f"  arguments: {shown[:400]}{'...' if len(shown) > 400 else ''}",
                  file=out)
        try:
            reply = input(f"{_c(_C, 'allow>')} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            return None
        return reply in ("y", "yes")


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
        policy=POLICIES.get(cfg.get("policy", "full"), FULL_FREEDOM),
        enable_meta_cognition=meta_cognition,
        confirmer=_TerminalConfirmer(),
    )
    # __post_init__ built its own verifier; make both points honour --fast so
    # the agent can never disagree with itself about whether a tool passed.
    agent.verifier = verifier
    agent.pipeline.verifier = verifier
    return agent


def _build_mode(cfg: dict, mode: str = "standard"):
    """Build the agent that a runtime mode names.

    This is the single seam the web harness and `auto run` share: a mode is an
    assembly of the same parts, not a separate program.

      standard — the full forging agent, with a store so sealed tools persist
      minimal  — two tools (bash + str_replace_editor), no forging
    """
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r} (have: {', '.join(MODES)})")

    if mode == "minimal":
        from .modes import MinimalAgent

        llm = OpenAICompatClient(
            model=cfg["model"], base_url=cfg["base"], api_key=cfg["key"],
            timeout=600, proxies=PROXIES if cfg["proxy"] else None,
        )
        # The control group carries a policy too, so an A/B against standard
        # compares like with like. Under the default preset nothing changes:
        # bash stays ungated, which is what the comparison depends on.
        return MinimalAgent(llm=llm, cwd=os.getcwd(),
                            policy=_policy_for(cfg.get("policy", "full")),
                            confirmer=_TerminalConfirmer())

    agent = _build(cfg)
    # The CLI's `forge` writes JSON artifacts by hand; the harness wants the
    # sealed tool to survive the session, so it gets a store.
    try:
        agent.store = ToolStore(os.environ.get("AUTOFORGE_DB") or None)
    except Exception as exc:                                   # noqa: BLE001
        print(f"{_c(_Y, 'note:')} tool store unavailable ({exc}); sealed tools "
              f"live for this session only")
    return agent


def _describe(cfg: dict, agent: ForgeAgent) -> None:
    print(_c(_D, f"model {cfg['model']}  |  {cfg['base']}  |  "
                 f"{'FAST' if cfg['fast'] else 'full'} checks  |  "
                 f"proxy={cfg['proxy']}  |  policy={cfg.get('policy', 'full')}"))


# ----------------------------------------------------------------------
# trace rendering (shared by chat and forge)
# ----------------------------------------------------------------------
def _show_trace(agent: ForgeAgent, from_idx: int, skip_streamed: bool = False) -> None:
    """The decision log for one task.

    With `skip_streamed`, the kinds the live line already narrated are dropped —
    so a run that was watched live ends with the few events that need saying
    twice, not a replay of everything already on screen.
    """
    err_rounds = {ev.get("round") for ev in agent.trace[from_idx:]
                  if ev.get("kind") == "forge_error"}
    for ev in agent.trace[from_idx:]:
        kind = ev.get("kind", "")
        if skip_streamed and kind in _LiveRun.STREAMED:
            continue
        if kind == "call":
            print(_c(_D, f"  -> {ev.get('tool')}"))
        elif kind == "forge_attempt":
            # Failures only: a success is reported once, by forge_done. A round
            # already narrated as a forge_error is not repeated here either.
            if ev.get("accepted") or ev.get("round") in err_rounds:
                continue
            err = str(ev.get("error") or "verification failed")[:64]
            print(f"{_c(_Y, '  round')} {ev.get('round')}: {err}")
        elif kind == "forge_done":
            ok = ev.get("ok")
            name = ev.get("name") or ev.get("tool") or "?"
            print(f"  {_c(_G, 'sealed') if ok else _c(_Y, 'rejected')} {name}")
        elif kind == "auto_quarantine":
            print(f"{_c(_Y, '  quarantined')} {ev.get('name', '?')} (failed review)")
        else:
            line = _LiveRun._milestone(kind, ev)
            if line:
                print(f"  {line}")


def _print_checks(result) -> None:
    for a in result.attempts:
        if a.error:
            print(f"  round {a.round}: {_c(_Y, 'error')} {a.error}")
        if a.report:
            for c in a.report.checks:
                tag = _c(_G, "PASS") if c.passed else _c(_Y, "FAIL")
                print(f"    [{tag}] {c.name}: {str(c.detail)[:96]}")


# ----------------------------------------------------------------------
# live progress — a slow model must read as "waiting", never as "hung"
# ----------------------------------------------------------------------
class _LiveRun:
    """Prints what the loop is doing while it does it.

    The reason this exists: `run` used to render the trace only after the whole
    task finished, so a model taking 40s per turn produced pure silence and no
    way to tell a slow request from a wedged process. Two signals fix that —
    an immediate line the moment each request goes out, and a ticking counter
    on that same line while the response is in flight.
    """

    WIDTH = 72

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self.t0 = time.time()
        self.live = bool(getattr(self.stream, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._waiting_since: float | None = None
        self._ticks = 0
        self._thread: threading.Thread | None = None
        # Where the run is, for `/status`: published as the loop moves, read by
        # whoever asks. Kept here rather than in the steering channel because
        # this object already sees every event.
        self._turn = 0
        self._last_tool: str | None = None
        self._forging: str | None = None
        # The round whose failure was already narrated as a forge_error, so the
        # attempt line that follows it doesn't say the same thing twice.
        self._err_round: int | None = None

    # -- plumbing ------------------------------------------------------
    def _write(self, text: str) -> None:
        with self._lock:
            self.stream.write(text)
            self.stream.flush()

    def _tickline(self, text: str) -> None:
        if self.live:
            self._write("\r" + text.ljust(self.WIDTH)[: self.WIDTH])
        else:
            self._write(text + "\n")

    def _clear(self) -> None:
        if self.live:
            self._write("\r" + " " * self.WIDTH + "\r")

    def _stamp(self) -> str:
        return time.strftime("%H:%M:%S")

    def _elapsed(self) -> str:
        return f"{time.time() - self.t0:.1f}s"

    # -- lifecycle -----------------------------------------------------
    def start(self) -> "_LiveRun":
        if self.live:
            self._thread = threading.Thread(target=self._beat, daemon=True)
            self._thread.start()
        return self

    def _beat(self) -> None:
        while not self._stop.wait(1.0):
            if self._waiting_since is None:
                continue
            secs = int(time.time() - self._waiting_since)
            if secs >= 1 and secs != self._ticks:
                self._ticks = secs
                self._tickline(f"      … waiting on model ({secs}s)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._clear()

    # -- the callback handed to ForgeAgent.run -------------------------
    def __call__(self, kind: str, payload: dict) -> None:
        if kind == "request":
            self._clear()
            self._turn = payload.get("turn") or self._turn
            self._waiting_since = time.time()
            self._ticks = 0
            self._write(f"  [{self._stamp()}] turn {payload.get('turn')} "
                        f"+{self._elapsed()}  asking the model…\n")
        elif kind == "call":
            self._clear()
            self._waiting_since = None
            self._last_tool = str(payload.get("tool"))
            self._write(f"  [{self._stamp()}] +{self._elapsed()}  "
                        f"-> {payload.get('tool')}\n")
        elif kind == "result":
            self._waiting_since = time.time()      # model turn resumes
            self._ticks = 0
        elif kind == "forge_start":
            # Forging costs a model turn per round — the longest silence in the
            # whole run, so it starts the heartbeat before the first request.
            self._clear()
            self._forging = str(payload.get("need", ""))[:48]
            self._waiting_since = time.time()
            self._ticks = 0
            self._write(f"  [{self._stamp()}] +{self._elapsed()}  "
                        f"forging {str(payload.get('need', ''))[:48]}…\n")
        elif kind == "forge_attempt":
            # Only failures are worth a line: a success is reported once, by
            # forge_done, so the reader never sees "round 1: accepted" followed
            # immediately by "sealed". Failures are the interesting case anyway
            # — they are why a forge takes more than one round.
            if not payload.get("accepted"):
                if payload.get("round") == self._err_round:
                    # Same failure, already reported as a forge_error above.
                    return
                self._clear()
                self._waiting_since = time.time()   # another round is coming
                err = str(payload.get("error") or "verification failed")[:60]
                self._write(f"  [{self._stamp()}] +{self._elapsed()}  "
                            f"round {payload.get('round')}: {err}\n")
        elif kind == "forge_error":
            # The exception that ended a round — louder than the attempt line,
            # so it takes the round slot and the attempt stays quiet.
            self._clear()
            self._waiting_since = None
            self._err_round = payload.get("round")
            self._forging = None
            futile = " (no point retrying)" if payload.get("futile") else ""
            self._write(f"  [{self._stamp()}] +{self._elapsed()}  "
                        f"forge error: {str(payload.get('error', ''))[:60]}{futile}\n")
        elif kind == "forge_done":
            self._clear()
            self._waiting_since = None
            self._forging = None
            ok = payload.get("ok")
            name = payload.get("name") or payload.get("need") or "?"
            rounds = payload.get("rounds")
            verdict = "sealed" if ok else "forge failed"
            self._write(f"  [{self._stamp()}] +{self._elapsed()}  "
                        f"{verdict} {name} ({rounds} round(s))\n")
        elif kind == "auto_quarantine":
            self._clear()
            self._write(f"  [{self._stamp()}] +{self._elapsed()}  "
                        f"quarantined {payload.get('name', '?')}\n")
        elif kind in ("amendment",):
            self._clear()
            self._write(f"  [{self._stamp()}] +{self._elapsed()}  "
                        f"amended self: {payload.get('field', '?')}\n")
        elif kind == "turn":
            self._waiting_since = None
        else:
            line = self._milestone(kind, payload)
            if line:
                self._clear()
                self._write(f"  [{self._stamp()}] +{self._elapsed()}  {line}\n")

    # -- the rare, load-bearing events ---------------------------------
    # Not every record deserves a line. These do: they change what the agent
    # *is* (its tools, its prompt, its team), and they were previously invisible
    # until the task ended — the exact silence this class exists to remove.
    @staticmethod
    def _milestone(kind: str, p: dict) -> str | None:
        if kind == "evolve":
            verdict = "improved" if p.get("improved") else "no improvement"
            vetoed = len(p.get("vetoed") or [])
            tail = f", {vetoed} mutant(s) vetoed" if vetoed else ""
            return f"evolved {p.get('tool', '?')}: {verdict}{tail}"
        if kind == "spawn":
            name = p.get("name") or p.get("child") or "child"
            if p.get("error"):
                return f"spawn {name} failed: {str(p['error'])[:60]}"
            return f"spawned {name} ({p.get('mode', 'child')})"
        if kind == "design_team":
            n, e = p.get("agents"), p.get("edges")
            note = " (degraded to solo)" if p.get("degraded") else ""
            return f"designed a team: {n} agent(s), {e} edge(s){note}"
        if kind == "retire":
            return f"retired {p.get('tool', '?')}: {str(p.get('rationale', ''))[:50]}"
        if kind == "promote_withheld":
            return f"promoted {p.get('name', p.get('tool', '?'))} out of quarantine"
        if kind in ("gpu_compile", "gpu_bench"):
            what = "compiled" if kind == "gpu_compile" else "benchmarked"
            ok = "ok" if p.get("ok") else "failed"
            return f"{what} {p.get('name', '?')} on GPU: {ok}"
        if kind == "forge_error":
            return f"forge error: {str(p.get('error', ''))[:60]}"
        if kind == "evaluate":
            return f"evaluated {p.get('tool', p.get('name', '?'))}"
        return None

    # Kinds `__call__` already narrates live. `_show_trace` skips them, so the
    # end-of-task summary never repeats what the reader just watched scroll by.
    STREAMED = ("request", "turn", "call", "result", "finish",
                "forge_start", "forge_attempt", "forge_done", "forge_error",
                "auto_quarantine", "amendment", "evolve", "spawn",
                "design_team", "retire", "promote_withheld",
                "gpu_compile", "gpu_bench", "evaluate")

    # -- the operator's channel -----------------------------------------
    def say(self, text: str) -> None:
        """Print one line of the *operator's* conversation with the run.

        A steering reply shares the progress lock, so it can never land in the
        middle of a heartbeat tick. It clears the tick first: the status line is
        being rewritten every second, and a reply that the next tick overwrote
        would be worse than no reply.
        """
        self._clear()
        self._write(f"  {text}\n")

    def snapshot(self) -> str:
        """Where the run is right now, in one line, for `/status`."""
        bits = [f"{self._elapsed()} elapsed", f"turn {self._turn or '-'}"]
        if self._waiting_since is not None:
            bits.append(f"waiting on the model {int(time.time() - self._waiting_since)}s")
        if self._forging:
            bits.append(f"forging {self._forging}")
        if self._last_tool:
            bits.append(f"last tool: {self._last_tool}")
        return "  ·  ".join(bits)

    def done(self, result) -> None:
        self._clear()
        n_calls = len(result.tool_calls) if isinstance(result.tool_calls, list) else result.tool_calls
        self._write(f"  [{self._stamp()}] +{self._elapsed()}  done — "
                    f"{result.turns} turn(s), {n_calls} tool call(s)\n")


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
HELP_BODY = """commands:
  /help      this list            /tools   the tool library + health
  /report    policy and amendments /trace   the agent's decision log
  /reset     forget the conversation (keeps forged tools)
  /quit      exit

while it is working: type a sentence to add it to the task mid-run,
  /status to ask where it is, /stop to end the run at the next step."""

BANNER = (f"{_c(_C, 'autoforge')} — an agent that writes, verifies and keeps its own tools.\n"
          f"Type a need in plain language. {_c(_D, '/help for commands, /quit to leave.')}")
LIVE_HINT = (f"{_c(_D, 'it does not lock the keyboard: while it runs, type to add to the task, ')}"
             f"{_c(_D, '/status to ask where it is, /stop to end the turn.')}")


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = _config(args)
    agent = _build(cfg)
    _describe(cfg, agent)
    print(BANNER)
    print(LIVE_HINT)

    # One reader owns stdin for the whole session. During a run its lines are
    # steering; between runs they are the next prompt. That is what makes
    # mid-run typing possible at all — the loop cannot block on a keyboard.
    steering = Steering().start()
    agent.steer = steering
    history: list = []

    try:
        while True:
            try:
                line = steering.take_line(f"\n{_c(_C, 'you>')} ")
            except KeyboardInterrupt:
                print()
                break
            if line == "":                 # end of input, not a blank line
                print()
                break
            line = line.strip()
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
            live = _LiveRun().start()
            steering.watch(live)          # replies and /status point at this run
            try:
                result = agent.run(line, history=history, progress=live)
            except KeyboardInterrupt:
                live.stop()
                print(f"\n{_c(_D, 'interrupted')}")
                continue
            except Exception as exc:                                   # noqa: BLE001
                live.stop()
                print(f"{_c(_Y, 'error:')} {type(exc).__name__}: {exc}")
                continue
            live.stop()
            live.done(result)

            print()
            _show_trace(agent, mark, skip_streamed=True)
            if result.content:
                print(f"\n{_c(_C, 'agent>')} {result.content}")
            if result.self_terminated:
                print(_c(_D, "(the agent decided the task was done)"))
            if getattr(result, "stopped_by_operator", False):
                print(_c(_Y, "(you stopped this run — the work above stands)"))
            history = result.messages
    finally:
        steering.close()

    print(_c(_D, f"bye — {len(agent.registry.names())} tool(s) this session"))
    return 0


def cmd_forge(args: argparse.Namespace) -> int:
    cfg = _config(args)
    agent = _build(cfg, meta_cognition=False)
    _describe(cfg, agent)
    need = " ".join(args.need)
    print(f"\n{_c(_D, 'need:')} {need}\n")

    mark = len(agent.trace)
    live = _LiveRun().start()
    agent._progress = live          # the pipeline records through the agent
    try:
        result = agent.pipeline.forge(need)
    finally:
        agent._progress = None
        live.stop()
    _show_trace(agent, mark, skip_streamed=True)
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
            ("fast", cfg["fast"], src.get("fast", "")),
            ("policy", cfg["policy"], src.get("policy", ""))]
    width = max(len(str(v)) for _n, v, _o in rows) + 2
    for name, value, origin in rows:
        print(f"  {name:<11}{str(value):<{width}}{_c(_D, origin)}")

    # Mirrors the agent's own my_capabilities: a policy that reads like a cage
    # while some of its fields are never consulted is worse than no policy.
    # Three answers, not two — a denial can close a door, only narrow it, or
    # turn it into a question, and printing the wrong one is its own small lie.
    preset = _policy_for(cfg["policy"])
    print(f"\n  {_c(_D, 'policy')}  {preset.describe()}")
    asked = [f for f in preset.denied if f in CONFIRM_REQUIRED]
    inert = preset.unenforced
    if not preset.denied:
        print(_c(_D, "  nothing is denied in this preset"))
    else:
        if asked:
            print(_c(_D, f"  runs only after asking you: {', '.join(asked)}"))
        if inert:
            print(_c(_Y, f"  not enforced by any code path: {', '.join(inert)}"))
            print(_c(_D, "  (partly enforced at best — DESIGN.md §8)"))
        if not asked and not inert:
            print(_c(_D, "  every denial in this preset is enforced"))

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


def cmd_web(args: argparse.Namespace) -> int:
    cfg = _config(args)
    from .web import serve

    token = args.token or os.environ.get("AUTOFORGE_WEB_TOKEN") or None
    try:
        serve(
            cfg,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            mode=args.mode,
            token=token,
        )
    except OSError as exc:
        print(f"{_c(_Y, 'error:')} cannot bind {args.host}:{args.port} — {exc}")
        print(f"{_c(_D, 'try:')} auto web --port {args.port + 1}")
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """One-shot task through a chosen mode — the headless twin of the UI."""
    cfg = _config(args)
    agent = _build_mode(cfg, args.mode)
    print(_c(_D, f"mode {args.mode}  |  model {cfg['model']}  |  {cfg['base']}  |  "
                 f"policy={cfg.get('policy', 'full')}"))
    task = " ".join(args.task)
    live = _LiveRun().start()
    try:
        result = agent.run(task, progress=live)
    finally:
        live.stop()
    live.done(result)
    if result.content:
        print(f"\n{result.content}")
    if result.self_terminated:
        print(_c(_D, f"(self-terminated: {result.termination_reason})"))
    return 0


def cmd_modes(args: argparse.Namespace) -> int:
    print("standard  — full forging agent: meta-tools, 5-check verification, "
          "evolution, spawning, persistence")
    print("minimal   — two tools (bash + str_replace_editor), no forging; "
          "the control group")
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
    p.add_argument("--policy", choices=list(POLICIES),
                   help="autonomy preset: full (default) or supervised "
                        "(AUTOFORGE_POLICY)")
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

    w = sub.add_parser("web", help="serve the harness UI (one command, no build step)")
    w.add_argument("--host", default=os.environ.get("AUTOFORGE_WEB_HOST", "127.0.0.1"),
                   help="bind address (default 127.0.0.1; use 0.0.0.0 to share)")
    w.add_argument("--port", type=int,
                   default=int(os.environ.get("AUTOFORGE_WEB_PORT") or "8765"),
                   help="port (default 8765; 0 picks a free one)")
    w.add_argument("--mode", choices=list(MODES),
                   default=os.environ.get("AUTOFORGE_MODE", "standard"),
                   help="runtime mode for new sessions")
    w.add_argument("--token", help="shared token needed to drive the agent "
                                   "(AUTOFORGE_WEB_TOKEN)")
    w.add_argument("--no-browser", action="store_true", help="do not open a browser")

    r = sub.add_parser("run", help="run one task through a chosen mode, headless")
    r.add_argument("task", nargs="+", help="the task, in plain language")
    r.add_argument("--mode", choices=list(MODES), default="standard")

    sub.add_parser("modes", help="list the runtime modes")
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
            "setup": cmd_setup, "config": cmd_config,
            "web": cmd_web, "run": cmd_run, "modes": cmd_modes}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
