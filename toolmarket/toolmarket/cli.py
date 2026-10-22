"""`tool-market` — the command that makes the substrate one step from a shell.

    tool-market web [--port 8000] [--host 127.0.0.1] [--no-seed]
    tool-market demo            the evolution transcript, in the terminal
    tool-market version

`web` is the point: one command from a checkout, or one command from a URL
(`uvx --from git+… tool-market web`), and the substrate is on screen. No config
file, no database to create, no seeding step the user has to know about.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _version() -> str:
    try:
        from importlib.metadata import version
        return f"tool-market {version('tool-market')}"
    except Exception:  # noqa: BLE001 - not installed, just running from a tree
        return "tool-market (uninstalled source tree)"


def _cmd_web(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "the console needs the api extra:\n"
            "    pip install 'tool-market[api]'\n"
            "or, without installing anything:\n"
            "    uvx --from 'tool-market[api]' tool-market web",
            file=sys.stderr,
        )
        return 2

    from toolmarket.web import build_app

    app = build_app(seed_demo=not args.no_seed)
    where = f"http://{args.host}:{args.port}"
    print(f"tool-market console · {where}  (ctrl-c to stop)")
    if args.host in ("0.0.0.0", "::"):
        print(f"  bound to {args.host}: reachable from other machines on this host")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run the end-to-end demo. Kept as a subcommand so the transcript the
    README shows is reachable from the installed command, not only from a
    checkout that happens to have examples/ in it."""
    try:
        from examples.demo_evolution import main as demo_main
    except ImportError:
        print(
            "the demo lives in the source tree (examples/demo_evolution.py); "
            "run it from a checkout:\n    python examples/demo_evolution.py",
            file=sys.stderr,
        )
        return 2
    demo_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tool-market",
        description="The enforced resource substrate for agent tools.",
    )
    p.add_argument("--version", action="version", version=_version())
    sub = p.add_subparsers(dest="cmd")

    w = sub.add_parser("web", help="Serve the console (the substrate on screen).")
    w.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 by default; 0.0.0.0 to reach it from outside.")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--no-seed", action="store_true",
                   help="Start with an empty substrate instead of the demo tools.")
    w.add_argument("--log-level", default="warning")
    w.set_defaults(func=_cmd_web)

    d = sub.add_parser("demo", help="Print the evolution transcript.")
    d.set_defaults(func=_cmd_demo)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
