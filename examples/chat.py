"""Shim: the REPL now lives in the installed CLI.

    auto                 # or: auto chat
    python examples/chat.py

This file stays so the documented example paths keep working.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.cli import main                                    # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["chat"]))
