"""Start the tool-market API with autoforge importable.

Why this file exists rather than a one-line uvicorn command: autoforge is not
installed into this interpreter, it is a sibling source tree. toolmarket
imports it (autoforge.tools.spec, autoforge.forge.verifier, ...), so both source
directories have to be on sys.path *before* uvicorn imports the app. A batch
file cannot express that, and -m uvicorn has no hook for it.

Run directly, or from autostart.bat at boot.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

AUTOFORGE = r"D:\\Users\\china\\Desktop\\项目_开发\\autoforge"
TOOLMARKET = r"D:\\Users\\china\\Desktop\\项目_开发\\tool-market"

for path in (AUTOFORGE, TOOLMARKET):
    if path not in sys.path:
        sys.path.insert(0, path)

# Durable by default.
#
# `make_store()` with no URL returns ResourceStore(":memory:"), which is the
# right default for a test and the wrong one for a shelf: every restart of this
# process silently empties the market, and the symptom is not an error. It is a
# shelf that lists one tool where it listed eight, which reads as "the other
# agents never published anything" rather than as "the database was in RAM".
#
# Set here rather than in the environment so a bare `python autostart_server.py`
# is durable too -- a guarantee that only holds when launched by the batch file
# is a guarantee that will be missing exactly when someone is debugging.
#
# An explicit TOOLMARKET_STORE still wins, so a container can point at Postgres
# without editing this file.
os.environ.setdefault(
    "TOOLMARKET_STORE",
    "sqlite:///C:/Users/china/toolmarket_data/toolmarket.db",
)

# Bind to loopback only. The public surface is ngrok's job, and exposing the
# API on 0.0.0.0 as well would mean two ways in, only one of which is
# rate-limited by the tunnel's own policy.
HOST = os.environ.get("TOOLMARKET_HOST", "127.0.0.1")
PORT = int(os.environ.get("TOOLMARKET_PORT", "8000"))


def main() -> int:
    import uvicorn

    # `served_app`, not `app`: the bare app has no rate limiter and no metrics
    # middleware. The deployment target is the wrapped one, so autostart must
    # serve the same object a container would.
    uvicorn.run(
        "toolmarket.api.main:served_app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
