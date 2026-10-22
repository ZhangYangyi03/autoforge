#!/usr/bin/env sh
# One command after clone: fetch the engine, install the substrate, run the demo.
#
#   ./run_demo.sh
#
# `pip install -e .` pulls autoforge from git, so no sibling checkout is needed.
set -e
cd "$(dirname "$0")"

py=""
for c in "${PYTHON:-}" python3 python; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1; then py=$c; break; fi
done
if [ -z "$py" ]; then
    echo "run_demo: no python3 on PATH -- install Python 3.10+ first" >&2
    exit 1
fi
if ! "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "run_demo: need Python 3.10+, got $("$py" -V 2>&1)" >&2
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "run_demo: git is required -- the engine is installed from a git URL" >&2
    exit 1
fi

echo "run_demo: installing the substrate and its engine dependency..." >&2
"$py" -m pip install --quiet -e .

echo "run_demo: running the end-to-end evolution demo" >&2
exec "$py" examples/demo_evolution.py
