#!/usr/bin/env bash
# The live Postgres suite, against a *running* compose stack. `make verify` after
# `make compose-up`.
#
# Why this is not `docker compose exec api pytest`: the runtime image is built
# deliberately without a compiler, without git and without pytest (see the
# Dockerfile) because it executes untrusted candidate tools — shipping a test
# runner inside it would ship a way to fetch and run arbitrary code into the
# exact container that is not supposed to have one. It also does not contain
# `tests/` (only `toolmarket/` is COPYed). So the suite runs in a one-off
# container from the same image, with the checkout bind-mounted and the `dev`
# extra installed for the duration of the run: same interpreter, same installed
# package, same database, nothing added to the shipped image.
#
# Usage:  ./deploy/verify.sh          (or: make verify)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# A bind mount needs a *host* path. Under git-bash/MSYS `pwd` answers /d/Users/…
# which Docker Desktop does not resolve; `pwd -W` answers D:/Users/…. On Linux
# there is no -W and `pwd` is already right.
REPO_ROOT="$(pwd -W 2>/dev/null || pwd)"

COMPOSE="${COMPOSE:-docker compose}"
DSN="postgresql://toolmarket:${POSTGRES_PASSWORD:-toolmarket}@postgres:5432/toolmarket"

echo "verify: live Postgres suite"
echo "  repo: $REPO_ROOT"
echo "  dsn:  postgresql://toolmarket:***@postgres:5432/toolmarket"
echo

$COMPOSE run --rm --no-deps --user root \
  -v "$REPO_ROOT:/src" -w /src \
  -e "TOOLMARKET_TEST_PG_DSN=$DSN" \
  --entrypoint sh api -c \
  'pip install -q pytest httpx && python -m pytest tests/test_store_pg.py -q -k Live'
rc=$?

echo
if [ "$rc" -eq 0 ]; then
  echo "verify: PASS"
else
  echo "verify: FAIL (exit $rc)"
fi
exit "$rc"
