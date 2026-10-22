#!/usr/bin/env bash
# End-to-end against a *running* stack. `make smoke` after `make compose-up`.
#
# This exists because the interesting failures are not "is it up" — they are
# "is the answer right". A container that serves /health and returns 404 for
# every resource, or accepts an async evolution and then loses the task id, looks
# perfectly healthy to anything that only curls a probe. So the script walks one
# resource through its whole documented life and asserts on the shape of each
# answer.
#
# Usage:  BASE=http://127.0.0.1:8000 ./deploy/smoke.sh
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
NAME="smoke_$(date +%s)"
PASS=0
FAIL=0

say()  { printf '  %s\n' "$*"; }
ok()   { PASS=$((PASS + 1)); printf '  ok   %s\n' "$*"; }
bad()  { FAIL=$((FAIL + 1)); printf '  FAIL %s\n' "$*"; }

# `check <description> <expected> <actual>`
check() {
  if [ "$2" = "$3" ]; then ok "$1 ($3)"; else bad "$1: expected $2, got $3"; fi
}

# The response body is captured in the *current directory*, not /tmp: this script
# runs on Windows under git-bash, where `curl` is the native binary and a POSIX
# path like /tmp/foo is not the same file the shell would read back. The failure
# is silent — curl exits 0 having written elsewhere — so the tests that inspect a
# body fail with "file not found" while the ones that only check a status pass.
SMOKE_BODY=".smoke-body.json"
trap 'rm -f "$SMOKE_BODY"' EXIT

code() { curl -s -o "$SMOKE_BODY" -w '%{http_code}' "$@"; }

# Field extraction without jq: the target machine is a Windows box with git-bash
# and curl, and requiring jq would make this script unrunnable exactly where it
# is most needed.
jget() {
  python -c "
import json, sys
try:
    d = json.load(open('$SMOKE_BODY'))
except Exception:
    print(''); raise SystemExit
for part in sys.argv[1].split('.'):
    d = d.get(part, '') if isinstance(d, dict) else ''
print(d if d is not None else '')
" "$1"
}

echo "smoke against $BASE"

echo "probes"
check "GET /health"  200 "$(code "$BASE/health")"
check "GET /ready"   200 "$(code "$BASE/ready")"
check "GET /metrics" 200 "$(code "$BASE/metrics")"
grep -q '^toolmarket_build_info' $SMOKE_BODY \
  && ok "metrics carry build_info" \
  || bad "metrics are missing toolmarket_build_info"
grep -q '^toolmarket_chain_intact 1' $SMOKE_BODY \
  && ok "event chain verifies" \
  || bad "event chain does not verify"
grep -q '^toolmarket_dependency_up{component="store"} 1' $SMOKE_BODY \
  && ok "store reachable at scrape time" \
  || bad "store reported down"

echo "register"
check "POST /resources" 201 "$(code -X POST "$BASE/resources" \
  -H 'content-type: application/json' -d "{
    \"name\": \"$NAME\",
    \"description\": \"Smoke-test resource.\",
    \"parameters\": {\"type\": \"object\",
                    \"properties\": {\"text\": {\"type\": \"string\"}},
                    \"required\": [\"text\"]},
    \"code\": \"def $NAME(text=''):\n    return '-'.join(text.lower().split())\n\",
    \"source\": \"human\",
    \"invariances\": [\"text\"],
    \"effect_signature\": \"pure\"
  }")"
RID="tool:$NAME"
check "registered id" "$RID" "$(python -c "import json;print(json.load(open('$SMOKE_BODY'))['id'])")"

echo "read paths"
check "GET /resources/{id}" 200 "$(code "$BASE/resources/$RID")"
grep -q '"capability_schema"' $SMOKE_BODY \
  && ok "view carries the capability schema" \
  || bad "view is missing capability_schema"
check "GET .../events"  200 "$(code "$BASE/resources/$RID/events")"
check "GET .../lineage" 200 "$(code "$BASE/resources/$RID/lineage")"
check "GET /stats"      200 "$(code "$BASE/stats")"

echo "lifecycle"
# The illegal move is asserted *first*, while the resource is still in `draft`:
# `draft -> active` skips probation, which the transition table refuses. Ordering
# matters because `probation -> retired` is legal, so testing the refusal after
# the legal move would be asserting on a redirect that is allowed.
check "draft -> active is refused" 409 "$(code -X POST "$BASE/resources/$RID/transition" \
  -H 'content-type: application/json' -d '{"to": "active"}')"
check "draft -> probation" 200 "$(code -X POST "$BASE/resources/$RID/transition" \
  -H 'content-type: application/json' -d '{"to": "probation", "reason": "smoke"}')"
check "probation -> active" 200 "$(code -X POST "$BASE/resources/$RID/transition" \
  -H 'content-type: application/json' -d '{"to": "active", "reason": "smoke"}')"
check "active -> draft is refused" 409 "$(code -X POST "$BASE/resources/$RID/transition" \
  -H 'content-type: application/json' -d '{"to": "draft"}')"

echo "invoke"
check "POST .../invoke" 200 "$(code -X POST "$BASE/resources/$RID/invoke" \
  -H 'content-type: application/json' -d '{"arguments": {"text": "Hello There"}}')"

echo "evolution (synchronous, stub proposer)"
check "POST .../evolve" 200 "$(code -X POST "$BASE/resources/$RID/evolve" \
  -H 'content-type: application/json' -d '{"goal": "tighten the trigger", "proposer": "stub"}')"
grep -q '"committed"' $SMOKE_BODY \
  && ok "evolve answered with a commit verdict" \
  || bad "evolve answer has no committed field"

echo "evolution (asynchronous)"
ASYNC_CODE="$(code -X POST "$BASE/resources/$RID/evolve/async" \
  -H 'content-type: application/json' -d '{"goal": "async smoke"}')"
check "POST .../evolve/async is 202" 202 "$ASYNC_CODE"
TASK_ID="$(python -c "
import json
try:
    print(json.load(open('$SMOKE_BODY')).get('task_id',''))
except Exception:
    print('')
")"
if [ -z "$TASK_ID" ]; then
  bad "no task_id returned"
else
  ok "task_id returned"
  TERMINAL=""
  for _ in $(seq 1 60); do
    code "$BASE/tasks/$TASK_ID" > /dev/null
    TERMINAL="$(python -c "
import json
try:
    print(json.load(open('$SMOKE_BODY')).get('terminal',''))
except Exception:
    print('')
")"
    if [ "$TERMINAL" = "True" ]; then break; fi
    sleep 1
  done
  if [ "$TERMINAL" = "True" ]; then
    ok "task reached a terminal state"
    say "state: $(python -c "import json;print(json.load(open('$SMOKE_BODY'))['state'])")"
  else
    bad "task never terminated (ast terminal=$TERMINAL)"
  fi
fi
check "unknown task is 404" 404 "$(code "$BASE/tasks/nope")"
check "unknown resource is 404" 404 "$(code "$BASE/resources/tool:nope")"

echo
echo "passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ]
