#!/usr/bin/env bash
# Roll the exposure backport out of the installed autoforge build.
#
# Restores the four files the backport modified to their pristine 0.4.2
# content, and removes tools/exposure.py (which did not exist before).
#
# After this the loop sends every visible schema in full again -- 60 schemas /
# 26,872 chars per request instead of 15 in full plus 45 one-line entries.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="C:/autoforge/venv/Lib/site-packages/autoforge"

echo "rolling back the exposure backport"
echo "  from : $SITE"
echo "  using: $HERE/rollback-exposure"
echo

for rel in tools/registry.py core/agent.py store.py agent.py; do
    src="$HERE/rollback-exposure/$rel"
    dst="$SITE/$rel"
    if [ ! -f "$src" ]; then
        echo "  !! missing backup $src" >&2
        exit 1
    fi
    cp "$src" "$dst"
    printf "  restored %-22s %s\n" "$rel" "$(md5sum "$dst" | cut -c1-12)"
done

# exposure.py was new; removing it is part of the rollback. But the registry
# now imports it lazily inside exposure(), so a leftover file is harmless --
# what matters is that the wiring is gone.
if [ -f "$SITE/tools/exposure.py" ]; then
    rm -f "$SITE/tools/exposure.py"
    echo "  removed  tools/exposure.py"
fi

# Drop stale bytecode so the restored source is what actually runs.
for pkg in "" core tools; do
    cache="$SITE/$pkg/__pycache__"
    [ -d "$cache" ] || continue
    find "$cache" -name 'agent.*' -o -name 'registry.*' -o -name 'store.*' \
        -o -name 'exposure.*' 2>/dev/null | while read -r f; do rm -f "$f"; done
done
echo "  cleared stale .pyc files"
echo
echo "done. The build is back to stock 0.4.2 behaviour."
echo "Re-apply with: C:/autoforge/venv/Scripts/python.exe install_exposure.py"
