"""Inspect the WorkBuddy desktop login files, without printing any secret.

Two files exist:
  workbuddy-desktop.info     (cn accounts)
  workbuddy-desktop-ai.info  (ai / global accounts)

wbproxy currently reads ~/.dsh/.xdpool/accounts.json, which is DSH's cache of
these. To make the tool standalone, it has to read the source. This prints the
shape -- key names, nesting, types -- and redacts every value that looks like a
token.
"""
import json
import os
import pathlib
import sys

AUTH_DIR = pathlib.Path(os.environ["LOCALAPPDATA"]) / "CodeBuddyExtension" / "Data" / "Public" / "auth"


def redact(key: str, value):
    if not isinstance(value, str):
        return value
    k = key.lower()
    if any(w in k for w in ("token", "secret", "password", "key", "cookie")):
        return f"<redacted len={len(value)}>"
    if len(value) > 60 and value.count(".") == 2:
        return f"<jwt-ish len={len(value)}>"
    return value


def walk(obj, path="", depth=0):
    if depth > 4:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                print(f"{'  '*depth}{k}: object({len(v)} keys)")
                walk(v, p, depth + 1)
            elif isinstance(v, list):
                print(f"{'  '*depth}{k}: list[{len(v)}]")
                if v and isinstance(v[0], dict):
                    walk(v[0], p + "[]", depth + 1)
                elif v:
                    print(f"{'  '*(depth+1)}first: {redact(k, v[0])!r}")
            else:
                print(f"{'  '*depth}{k}: {type(v).__name__} = {redact(k, v)!r}")


def main() -> int:
    if not AUTH_DIR.exists():
        print(f"missing: {AUTH_DIR}")
        return 1
    for f in sorted(AUTH_DIR.glob("*.info")):
        print("=" * 74)
        print(f"{f.name}   {f.stat().st_size} bytes")
        raw = f.read_text(encoding="utf-8", errors="replace")
        print(f"  first 40 bytes: {raw[:40]!r}")
        try:
            doc = json.loads(raw)
        except Exception as exc:
            print(f"  not JSON: {exc}")
            continue
        print(f"  top-level: {list(doc) if isinstance(doc, dict) else type(doc).__name__}")
        walk(doc)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
