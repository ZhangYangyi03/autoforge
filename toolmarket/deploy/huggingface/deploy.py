#!/usr/bin/env python
"""Publish this repo's API to a HuggingFace Docker Space.

    python deploy/huggingface/deploy.py              # create/reuse, push, verify
    python deploy/huggingface/deploy.py --no-wait     # push and return
    python deploy/huggingface/deploy.py --name other   # a different Space name

Why a script rather than a git remote: a Space is a *separate repository* whose
root must contain the Dockerfile, and whose README.md must carry Space
frontmatter. Pushing this repo to it directly would either publish the whole
project (including `experiments/`, the paper artifacts and `.deploy.env`) or need
careful subtree surgery every time. So the Space is treated as a build artifact:
a projection of a known file list, assembled into a temp directory and pushed from
there. The list is explicit and reviewable, which is the property that matters —
the alternative is a deploy that silently publishes a credential because a new
file appeared in the source tree.

Credentials are read from `.deploy.env` at the repo root (gitignored):

    HF_TOKEN=hf_...
    HF_NAMESPACE=<user-or-org>

Nothing is written to disk outside the temp directory, and the token is never
printed — it only ever appears in the argument of a single `git push`, which is
also the one place it could leak via `ps`. Said out loud rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".deploy.env"

#: Two ways to publish, and neither is free. Verified against the API rather than
#: assumed: creating *either* a Gradio or a Docker Space on a free account fails
#:
#:     402: Static Spaces are free for everyone, but hosting Gradio and Docker
#:     Spaces on free cpu-basic requires a PRO subscription.
#:
#: Only Static Spaces (no server) are free, and a static Space cannot run this
#: API. So the modes below differ in how the process starts and what the Space
#: builds from — not in what they cost:
#:
#:   * `docker` is the canonical artefact: the repository's own Dockerfile, the
#:     same image `docker compose up` builds and CI smoke-tests.
#:   * `gradio` skips the image build, which makes it faster and usable on hosts
#:     without a container runtime. It is the default for that reason.
#:
#: Both serve *the same app*: `toolmarket.api.main.create_served_app`, with the
#: rate limiter and the metrics middleware.
MODES: dict[str, dict[str, object]] = {
    "gradio": {
        "sdk": "gradio",
        "card": "gradio-README.md",
        # The Space's `app_file` is `app.py`; the source of truth stays named for
        # what it is, so the copy is explicit here rather than a rename in git.
        "copies": {"gradio_app.py": "app.py",
                   "gradio-requirements.txt": "requirements.txt"},
        "payload": ("toolmarket",),
    },
    "docker": {
        "sdk": "docker",
        "card": "README.md",
        "copies": {},
        "payload": ("Dockerfile", "pyproject.toml", "toolmarket"),
    },
}
HERE = Path(__file__).resolve().parent


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"missing {path}; see the module docstring for the format")
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def _api(url: str, token: str, payload: dict | None = None,
         method: str | None = None) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"Authorization": f"Bearer {token}",
                 **({"Content-Type": "application/json"} if data else {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body


def whoami(token: str) -> str:
    status, body = _api("https://huggingface.co/api/whoami-v2", token)
    if status != 200 or not isinstance(body, dict):
        sys.exit(f"token rejected by HuggingFace ({status}): {body}")
    return str(body.get("name") or "")


def ensure_space(token: str, namespace: str, name: str, sdk: str) -> str:
    """Create the Space if it does not exist. Returns its repo id.

    An existing Space is reused whatever its SDK: the frontmatter in the card
    decides how HuggingFace builds it, and re-creating the repo would discard its
    history for no reason. A 402 here is the free-account Docker restriction and
    is reported with the explanation rather than as a bare status code.
    """
    repo = f"{namespace}/{name}"
    status, body = _api(f"https://huggingface.co/api/spaces/{repo}", token)
    if status == 200:
        print(f"space {repo} already exists")
        return repo
    if status != 404:
        sys.exit(f"unexpected response reading {repo}: {status} {body}")
    status, body = _api("https://huggingface.co/api/repos/create", token, {
        "type": "space",
        "name": name,
        "organization": namespace if namespace else None,
        "private": False,
        # The SDK is what makes this a Docker or a Gradio Space at creation time;
        # the frontmatter in the card repeats it, and the card wins thereafter.
        "sdk": sdk,
    })
    if status not in (200, 201):
        if status == 402:
            sys.exit(
                f"could not create space {repo}: HuggingFace declined ({status}).\n"
                f"  {body}\n"
                "  Both Gradio and Docker Spaces need a PRO subscription on\n"
                "  cpu-basic; only Static Spaces (no server) are free, and a\n"
                "  Static Space cannot run this API. Nothing about `--mode`\n"
                "  changes that. The free paths are `docker compose up` locally\n"
                "  and the Render blueprint (see deploy/render/render.yaml)."
            )
        sys.exit(f"could not create space {repo}: {status} {body}")
    print(f"created space {repo} (sdk={sdk})")
    return repo


def _run(args: list[str], cwd: Path) -> None:
    proc = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        # The token is in the push URL; scrub it from anything we print.
        stderr = proc.stderr.replace(os.environ.get("_HF_TOK", "\0"), "***")
        sys.exit(f"{' '.join(args[:2])} failed:\n{stderr}")


def assemble(work: Path, mode: str) -> None:
    spec = MODES[mode]
    for entry in spec["payload"]:  # type: ignore[union-attr]
        src = REPO_ROOT / entry
        dst = work / entry
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache"))
        else:
            shutil.copy2(src, dst)
    for src_name, dst_name in (spec["copies"] or {}).items():  # type: ignore[union-attr]
        shutil.copy2(HERE / src_name, work / dst_name)
    shutil.copy2(HERE / str(spec["card"]), work / "README.md")
    # A Space-local .dockerignore. Written rather than copied because the repo's
    # version excludes `deploy/` and the paper directories that are not in the
    # Space in the first place, and an ignore file listing files that do not
    # exist is a file nobody can reason about later.
    (work / ".dockerignore").write_text(
        "__pycache__\n*.pyc\n.git\n.pytest_cache\n", encoding="utf-8")


def push(work: Path, repo: str, token: str) -> None:
    env = dict(os.environ)
    # Any credential prompt would hang a non-interactive push forever. Fail
    # instead: the token is right there, so a prompt means something else is wrong.
    env["GIT_TERMINAL_PROMPT"] = "0"
    os.environ["_HF_TOK"] = token
    _run(["git", "init", "-q", "-b", "main"], work)
    _run(["git", "config", "user.email", "deploy@toolmarket.local"], work)
    _run(["git", "config", "user.name", "toolmarket deploy"], work)
    (work / ".git" / "config").write_text(
        (work / ".git" / "config").read_text(encoding="utf-8")
        + "\n[credential]\n\thelper =\n", encoding="utf-8")
    _run(["git", "add", "-A"], work)
    _run(["git", "commit", "-q", "-m",
          "Deploy toolmarket API to a Docker Space"], work)
    url = f"https://{repo.split('/')[0]}:{token}@huggingface.co/spaces/{repo}"
    proc = subprocess.run(["git", "push", "-q", "--force", url, "main:main"],
                          cwd=str(work), capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.exit("push failed:\n"
                 + proc.stderr.replace(token, "***").strip())
    print(f"pushed to {repo}")


def wait_for_runtime(token: str, repo: str, timeout: float = 900.0) -> str:
    """Poll until the Space is running or has failed. Returns the final stage.

    Polling rather than sleeping a fixed amount: a Docker Space build takes
    anywhere from one to eight minutes depending on whether the layer cache is
    warm, and a script that guesses will either report success before the build
    finished or wait eight minutes every time.
    """
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        status, body = _api(f"https://huggingface.co/api/spaces/{repo}", token)
        if status == 200 and isinstance(body, dict):
            stage = str((body.get("runtime") or {}).get("stage") or "?")
            if stage != last:
                print(f"  [{time.strftime('%H:%M:%S')}] {stage}")
                last = stage
            if stage in ("RUNNING", "BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR",
                         "PAUSED", "STOPPED"):
                return stage
        time.sleep(10)
    return last or "TIMEOUT"


def tail_logs(token: str, repo: str, kind: str = "build", lines: int = 40) -> None:
    url = f"https://huggingface.co/api/spaces/{repo}/logs/{kind}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not read {kind} logs: {exc})")
        return
    # The endpoint is SSE-ish; keep the payload lines and drop the framing.
    keep = [l[6:] if l.startswith("data: ") else l for l in text.splitlines()
            if l.strip() and l.strip() != "data:"]
    print(f"--- {kind} log (last {lines}) ---")
    for line in keep[-lines:]:
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="toolmarket")
    ap.add_argument("--mode", choices=sorted(MODES), default="gradio",
                    help="gradio (free) or docker (needs a PRO account)")
    ap.add_argument("--namespace", default=None,
                    help="HF user or org; defaults to the token's own account")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args()

    env = load_env()
    token = env.get("HF_TOKEN", "")
    if not token:
        sys.exit(".deploy.env has no HF_TOKEN")
    namespace = args.namespace or env.get("HF_NAMESPACE") or whoami(token)
    print(f"publishing as {namespace}/{args.name} (mode={args.mode})")

    repo = ensure_space(token, namespace, args.name, str(MODES[args.mode]["sdk"]))

    work = Path(tempfile.mkdtemp(prefix="hf-space-"))
    try:
        assemble(work, args.mode)
        push(work, repo, token)
    finally:
        if args.keep_workdir:
            print(f"workdir kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    url = f"https://huggingface.co/spaces/{repo}"
    if args.no_wait:
        print(f"pushed; check {url}")
        return 0

    stage = wait_for_runtime(token, repo)
    print(f"final stage: {stage}")
    if stage != "RUNNING":
        tail_logs(token, repo, "build")
        print(f"see {url}")
        return 1

    # A running Space that serves errors is not a successful deploy. The health
    # endpoint is process-local and touches no dependency, so it is the honest
    # signal that the container is up; /ready and /metrics are checked because a
    # Space that 500s on either is a Space whose instrumentation is broken.
    #
    # The hostname is the Space's *subdomain* form; the embed form is different
    # and returns the UI shell rather than the API, which is why this is spelled
    # out instead of derived from the page URL.
    slug = repo.replace("/", "-").replace("_", "-").replace(".", "-").lower()
    base = f"https://{slug}.hf.space"
    import urllib.error as _ue
    for path in ("/health", "/ready", "/metrics"):
        try:
            with urllib.request.urlopen(base + path, timeout=90) as resp:
                body = resp.read(400).decode("utf-8", "replace")
                print(f"GET {path} -> {resp.status}: {body.splitlines()[0][:120]}")
        except _ue.HTTPError as exc:
            print(f"GET {path} -> {exc.code}")
        except Exception as exc:  # noqa: BLE001
            print(f"GET {path} -> {exc}")
    print(f"\nlive at {base}\ndocs at {base}/docs\nui at {base}/ui\nspace page: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
