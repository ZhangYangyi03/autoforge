"""HuggingFace Space entrypoint — the app, served from a plain Python process.

**On HuggingFace's free plan this cannot run.** The API's own words, verified by
asking it:

    POST /api/repos/create {"type": "space", "sdk": "gradio"}  ->  402
    "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on
     free cpu-basic requires a PRO subscription."

So on a free account the only thing HuggingFace will host is a *Static* Space —
HTML and JavaScript with no server — which cannot run this API at all. Both the
Gradio and the Docker route need a PRO subscription; neither is a cheaper
alternative to the other. This file is therefore not "the free path":

  * on a PRO account it is the faster of the two, because nothing has to be built
    from a Dockerfile;
  * on any other host that runs a Python process (a VM, a container platform that
    builds from a repo, a laptop) it is how the API is served with a UI attached;
  * `--mode docker` remains the canonical artefact, because it is the same image
    `docker compose up` builds and CI smoke-tests.

The deployment story that is actually free lives in the README (`Render`, and
`docker compose` locally). This file exists so that the Space deployments are one
command when the account allows them, not as an argument that they are free.

The API served here is `toolmarket.api.main.create_served_app` — rate limiter and
metrics middleware included — not a copy of it. Every route from the README is at
the root, and `/ui` is a small panel that fetches those routes over HTTP.

    # locally, exactly as a Space runs it
    python deploy/huggingface/gradio_app.py

    # and then
    curl localhost:7860/health
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The Space repo has this file at its root next to `toolmarket/`, and the Space
# build does not `pip install` the project itself — so the package is imported
# from the checkout. Explicit rather than relying on the cwd being on sys.path,
# which is a detail of how each runner invokes the script.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_api():
    """The served app, with the Space's own configuration."""
    # Single process, so: in-memory substrate, in-process cache, inline queue.
    # Set before the app is built, because the registry, the cache and the queue
    # all read their backend from the environment at construction time.
    os.environ.setdefault("TOOLMARKET_STORE", ":memory:")
    os.environ.setdefault("TASK_QUEUE", "inline")
    os.environ.setdefault("CACHE_TTL", "30")
    # A public URL with no auth in front of it. 120 requests/minute/client is
    # generous for a demo and the difference between a Space that survives being
    # posted somewhere and one that gets its CPU quota suspended. The proxy in
    # front of a Space sets X-Forwarded-For, so the limit is per visitor rather
    # than per Space — which is the whole point of the header being trusted here
    # and nowhere else.
    os.environ.setdefault("RATE_LIMIT", "120")
    os.environ.setdefault("RATE_LIMIT_WINDOW", "60")
    os.environ.setdefault("TRUST_PROXY", "1")

    from toolmarket.api.main import create_served_app

    return create_served_app()


def build_ui(api):
    """Mount a small Gradio panel on the API.

    Deliberately minimal: the dashboard and the protocol-conformant views are the
    console in `toolmarket/web.py`, and duplicating them here would be a second
    implementation to keep in sync. This panel answers one question — "is this
    thing alive, and what does it say about itself" — by fetching the API's own
    endpoints, which is the same thing `curl` would do.
    """
    import gradio as gr
    import json
    import urllib.request

    def _get(path: str) -> str:
        # Through the socket, not through an internal call: the panel then proves
        # the served surface works, which is the only interesting claim. Inlining
        # the call would still render a nice page while /health returned 500.
        port = os.environ.get("PORT", "7860")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=15) as r:
                body = r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            return f"{path} -> {type(exc).__name__}: {exc}"
        try:
            return json.dumps(json.loads(body), indent=2)[:4000]
        except json.JSONDecodeError:
            return body[:4000]

    with gr.Blocks(title="toolmarket") as demo:
        gr.Markdown(
            "# toolmarket\n"
            "A protocol-registered resource platform for evolvable agent tools.\n\n"
            "The API is at the root of this Space: `/docs` for the interactive "
            "schema, `/metrics` for Prometheus exposition, `/health` and `/ready` "
            "for the probes. The buttons below fetch those endpoints over HTTP, so "
            "what you see is what a client would get."
        )
        out = gr.Code(label="response", language="json")
        with gr.Row():
            for label, path in (("health", "/health"), ("ready", "/ready"),
                                ("stats", "/stats"), ("resources", "/resources")):
                gr.Button(label).click(lambda p=path: _get(p), outputs=out)
        with gr.Row():
            gr.Button("metrics (head)").click(
                lambda: _get("/metrics"), outputs=out)
            gr.Button("open /docs").click(
                None, js="() => window.open('/docs', '_blank')")
        demo.load(lambda: _get("/health"), outputs=out)
    return demo


def main() -> int:
    import uvicorn
    import gradio as gr

    api = build_api()
    # `/ui` rather than `/`: the API's root is its index document, and a UI that
    # stole the root would make `/` mean two different things depending on
    # whether the reader was looking at the code or the browser.
    app = gr.mount_gradio_app(api, build_ui(api), path="/ui")

    port = int(os.environ.get("PORT", "7860"))
    print(f"serving toolmarket on 0.0.0.0:{port}  (ui at /ui, docs at /docs)",
          flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
