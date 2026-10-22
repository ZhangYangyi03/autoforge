# syntax=docker/dockerfile:1.7
#
# Two stages, and the split is about what the *runtime* image is allowed to
# contain rather than about size. The build stage needs git (the pinned engine
# is a git dependency) and a compiler toolchain; the runtime stage needs neither,
# and shipping them means shipping a way to fetch and execute arbitrary code into
# a container that runs untrusted candidate tools. The final image has no git,
# no compiler and no network tooling beyond what Python itself provides.
#
# The engine pin in pyproject.toml is honoured here for free: `pip install .`
# reads the same `autoforge @ git+...@<sha>` line the README's reproduction
# instructions do, so the image and the documented reproduction cannot drift.

# -- build ------------------------------------------------------------------
FROM python:3.11-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# pyproject first so a source-only edit does not invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY toolmarket/ ./toolmarket/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
 && pip install ".[api,postgres,redis,worker]"

# -- runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    # The substrate's own defaults. Overridden per service by compose; these are
    # what you get from a bare `docker run`, and they are chosen so a bare run
    # *works*: in-memory store, in-process cache, inline queue.
    #
    # `REDIS_URL` is deliberately *unset* rather than "none". `none` means
    # NullCache, which stores nothing — and task records live in the cache, so a
    # container started that way would accept an async evolution, hand back a task
    # id, and 404 on every poll. Unset means MemoryCache, which is the right
    # default for a single process: it remembers the task it just accepted.
    # `docker compose` overrides this with the real Redis.
    TOOLMARKET_STORE=":memory:" \
    TASK_QUEUE="inline" \
    CACHE_TTL=30 \
    PORT=8000

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app

COPY --from=build /opt/venv /opt/venv

USER app
WORKDIR /home/app

EXPOSE 8000

# A healthcheck that uses the interpreter already in the image rather than
# installing curl purely to probe itself, and that hits `/health` (which touches
# nothing) instead of `/ready` — a container whose health depends on Postgres
# would be marked unhealthy during a database restart and killed by anything
# that acts on that, which is how a database blip becomes a full outage.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=4 \
  CMD python -c "import os,sys,urllib.request; \
      sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health', timeout=3).status==200 else 1)"

# `served_app` is the instrumented app: rate limiter + metrics middleware on top
# of the routes. Running `app` here would serve the same endpoints with no
# /metrics traffic ever being counted, which is the kind of difference nobody
# notices until the dashboard is flat during an incident.
CMD ["sh", "-c", "exec uvicorn toolmarket.api.main:served_app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
