# The stack as a dependency graph rather than a script.
#
# Why this exists next to docker-compose.yml, stated honestly: for a purely local
# run, compose is the better tool and this file does not change that. What it adds
# is that the topology is a *graph with a plan* — `terraform plan` says what will
# change before anything does, and the same variables can be re-pointed at a
# remote provider without rewriting the topology. Compose has neither property.
# The two descriptors are kept equivalent on purpose; where they differ, it is
# noted at the point of difference.
#
# The dependency edges are mostly implicit (a container references the network
# and the image it needs) and one is deliberately absent: nothing here waits for
# Postgres to be *healthy* before starting the API, because the app already
# tolerates that race — `make_store` calls `store.wait_ready(attempts=30,
# delay=1.0)` for the Postgres backend. Modelling it here as well would be the
# second copy of one guarantee, and the copy is the one that goes stale.

provider "docker" {}

# -- shared plumbing --------------------------------------------------------

resource "docker_network" "substrate" {
  name = "${var.stack_name}-net"

  # A user-defined network, not the default bridge, for one reason that matters:
  # name resolution. On a user-defined network containers resolve each other by
  # alias, so the DSNs below can say `postgres:5432` and survive the container
  # being replaced with a different IP. On the default bridge they cannot, and
  # the DSN would have to be rewritten from an IP on every rebuild.
}

resource "docker_volume" "pgdata" {
  name = "${var.stack_name}-pgdata"
}

# -- images -----------------------------------------------------------------

resource "docker_image" "postgres" {
  name = "postgres:16-alpine"
  # Kept after `destroy`: re-pulling 80 MB to prove a teardown worked is a tax
  # on the next `apply`, and the image is not what this config owns.
  keep_locally = true
}

resource "docker_image" "redis" {
  name         = "redis:7-alpine"
  keep_locally = true
}

resource "docker_image" "app" {
  name = var.app_image_tag

  build {
    context    = abspath("${path.module}/..")
    dockerfile = "Dockerfile"
    # The runtime stage: no git, no compiler, no way to fetch and execute code
    # from inside the container that runs untrusted candidate tools.
    target = "runtime"
  }

  # Rebuild when the inputs to the image change — and there are three of them,
  # not the two that are easy to remember.
  #
  #   * `Dockerfile` — the stages themselves.
  #   * `pyproject.toml` — the dependency pins, including the autoforge commit.
  #     That pin is the one input that changes the enforcement engine's behaviour
  #     without changing a line of this repository.
  #   * the package source. `COPY toolmarket/ ./toolmarket/` puts every module in
  #     the image, so a one-line fix to a Python file is an input to the build
  #     exactly as much as the Dockerfile is. Omitting this is the trap: Terraform
  #     sees an unchanged image `name`, keeps the stale image, and the symptom is
  #     a fix that passes every local test and is silently absent from the
  #     container — which is the worst possible shape for this bug, because the
  #     evidence points at the code rather than at the image.
  #
  # `fileset` + `filesha256` rather than the `archive_file` data source: this
  # needs the built-in provider only, so `terraform init` stays a one-provider
  # download, and hashing a sorted file list is deterministic without writing an
  # intermediate archive anywhere.
  triggers = {
    dockerfile = filesha256("${path.module}/../Dockerfile")
    pyproject  = filesha256("${path.module}/../pyproject.toml")
    source = sha256(join("", [
      for f in sort(fileset("${path.module}/../toolmarket", "**/*.py")) :
      filesha256("${path.module}/../toolmarket/${f}")
    ]))
  }
}

# -- data -------------------------------------------------------------------

resource "docker_container" "postgres" {
  name     = "${var.stack_name}-postgres"
  image    = docker_image.postgres.image_id
  restart  = "unless-stopped"
  must_run = true

  env = [
    "POSTGRES_USER=toolmarket",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=toolmarket",
  ]

  volumes {
    volume_name    = docker_volume.pgdata.name
    container_path = "/var/lib/postgresql/data"
  }

  # `-U` and `-d` are not optional. Without them this runs as the `postgres`
  # superuser against the `postgres` database and reports healthy — which is
  # exactly the state in which the app's own database does not exist yet, so the
  # probe passes while every request fails.
  healthcheck {
    test         = ["CMD-SHELL", "pg_isready -U toolmarket -d toolmarket"]
    interval     = "5s"
    timeout      = "5s"
    retries      = 12
    start_period = "10s"
  }

  networks_advanced {
    name = docker_network.substrate.name
    # The alias, not the container name: the DSNs say `postgres`, and binding
    # that here means `stack_name` can change without touching a connection
    # string. The container name carries the prefix for a human reading
    # `docker ps`; the alias is for the code.
    aliases = ["postgres"]
  }
}

resource "docker_container" "redis" {
  name     = "${var.stack_name}-redis"
  image    = docker_image.redis.image_id
  restart  = "unless-stopped"
  must_run = true

  # Persistence off, because nothing here treats Redis as durable: the cache is a
  # cache, task records are operational, and a broker's queue is re-submittable.
  # Persisting it would buy an fsync per evolution for the privilege of restoring
  # a stale cache across restarts.
  #
  # `--save` with **no argument**, not `--save ""`. They are the same directive
  # to Redis — a `save` line with no arguments clears the save points — but the
  # provider rejects an empty string in `command` outright ("values for command
  # may not be empty"), and it is right to: `command` becomes argv directly, with
  # no shell to collapse a quoted empty string, so `""` is not a value Redis
  # could distinguish from a bug. Verified against redis:7-alpine that the
  # no-argument form leaves `CONFIG GET save` empty and the server alive.
  command = [
    "redis-server", "--save", "--appendonly", "no",
    "--maxmemory", var.redis_maxmemory, "--maxmemory-policy", "allkeys-lru",
  ]

  healthcheck {
    test         = ["CMD", "redis-cli", "ping"]
    interval     = "5s"
    timeout      = "3s"
    retries      = 12
    start_period = "5s"
  }

  networks_advanced {
    name    = docker_network.substrate.name
    aliases = ["redis"]
  }
}

# -- application ------------------------------------------------------------

locals {
  # One definition of the substrate's wiring, consumed by both app containers.
  # The API and the worker must agree about which database, which cache and which
  # broker they are talking to; restating this per container is how the two drift
  # and how an evolution gets enqueued to a broker nobody is consuming.
  substrate_env = [
    "TOOLMARKET_STORE=postgresql://toolmarket:${var.postgres_password}@postgres:5432/toolmarket",
    "REDIS_URL=redis://redis:6379/0",
    "TASK_QUEUE=${var.task_queue}",
    "CELERY_BROKER_URL=redis://redis:6379/1",
    "CELERY_RESULT_BACKEND=redis://redis:6379/2",
    "CACHE_TTL=30",
    "RATE_LIMIT=120",
    "RATE_LIMIT_WINDOW=60",
    # Trusted only because the sole way in is this network. On a host where the
    # API is reachable directly this would let any client forge its own identity
    # and choose its own rate-limit bucket.
    "TRUST_PROXY=1",
    "PYTHONUNBUFFERED=1",
  ]
}

resource "docker_container" "api" {
  name     = "${var.stack_name}-api"
  image    = docker_image.app.image_id
  restart  = "unless-stopped"
  must_run = true

  env = concat(local.substrate_env, ["PORT=8000"])

  ports {
    internal = 8000
    external = var.api_port
  }

  networks_advanced {
    name = docker_network.substrate.name
  }

  # Readiness, not liveness, is what a caller should act on — but this is a
  # *container* healthcheck, and acting on it means restarting. An app whose
  # readiness is down because Postgres is restarting must not be restarted, or a
  # database blip becomes a full outage. So the check is process-local and
  # `/ready` is left to whatever is actually routing traffic.
  healthcheck {
    test = ["CMD-SHELL",
      "python -c \"import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)\"",
    ]
    interval     = "15s"
    timeout      = "5s"
    retries      = 4
    start_period = "20s"
  }
}

resource "docker_container" "worker" {
  # `count`, not `task_queue == "celery" ? 1 : 0` on the same resource, so the
  # container simply does not exist in inline mode rather than existing and being
  # inert. An inert worker is a container that shows up in `docker ps` and in
  # every resource count while doing nothing.
  count = var.task_queue == "celery" ? 1 : 0

  name     = "${var.stack_name}-worker"
  image    = docker_image.app.image_id
  restart  = "unless-stopped"
  must_run = true

  env = local.substrate_env

  # The image's CMD is uvicorn; this replaces it. `--concurrency=1` and
  # `--prefetch-multiplier=1` rather than the default of `ncpu`: a laptop with
  # 3 GB free does not want eight resident copies of the substrate serving a
  # queue that is empty, and an unprefetched task is one that a slow worker has
  # not already claimed from a peer.
  command = [
    "celery", "-A", "toolmarket.worker:celery_app", "worker",
    "--loglevel=info", "--concurrency=1", "--prefetch-multiplier=1",
  ]

  networks_advanced {
    name = docker_network.substrate.name
  }

  # `inspect ping` is the only probe that proves the worker is *consuming*
  # rather than merely alive: a worker whose broker connection has died still
  # has a running process, and a healthcheck that only proves liveness would
  # report that corpse as healthy.
  #
  # `$HOSTNAME` with ONE dollar, which is not a typo for the `$$HOSTNAME` in
  # docker-compose.yml. Compose does its own `$VAR` interpolation and therefore
  # needs `$$` to escape a literal dollar; Terraform does no such interpolation
  # (only `${...}` is special to HCL) and passes `$$` through to the container
  # verbatim. The container's `sh` then reads `$$` as its own PID, so `-d`
  # addressed `celery@1234HOSTNAME`, a node that never exists — the probe failed
  # forever with "No nodes replied within time constraint" while the worker was
  # perfectly healthy. Porting a healthcheck between the two tools means
  # re-checking every dollar sign, because the same text is not the same string.
  healthcheck {
    test = ["CMD-SHELL",
      "celery -A toolmarket.worker:celery_app inspect ping -d celery@$HOSTNAME --timeout 10",
    ]
    interval     = "30s"
    timeout      = "15s"
    retries      = 3
    start_period = "30s"
  }
}
