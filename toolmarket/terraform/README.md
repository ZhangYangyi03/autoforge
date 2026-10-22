# Terraform: the stack as a graph with a plan

`docker-compose.yml` and this directory describe the same five services. This is
not a replacement for compose and does not pretend to be — for a purely local
run, compose is the better tool: shorter, no state file, no provider download.
What Terraform adds is two properties compose does not have.

**A plan.** `terraform plan` prints every change before any of it happens. That
matters when the thing you are about to change is a database volume whose name
determines whether the data survives.

**A separable topology.** The service graph here is expressed against variables,
not against Docker. Re-pointing it at a remote provider — the same five roles as
an ECS cluster, a Render stack, a Kubernetes namespace — is a change of provider
block and resource types, not a re-derivation of what talks to what. That is the
part that transfers off a laptop, and it is why this exists at all.

## Run it

```sh
cd terraform
terraform init
terraform plan          # read this
terraform apply         # builds the image, starts five containers
```

The same three steps are wrapped as `make tf-init`, `make tf-plan` and
`make tf-apply`, which pass `api_port=$(TF_API_PORT)` — see the port note below.
`make tf-ready` is the verification step, and it takes its URL from the state
rather than from a constant:

```sh
make tf-ready           # curl the ready_url this stack actually published
```

Then verify it actually found the backends rather than falling back:

```sh
terraform output -raw ready_url | xargs curl -s
```

`/ready` reports the backend it connected to. The whole point of this stack over
the free-tier cloud one is that the answer must be `postgres` and `redis`, not
`sqlite` and `memory`.

Tear down:

```sh
terraform destroy       # keeps the Postgres image and the volume; see below
```

## What it creates

| Resource | Purpose |
|---|---|
| `docker_network.substrate` | User-defined network, so containers resolve each other by alias |
| `docker_volume.pgdata` | Postgres data |
| `docker_image.app` | Built from `../Dockerfile`, target `runtime` |
| `docker_container.postgres` | `postgres:16-alpine`, healthchecked on the app's own DB |
| `docker_container.redis` | `redis:7-alpine`, no persistence, LRU at 128 MB |
| `docker_container.api` | Uvicorn, published on `api_port` |
| `docker_container.worker` | Celery, `count = 0` when `task_queue = "inline"` |

## Decisions worth knowing

**The API does not wait for Postgres to be healthy, and that is deliberate.**
The app already tolerates the race — `make_store` calls
`store.wait_ready(attempts=30, delay=1.0)` for the Postgres backend. Adding a
Terraform-level wait would be a second copy of one guarantee, and the copy is the
one that goes stale.

**Container names are prefixed with `stack_name`; DNS aliases are not.** The DSNs
say `postgres:5432` and `redis:6379` because those are the aliases. Change
`stack_name` to run a second copy of the stack alongside the first and every
connection string stays valid.

**The app image rebuilds on `Dockerfile` and `pyproject.toml` hashes.** The
second one is not optional: `pyproject.toml` pins autoforge to a commit, and a
pin that moves changes the enforcement engine's behaviour without changing a line
of this repository. Without the trigger, Terraform would see an unchanged image
name, keep the stale one, and the symptom would be a fix that passes locally and
is silently ignored in the container.

**The API is published on 18001 here, not 8000, and that is the same rule the
Kubernetes path follows.** `docker compose up` owns 8000. A Terraform stack that
also claimed it would fail to bind — or worse, `make tf-ready` would curl
`127.0.0.1:8000`, get a 200 from the *compose* container, and report a stack it
never created as healthy. The target's whole job is to say which backends the
API actually found, so answering from the wrong API is the one failure it cannot
afford. Hence a port of its own (`TF_API_PORT`, overridable) and a URL read out
of the state instead of typed twice.

**State is gitignored and must stay that way.** `terraform.tfstate` holds the
Postgres password in cleartext, because that is what the resource declares. It is
also why `postgres_password` is marked `sensitive` — that keeps it out of plan
output and CI logs, which is a different leak from the state file and the more
common one.

**`keep_locally = true` on the base images.** `destroy` removes containers and
volumes but not the 80 MB of Postgres and Redis you will re-pull on the next
apply. The images are not what this config owns.

## Not here

No remote backend, no locking, no workspaces, no `terraform fmt` in CI. A local
single-operator stack needs none of them, and adding a backend that nothing reads
is the kind of configuration that looks mature and is never exercised.
