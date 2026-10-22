# Every knob is a variable with a default, so `terraform apply` with no tfvars
# brings the whole stack up. The defaults are deliberately the same values
# docker-compose.yml uses: two deployment descriptors for one stack that
# disagreed about a port would be a bug farm, and the only thing worse than no
# second descriptor is one that is subtly different from the first.

variable "stack_name" {
  description = "Prefix for every resource this config creates. Change it to run a second copy of the stack side by side; the container DNS aliases stay fixed at `postgres`/`redis` so the DSNs do not have to move with it."
  type        = string
  default     = "toolmarket-tf"
}

variable "api_port" {
  description = "Host port the API is published on."
  type        = number
  default     = 8000
}

variable "postgres_password" {
  description = "Password for the toolmarket role. Local-only default; override it for anything reachable."
  type        = string
  default     = "toolmarket"
  sensitive   = true
}

variable "redis_maxmemory" {
  description = "Redis maxmemory. 128 MB matches compose: single resource records with a 30 s TTL, so headroom past this buys entries that expire before they are read."
  type        = string
  default     = "128mb"
}

variable "app_image_tag" {
  description = "Tag for the image built from the repository Dockerfile."
  type        = string
  default     = "toolmarket:tf"
}

variable "task_queue" {
  description = "`celery` runs a separate worker container; `inline` runs evolutions on a thread of the API process and needs no worker. Celery is the default here because standing up the worker is the point of doing this with Terraform rather than compose."
  type        = string
  default     = "celery"

  validation {
    condition     = contains(["celery", "inline"], var.task_queue)
    error_message = "task_queue must be `celery` or `inline`."
  }
}
