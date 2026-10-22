# Provider pinning. `~> 3.0` rather than `>= 3.0`: this config uses the v3
# single-`build`-block form of `docker_image`, and v4 made that a list. A `>=`
# constraint would let a fresh `terraform init` pull v4 and fail at plan time
# with an error that reads like a syntax mistake in this file rather than a
# provider major bump — the least diagnosable kind of breakage, because the file
# in front of you is correct for the version it was written against.
terraform {
  required_version = ">= 1.5"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}
