# Changelog

All notable changes are documented here. This file starts with the 2026-11
GitHub portfolio audit and is itemised so a reviewer can verify each entry
against the commit.

## [Unreleased] — 2026-11 portfolio audit fixes

- Fixed `CITATION.cff`: it declared Apache-2.0 while `LICENSE` and `pyproject.toml` are MIT.
- Fixed a red Python 3.10 CI job. Two tests hard-coded `znver5` as an ISA the host cannot have, but current GitHub `ubuntu-latest` runners can resolve `-march=native` to `znver5`, where `runs_here()` correctly takes the `target == info.arch` fast path. The tests now use a target no compiler can resolve, and the matching-arch branch has its own explicit assertion. Product code was already correct.
- Added the "Independent reproduction" issue template, a portfolio-status contact link, and a reproduction section to `CONTRIBUTING.md`.
- Added this changelog.

## Audit context

- Full portfolio assessment: `GitHub-项目综合评价.md` (overall 6.8/10; external validation 1.5/10). At audit time `autoforge` was one of two repositories whose latest CI run was red.
- Canonical maintainer identity: `ZhangYangyi03` / Yangyi Zhang.
- Nothing here claims external validation. The reproduction template is an invitation, not a result.
