"""Delegation backends: which model, provider and endpoint a child runs on.

Ported from hermes-agent's ``agent/delegation_provider.py`` + ``plugins/delegation/default``
(MIT). The split is the whole idea, and it is worth stating because the obvious
alternative is worse:

    core owns the lifecycle  -- spawn, isolation, roles, depth, timeout, state
    a backend owns the route -- which model/endpoint/effort the child uses

Without that split, supporting a second provider means editing the spawn path,
and every new provider is a chance to break the first one. With it, adding a
route is a small class that answers one question and can be swapped per task.

A configured backend that is not registered FAILS CLOSED: the spawn is refused
with the name of the backend and the fix. Quietly falling back to "default"
would run the child somewhere the caller did not ask for, and the caller would
only find out from the bill.

Route keys, deliberately the same vocabulary as hermes so a config file ports:
    model, provider, base_url, api_key, api_mode, reasoning_effort,
    request_overrides, fallback_providers, label
An empty route means "inherit the parent's", which is the safe default: a child
running on the parent's model is a known quantity.
"""
from __future__ import annotations

import abc
from typing import Any, Mapping

#: Per-task names core owns. A backend may add parameters; it never redefines
#: these, and the raw credential fields never become model-facing.
RESERVED_TASK_PARAMETERS = frozenset({
    "goal", "context", "index", "role", "toolsets", "timeout_seconds",
    "provider", "model", "base_url", "api_key", "api_mode",
})

#: What a route dict may carry. Anything else is a typo, and a typo silently
#: ignored is a child running on the wrong model with no error to point at.
ROUTE_KEYS = frozenset({
    "model", "provider", "base_url", "api_key", "api_mode",
    "reasoning_effort", "request_overrides", "fallback_providers", "label",
})

REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})


class DelegationRouteError(Exception):
    """A backend was asked for and cannot serve; the spawn must be refused."""


class DelegationProvider(abc.ABC):
    """Answers one question: which route does this task take?

    Everything crossing this interface is plain JSON data -- no live agent
    objects -- so a provider can run in-process or out of it unchanged.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def summary(self) -> str:
        return ""

    def is_available(self) -> bool:
        """Whether this backend can serve right now. Raising is treated as False."""
        return True

    def task_schema(self) -> dict[str, Any]:
        """Extra per-task parameters this backend adds. Empty by default."""
        return {}

    @abc.abstractmethod
    def route_task(self, task: Mapping[str, Any],
                   config: Mapping[str, Any]) -> dict[str, Any]:
        """The route for one task. Raises DelegationRouteError to refuse it."""

    def settings(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """This backend's own settings, kept under delegation.backends.<name>."""
        backends = config.get("backends") if isinstance(config, Mapping) else None
        if isinstance(backends, Mapping):
            own = backends.get(self.name)
            if isinstance(own, Mapping):
                return own
        return {}


def validate_route(route: Mapping[str, Any], *, backend: str = "") -> dict[str, Any]:
    """Check a route before it is used: unknown keys and bad efforts refuse."""
    if not isinstance(route, Mapping):
        raise DelegationRouteError(
            f"backend {backend or '?'!r} returned {type(route).__name__}, not a route dict")
    unknown = set(route) - ROUTE_KEYS
    if unknown:
        raise DelegationRouteError(
            f"backend {backend or '?'!r} returned unknown route key(s) "
            f"{sorted(unknown)}; known: {sorted(ROUTE_KEYS)}")
    effort = route.get("reasoning_effort")
    if effort is not None and effort not in REASONING_EFFORTS:
        raise DelegationRouteError(
            f"reasoning_effort {effort!r} is not one of {sorted(REASONING_EFFORTS)}")
    return dict(route)


class DefaultDelegationProvider(DelegationProvider):
    """Every task takes the `delegation` section, else the parent's own route."""

    @property
    def name(self) -> str:
        return "default"

    @property
    def display_name(self) -> str:
        return "Default"

    @property
    def summary(self) -> str:
        return ("Every child uses the delegation.model/provider setting, "
                "else the parent's model.")

    def route_task(self, task, config):
        return validate_route(
            {k: v for k, v in config.items() if k in ROUTE_KEYS}, backend=self.name)


class Registry:
    """Name -> provider, plus the fail-closed resolution the rest of the code uses."""

    def __init__(self) -> None:
        self._providers: dict[str, DelegationProvider] = {}

    def register(self, provider: DelegationProvider, replace: bool = True) -> None:
        key = provider.name.strip().lower()
        if not key:
            raise DelegationRouteError("a delegation backend needs a name")
        if not replace and key in self._providers:
            raise DelegationRouteError(f"backend {key!r} is already registered")
        self._providers[key] = provider

    def names(self) -> list[str]:
        return sorted(self._providers)

    def get(self, name: str) -> DelegationProvider | None:
        return self._providers.get(str(name or "").strip().lower())

    def active(self, config: Mapping[str, Any] | None = None) -> DelegationProvider:
        """The backend the config selects, or raise naming the fix.

        Unset means `default`. A named-but-missing or unavailable backend is an
        error, never a silent fallback -- see the module docstring.
        """
        config = config or {}
        raw = config.get("backend") if isinstance(config, Mapping) else None
        name = str(raw).strip().lower() if isinstance(raw, str) and str(raw).strip() else "default"
        provider = self.get(name)
        if provider is None:
            raise DelegationRouteError(
                f"delegation backend {name!r} is not registered (have: "
                f"{self.names() or 'none'}). Register it, or set "
                f"delegation.backend to one of those.")
        try:
            ok = provider.is_available()
        except Exception as exc:  # noqa: BLE001 - a backend that throws is unavailable
            raise DelegationRouteError(
                f"delegation backend {name!r}.is_available() raised "
                f"{type(exc).__name__}: {exc}") from exc
        if not ok:
            raise DelegationRouteError(
                f"delegation backend {name!r} is registered but reports itself "
                f"unavailable. Fix its configuration or credentials, or switch "
                f"delegation.backend.")
        return provider

    def report(self, config: Mapping[str, Any] | None = None) -> str:
        lines = []
        for name in self.names():
            p = self._providers[name]
            try:
                ok = p.is_available()
            except Exception:  # noqa: BLE001
                ok = False
            lines.append(f"  {name:<16} {'available' if ok else 'UNAVAILABLE'}"
                         f"  {p.summary}")
        try:
            chosen = self.active(config).name
        except DelegationRouteError as exc:
            chosen = f"NONE ({exc})"
        return "Delegation backends:\n" + ("\n".join(lines) or "  (none)") + \
               f"\nActive: {chosen}"


def default_registry() -> Registry:
    reg = Registry()
    reg.register(DefaultDelegationProvider())
    return reg


__all__ = [
    "DelegationProvider", "DefaultDelegationProvider", "DelegationRouteError",
    "Registry", "default_registry", "validate_route",
    "RESERVED_TASK_PARAMETERS", "ROUTE_KEYS", "REASONING_EFFORTS",
]
