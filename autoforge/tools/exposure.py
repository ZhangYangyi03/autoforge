"""Which tool schemas ride in every request, and which are merely listed.

The library is not the prompt. Those became the same thing by accident: every
tool the registry exposes is serialised into every model call, and nothing ever
looked at whether the tool had been used. Measured on this machine 2026-09-28:
204 visible schemas, 106,503 characters, ~33k tokens, paid on every request of
every turn -- of which 61 tools had never been called once, and 147 of the 148
tools ever called sat behind them.

The fix is not deletion and not a smaller library. A tool that is never called
is still worth keeping -- it was forged for a need that may come back -- but it
does not need its full parameter schema in front of the model to be *findable*.
So exposure splits in two:

    hot   -- called within the window, plus a floor of tools that carry the
             interaction itself (run_python, forge_tool, describe_tool, ...).
             Full schema, as before.
    cold  -- everything else. Name, parameters and a one-line description:
             enough to recognise the tool and ask for it by name.

A cold tool is not hidden and not retired. It costs a line, it is callable as
soon as the model names it, and `describe_tool` hands back the full schema on
request. The saving is real only because the model does not need to know the
argument shapes of 61 tools it is not calling this turn -- and the price of
being wrong is one extra call, not a lost capability.

A third tier is deliberately absent: dropping the name entirely and making the
model guess from a search box. The market does that, and the reason a name-and-
line list beats it is that it costs 20 characters and removes the whole class of
failure where the tool exists and the model never learns it does.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

#: The tools that carry the interaction itself. A floor, not a preference: these
#: are the ones a model reaches for when it does not yet know what it needs, so
#: they must never be demoted to a name on a list. `run_python` is where a
#: one-off lookup goes; `forge_tool`/`describe_tool`/`list_tools` are how a cold
#: tool is found again; `remember`/`recall` carry facts across sessions;
#: `terminate` ends the run. Everything else earns its place by being called.
HOT_FLOOR: frozenset[str] = frozenset({
    "run_python", "describe_tool", "list_tools", "forge_tool", "terminate",
    "remember", "recall",
})

#: Description characters kept in the cold list. Long enough to say what the
#: tool is for, short enough that 120 of them cost less than ten hot schemas.
COLD_DESC_CHARS = 100

#: A cold MCP tool is shown per server once a server has this many. Below it the
#: individual lines are already short enough to be worth reading one by one.
MCP_COLLAPSE_MIN = 5

#: Prefix every mounted MCP tool carries. The server name sits between the
#: prefix and the next "__" -- mcp__rea__open_binary -> rea.
MCP_PREFIX = "mcp__"

#: Parameter names shown in the cold list. The point of showing any is that
#: "read_magic_value(path)" is recognisable where "read_magic_value" alone is
#: only a guess; the point of capping is that the list is one line per tool.
COLD_PARAM_NAMES = 4


def _names(schema: dict[str, Any]) -> list[str]:
    fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    params = fn.get("parameters") if isinstance(fn, dict) else None
    if not isinstance(params, dict):
        return []
    props = params.get("properties")
    return list(props) if isinstance(props, dict) else []


def _schema_name(schema: dict[str, Any]) -> str:
    fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    return str(fn.get("name") or "")


def _schema_desc(schema: dict[str, Any]) -> str:
    fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    return str(fn.get("description") or "").strip().split("\n")[0]


def cold_line(name: str, schema: dict[str, Any],
              desc_override: str = "") -> str:
    """One line standing in for a tool that is not being handed over in full."""
    params = _names(schema)
    shown = ", ".join(params[:COLD_PARAM_NAMES])
    if len(params) > COLD_PARAM_NAMES:
        shown += ", ..."
    desc = (desc_override or _schema_desc(schema)).strip().replace("\n", " ")
    if len(desc) > COLD_DESC_CHARS:
        desc = desc[:COLD_DESC_CHARS - 1].rstrip() + "\u2026"
    sig = f"{name}({shown})" if shown else f"{name}()"
    return f"  - {sig} \u2014 {desc}" if desc else f"  - {sig}"


def split(
    schemas: Sequence[dict[str, Any]],
    days_since: Callable[[str], float | None],
    *,
    window_days: float = 7.0,
    floor: Iterable[str] = HOT_FLOOR,
    protect: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    """Split schemas into (hot, cold lines, counts).

    `days_since(name)` returns how long ago the tool was last called, or None if
    there is no record of it ever being called. A tool with no record is cold --
    that is the whole point: never called is the strongest evidence available
    that its argument list does not need to be in front of the model this turn.

    `window_days` bounds the number of schemas by *behaviour*, not by a count.
    A hard cap needs a tie-break, and every tie-break it could have (name order,
    cost, age) demotes a tool for a reason unrelated to whether it is wanted. A
    window demotes for exactly one reason -- nothing reached for it in a week --
    and self-corrects the moment something does.

    `protect` is the caller's escape hatch: names that must stay hot this run
    whatever the ledger says (a tool the task text names, a topology's role
    whitelist). Kept separate from `floor` so a caller's reason and the
    framework's reason are not folded into one set.
    """
    keep = set(floor) | set(protect)
    hot: list[dict[str, Any]] = []
    lines: list[str] = []
    # Cold MCP tools are collapsed per server. Measured on this host
    # 2026-11-25: 207 mcp__rea__ entries, never called once, cost 33,681
    # characters of cold list -- about 11k tokens on every request, to describe
    # a reverse-engineering server that was not even started. A cold line earns
    # its place by being recognisable and askable-for-by-name; 207 of them in a
    # row is a wall nobody reads. One line per server keeps the whole capability
    # visible (mcp_servers() lists them all) at 1/200th of the price.
    groups: dict[str, list[str]] = {}
    for schema in schemas:
        name = _schema_name(schema)
        age = days_since(name)
        if name in keep or (age is not None and age <= window_days):
            hot.append(schema)
            continue
        if name.startswith(MCP_PREFIX):
            server = name[len(MCP_PREFIX):].split("__", 1)[0] or "?"
            groups.setdefault(server, []).append(name)
            continue
        lines.append(cold_line(name, schema))
    for server, names in sorted(groups.items()):
        if len(names) < MCP_COLLAPSE_MIN:
            for n in names:
                schema = next((x for x in schemas if _schema_name(x) == n), None)
                if schema is not None:
                    lines.append(cold_line(n, schema))
            continue
        lines.append(
            "  - mcp__%s__* \u2014 %d tools from the %r MCP server, none called "
            "yet; mcp_servers() lists them, and any one is callable by name."
            % (server, len(names), server))
    stats = {
        "hot": len(hot),
        "cold": len(lines),
        "total": len(schemas),
    }
    return hot, lines, stats


def manifest_header(stats: dict[str, int], window_days: float) -> str:
    return (
        f"Tools you can call but are not carrying in full ({stats['cold']} of "
        f"{stats['total']}): not called in {window_days:g} days, so only their "
        "name, arguments and one line are shown. They are callable as they are "
        "\u2014 use one directly, or call describe_tool(name) for its full "
        "parameter schema first."
    )


def task_hint_line(pairs: "Iterable[tuple[str, str]]", stats: dict[str, int] | None = None) -> str:
    """The ranker's shortlist, said out loud in the request.

    Measured 2026-11-25 on this machine: `_task_tool_candidates` had ranked
    tools for 112 user turns and the model called one of its picks twice. The
    ranking was not the problem -- it was never spoken. The candidates were put
    into `registry.protect`, which only decides whether a schema rides in full,
    so a tool the ranker found and the model never noticed stayed exactly as
    invisible as before.

    Four names and their one-line descriptions cost about 400 characters. That
    is cheaper than filling the forge slot for a capability that already exists
    under a name nobody said.
    """
    seen: list[str] = []
    seen_names: set[str] = set()
    for name, desc in pairs:
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        # An MCP or builtin name is already advertised in full elsewhere; the
        # hint is for the tool that would otherwise be looked for and missed.
        d = (desc or "").strip().replace("\n", " ")
        if len(d) > 90:
            d = d[:89].rstrip() + "\u2026"
        seen.append(f"  - {name} \u2014 {d}" if d else f"  - {name}")
    if not seen:
        return ""
    head = ("Tools that fit this task, from this machine's own library "
            "(ranked from the task text, not guesses):")
    return head + "\n" + "\n".join(seen) + \
        "\nCall one directly; describe_tool(name) gives its full schema first."
