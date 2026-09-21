# The Windows host side of the distributed node

These files used to live only in `C:\Users\china\autoforge_node\`, which means
the one half of the deployment that actually has to survive a reboot was not
versioned anywhere. A node whose start-up path exists on exactly one machine is
a node nobody can rebuild, and the failure mode is silent: everything works
until the machine is replaced.

## What is here

    start_market.cmd        the shelf (:8000). Binds 0.0.0.0 by design -- the
                            peer machine reads it directly. Logs are appended,
                            not discarded.
    start_node.cmd          the execution node (:8077). Its /market prefix is
                            the tunneled door and keeps its bearer-token gate.
    install_elevated.cmd/.ps1   one elevated run: authorise the peer's ssh key,
                            register both tasks, start them, verify.
    install_worker.cmd/.ps1     one elevated run: install the privileged worker.
    elevated_drain.ps1      the privileged worker itself.

## Why this shape

Two facts were measured on 2026-09-21 and they are the whole design:

1. **A process the agent starts dies with the agent's call.** The sandbox reaps
   its children when the call returns, so `:8000` and `:8077` lived for a few
   seconds at a time. "The node is up" was true in the same way a stopped clock
   is right twice a day. Durability therefore cannot come from the agent: it
   comes from a scheduled task registered once, elevated.

2. **An elevated prompt cannot be answered by the agent.** UAC exists so that a
   non-elevated process cannot click its own consent dialog. The workaround is
   not to defeat it; it is to need it rarely. Hence a queue.

## The queue, and the honest cost of it

`elevated_queue\in\<id>.json` with `{"cmd": "..."}`; the worker runs it as the
task's user and writes `out\<id>.json` with `exit`, `stdout`, `stderr`, `ran_as`,
`admin`. `done\` keeps the request.

**Anyone who can write to `in\` can run code at the worker's privilege.** That is
not a flaw to be papered over, it is the entire security model, so it is stated
where it cannot be missed and the ACL is the thing to audit:

    icacls C:\Users\china\autoforge_node\elevated_queue

It is set to the interactive user, Administrators and SYSTEM -- no `Users`, no
`Everyone`. A directory was chosen over a listener on purpose: a root process
holding an open port is an unauthenticated remote-execution service, and this
host already has a public tunnel in front of it.

## Which door is which

    :8000   the shelf            LAN-open on purpose; the peer reads it directly
    :8077   the node's own API   LAN-open
    :8077/market/<path>          the shelf, through the node, token-gated
    ngrok   -> :8077             so EVERY tunneled request passes the token gate

The token gate is not paranoia about the LAN. It exists because the tunnel
publishes `:8077` to the internet, and `toolmarket` is unauthenticated by its own
design ("the LAN boundary is the auth"). A shelf that leans on a LAN boundary
must not be reachable from behind a public URL without something in the way.
`AUTOFORGE_NODE_MARKET_OPEN=1` removes that something, which is why it is set
only on a host with no tunnel.
