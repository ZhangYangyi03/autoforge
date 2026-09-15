# Working in this repo as an agent

## More than one session may be here at once

Two `auto` processes started in this repo cannot see each other. There is no
socket and no notification: each run reads state when its own turn takes it
there. The expensive failure is not that two sessions disagree — it is that
they never notice. Both edit the same file, the second write wins, and the
first session goes on describing a tree that no longer exists.

So there is a bus, and it is a file.

    auto bus read --as <your-name> --board autoforge   # what arrived since you last looked
    auto bus send --as <your-name> --board autoforge --to <name> "..."
    auto bus who                                       # who else is registered
    auto bus boards                                    # boards and their depth
    auto bus tail --board autoforge                    # last 20, does not advance

Read it at the start of a turn, and again before you commit. `read` advances a
per-reader cursor stored on disk, so a second call returns only what is new —
`--peek` looks without advancing, `--all` ignores the cursor.

Messages are plain text. Useful kinds, by convention: `claim` when you are about
to edit a file, `ask` for a question you need answered, `answer` for the
answer, `done` when you have stopped touching something.

State lives in `$AUTOFORGE_BUS_DIR`, else the first mount that already exists —
on this machine that is `%LOCALAPPDATA%\hermes\agent-bus`, the board the Hermes
sessions read — else the tool store's per-user home. The order matters: the last
fallback is normally empty, so a session that resolves there reads zero messages
and concludes nobody is talking. `auto bus register --as <name> --note "..."`
records what you are holding, so `who` is a live answer rather than a guess.

## Claim a lane before you edit a shared file

The bus is how a claim is announced; this is what to announce.

Give each session a file-level lane and say out loud which one you took before
writing to it. Adding a *new* file is always safe. Editing a file another
session has claimed is what produces the second-write-wins failure above, so it
is worth one message to avoid.

Report what you find rather than re-deriving it: a session that independently
reproduces a peer's result and gets a different answer has found something worth
more than either result alone.

## The bus is not a substitute for verification

A peer's summary is a claim, not evidence — including a claim that a fix
worked, or that a test suite is green. Re-run the thing. When two sessions
disagree, the disagreement is the finding.
