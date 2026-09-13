"""`python -m autoforge ...` — the invocation the OS scheduler is given.

`schedule.wake_command` emits `python -m autoforge tick --quiet`, and the
README's unattended story tells an operator to register exactly that. Without
this file the command is not a slow path or a wrong path, it is no path:
`No module named autoforge.__main__`. `install_system_task` would register a
task that fails every thirty minutes, silently, forever.

Kept to two lines on purpose. Its whole job is to make the module runnable;
anything that lives here is a subcommand that cannot be tested through `main`.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
