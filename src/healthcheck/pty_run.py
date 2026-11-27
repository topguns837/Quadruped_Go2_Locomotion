# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a command in a pseudo-terminal and type into it when it prints something: lets the health check exercise the
interactive paths (SPACE-to-confirm prompts, Ctrl+C handling) that need a real terminal.

    python pty_run.py [--once PATTERN KEYS]... [--auto PATTERN KEYS]... [--delay S] [--timeout S] -- CMD [ARGS...]

``--once``  type KEYS the first time PATTERN (a regex) appears in the output.
``--auto``  type KEYS every time PATTERN appears again (e.g. every "Press SPACE" prompt).
KEYS accepts the escapes \\n, \\t, \\x03 (Ctrl+C) and the word SPACE. Keys are typed ``--delay`` seconds (default 0.6)
after the pattern, because the program may flush pending input right after printing its prompt.
The command's output is copied to stdout; the exit code is the command's (124 on timeout).
"""

from __future__ import annotations

import argparse
import os
import pty
import re
import select
import signal
import sys
import time


def decode_keys(text: str) -> bytes:
    return text.replace("SPACE", " ").encode().decode("unicode_escape").encode("latin-1")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("pty_run.py: missing '--' before the command", file=sys.stderr)
        return 2
    cut = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--once", nargs=2, action="append", default=[], metavar=("PATTERN", "KEYS"))
    ap.add_argument("--auto", nargs=2, action="append", default=[], metavar=("PATTERN", "KEYS"))
    ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv[:cut])
    cmd = argv[cut + 1:]
    if not cmd:
        print("pty_run.py: no command given", file=sys.stderr)
        return 2

    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(cmd[0], cmd)
    once = [[re.compile(p), decode_keys(k), False] for p, k in args.once]
    auto = [[re.compile(p), decode_keys(k), 0] for p, k in args.auto]
    pending: list[tuple[float, bytes]] = []
    buf = ""
    deadline = time.monotonic() + args.timeout
    status = None
    while True:
        now = time.monotonic()
        if now > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            print("\n[pty_run] timeout", flush=True)
            return 124
        for item in [p for p in pending if p[0] <= now]:
            os.write(fd, item[1])
            pending.remove(item)
        ready, _, _ = select.select([fd], [], [], 0.05)
        if ready:
            try:
                data = os.read(fd, 4096)
            except OSError:
                data = b""
            if not data:
                break
            text = data.decode(errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            buf += text
            for rule in once:
                if not rule[2] and rule[0].search(buf):
                    rule[2] = True
                    pending.append((time.monotonic() + args.delay, rule[1]))
            for rule in auto:
                n = len(rule[0].findall(buf))
                while rule[2] < n:
                    rule[2] += 1
                    pending.append((time.monotonic() + args.delay, rule[1]))
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    sys.exit(main())
