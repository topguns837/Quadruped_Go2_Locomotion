# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The pty driver types into interactive programs and reports their exit code."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pty_run  # noqa: E402


def test_decode_keys():
    assert pty_run.decode_keys("SPACE") == b" "
    assert pty_run.decode_keys("\\x03") == b"\x03"
    assert pty_run.decode_keys("a\\n") == b"a\n"


def test_once_types_after_the_prompt_and_exit_code_is_returned(capsys):
    cmd = ["bash", "-c", "echo READY; read -r -n1 k; echo \"got[$k]\"; exit 7"]
    rc = pty_run.main(["--once", "READY", "x", "--delay", "0.2", "--timeout", "20", "--"] + cmd)
    assert rc == 7
    assert "got[x]" in capsys.readouterr().out


def test_auto_answers_every_prompt(capsys):
    cmd = ["bash", "-c", "for i in 1 2 3; do echo PROMPT; read -r -n1 k; done; echo done"]
    assert pty_run.main(["--auto", "PROMPT", "y", "--delay", "0.2", "--timeout", "30", "--"] + cmd) == 0
    assert "done" in capsys.readouterr().out


def test_timeout_kills_a_stuck_command():
    assert pty_run.main(["--timeout", "1", "--", "bash", "-c", "sleep 30"]) == 124


def test_usage_errors():
    assert pty_run.main(["--once", "a", "b"]) == 2
    assert pty_run.main(["--"]) == 2
