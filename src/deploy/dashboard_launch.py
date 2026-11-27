# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Start the live-graph dashboard (src/common/live_dashboard.py + a feh viewer) from inside a deploy run.

This replaces the separate tmux dashboard / viewer panes: with `--live_dashboard` (or `live_dashboard: true` in a
deploy profile) deploy_real.py spawns both itself, pointed at the run's own --live_plot log folder, and tears them
down at exit. Same subprocess + atexit pattern as launch_slider() and preflight_check.launch_live_dashboard().

Deliberately free of torch / SDK imports so it is unit-testable anywhere (see tests/test_dashboard_launch.py):
the process launcher, `which`, sleeping and atexit registration are all injectable.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import threading
import time

DEFAULT_OUTPUT = "/tmp/hardware_dashboard.png"  # not training's /tmp/dashboard.png, so the two never collide
# Training-only panels: deploy never logs them, so they would just render blank.
EXCLUDE_PANELS = "Mean reward,Mean episode length,Velocity tracking error"


def _dashboard_script() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common", "live_dashboard.py")


def dashboard_argv(python: str, script: str, logdir: str, output_png: str, interval: float = 0.5) -> list[str]:
    """Command line for the PNG-writing dashboard. --max-points 500 / --y-range-multiplier 2.0 are the same as the
    hardware tmux pane used: deploy logs per control tick, so the default 100-point window is only ~2 s."""
    return [
        python, script,
        "--logdir", logdir,
        "--output", output_png,
        "--interval", str(interval),
        "--max-points", "500",
        "--exclude-panels", EXCLUDE_PANELS,
        "--y-range-multiplier", "2.0",
    ]


def viewer_argv(output_png: str, interval: float = 0.5) -> list[str]:
    return ["feh", "--scale-down", "--reload", str(interval), output_png]


def _open_viewer_when_ready(
    dashboard_proc, output_png: str, interval: float, startup_timeout: float, *, popen, register, sleep
) -> None:
    """Wait for the dashboard's first frame, then open feh. `feh --reload` does NOT wait for a missing file (it
    exits immediately), so launching it back-to-back with the generator never works -- poll for the PNG first."""
    waited, step = 0.0, 0.2
    while not os.path.exists(output_png):
        if dashboard_proc.poll() is not None:
            print(f"[WARN] live_dashboard.py exited early (code {dashboard_proc.returncode}) before producing "
                  f"{output_png}; not opening the viewer. Check its output above.")
            return
        if waited >= startup_timeout:
            print(f"[WARN] No dashboard frame at {output_png} after {startup_timeout:.0f}s; not opening the viewer "
                  "(the generator is still running -- open the PNG yourself).")
            return
        sleep(step)
        waited += step
    register(popen(viewer_argv(output_png, interval)).terminate)
    print(f"[INFO] Live dashboard viewer opened on {output_png}.")


def start(
    logdir: str,
    output_png: str = DEFAULT_OUTPUT,
    viewer: bool = True,
    interval: float = 0.5,
    startup_timeout: float = 60.0,
    *,
    python: str | None = None,
    popen=subprocess.Popen,
    which=shutil.which,
    environ=None,
    register=atexit.register,
    sleep=time.sleep,
    spawn_thread: bool = True,
):
    """Spawn the dashboard (and, if `viewer`, feh once the first frame exists). Returns the dashboard process.

    The viewer wait runs on a daemon thread by default so a slow matplotlib import never delays the deploy
    startup / pre-release checklist. Never raises for a missing display or feh: the run matters more than the graphs.
    """
    environ = os.environ if environ is None else environ
    try:  # a PNG left by an earlier run would be shown immediately as a stale frame
        os.remove(output_png)
    except FileNotFoundError:
        pass
    proc = popen(dashboard_argv(python or sys.executable, _dashboard_script(), logdir, output_png, interval))
    register(proc.terminate)
    print(f"[INFO] Live dashboard generator started: {logdir} -> {output_png}")
    if not viewer:
        return proc
    if not environ.get("DISPLAY"):
        print(f"[WARN] No DISPLAY: not opening the viewer. Open {output_png} yourself.")
        return proc
    if which("feh") is None:
        print(f"[WARN] feh is not installed: not opening the viewer. Open {output_png} yourself.")
        return proc
    args = (proc, output_png, interval, startup_timeout)
    kw = {"popen": popen, "register": register, "sleep": sleep}
    if spawn_thread:
        threading.Thread(target=_open_viewer_when_ready, args=args, kwargs=kw, daemon=True).start()
    else:
        _open_viewer_when_ready(*args, **kw)
    return proc
