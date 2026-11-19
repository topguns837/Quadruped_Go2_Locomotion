# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""deploy/dashboard_launch.py: spawns the live-graph dashboard (+ feh viewer) for a deploy run.

Pure Python (no torch / SDK / display needed): processes, `which`, sleeping and atexit are all injected, so every
branch is tested without launching anything.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import dashboard_launch as dl  # noqa: E402


class FakeProc:
    def __init__(self, argv, alive=True):
        self.argv = argv
        self.alive = alive
        self.terminated = False
        self.returncode = None if alive else 1

    def poll(self):
        return None if self.alive else self.returncode

    def terminate(self):
        self.terminated = True


class Harness:
    """Injectable fakes + a record of what was launched / registered."""

    def __init__(self, *, display=":0", feh="/usr/bin/feh", dashboard_alive=True, png_appears_after=0, png=None):
        self.procs: list[FakeProc] = []
        self.registered: list = []
        self.sleeps = 0
        self.png = png
        self.png_appears_after = png_appears_after  # sleeps before the dashboard "writes" its first frame
        self.dashboard_alive = dashboard_alive
        self.environ = {"DISPLAY": display} if display else {}
        self.feh = feh

    def popen(self, argv, **_kw):
        is_dash = "live_dashboard.py" in " ".join(argv)
        p = FakeProc(argv, alive=self.dashboard_alive if is_dash else True)
        self.procs.append(p)
        return p

    def which(self, name):
        return self.feh if name == "feh" else None

    def sleep(self, _s):
        self.sleeps += 1
        if self.png and self.sleeps >= self.png_appears_after:
            open(self.png, "w").close()

    def start(self, logdir="logs/hardware_dashboard/run", output=None, **kw):
        return dl.start(
            logdir, output or self.png, popen=self.popen, which=self.which, environ=self.environ,
            register=self.registered.append, sleep=self.sleep, spawn_thread=False, python="PY", **kw,
        )

    def launched(self, needle):
        return [p for p in self.procs if needle in " ".join(p.argv)]


class ArgvBuilders(unittest.TestCase):
    def test_dashboard_argv(self):
        argv = dl.dashboard_argv("PY", "S.py", "logs/x", "/tmp/o.png", interval=0.5)
        self.assertEqual(argv[:2], ["PY", "S.py"])
        self.assertEqual(argv[argv.index("--logdir") + 1], "logs/x")
        self.assertEqual(argv[argv.index("--output") + 1], "/tmp/o.png")
        self.assertEqual(argv[argv.index("--interval") + 1], "0.5")
        # training-only panels would render blank for deploy data, so they are excluded (same as the tmux pane)
        self.assertEqual(argv[argv.index("--exclude-panels") + 1], dl.EXCLUDE_PANELS)
        self.assertIn("Mean reward", dl.EXCLUDE_PANELS)
        self.assertEqual(argv[argv.index("--max-points") + 1], "500")
        self.assertEqual(argv[argv.index("--y-range-multiplier") + 1], "2.0")

    def test_viewer_argv(self):
        self.assertEqual(dl.viewer_argv("/tmp/o.png", 0.5), ["feh", "--scale-down", "--reload", "0.5", "/tmp/o.png"])

    def test_default_output_is_hardware_specific(self):
        self.assertEqual(dl.DEFAULT_OUTPUT, "/tmp/hardware_dashboard.png")  # not the training PNG


class Start(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.png = os.path.join(self.tmp, "d.png")

    def test_happy_path_starts_dashboard_then_viewer(self):
        h = Harness(png=self.png, png_appears_after=2)
        h.start(viewer=True)
        self.assertEqual(len(h.launched("live_dashboard.py")), 1)
        feh = h.launched("feh")
        self.assertEqual(len(feh), 1)
        self.assertEqual(feh[0].argv[-1], self.png)
        # feh must come AFTER the dashboard (it does not wait for a missing PNG)
        self.assertLess(h.procs.index(h.launched("live_dashboard.py")[0]), h.procs.index(feh[0]))
        self.assertEqual(len(h.registered), 2)  # both children registered for cleanup at exit

    def test_viewer_false_launches_no_feh(self):
        h = Harness(png=self.png)
        h.start(viewer=False)
        self.assertEqual(len(h.launched("live_dashboard.py")), 1)
        self.assertEqual(h.launched("feh"), [])

    def test_no_display_skips_viewer_but_keeps_dashboard(self):
        h = Harness(display=None, png=self.png)
        h.start(viewer=True)
        self.assertEqual(len(h.launched("live_dashboard.py")), 1)
        self.assertEqual(h.launched("feh"), [])

    def test_feh_not_installed_skips_viewer(self):
        h = Harness(feh=None, png=self.png)
        h.start(viewer=True)
        self.assertEqual(h.launched("feh"), [])

    def test_stale_png_removed_before_start(self):
        # a leftover PNG from an earlier run would make feh open a stale frame immediately
        open(self.png, "w").close()
        h = Harness(png=None)  # the fake dashboard never writes a frame in this test
        h.start(viewer=False, output=self.png)
        self.assertFalse(os.path.exists(self.png))

    def test_dashboard_dies_early_no_viewer(self):
        h = Harness(dashboard_alive=False, png=None)
        h.start(viewer=True, output=self.png)
        self.assertEqual(h.launched("feh"), [])

    def test_timeout_waiting_for_first_frame_no_viewer(self):
        h = Harness(png=None)  # PNG never appears
        h.start(viewer=True, output=self.png, startup_timeout=1.0)
        self.assertEqual(h.launched("feh"), [])
        self.assertGreater(h.sleeps, 0)  # it polled rather than giving up instantly

    def test_returns_dashboard_process(self):
        h = Harness(png=self.png)
        proc = h.start(viewer=False)
        self.assertIs(proc, h.launched("live_dashboard.py")[0])


if __name__ == "__main__":
    unittest.main()
