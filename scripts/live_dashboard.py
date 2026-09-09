# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Live-updating dashboard of RSL-RL training metrics.

Reads the same TensorBoard event file (`events.out.tfevents.*`) RSL-RL's OnPolicyRunner already writes
every iteration -- via tensorboard's own EventAccumulator, reused here purely as a log-file reader, not as
the TensorBoard server/UI -- and renders it as a live matplotlib window. Purely read-only with respect to
training: it never touches the training process, the environment, or the checkpoint files.

Usage (interactive window -- needs a working matplotlib GUI backend, e.g. run with the host's own
Python rather than Isaac Sim's bundled one, which has no tkinter/Qt/GTK bindings):
    python3 scripts/live_dashboard.py --logdir logs/rsl_rl/go2_with_pitch_lean_and_height_control_ppo

Usage (no GUI backend available, e.g. inside the Isaac Sim container -- saves a PNG instead, view it
with an auto-reloading image viewer like `feh --reload <interval> <path>`):
    isaaclab -p scripts/live_dashboard.py --logdir logs/rsl_rl/go2_with_pitch_lean_and_height_control_ppo \\
        --output /tmp/dashboard.png
"""

import argparse
import glob
import os
import time

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# (panel title, [(tag, line label), ...]) -- one subplot per entry, one line per tag.
PANELS = [
    ("Mean reward", [("Train/mean_reward", "reward")]),
    ("Episode terminations", [
        ("Episode_Termination/invalid_state", "invalid_state"),
        ("Episode_Termination/body_contact", "body_contact"),
        ("Episode_Termination/time_out", "time_out"),
    ]),
    ("Mean episode length", [("Train/mean_episode_length", "length")]),
    ("Mean action noise std", [("Policy/mean_noise_std", "std")]),
    ("PPO losses", [
        ("Loss/value_function", "value_function"),
        ("Loss/surrogate", "surrogate"),
    ]),
    ("Linear vel x: commanded vs actual", [
        ("Metrics/base_velocity/cmd_lin_vel_x", "commanded"),
        ("Metrics/base_velocity/actual_lin_vel_x", "actual"),
    ]),
    ("Linear vel y: commanded vs actual", [
        ("Metrics/base_velocity/cmd_lin_vel_y", "commanded"),
        ("Metrics/base_velocity/actual_lin_vel_y", "actual"),
    ]),
    ("Angular vel z: commanded vs actual", [
        ("Metrics/base_velocity/cmd_ang_vel_z", "commanded"),
        ("Metrics/base_velocity/actual_ang_vel_z", "actual"),
    ]),
    ("Velocity tracking error", [
        ("Metrics/base_velocity/error_vel_xy", "error_vel_xy"),
        ("Metrics/base_velocity/error_vel_yaw", "error_vel_yaw"),
    ]),
]


def resolve_logdir(path: str) -> str:
    """If `path` is an experiment root (holds timestamped run subdirs), return the most recently
    modified one that actually has an events file. Otherwise return `path` unchanged."""
    if not os.path.isdir(path):
        return path
    if glob.glob(os.path.join(path, "events.out.tfevents.*")):
        return path
    run_dirs = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
        and glob.glob(os.path.join(path, name, "events.out.tfevents.*"))
    ]
    if not run_dirs:
        return path
    return max(run_dirs, key=os.path.getmtime)


class LiveDashboard:
    def __init__(self, logdir: str, max_points: int = 100):
        self.logdir = logdir
        self.max_points = max_points
        # size_guidance scalars=0 means "keep all points", not TensorBoard's default reservoir-sampled
        # subset -- we read the true full history each reload, then slice to the last max_points in
        # _update (below) for display, so "last N steps" always means the N most recent, not a sample
        # spread across the whole run.
        self.accumulator = EventAccumulator(logdir, size_guidance={"scalars": 0})

        self.fig, axes = plt.subplots(3, 3, figsize=(16, 10))
        self.fig.canvas.manager.set_window_title(f"Live training dashboard -- {logdir}")
        self.axes = axes.flatten()
        self.lines: dict[str, dict[str, plt.Line2D]] = {}

        for ax, (title, tags) in zip(self.axes, PANELS):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("step")
            self.lines[title] = {}
            for tag, label in tags:
                (line,) = ax.plot([], [], label=label)
                self.lines[title][tag] = line
            if len(tags) > 1:
                ax.legend(fontsize=8)
        self.fig.tight_layout()

    def _update(self, _frame):
        self.accumulator.Reload()
        available = set(self.accumulator.Tags().get("scalars", []))

        for ax, (title, tags) in zip(self.axes, PANELS):
            changed = False
            for tag, _label in tags:
                if tag not in available:
                    continue
                events = self.accumulator.Scalars(tag)[-self.max_points :]
                steps = [e.step for e in events]
                values = [e.value for e in events]
                self.lines[title][tag].set_data(steps, values)
                changed = True
            if changed:
                ax.relim()
                ax.autoscale_view()

        return [line for panel in self.lines.values() for line in panel.values()]

    def run(self, interval_ms: int, output_path: str | None = None):
        if output_path is not None:
            # No working interactive matplotlib backend in this environment (e.g. Isaac Sim's bundled
            # Python has no tkinter/Qt/GTK bindings) -- periodically re-render and save to a PNG instead,
            # viewed externally with an auto-reloading image viewer (e.g. `feh --reload <interval> <path>`),
            # a plain apt-installed binary with no Python-side GUI-toolkit entanglement.
            frame = 0
            print(f"[INFO] No interactive window -- saving to {output_path} every {interval_ms / 1000:.1f}s")
            try:
                while True:
                    self._update(frame)
                    self.fig.savefig(output_path)
                    frame += 1
                    time.sleep(interval_ms / 1000)
            except KeyboardInterrupt:
                pass
        else:
            # Kept as an attribute so the animation object isn't garbage-collected mid-run (a common
            # FuncAnimation footgun -- it silently stops updating if the reference is dropped).
            self._anim = animation.FuncAnimation(
                self.fig, self._update, interval=interval_ms, cache_frame_data=False
            )
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Live-updating dashboard of RSL-RL training metrics.")
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help=(
            "Either a specific run directory (contains events.out.tfevents.*), or an experiment root "
            "(e.g. logs/rsl_rl/<experiment_name>) -- in which case the most recently modified run is used."
        ),
    )
    parser.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=100,
        help="Show only the most recent N logged steps per line, instead of the full training history.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "If set, instead of opening an interactive window, periodically save the dashboard as a PNG "
            "to this path -- for environments without a working matplotlib GUI backend (e.g. Isaac Sim's "
            "bundled Python, which has no tkinter/Qt/GTK bindings). View the file with an auto-reloading "
            "image viewer, e.g. `feh --reload <interval> <path>`."
        ),
    )
    args = parser.parse_args()

    if args.output is not None:
        # Force the plain rasterizing backend explicitly. Without this, matplotlib's automatic backend
        # selection picks whatever interactive toolkit is importable (e.g. PyQt5, if installed) as soon as
        # a figure is created below -- regardless of whether we ever call plt.show() -- and in this
        # container that Qt backend crashes on missing system XCB dependencies before we ever reach the
        # savefig() loop. Agg has no such dependency; it's the same backend that was already working fine
        # for rendering before PyQt5 was ever installed.
        matplotlib.use("Agg")

    logdir = resolve_logdir(args.logdir)
    print(f"[INFO] Watching: {logdir}")

    dashboard = LiveDashboard(logdir, max_points=args.max_points)
    dashboard.run(interval_ms=int(args.interval * 1000), output_path=args.output)


if __name__ == "__main__":
    main()
