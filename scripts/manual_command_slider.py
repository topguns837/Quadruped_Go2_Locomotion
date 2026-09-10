# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slider window for scripts/rsl_rl/play.py's --manual_commands mode.

Runs under the container's plain system Python (`python3`), never `isaaclab -p` -- Isaac Sim's bundled
Python has no tkinter/Qt/GTK bindings at all (see docker/Dockerfile's note next to the python3-tk apt
install). This script and play.py are two separate processes that never import each other; they only agree
on a shared JSON file (--command_file) that this script writes to and play.py polls.

Usage (normally auto-launched by play.py's --manual_commands mode, but also runnable standalone, e.g. to
reopen the window after closing it by accident, as long as play.py is still polling the same file).
Use the absolute path, not bare "python3" -- this container's ~/.bashrc aliases the bare name to Isaac
Sim's bundled Python (no tkinter) for interactive shells:
    /usr/bin/python3 scripts/manual_command_slider.py --command_file /tmp/manual_command.json
"""

import argparse
import json
import os
import tkinter as tk

# (label, key, min, max, initial, units) -- one slider per command component, in the exact order
# play.py's poll loop expects: [lin_vel_x, lin_vel_y, ang_vel_z, pitch, lean, height]. Ranges match
# CommandsCfg.ranges in quadruped_go2_locomotion_env_cfg.py (verified directly against that file, not
# CLAUDE.md, whose pitch/lean ranges are swapped relative to the actual code).
SLIDERS = [
    ("lin_vel_x", "lin_vel_x (m/s)", -1.0, 1.0, 0.0),
    ("lin_vel_y", "lin_vel_y (m/s)", -1.0, 1.0, 0.0),
    ("ang_vel_z", "ang_vel_z (rad/s)", -1.0, 1.0, 0.0),
    ("pitch", "pitch (rad)", -0.6, 0.6, 0.0),
    ("lean", "lean (rad)", -0.3, 0.3, 0.0),
    ("height", "height (m)", 0.2, 0.4, 0.3),
]


def build_command_dict(lin_vel_x, lin_vel_y, ang_vel_z, pitch, lean, height) -> dict:
    """Pure function (no Tkinter/file I/O) so it's directly unit-testable."""
    return {
        "lin_vel_x": lin_vel_x,
        "lin_vel_y": lin_vel_y,
        "ang_vel_z": ang_vel_z,
        "pitch": pitch,
        "lean": lean,
        "height": height,
    }


def write_command_file(path: str, command: dict) -> None:
    """Atomically write `command` as JSON to `path` (temp file + os.replace), so a concurrent reader
    (play.py's poll loop) never sees a partially-written file."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(command, f)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Slider window for play.py's --manual_commands mode.")
    parser.add_argument("--command_file", type=str, default="/tmp/manual_command.json")
    args = parser.parse_args()

    root = tk.Tk()
    root.title("Manual command")

    values = {key: initial for key, _label, _lo, _hi, initial in SLIDERS}

    def on_change(key, value):
        values[key] = float(value)
        write_command_file(args.command_file, build_command_dict(**values))

    for key, label, lo, hi, initial in SLIDERS:
        tk.Label(root, text=label).pack(anchor="w", padx=8)
        scale = tk.Scale(
            root,
            from_=lo,
            to=hi,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda v, key=key: on_change(key, v),
        )
        scale.set(initial)
        scale.pack(padx=8, pady=(0, 8))

    # Write the initial (all-default) command immediately, so play.py's poller has a valid file to read
    # even before any slider is touched.
    write_command_file(args.command_file, build_command_dict(**values))

    root.mainloop()


if __name__ == "__main__":
    main()
