# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded random command schedules, so Isaac and MuJoCo can be driven with the exact same commands.

A schedule is a list of fixed-command segments in the same YAML format as `src/sim2sim/scenarios/*.yaml`, so every
runner that reads a scenario (`sim2sim_mujoco.py --scenario`, `play.py --scenario`) can replay one unchanged.

Each segment draws every command component uniformly from the range the policy was trained on (the
`UniformVelocityCommandCfgWithPitch` ranges in `env_cfg.py`), and a `standing_fraction` of the
segments are an all-zero command, like the training config's `rel_standing_envs` (0.12).

Pure numpy/yaml: no simulator, no policy.

    python3 src/sim2sim/random_commands.py --seed 0 --segments 20 --segment_s 4 --out /tmp/random_seed0.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

# Training command ranges: lin_vel_x, lin_vel_y, ang_vel_z in [-1, 1] (m/s, m/s, rad/s); pitch (ang_pos_y) in
# [-0.6, 0.6] rad; lean (ang_pos_x) in [-0.3, 0.3] rad. Order matches the 5-wide command vector.
COMMAND_RANGES: tuple[tuple[float, float], ...] = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-0.6, 0.6), (-0.3, 0.3))
COMMAND_NAMES = ("lin_x", "lin_y", "ang_z", "pitch", "lean")
TRAINING_STANDING_FRACTION = 0.12


def generate_segments(
    seed: int,
    num_segments: int = 20,
    segment_s: float = 4.0,
    standing_fraction: float = TRAINING_STANDING_FRACTION,
    scale: float = 1.0,
) -> list[dict]:
    """Return `num_segments` command segments (dicts with name / duration / command), deterministic in `seed`.

    `scale` shrinks every range around zero (e.g. 0.5 draws from half the training range) for a gentler test.
    """
    rng = np.random.default_rng(seed)
    segments = []
    for i in range(num_segments):
        standing = rng.random() < standing_fraction
        cmd = [0.0] * 5
        draws = [float(rng.uniform(lo, hi)) * scale for lo, hi in COMMAND_RANGES]
        if not standing:
            cmd = draws
        segments.append(
            {
                "name": f"r{i:02d}_{'stand' if standing else 'cmd'}",
                "duration": float(segment_s),
                "command": [round(v, 4) for v in cmd],
            }
        )
    return segments


def write_scenario(path: str, segments: list[dict], header: str = "") -> None:
    """Write segments in the scenario YAML format (flow style, one segment per line)."""
    lines = [f"# {line}" for line in header.splitlines()] if header else []
    lines.append("# command = [lin_vel_x (m/s), lin_vel_y (m/s), ang_vel_z (rad/s), pitch (rad), lean (rad)]")
    lines.append("segments:")
    for s in segments:
        cmd = ", ".join(f"{v}" for v in s["command"])
        lines.append(f"  - {{name: {s['name']}, duration: {s['duration']}, command: [{cmd}]}}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segments", type=int, default=20, help="number of fixed-command segments")
    parser.add_argument("--segment_s", type=float, default=4.0, help="seconds per segment")
    parser.add_argument("--standing_fraction", type=float, default=TRAINING_STANDING_FRACTION)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every training range (1.0 = full range)")
    parser.add_argument("--out", required=True, help="scenario YAML to write")
    args = parser.parse_args()
    segs = generate_segments(args.seed, args.segments, args.segment_s, args.standing_fraction, args.scale)
    write_scenario(
        args.out, segs,
        f"Random command schedule: seed={args.seed} segments={args.segments} segment_s={args.segment_s} "
        f"standing_fraction={args.standing_fraction} scale={args.scale}",
    )
    print(f"[INFO] wrote {len(segs)} segments ({sum(s['duration'] for s in segs):.0f} s) to {args.out}")


if __name__ == "__main__":
    main()
