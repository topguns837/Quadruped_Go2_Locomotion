# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn a real hardware (or sim) policy trace into a replayable scenario YAML.

The deploy trace records the command the robot was actually given each tick inside its observation
(`obs[:, 6:11]` = [lin_x, lin_y, ang_z, pitch, lean]; see the 47-dim layout in deploy_real.build_observation).
Manual-slider commands are piecewise-constant, so this collapses consecutive equal commands into
constant-command segments and writes them with random_commands.write_scenario -- the SAME format
sim2sim_mujoco.py --scenario and play.py --scenario already consume. Replaying that scenario in MuJoCo /
Isaac drives the policy with the exact commands from the real run, so the hardware behaviour (e.g. the
shaking) can be reproduced in sim. Pure numpy -- no Isaac Sim, no torch.

Usage:
    python3 src/sim2sim/scenario_from_trace.py logs/hardware_dashboard/<ts>/policy_trace.npz --out run.yaml
    python3 src/sim2sim/scenario_from_trace.py <trace.npz> --out run.yaml --round 2 --min_seg_s 0.2
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import random_commands  # same dir on sys.path when run as a script

CMD_SLICE = slice(6, 11)  # command terms within the 47-dim observation


def segments_from_trace(path: str, decimals: int, min_seg_s: float) -> tuple[list[dict], float, float]:
    with np.load(path, allow_pickle=True) as z:
        obs = np.asarray(z["obs"], dtype=float)
        t = np.asarray(z["t"], dtype=float) if "t" in z.files else None
    cmd = np.round(obs[:, CMD_SLICE], decimals)
    dt = float(np.median(np.diff(t))) if t is not None and len(t) > 1 else 0.02
    # group consecutive identical (rounded) command rows
    segs: list[dict] = []
    start = 0
    for i in range(1, len(cmd) + 1):
        if i == len(cmd) or not np.array_equal(cmd[i], cmd[start]):
            dur = (i - start) * dt
            segs.append({"command": cmd[start].tolist(), "duration": dur, "_n": i - start})
            start = i
    # merge too-short segments into the previous one (slider jitter / single-tick blips)
    merged: list[dict] = []
    for s in segs:
        if merged and s["duration"] < min_seg_s:
            merged[-1]["duration"] += s["duration"]
        else:
            merged.append(s)
    # name them and round durations for a tidy YAML
    out = []
    for k, s in enumerate(merged):
        c = s["command"]
        moving = any(abs(v) > 10.0 ** (-decimals) for v in c)
        out.append({"name": f"h{k:02d}_{'cmd' if moving else 'stand'}",
                    "command": c, "duration": round(s["duration"], 2)})
    total = sum(s["duration"] for s in out)
    return out, total, dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="policy_trace.npz from deploy_real.py")
    ap.add_argument("--out", required=True, help="scenario YAML to write")
    ap.add_argument("--round", type=int, default=2, help="decimals to round commands to before grouping")
    ap.add_argument("--min_seg_s", type=float, default=0.2, help="merge segments shorter than this into the previous")
    args = ap.parse_args()
    segs, total, dt = segments_from_trace(args.trace, args.round, args.min_seg_s)
    header = (f"Scenario reconstructed from the real command log {os.path.basename(args.trace)}\n"
              f"{len(segs)} segments, {total:.1f} s total (source dt {dt * 1000:.1f} ms). Replays the exact "
              f"commands the robot was given; use with sim2sim_mujoco.py --scenario or play.py --scenario.")
    random_commands.write_scenario(args.out, segs, header=header)
    print(f"[wrote] {args.out}: {len(segs)} segments, {total:.1f} s")
    for s in segs[:40]:
        print(f"  {s['name']:14s} {s['duration']:5.2f}s  {s['command']}")


if __name__ == "__main__":
    main()
