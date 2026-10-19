# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feeds a recording from collect_still_observations.py through the exported policy completely offline --
pure Python, no Isaac Sim/Kit, no unitree_sdk2py, no DDS, no robot, and deliberately NO safety code
(no clamp_step, no tether_limit, no watchdogs): the entire point is seeing the policy's raw, unfiltered
opinion on real sensor data before any safety layer ever gets a chance to intervene.

last_action is NOT in the recording (there was no policy running during collection) -- it is
reconstructed here, starting at zero and updating from the POLICY'S OWN previous output each tick, same
as it would behave live. This makes the replay closed-loop in the action-feedback sense (the obs the
policy sees include its own prior output) even though the underlying physical states are pre-recorded,
not reactive to what the policy decides.

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh deploy/replay_policy_offline.py --obs_log logs/still_observations/<ts>.npz
    /workspace/isaaclab/_isaac_sim/python.sh deploy/replay_policy_offline.py --obs_log <path> --policy_path <override>
    /workspace/isaaclab/_isaac_sim/python.sh deploy/replay_policy_offline.py --obs_log <path> --out /tmp/replay_trace.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import deploy_real as dr

ACTION_WARN_THRESHOLD = 1.0  # |action| above this -> flagged; action_scale=0.25 so this is a 0.25 rad (~14 deg) offset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline, safety-code-free replay of a recorded observation log through the exported policy."
    )
    parser.add_argument("--config", type=str, default="deploy/configs/go2_locomotion.yaml")
    parser.add_argument("--obs_log", type=str, required=True, help=".npz written by collect_still_observations.py")
    parser.add_argument(
        "--policy_path", type=str, default=None,
        help="Overrides the config file's policy_path (torch.jit.load-able exported policy).",
    )
    parser.add_argument(
        "--action_warn_threshold", type=float, default=ACTION_WARN_THRESHOLD,
        help="Print how many ticks had any |action| component exceed this (diagnostic only, never aborts "
        "-- this script intentionally has no safety code).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="If set, saves the full obs/action sequence to this .npz, using the same 'obs'/'action' field "
        "names sim2sim/compare_traces.py already expects, so that tool can plot this replay too.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # network_interface is irrelevant here (nothing ever touches the network) but load_config requires it
    # to be set somewhere -- satisfy that validation with a clearly-labeled placeholder.
    cfg = dr.load_config(args.config, cli_network_interface="unused-offline-replay")
    sdk_to_isaac, _ = dr.build_joint_index_maps(cfg)
    default_joint_pos_isaac_order = [cfg.default_joint_pos[name] for name in cfg.isaac_joint_order]

    log = np.load(args.obs_log)
    joint_pos_log = log["joint_pos"]  # (N, 12) SDK order
    joint_vel_log = log["joint_vel"]  # (N, 12) SDK order
    quat_wxyz_log = log["quat_wxyz"]  # (N, 4)
    gyro_log = log["gyro"]  # (N, 3)
    sportmode_velocity_log = log["sportmode_velocity"]  # (N, 3)
    manual_cmd_log = log["manual_cmd"]  # (N, 5)
    num_ticks = joint_pos_log.shape[0]
    print(f"[INFO] Loaded {num_ticks} ticks ({num_ticks * float(log['control_dt']):.1f}s) from {args.obs_log}")

    policy_path = args.policy_path or cfg.policy_path
    print(f"[INFO] Loading policy from: {policy_path}")
    policy = torch.jit.load(policy_path)
    policy.eval()

    state = dr.LatestRobotState(num_joints=12)
    last_action = [0.0] * 12
    obs_history: list[list[float]] = []
    action_history: list[list[float]] = []

    for i in range(num_ticks):
        state.joint_pos = joint_pos_log[i].tolist()
        state.joint_vel = joint_vel_log[i].tolist()
        state.quat_wxyz = tuple(quat_wxyz_log[i].tolist())
        state.gyro = tuple(gyro_log[i].tolist())
        state.sportmode_velocity = tuple(sportmode_velocity_log[i].tolist())
        manual_cmd = manual_cmd_log[i].tolist()

        obs = dr.build_observation(cfg, state, sdk_to_isaac, default_joint_pos_isaac_order, manual_cmd, last_action)
        with torch.inference_mode():
            action = policy(obs)[0].tolist()  # raw policy output, no clamp/tether/watchdog applied

        obs_history.append(obs[0].tolist())
        action_history.append(action)
        last_action = action  # feeds back exactly as it would live (see module docstring)

    actions = np.array(action_history)  # (N, 12)
    print("\n" + "=" * 78)
    print(f"OFFLINE POLICY REPLAY -- {policy_path}")
    print("=" * 78)
    print(f"{'joint':16s}{'min':>10s}{'max':>10s}{'mean':>10s}{'std':>10s}")
    for j, name in enumerate(cfg.isaac_joint_order):
        col = actions[:, j]
        print(f"{name:16s}{col.min():10.3f}{col.max():10.3f}{col.mean():10.3f}{col.std():10.3f}")
    print("-" * 78)

    over_threshold = np.any(np.abs(actions) > args.action_warn_threshold, axis=1)
    num_over = int(over_threshold.sum())
    if num_over == 0:
        print(f"[OK]   No tick had any |action| exceed {args.action_warn_threshold:.2f}.")
    else:
        bad_ticks = np.nonzero(over_threshold)[0]
        preview = ", ".join(str(t) for t in bad_ticks[:10])
        more = f" (+{len(bad_ticks) - 10} more)" if len(bad_ticks) > 10 else ""
        print(f"[WARN] {num_over}/{num_ticks} ticks had some |action| exceed {args.action_warn_threshold:.2f} "
              f"(action_scale={cfg.action_scale} -> {args.action_warn_threshold * cfg.action_scale:.3f} rad "
              f"offset): ticks [{preview}]{more}")
    print("=" * 78)
    print("This replay ran with NO safety code (no slew-rate clamp, no tether, no watchdogs) -- it shows "
          "the policy's raw opinion only. A clean result here does not replace the real-hardware "
          "checklist's own safety layers; it only rules out an obviously broken policy before testing them.")

    if args.out:
        np.savez_compressed(args.out, obs=np.array(obs_history), action=actions)
        print(f"[INFO] Saved full obs/action trace to {args.out} (viewable with sim2sim/compare_traces.py).")


if __name__ == "__main__":
    main()
