# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Records real sensor data from a STILL (standing, nothing hoisted, nothing moved by hand) robot to a
local .npz file, with zero commands ever sent -- read-only, structurally incapable of commanding the
robot. Same safety pattern as deploy/preflight_check.py: this file deliberately does NOT import
ChannelPublisher, LowCmd_, or anything else that could send data to the robot, so it's safe to run
against a powered, standing robot before ever touching release_motion_control() or LowCmd publishing.

Purpose: get a short recording of real, steady-state sensor data (joint positions/velocities, IMU
orientation/gyro, sportmode velocity) with no robot motion and no policy running, so it can be fed
through the exported policy completely offline afterward (see replay_policy_offline.py) to sanity-check
that the policy's raw action outputs are quiet/small for a steady-state input -- before ever trusting it
near real motors. This is a steady-state recording, not a dynamic/variety dataset: it does not involve
hoisting or moving the robot by hand.

Reuses deploy_real.py's own helpers directly (config loading, state buffer, quaternion math, manual
command slider) rather than reimplementing them.

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh deploy/collect_still_observations.py --network_interface enp3s0
    /workspace/isaaclab/_isaac_sim/python.sh deploy/collect_still_observations.py --network_interface enp3s0 --duration_s 20

Testable right now, no robot needed, against deploy/fake_robot.py:
    # terminal 1:
    /workspace/isaaclab/_isaac_sim/python.sh deploy/fake_robot.py --network_interface lo
    # terminal 2:
    /workspace/isaaclab/_isaac_sim/python.sh deploy/collect_still_observations.py --network_interface lo --duration_s 5
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime

import numpy as np

import deploy_real as dr
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record real sensor data from a still, standing robot (no commands ever sent)."
    )
    parser.add_argument("--config", type=str, default="deploy/configs/go2_locomotion.yaml")
    parser.add_argument("--network_interface", type=str, default=None, help="Overrides the config file's value.")
    parser.add_argument("--duration_s", type=float, default=10.0, help="Recording length in seconds.")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output .npz path. Default: logs/still_observations/<timestamp>.npz",
    )
    parser.add_argument("--status_interval_s", type=float, default=1.0, help="Seconds between printed status lines.")
    parser.add_argument(
        "--manual_commands", action="store_true", default=False,
        help="Launch the same command slider deploy_real.py's --manual_commands uses, and record whatever is "
        "set there each tick, instead of recording an all-zero command throughout. The robot itself still "
        "never moves (nothing is ever published) -- this only changes what command value is written "
        "alongside each tick's sensor data. Off by default: for a still-standing recording, an all-zero "
        "command is the honest label for the data.",
    )
    parser.add_argument("--command_file", type=str, default="/tmp/manual_command.json")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = dr.load_config(args.config, args.network_interface)

    state = dr.LatestRobotState(num_joints=12)

    manual_cmd = dr.ManualCommandBuffer()
    if args.manual_commands:
        threading.Thread(
            target=dr._manual_command_file_poll_loop, args=(manual_cmd, args.command_file), daemon=True
        ).start()
        dr.launch_slider(args.command_file)
        print(f"[INFO] Manual command mode: slider window launched (writing to {args.command_file}).")

    ChannelFactoryInitialize(0, cfg.network_interface)

    def _on_low_state(msg: LowState_):
        state.update_from_low_state(msg)

    def _on_sportmode_state(msg: SportModeState_):
        state.update_from_sportmode_state(msg)

    low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_sub.Init(_on_low_state, 10)
    sportmode_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sportmode_sub.Init(_on_sportmode_state, 10)

    print(f"[INFO] Subscribed on interface '{cfg.network_interface}'. This script NEVER publishes anything "
          "-- it cannot move the robot. Keep the robot STILL and standing throughout -- this is a "
          "steady-state recording, not a dynamic one. Ctrl+C to stop early (partial data is still saved).")
    print(f"[INFO] Waiting for first LowState message...")
    while not state.ready:
        time.sleep(0.05)
    print("[INFO] Robot state stream connected. Recording...")

    joint_pos_log: list[list[float]] = []
    joint_vel_log: list[list[float]] = []
    quat_wxyz_log: list[list[float]] = []
    gyro_log: list[list[float]] = []
    sportmode_velocity_log: list[list[float]] = []
    manual_cmd_log: list[list[float]] = []

    control_dt = cfg.control_dt
    start_time = time.monotonic()
    last_status = start_time
    step = 0
    try:
        while True:
            tick_start = time.perf_counter()
            snap = state.snapshot()
            cmd = manual_cmd.get()

            joint_pos_log.append(list(snap.joint_pos))
            joint_vel_log.append(list(snap.joint_vel))
            quat_wxyz_log.append(list(snap.quat_wxyz))
            gyro_log.append(list(snap.gyro))
            sportmode_velocity_log.append(list(snap.sportmode_velocity))
            manual_cmd_log.append(list(cmd))
            step += 1

            now = time.monotonic()
            if now - last_status >= args.status_interval_s:
                lean, pitch = dr.quat_to_pitch_lean(snap.quat_wxyz)
                elapsed = now - start_time
                print(f"[REC] T+{elapsed:5.1f}s  samples={step:5d}  "
                      f"pitch={pitch:+.3f}rad lean={lean:+.3f}rad  gyro={tuple('%.3f' % v for v in snap.gyro)}")
                last_status = now

            elapsed = now - start_time
            if elapsed >= args.duration_s:
                break

            sleep_time = control_dt - (time.perf_counter() - tick_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C -- stopping early, saving what was collected so far.")

    if step == 0:
        print("[WARN] No samples collected -- not writing an output file.")
        return

    out_path = args.out or os.path.join("logs", "still_observations", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        joint_pos=np.array(joint_pos_log, dtype=np.float32),  # (N, 12), SDK order
        joint_vel=np.array(joint_vel_log, dtype=np.float32),  # (N, 12), SDK order
        quat_wxyz=np.array(quat_wxyz_log, dtype=np.float32),  # (N, 4)
        gyro=np.array(gyro_log, dtype=np.float32),  # (N, 3), body frame
        sportmode_velocity=np.array(sportmode_velocity_log, dtype=np.float32),  # (N, 3), body frame
        manual_cmd=np.array(manual_cmd_log, dtype=np.float32),  # (N, 5): lin_x, lin_y, ang_z, pitch, lean
        control_dt=control_dt,
        sdk_joint_order=np.array(cfg.sdk_joint_order),
    )
    print(f"[INFO] Saved {step} samples ({step * control_dt:.1f}s) to {out_path}")
    print(f"[INFO] Replay offline with: deploy/replay_policy_offline.py --obs_log {out_path}")


if __name__ == "__main__":
    main()
