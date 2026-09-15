# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic Go2 DDS publisher for testing deploy_real.py without a physical robot.

Publishes plausible `LowState_`/`SportModeState_` messages on the loopback interface (or any interface you
point it at) so deploy_real.py's subscribers actually receive real DDS traffic -- not zeroed stand-in state
like --dry_run uses -- exercising the real message-parsing/observation-assembly/inference/publish path
end-to-end. Also subscribes to `rt/lowcmd` and logs what it receives, so you can see deploy_real.py's
actual computed targets (CRC-valid or not, sane joint positions or not) without a robot to send them to.

Does NOT implement the SportClient/MotionSwitcherClient RPC service side (CheckMode/ReleaseMode) -- that's
a request/response service protocol, not a pub/sub topic, and convincingly faking it is a separate, deeper
piece of work. Run deploy_real.py's `release_motion_control()` against this fake robot and it will correctly
fail loud after retrying (RPC calls time out, `code != 0`, never silently treated as "released" -- see that
function's own comments) -- this is a genuine, useful verification of that fail-safe behavior, not a gap in
this tool.

Usage (see deploy.md's hardware usage guide for the full picture):
    /workspace/isaaclab/_isaac_sim/python.sh deploy/fake_robot.py --network_interface lo
"""

from __future__ import annotations

import argparse
import time

import yaml

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_, unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.thread import RecurrentThread


def parse_args():
    parser = argparse.ArgumentParser(description="Publish synthetic Go2 DDS state for testing deploy_real.py.")
    parser.add_argument("--network_interface", type=str, default="lo")
    parser.add_argument(
        "--config",
        type=str,
        default="deploy/configs/go2_locomotion.yaml",
        help="Read default_joint_pos/sdk_joint_order from here so the synthetic LowState looks like a "
        "robot standing at its actual default pose, not all-zeros.",
    )
    parser.add_argument("--rate_hz", type=float, default=50.0, help="Publish rate for LowState/SportModeState.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sdk_joint_order = cfg["sdk_joint_order"]
    default_joint_pos = cfg["default_joint_pos"]
    default_q_sdk_order = [default_joint_pos[name] for name in sdk_joint_order]

    ChannelFactoryInitialize(0, args.network_interface)

    low_state_pub = ChannelPublisher("rt/lowstate", LowState_)
    low_state_pub.Init()
    sportmode_pub = ChannelPublisher("rt/sportmodestate", SportModeState_)
    sportmode_pub.Init()

    received_lowcmd_count = 0

    def _on_low_cmd(msg: "LowCmd_"):
        nonlocal received_lowcmd_count
        received_lowcmd_count += 1
        if received_lowcmd_count % 50 == 1:
            qs = [f"{m.q:.3f}" for m in msg.motor_cmd[:12]]
            print(f"[FAKE_ROBOT] received LowCmd #{received_lowcmd_count}: head={list(msg.head)} "
                  f"level_flag={msg.level_flag} mode0={msg.motor_cmd[0].mode} crc={msg.crc} q={qs}")

    low_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    low_cmd_sub.Init(_on_low_cmd, 10)

    def _publish_state():
        low_state = unitree_go_msg_dds__LowState_()
        for i, q in enumerate(default_q_sdk_order):
            low_state.motor_state[i].q = q
            low_state.motor_state[i].dq = 0.0
        low_state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]  # identity: level, facing +x
        low_state.imu_state.gyroscope = [0.0, 0.0, 0.0]
        low_state_pub.Write(low_state)

        sportmode_state = unitree_go_msg_dds__SportModeState_()
        sportmode_state.velocity = [0.0, 0.0, 0.0]
        sportmode_state.position = [0.0, 0.0, 0.3]
        sportmode_pub.Write(sportmode_state)

    interval = 1.0 / args.rate_hz
    thread = RecurrentThread(interval=interval, target=_publish_state, name="fake_robot_publish")
    thread.Start()

    print(f"[FAKE_ROBOT] Publishing synthetic LowState/SportModeState on '{args.network_interface}' "
          f"at {args.rate_hz} Hz. Ctrl+C to stop.")
    print("[FAKE_ROBOT] NOTE: does not answer CheckMode/ReleaseMode RPC calls -- deploy_real.py's "
          "release_motion_control() will correctly time out and fail loud against this tool (see module "
          "docstring). That's expected, not a bug in this fake robot.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"\n[FAKE_ROBOT] Stopping. Received {received_lowcmd_count} LowCmd messages total.")
        thread.Wait(1.0)


if __name__ == "__main__":
    main()
