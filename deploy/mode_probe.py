# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only probe of the robot's motion-service state, per-motor state, and handheld-controller buttons.

Cannot move the robot: it only calls MotionSwitcherClient.CheckMode() (a query -- never SelectMode or
ReleaseMode), and never imports a LowCmd publisher or SportClient. Safe to run at any point, including while
the robot is standing in normal sport mode.

What it answers (see deploy/Checklist.md, Phase 2):
  - Which onboard motion service is active right now (`normal` / `ai` / `advanced`, or none = released).
  - Per-motor mode / q / tau_est / lost / temperature, once -- tau_est is the robot's own estimate of the
    torque it is applying, the number that showed ~0 on every joint during the first hardware attempts.
  - Which handheld-controller buttons are pressed, live. Press L2+B to confirm the controller e-stop data
    path exists (deploy_real.py treats L2+B as an emergency stop).

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh deploy/mode_probe.py --network_interface enp2s0
"""

from __future__ import annotations

import argparse
import time

import deploy_real as dr
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only motion-service / motor / controller probe.")
    parser.add_argument("--config", type=str, default="deploy/configs/go2_locomotion.yaml")
    parser.add_argument("--network_interface", type=str, default=None, help="Overrides the config file's value.")
    parser.add_argument(
        "--watch_s", type=float, default=15.0,
        help="Seconds to keep echoing controller button changes after the one-shot report (0 to skip).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = dr.load_config(args.config, args.network_interface)

    ChannelFactoryInitialize(0, cfg.network_interface)

    latest = {}

    def _on_low_state(msg: LowState_):
        latest["msg"] = msg

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(_on_low_state, 10)

    msc = MotionSwitcherClient()
    msc.SetTimeout(10.0)
    msc.Init()
    code, result = msc.CheckMode()
    print("=" * 74)
    if code != 0:
        print(f"[PROBE] CheckMode() failed (code={code}) -- motion-switcher service not answering. "
              "The robot's state is UNKNOWN; do not assume it is released.")
    elif not result or not result.get("name"):
        print("[PROBE] No onboard motion service active (released). LowCmd is authoritative.")
    else:
        print(f"[PROBE] Active onboard motion service: {result!r}  "
              "(normal = sport_mode, ai = ai_sport, advanced = advanced_sport)")

    deadline = time.monotonic() + 3.0
    while "msg" not in latest and time.monotonic() < deadline:
        time.sleep(0.05)
    if "msg" not in latest:
        print("[PROBE] No LowState received within 3 s -- check network_interface, robot power, cabling.")
        return

    msg = latest["msg"]
    print("\n[PROBE] Per-motor state (SDK order):")
    for i, name in enumerate(cfg.sdk_joint_order):
        ms = msg.motor_state[i]
        print(f"  idx={i:2d} {name:16s} mode={ms.mode}  q={ms.q:+.4f}  dq={ms.dq:+.4f}  "
              f"tau_est={ms.tau_est:+.3f}  temperature={ms.temperature}  lost={ms.lost}")
    print(f"\n[PROBE] imu temperature={msg.imu_state.temperature}")
    print("=" * 74)

    if args.watch_s <= 0:
        return
    print(f"\n[PROBE] Watching controller buttons for {args.watch_s:.0f} s. Press L2+B to test the e-stop path "
          "(Ctrl+C to stop early)...")
    seen: set[str] = set()
    end = time.monotonic() + args.watch_s
    try:
        while time.monotonic() < end:
            buttons = dr.decode_controller_buttons(bytes(latest["msg"].wireless_remote))
            pressed = {n for n, v in buttons.items() if v}
            if pressed != seen:
                note = "   <-- L2+B e-stop combo detected" if dr.estop_pressed(bytes(latest["msg"].wireless_remote)) else ""
                print(f"[PROBE] pressed: {sorted(pressed) if pressed else '(none)'}{note}")
                seen = pressed
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
