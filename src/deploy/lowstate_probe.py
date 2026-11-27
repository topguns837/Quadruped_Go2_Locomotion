# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only, torch-free DDS probe: subscribes to rt/lowstate and rt/sportmodestate and prints their rate and
a few values. Needs only unitree_sdk2py + cyclonedds, so it runs on the robot's Jetson (no torch there).

Structurally incapable of commanding the robot: it imports no publisher and no LowCmd type, same rule as
preflight_check.py. Use it to prove that a given machine, on a given interface, actually receives the robot's
low-level DDS before building anything on top of that.

Usage (Jetson, inside ~/relay_env, and do NOT source the ROS setup in that shell):
    python src/deploy/lowstate_probe.py --network_interface eth0 --duration 10
"""

from __future__ import annotations

import argparse
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only rt/lowstate probe.")
    parser.add_argument("--network_interface", type=str, required=True)
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to listen.")
    args = parser.parse_args()

    lock = threading.Lock()
    stamps: list[float] = []
    latest = {}
    sport_count = [0]

    def on_low_state(msg: LowState_) -> None:
        now = time.monotonic()
        with lock:
            stamps.append(now)
            latest["q"] = [ms.q for ms in msg.motor_state[:12]]
            latest["dq"] = [ms.dq for ms in msg.motor_state[:12]]
            latest["tau"] = [ms.tau_est for ms in msg.motor_state[:12]]
            latest["quat"] = tuple(msg.imu_state.quaternion)

    def on_sport_state(msg: SportModeState_) -> None:
        with lock:
            sport_count[0] += 1
            latest["sport_vel"] = tuple(msg.velocity)

    ChannelFactoryInitialize(0, args.network_interface)
    low_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_sub.Init(on_low_state, 10)
    sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sport_sub.Init(on_sport_state, 10)

    print(f"[PROBE] Listening on '{args.network_interface}' for {args.duration:.0f} s (read-only)...")
    end = time.monotonic() + args.duration
    while time.monotonic() < end:
        time.sleep(1.0)
        with lock:
            n = len(stamps)
            recent = [t for t in stamps if t > time.monotonic() - 1.0]
            q = latest.get("q")
        rate = len(recent)
        head = "no LowState yet" if q is None else "q[0:3]=" + " ".join(f"{v:+.3f}" for v in q[:3])
        print(f"[PROBE] lowstate total={n} last-second={rate} Hz  {head}")

    with lock:
        n = len(stamps)
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        q, quat = latest.get("q"), latest.get("quat")
        sport_n = sport_count[0]
    print()
    if n == 0:
        print("[PROBE] RESULT: NO LowState received. This machine cannot see the robot's DDS on that interface.")
        return 1
    mean_hz = n / args.duration
    print(f"[PROBE] RESULT: received {n} LowState msgs (~{mean_hz:.0f} Hz), {sport_n} SportModeState msgs.")
    if gaps:
        gaps.sort()
        print(f"[PROBE] LowState gaps ms: median {gaps[len(gaps) // 2] * 1000:.1f}  max {gaps[-1] * 1000:.1f}")
    print(f"[PROBE] joint q (SDK order, rad): {[round(v, 3) for v in q]}")
    print(f"[PROBE] imu quaternion (w,x,y,z): {tuple(round(v, 3) for v in quat)}")
    print("[PROBE] Sanity: a robot standing still shows ~500 Hz, quaternion w near 1, thigh/calf angles plausible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
