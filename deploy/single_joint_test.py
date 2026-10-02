# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal LowCmd command-path test: drive ONE joint gently and check that the robot actually applies torque.

Modeled on Unitree's own safe low-level test (unitree_sdk2_python README: one joint held at kp=10, kd=1).
Everything else is left limp with only light damping (kp=0), so a hoisted robot's legs just hang. Each
nudge (+N deg, back, -N deg, back, relative to where the joint is when the test starts) is applied only
after a SPACE press, and the script reports the joint's actual motion and the robot's own `tau_est` for it.

Why it exists: on first hardware use every joint reported tau_est ~0 while a full-body ramp ran at kp~25,
with a slow red head LED. This isolates "does the robot obey a LowCmd at all" from everything else in
deploy_real.py. PASS = the joint follows the nudge and tau_est responds. FAIL (no torque, no motion) points
at the robot (fault / protection), not at this project's control code.

Same safety rules as deploy_real.py (see deploy/Checklist.md): the robot must be prone on the ground or
already hanging, never standing, and it releases the onboard motion service first if still active (with the
same checklist prompt). Controller L2+B and Ctrl+C both stop it and send a damping command.

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh deploy/single_joint_test.py --network_interface enp2s0 --joint RL_hip_joint
"""

from __future__ import annotations

import argparse
import math
import threading
import time

import deploy_real as dr
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

# A step is judged to have "responded" if the robot reported at least this much torque on the joint during
# it. PD torque for a 5 deg error at kp=10 is ~0.9 Nm, so 0.2 Nm leaves margin for friction/rounding.
TAU_RESPONSE_THRESHOLD_NM = 0.2
SETTLE_S = 1.0  # how long to let each nudge act before reading position/torque


class SingleJointCommand:
    """Lock-protected command shared between the main thread (writer) and the 500 Hz publish thread."""

    def __init__(self, test_idx: int):
        self._lock = threading.Lock()
        self.test_idx = test_idx
        self._q = 0.0
        self._kp = 0.0
        self._kd = 0.0
        self._other_kd = 0.0

    def set(self, q: float, kp: float, kd: float, other_kd: float) -> None:
        with self._lock:
            self._q, self._kp, self._kd, self._other_kd = q, kp, kd, other_kd

    def get(self) -> tuple[float, float, float, float]:
        with self._lock:
            return self._q, self._kp, self._kd, self._other_kd


def publish_single(low_cmd, cmd: SingleJointCommand, publisher, crc) -> None:
    q, kp, kd, other_kd = cmd.get()
    for i in range(12):
        motor = low_cmd.motor_cmd[i]
        if i == cmd.test_idx:
            motor.q, motor.kp, motor.kd = q, kp, kd
        else:
            motor.kp, motor.kd = 0.0, other_kd
    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-joint LowCmd command-path test (see module docstring).")
    parser.add_argument("--config", type=str, default="deploy/configs/go2_locomotion.yaml")
    parser.add_argument("--network_interface", type=str, default=None, help="Overrides the config file's value.")
    parser.add_argument("--joint", type=str, default="RL_hip_joint", help="Joint name from sdk_joint_order.")
    parser.add_argument("--kp", type=float, default=10.0, help="Unitree's own safe test gain.")
    parser.add_argument("--kd", type=float, default=1.0, help="Unitree's own safe test gain.")
    parser.add_argument("--nudge_deg", type=float, default=5.0)
    parser.add_argument("--other_kd", type=float, default=1.0, help="Light damping on all OTHER joints (kp=0).")
    parser.add_argument(
        "--skip_release", action="store_true", default=False,
        help="Testing seam, only accepted with --network_interface lo (fake robot).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = dr.load_config(args.config, args.network_interface)
    if args.skip_release and cfg.network_interface != "lo":
        raise ValueError("--skip_release is only accepted with --network_interface lo (the fake robot).")
    if args.joint not in cfg.sdk_joint_order:
        raise ValueError(f"--joint must be one of {cfg.sdk_joint_order}, got {args.joint!r}")
    test_idx = cfg.sdk_joint_order.index(args.joint)

    state = dr.LatestRobotState(num_joints=12)
    ChannelFactoryInitialize(0, cfg.network_interface)
    crc = CRC()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(state.update_from_low_state, 10)
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()

    print(f"[INFO] Waiting for first LowState message on interface {cfg.network_interface}...")
    while not state.ready:
        time.sleep(0.05)

    if args.skip_release:
        print("[WARN] --skip_release: skipping release_motion_control() (loopback/fake robot only).")
    else:
        dr.release_motion_control(before_release=lambda: dr.wait_for_space(dr.PRE_RELEASE_CHECKLIST, flush=True))

    low_cmd = dr.init_low_cmd(cfg)
    for i in range(12):  # init_low_cmd sets cfg.kp on every leg; this test must start with NO gains anywhere.
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 0.0
    cmd = SingleJointCommand(test_idx)
    thread = RecurrentThread(
        interval=0.002, target=publish_single, name="single_joint_publish", args=(low_cmd, cmd, publisher, crc)
    )
    thread.Start()

    def poll() -> None:
        snap = state.snapshot()
        if snap.age_s() > dr.LINK_LOST_AFTER_S:
            raise dr.LinkLost(f"No LowState for {snap.age_s():.1f} s -- stopping, not resuming.")
        if dr.estop_pressed(snap.wireless_remote):
            raise dr.EmergencyStop("Controller e-stop (L2+B) pressed.")

    results: list[tuple[str, bool, bool]] = []
    try:
        print("[INFO] Zero-gain heartbeat (no torque) for 1 s...")
        end = time.monotonic() + dr.ZERO_GAIN_HEARTBEAT_S
        while time.monotonic() < end:
            poll()
            time.sleep(0.02)

        q0 = state.snapshot().joint_pos[test_idx]
        nudge = math.radians(args.nudge_deg)
        print(f"[INFO] Test joint {args.joint} (SDK idx {test_idx}) starts at {math.degrees(q0):+.1f} deg. "
              f"Gains on it: kp={args.kp}, kd={args.kd}. Others: kp=0, kd={args.other_kd}.")
        cmd.set(q0, args.kp, args.kd, args.other_kd)  # target == actual: no error, so no torque yet.

        for label, offset in (("+nudge", nudge), ("back to start", 0.0), ("-nudge", -nudge), ("back to start", 0.0)):
            dr.wait_for_space(f"\n[CONFIRM] Press SPACE to command {label} ({math.degrees(offset):+.1f} deg "
                              f"from start)...", poll=poll, flush=False)
            target = q0 + offset
            cmd.set(target, args.kp, args.kd, args.other_kd)
            peak_tau = 0.0
            window_end = time.monotonic() + SETTLE_S
            while time.monotonic() < window_end:
                poll()
                peak_tau = max(peak_tau, abs(state.snapshot().tau_est[test_idx]))
                time.sleep(0.01)
            actual = state.snapshot().joint_pos[test_idx]
            err = abs(target - actual)
            moved = abs(actual - q0) if offset != 0.0 else None
            responded = peak_tau >= TAU_RESPONSE_THRESHOLD_NM
            followed = (moved >= 0.5 * abs(offset)) if moved is not None else (err < 0.5 * nudge)
            print(f"[RESULT] {label:14s} target={math.degrees(target):+7.2f} deg  actual={math.degrees(actual):+7.2f} deg  "
                  f"remaining_err={math.degrees(err):5.2f} deg  peak|tau_est|={peak_tau:.3f} Nm  "
                  f"torque={'YES' if responded else 'NO'}  followed={'YES' if followed else 'NO'}")
            results.append((label, responded, followed))

        print("\n" + "=" * 74)
        if all(r and f for _, r, f in results):
            print("[PASS] The robot applied torque and followed every nudge: LowCmd is being obeyed on this joint.")
        elif not any(r for _, r, _ in results):
            print("[FAIL] No torque reported on any step. The robot is not obeying LowCmd: check the head LED "
                  "and the Go app for a fault. This is robot-side, not this project's control code. Stop here.")
        else:
            print("[PARTIAL] Some steps responded, some did not. Inspect the RESULT lines above before going on.")
        print("=" * 74)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received -- stopping.")
    except dr.EmergencyStop as e:
        print(f"\n[ESTOP] {e} -- stopping.")
    except dr.LinkLost as e:
        print(f"\n[LINK] {e}")
    finally:
        thread.Wait(1.0)
        print("[INFO] Sending damping command before exit (kp=0, kd=3 on all joints).")
        damp_cmd = dr.init_low_cmd(cfg)
        for i in range(12):
            damp_cmd.motor_cmd[i].q = 0.0
            damp_cmd.motor_cmd[i].kp = 0.0
            damp_cmd.motor_cmd[i].kd = dr.HOLD_DAMPING_KD
        damp_cmd.crc = crc.Crc(damp_cmd)
        for _ in range(100):
            publisher.Write(damp_cmd)
            time.sleep(0.005)


if __name__ == "__main__":
    main()
