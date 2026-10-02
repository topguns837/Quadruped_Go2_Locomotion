# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware-free end-to-end check that each joint's command reaches the motor that IS that joint.

Runs the REAL deploy_real.publish_low_cmd against stand-in message objects (no DDS, no robot, no network),
with every Isaac slot's target encoded as its own index, then checks the motor for each named joint received
that joint's target and no other. Also checks the gather direction used for reading state.

Why it exists: on first hardware use, publish_low_cmd sent 8 of 12 joints the WRONG joint's target (an index
list that was a valid permutation but in the wrong direction), which a permutation-only check never caught.
Run this before every hardware session (deploy/Checklist.md, section A); it must print 12/12 correct.

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh deploy/verify_joint_mapping.py
"""

from __future__ import annotations

import argparse
import sys

import deploy_real as dr


class _Motor:
    q = 0.0
    kp = 0.0
    kd = 0.0


class _Cmd:
    def __init__(self):
        self.motor_cmd = [_Motor() for _ in range(20)]
        self.crc = 0


class _Crc:
    def Crc(self, cmd):
        return 0


class _Publisher:
    def Write(self, cmd):
        pass


class _Buf:
    def __init__(self, target):
        self._target = target

    def get(self):
        return list(self._target)

    def get_kp(self):
        return 1.0

    def get_kd(self):
        return 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify joint index maps end to end (no hardware).")
    parser.add_argument("--config", type=str, default="deploy/configs/go2_locomotion.yaml")
    args = parser.parse_args()

    cfg = dr.load_config(args.config, "lo")
    sdk_to_isaac, isaac_to_sdk = dr.build_joint_index_maps(cfg)  # also raises if the maps are inconsistent

    target = [float(i) for i in range(12)]  # Isaac slot i commands the value i
    cmd = _Cmd()
    dr.publish_low_cmd(cmd, _Buf(target), isaac_to_sdk, _Publisher(), _Crc())

    wrong = 0
    print("WRITE path (Isaac-ordered targets -> SDK motors):")
    for i, name in enumerate(cfg.isaac_joint_order):
        motor = cfg.sdk_joint_order.index(name)
        got = cmd.motor_cmd[motor].q
        ok = got == float(i)
        wrong += not ok
        got_name = cfg.isaac_joint_order[int(got)] if 0 <= int(got) < 12 else "?"
        print(f"  {name:16s} -> motor {motor:2d}  received target of {got_name:16s} {'ok' if ok else 'WRONG'}")

    print("READ path (SDK motors -> Isaac-ordered observation):")
    sdk_values = [100.0 + s for s in range(12)]  # motor s reads 100+s
    for i, name in enumerate(cfg.isaac_joint_order):
        read = sdk_values[sdk_to_isaac[i]]
        ok = read == 100.0 + cfg.sdk_joint_order.index(name)
        wrong += not ok
        if not ok:
            print(f"  {name:16s} reads the wrong motor  WRONG")
    if wrong == 0:
        print("  all 12 read slots map to the right motor  ok")

    print(f"\nRESULT: {12 - min(wrong, 12)}/12 correct" if wrong else "\nRESULT: 12/12 correct (read and write)")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
