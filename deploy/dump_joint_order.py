# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-off helper: prints the real Isaac Lab articulation joint order for the Go2+arm asset, so
deploy/configs/go2_locomotion.yaml's `isaac_joint_order` can be filled in with ground truth instead of a
guess. No robot needed -- this only needs the sim (Isaac Lab's own articulation loading), same GPU-safety
rules as any other sim launch in this repo apply (check `nvidia-smi` first, don't run this while a
training/play session is already using the GPU).

Usage:
    isaaclab -p deploy/dump_joint_order.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print the loaded Go2 articulation's joint order.")
parser.add_argument("--task", type=str, default="Quadruped-Locomotion-Go2-Play", help="Task name to load.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import quadruped_go2_locomotion.tasks  # noqa: F401


def main():
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    robot = env.unwrapped.scene["robot"]
    print("\n[INFO] Isaac Lab articulation joint order (this is what deploy/configs/go2_locomotion.yaml's "
          "isaac_joint_order must match, in this exact order):")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i:2d}: {name}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
