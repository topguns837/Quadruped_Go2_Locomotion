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
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

import sys

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab_tasks.utils.hydra import hydra_task_config

import quadruped_go2_locomotion.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """`agent_cfg` is required by hydra_task_config's calling convention (matches scripts/rsl_rl/play.py)
    but unused here -- this script only needs env_cfg to construct the environment and read its
    articulation's joint order, no RL runner involved."""
    del agent_cfg
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    print("\n[INFO] Isaac Lab articulation joint order (this is what deploy/configs/go2_locomotion.yaml's "
          "isaac_joint_order must match, in this exact order):")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i:2d}: {name}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
