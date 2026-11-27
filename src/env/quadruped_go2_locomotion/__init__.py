# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

# Importing this package registers VisionActorCritic in RSL-RL's runner namespace so the stairs vision
# agent config's class_name resolves (see modules/__init__.py). Must run before any runner is built.
from . import modules  # noqa: F401

##
# Register Gym environments.
##


gym.register(
    id="Quadruped-Locomotion-Go2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:QuadrupedLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Quadruped-Locomotion-Go2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:QuadrupedLocomotionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Quadruped-Locomotion-Go2-Stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_go2_locomotion_stairs_env_cfg:QuadrupedLocomotionStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VisionPPORunnerCfg",
    },
)

gym.register(
    id="Quadruped-Locomotion-Go2-Stairs-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_go2_locomotion_stairs_env_cfg:QuadrupedLocomotionStairsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VisionPPORunnerCfg",
    },
)
