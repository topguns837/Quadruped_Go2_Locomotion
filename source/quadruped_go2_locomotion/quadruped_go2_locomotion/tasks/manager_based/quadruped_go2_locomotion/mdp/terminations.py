# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from . import diagnostics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def invalid_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate envs whose root or joint state has gone non-finite (NaN/Inf).

    Safety net for physics-solver edge cases upstream of anything else in the MDP (e.g. a contact-solver
    excursion), independent of whatever specific reward/observation bug they might otherwise crash PPO
    through. Logs which field and env triggered it (mdp/diagnostics.py) so a recurrence is pinpointable.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    fields = {
        "invalid_state.root_pos_w": asset.data.root_pos_w,
        "invalid_state.root_quat_w": asset.data.root_quat_w,
        "invalid_state.joint_pos": asset.data.joint_pos,
        "invalid_state.joint_vel": asset.data.joint_vel,
    }

    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name, value in fields.items():
        diagnostics.log_step(env, name, value)
        done |= ~torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)
    return done
