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


def root_displacement_excessive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_displacement: float = 10.0,
) -> torch.Tensor:
    """Terminate envs whose root has moved more than `max_displacement` m from its own env origin.

    `invalid_state` above only catches NaN/Inf. It does NOT catch a PhysX contact-solver excursion that
    flings the root to an absurd but still-finite position -- confirmed happening live (root_pos_w hit
    -2005 m in one env during a real training run), which `torch.isfinite()` passes, so that env kept
    running for the rest of its episode in a nonsense state: `base_height_l2_pitch` (and potentially other
    state-dependent rewards/observations) kept computing off a meaningless position. This is the fix for
    that specific gap -- see mdp.rewards.base_height_l2_pitch's docstring for the reward-side half of the
    same incident and why both are needed (that clamp bounds the DAMAGE from one bad step; this stops the
    env from compounding it for the rest of the episode).

    Checked relative to the env's own origin, not world-absolute position: envs are spatially tiled across
    one shared PhysX world, so each one's legitimate operating area is centered on `env.scene.env_origins`,
    not the world origin.

    max_displacement=10.0 is deliberately generous for this task's 8x8m terrain (half-width 4m, plus
    margin for a jump/fall) -- it exists to catch a genuine solver explosion, not to clip ordinary walking.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    local_pos = asset.data.root_pos_w - env.scene.env_origins
    displacement = torch.linalg.norm(local_pos, dim=-1)
    diagnostics.log_step(env, "invalid_state.root_displacement", displacement)
    return displacement > max_displacement
