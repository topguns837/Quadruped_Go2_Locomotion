# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-randomisation events whose strength follows the curriculum difficulty scale.

Each function reads ``env.difficulty_scale`` (a float in [0, 1], written by ``curriculums.difficulty_scale``) and
multiplies its randomisation range by it: at 0 the event does nothing, at 1 it reaches the full range. The scale
defaults to 0 when no curriculum term has run, so a config without the curriculum trains with none of these.

The stock Isaac Lab mass and centre-of-mass events are ``startup`` events, drawn once per environment, so their range
cannot follow a curriculum. These are ``reset`` events, redrawn every episode. Both are written so that repeated
draws never accumulate: they always start from the default value and set the result, never add to the last draw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _scale(env: ManagerBasedEnv) -> float:
    return float(getattr(env, "difficulty_scale", 0.0))


def randomize_base_mass_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    max_abs_kg: float,
    min_mass: float = 0.5,
):
    """Add ``U(-max_abs_kg * s, +max_abs_kg * s)`` kg to the selected bodies, with the inertia scaled to match.

    Mirrors ``isaaclab.envs.mdp.randomize_rigid_body_mass`` (operation="add", recompute_inertia=True): the draw starts
    from ``default_mass`` each time, so redrawing every episode does not accumulate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # the physx view works on CPU tensors
    env_ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    limit = max_abs_kg * _scale(env)
    masses = asset.root_physx_view.get_masses()
    default_mass = asset.data.default_mass[env_ids[:, None], body_ids].clone()
    offset = math_utils.sample_uniform(-limit, limit, (len(env_ids), len(body_ids)), device="cpu")
    masses[env_ids[:, None], body_ids] = torch.clamp(default_mass + offset, min=min_mass)
    asset.root_physx_view.set_masses(masses, env_ids)

    # a heavier trunk is also harder to rotate: scale the default inertia by the same ratio
    ratios = masses[env_ids[:, None], body_ids] / default_mass
    inertias = asset.root_physx_view.get_inertias()
    inertias[env_ids[:, None], body_ids] = asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
    asset.root_physx_view.set_inertias(inertias, env_ids)


class randomize_base_com_curriculum(ManagerTermBase):
    """Offset the centre of mass of the selected bodies by ``U(-max_xyz * s, +max_xyz * s)`` metres per axis.

    ``isaaclab.envs.mdp.randomize_rigid_body_com`` does ``com += sample``, which at ``reset`` frequency would random
    walk the centre of mass further away every episode. This keeps the original centre of mass (read on the first
    call, before anything has changed it) and sets ``default + offset`` instead.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._default_coms: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        max_xyz: tuple[float, float, float],
    ):
        asset: Articulation = env.scene[asset_cfg.name]
        env_ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
        if asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

        coms = asset.root_physx_view.get_coms().clone()
        if self._default_coms is None:
            self._default_coms = coms.clone()

        limit = torch.tensor(max_xyz, device="cpu") * _scale(env)
        offset = math_utils.sample_uniform(-limit, limit, (len(env_ids), 1, 3), device="cpu")
        coms[env_ids[:, None], body_ids, :3] = self._default_coms[env_ids[:, None], body_ids, :3] + offset
        asset.root_physx_view.set_coms(coms, env_ids)


def push_force_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_force: float,
    max_torque: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply a random external force and torque ``U(-max * s, +max * s)`` to the selected bodies.

    Mirrors ``isaaclab.envs.mdp.apply_external_force_torque``: the wrench persists until the next interval draw.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    s = _scale(env)
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(-max_force * s, max_force * s, size, asset.device)
    torques = math_utils.sample_uniform(-max_torque * s, max_torque * s, size, asset.device)
    # only applied when asset.write_data_to_sim() runs in the environment step
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torques, body_ids=asset_cfg.body_ids, env_ids=env_ids
    )
