# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


class difficulty_scale(ManagerTermBase):
    """Curriculum on the strength of the domain randomisation: a scale ``s`` in [0, 1] that follows how well the policy survives.

    Every episode that ends is either a time-out (the robot stayed up for the whole episode) or a fall. Episodes are
    counted in windows of ``dwell_steps`` environment steps; once a window is complete, the survival rate decides:

    * ``>= up_thresh``   -> ``s += step`` (the policy handles this level, make it harder)
    * ``<= down_thresh`` -> ``s -= step`` (too hard, back off)
    * in between         -> hold (the gap is hysteresis, so ``s`` does not flap on noise)

    The scale is published as ``env.difficulty_scale`` for the events in ``events.py`` and returned so it is logged as
    ``Curriculum/difficulty``. ``settle_steps`` at the start of each window are not counted: episodes that began
    before the last change were drawn at the old ``s``.

    Only envs that actually ended an episode count (``termination_manager.dones``); the first reset, where no episode
    has ended, would otherwise read as everything falling.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        env.difficulty_scale = 0.0
        self._window_start = 0
        self._survived = 0
        self._total = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        up_thresh: float = 0.8,
        down_thresh: float = 0.4,
        step: float = 0.05,
        dwell_steps: int = 6000,
        settle_steps: int = 2000,
        min_episodes: int = 500,
    ) -> torch.Tensor:
        now = int(env.common_step_counter)
        ended = env.termination_manager.dones[env_ids]
        if now - self._window_start >= settle_steps and bool(ended.any()):
            self._survived += int(env.termination_manager.time_outs[env_ids][ended].sum())
            self._total += int(ended.sum())

        if now - self._window_start >= dwell_steps and self._total >= min_episodes:
            rate = self._survived / self._total
            if rate >= up_thresh:
                env.difficulty_scale = min(1.0, env.difficulty_scale + step)
            elif rate <= down_thresh:
                env.difficulty_scale = max(0.0, env.difficulty_scale - step)
            self._window_start, self._survived, self._total = now, 0, 0

        return torch.tensor(env.difficulty_scale)


def get_curriculum_state(env) -> dict:
    """Snapshot the curriculum state that lives in the running process and is NOT part of an RSL-RL checkpoint.

    ``difficulty_scale`` (the domain-randomisation strength) and the per-env terrain levels are plain attributes of
    the environment, so a restarted process would begin again at difficulty 0 and random terrain levels. Batched
    training (src/training/train_batched.sh) saves this next to every checkpoint (``curriculum_<N>.pt``) and restores it
    on resume with :func:`set_curriculum_state`.

    Args:
        env: The environment (wrapped or not).
    """
    base = env.unwrapped
    state = {"difficulty_scale": float(getattr(base, "difficulty_scale", 0.0))}
    terrain = base.scene.terrain
    if getattr(terrain, "terrain_origins", None) is not None and hasattr(terrain, "terrain_levels"):
        state["terrain_levels"] = terrain.terrain_levels.detach().cpu().clone()
    return state


def set_curriculum_state(env, state: dict) -> None:
    """Restore a snapshot from :func:`get_curriculum_state`; the caller should reset the env afterwards.

    The terrain levels are only restored when the env count matches the snapshot (a different ``--num_envs`` between
    batches would make per-env levels meaningless); the difficulty scale is always restored.
    """
    base = env.unwrapped
    base.difficulty_scale = float(state["difficulty_scale"])
    terrain = base.scene.terrain
    levels = state.get("terrain_levels")
    if levels is None or getattr(terrain, "terrain_origins", None) is None:
        return
    if levels.shape != terrain.terrain_levels.shape:
        print(f"[WARN] curriculum: terrain levels not restored (saved {tuple(levels.shape)}, env {tuple(terrain.terrain_levels.shape)})")
        return
    terrain.terrain_levels[:] = levels.to(terrain.terrain_levels.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
