# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step diagnostic file logging.

Writes one compact line per call to <env.cfg.log_dir>/step_diagnostics.log, every training step, so a
future non-finite (NaN/Inf) crash can be diagnosed by reading the trail leading up to it instead of
re-running training to reproduce it. No-ops (and never raises) when log_dir isn't set, e.g. zero_agent.py.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .obs_filters import alpha_from_cutoff

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Bounds step_diagnostics.log's total size per run -- it used to grow unboundedly (measured up to 1.1 GB on a
# long run) since it appended one line per logged term per step forever. ~7 logged terms x ~70 bytes/line/step
# means 20 MB is ~40000 steps of trailing history -- far more than needed to see the lead-up to a crash -- and
# x 3 backups caps the worst case at 80 MB per run instead of unbounded growth. Oldest rotated file is
# step_diagnostics.log.3; RotatingFileHandler deletes it once a 4th rotation would be needed, so the total is a
# hard cap, not just a rename.
STEP_DIAGNOSTICS_MAX_BYTES = 20 * 1024 * 1024
STEP_DIAGNOSTICS_BACKUP_COUNT = 3

# One open file handle per log_dir, reused across calls/terms instead of reopening every line.
_LOG_FILES: dict[str, "_StepLogger"] = {}


class _StepLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        # logging.getLogger's name registry is itself keyed by log_dir, so a second _StepLogger built for the
        # same log_dir (shouldn't happen given the _LOG_FILES cache above, but kept defensive) reuses the same
        # handler instead of attaching a duplicate one.
        logger = logging.getLogger(f"step_diagnostics.{log_dir}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = RotatingFileHandler(
                os.path.join(log_dir, "step_diagnostics.log"),
                maxBytes=STEP_DIAGNOSTICS_MAX_BYTES,
                backupCount=STEP_DIAGNOSTICS_BACKUP_COUNT,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        self._logger = logger

    def write(self, line: str) -> None:
        self._logger.info(line)


def log_step(env: ManagerBasedRLEnv, source: str, value: torch.Tensor) -> None:
    """Append one diagnostic line for ``value`` (shape ``(num_envs, ...)``) under ``source``.

    Purely diagnostic: never modifies ``value``, safe to call every step from any reward/observation term.
    """
    log_dir = getattr(env.cfg, "log_dir", None)
    if not log_dir:
        return
    logger = _LOG_FILES.setdefault(log_dir, _StepLogger(log_dir))

    finite = torch.isfinite(value)
    per_env_finite = finite.reshape(finite.shape[0], -1).all(dim=1)
    step = int(env.common_step_counter)

    if bool(per_env_finite.all().item()):
        stats = torch.stack([value.min(), value.max(), value.mean()]).tolist()
        logger.write(f"{step}\tOK\t{source}\tmin={stats[0]:.6g}\tmax={stats[1]:.6g}\tmean={stats[2]:.6g}")
    else:
        bad_envs = torch.nonzero(~per_env_finite, as_tuple=False).flatten().tolist()
        logger.write(f"{step}\tNONFINITE\t{source}\tbad_envs={bad_envs}")


# Logged wrappers for the policy observation terms, named at module level (not returned by a factory
# closure) because Isaac Lab's config system serializes ObsTerm.func as "module:function_name" to dump
# params/env.yaml and to reconstruct configs via Hydra — a closure isn't a real module attribute, so it
# fails that round-trip. Each of these just forwards to the underlying isaaclab.envs.mdp function and logs
# the result unchanged (see log_step).


def logged_base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    value = isaaclab_mdp.base_lin_vel(env, asset_cfg)
    log_step(env, "obs.base_lin_vel", value)
    return value


def logged_base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    value = isaaclab_mdp.base_ang_vel(env, asset_cfg)
    log_step(env, "obs.base_ang_vel", value)
    return value


def logged_projected_gravity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    value = isaaclab_mdp.projected_gravity(env, asset_cfg)
    log_step(env, "obs.projected_gravity", value)
    return value


def logged_joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    value = isaaclab_mdp.joint_pos_rel(env, asset_cfg)
    log_step(env, "obs.joint_pos", value)
    return value


def logged_joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    value = isaaclab_mdp.joint_vel_rel(env, asset_cfg)
    log_step(env, "obs.joint_vel", value)
    return value


class JointVelFiltered(ManagerTermBase):
    """joint_vel_rel with matched sensor noise added, THEN a first-order EMA low-pass -- the sim-to-real-consistent
    joint-velocity observation.

    Noise then filter, in THAT order, so training sees exactly what deploy sees: the real robot's joint_vel is noisy
    and deploy_real.ObsEmaFilter low-passes that noisy signal, so here the noise is added BEFORE the EMA (not by the
    observation manager afterwards). Set this term's ``noise=None`` in the ObsTerm -- the manager must not add more.

    The real hardware joint_vel noise floor measured ~0.035 rad/s std (uniform-equivalent ~0.06); ``noise_amp`` of
    ~0.3 is a conservative several-times-that to cover dynamic conditions -- NOT the old adversarial 1.5. With
    history stacking the policy learns to infer the true velocity from the position history, so it no longer
    amplifies this (small) noise into the ~25 Hz buzz. ``cutoff_hz=None`` skips the EMA (noise-only).

    Per-env EMA state re-seeds from the next frame after an env reset (no startup transient); mirrors
    deploy_real.ObsEmaFilter's lazy seeding.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ema: torch.Tensor | None = None
        self._seeded: torch.Tensor | None = None

    def reset(self, env_ids=None):
        if self._seeded is None:
            return
        if env_ids is None:
            self._seeded.zero_()
        else:
            self._seeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        noise_amp: float = 0.3,
        cutoff_hz: float | None = None,
    ) -> torch.Tensor:
        jv = isaaclab_mdp.joint_vel_rel(env, asset_cfg)
        if noise_amp and noise_amp > 0.0:
            jv = jv + (torch.rand_like(jv) * 2.0 - 1.0) * noise_amp
        if cutoff_hz:
            a = alpha_from_cutoff(cutoff_hz, env.step_dt)
            if self._ema is None:
                self._ema = jv.clone()
                self._seeded = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            else:
                self._ema = self._ema + a * (jv - self._ema)
                unseeded = ~self._seeded  # envs reset since their last filtered frame: re-seed, no transient
                if bool(unseeded.any()):
                    self._ema[unseeded] = jv[unseeded]
                    self._seeded[unseeded] = True
            jv = self._ema.clone()
        log_step(env, "obs.joint_vel", jv)
        return jv


def logged_last_action(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    value = isaaclab_mdp.last_action(env, action_name)
    log_step(env, "obs.actions", value)
    return value


def logged_generated_commands(env: ManagerBasedRLEnv, command_name: str | None = None) -> torch.Tensor:
    value = isaaclab_mdp.generated_commands(env, command_name)
    log_step(env, "obs.velocity_commands", value)
    return value
