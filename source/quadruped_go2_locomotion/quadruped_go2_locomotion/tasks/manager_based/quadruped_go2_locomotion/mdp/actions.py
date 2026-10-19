# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint position action with a random control-latency, so training sees what the robot's command path does.

On the real robot a policy output reaches the motors some time after the observation it was computed from
(DDS transport, the 500 Hz publish thread picking the target up, motor-driver update). Training with a perfectly
instantaneous action path teaches a policy that is too sensitive to timing. This term applies each processed action
0..`max_delay_steps` control steps (20 ms each) late, with the delay drawn per environment at every reset.

Only the physical command is delayed. `raw_actions` (what the `last_action` observation reads) stays the policy's own
most recent output, exactly like the robot's deploy loop, which feeds back the action it just computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedJointPositionAction(JointPositionAction):
    """`JointPositionAction` whose applied target lags the policy output by a per-environment random delay."""

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._max_delay = int(cfg.max_delay_steps)
        # history of processed (target) actions, newest at index 0
        self._history = self._offset.unsqueeze(1).repeat(1, self._max_delay + 1, 1).clone()
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_index = torch.arange(self.num_envs, device=self.device)
        self._resample_delay(slice(None))

    def _resample_delay(self, env_ids: Sequence[int] | slice | torch.Tensor) -> None:
        n = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        self._delay[env_ids] = torch.randint(0, self._max_delay + 1, (n,), device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)  # raw_actions and the undelayed processed target
        if self._max_delay == 0:
            return
        self._history = torch.roll(self._history, shifts=1, dims=1)
        self._history[:, 0] = self._processed_actions
        self._processed_actions = self._history[self._env_index, self._delay]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # until the first policy output arrives the robot holds the default pose, as the deploy ramp ends there
        self._history[ids] = self._offset[ids].unsqueeze(1)
        self._resample_delay(ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`DelayedJointPositionAction`."""

    class_type: type = DelayedJointPositionAction

    max_delay_steps: int = 2
    """Largest action latency in control steps (50 Hz, so 2 = 40 ms). The per-env delay is uniform in 0..this."""
