# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-torch maths for the base-height command (no Isaac Sim imports, so it is unit-testable anywhere).

Height is the sixth command component, ``cmd[:, 5]`` (metres, world-z of the base). It is a *command and reward
only*: the policy observes the COMMANDED height (known on the robot, like the velocity command) and is rewarded
against the simulator's ground-truth ``root_pos_w[:, 2]``. Actual height is never an observation, because the
real robot cannot measure it -- so this is open-loop on hardware and the policy must infer the achieved height
from proprioception.

Unlike lin_x/lin_y/ang_z/pitch/lean, height is NOT zeroed for standing envs: a standing robot still has a stance
height to hold. The hard stand-still gate (rewards._any_commanded) reads only columns 0-4, so a non-zero height
never turns that gate off.
"""

from __future__ import annotations

import torch

HEIGHT_COLUMN = 5


def sample_target_height(
    num: int, height_range: tuple[float, float] | None, nominal: float, device: str | torch.device
) -> torch.Tensor:
    """(num,) target heights ~ U(low, high), or all ``nominal`` when no range is configured."""
    if height_range is None:
        return torch.full((num,), float(nominal), device=device)
    low, high = height_range
    if low > high:
        raise ValueError(f"height range must be (low, high) with low <= high, got {height_range}")
    return torch.empty(num, device=device).uniform_(low, high)


def adaptive_height_target(
    cmd: torch.Tensor,
    pitch_sensitivity: float,
    lean_sensitivity: float,
    fallback_height: float | None = None,
) -> torch.Tensor:
    """Per-env height the reward should score against: commanded height plus the pitch/lean allowance.

    ``cmd`` is the command buffer, (N, 6) with height in column 5. A legacy 5-wide buffer (no height command)
    uses ``fallback_height`` instead and raises if none is given, so a missing height command can never silently
    become a target of 0 m.
    """
    if cmd.shape[1] > HEIGHT_COLUMN:
        base = cmd[:, HEIGHT_COLUMN]
    elif fallback_height is not None:
        base = torch.full_like(cmd[:, 0], float(fallback_height))
    else:
        raise ValueError(
            f"command has {cmd.shape[1]} columns, so there is no height command (column {HEIGHT_COLUMN}); "
            "pass fallback_height to use a fixed target"
        )
    return base + pitch_sensitivity * torch.abs(cmd[:, 3]) + lean_sensitivity * torch.abs(cmd[:, 4])
