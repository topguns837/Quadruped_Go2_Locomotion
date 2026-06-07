# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .curriculums import *  # noqa: F401, F403

# Extended command terms with pitch (lean) support
from .commands import (
    UniformVelocityCommandCfgWithPitch,
    UniformVelocityCommandWithPitch,
    get_pitch_command,
    get_lean_command,
    get_lin_pos_z_command,
)

# Pitch tracking reward
from .rewards import (
    track_pitch_exp,
    track_lean_exp,
    base_height_l2_pitch,
    hip_crossing_l2,
    foot_sliding_exp,
    foot_lift_exp,  # noqa: F401
)
