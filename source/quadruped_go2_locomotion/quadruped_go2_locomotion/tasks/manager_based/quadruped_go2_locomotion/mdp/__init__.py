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

# Safety-net termination for non-finite (NaN/Inf) simulation state
from .terminations import invalid_state  # noqa: F401

# Per-step diagnostic file logging (see mdp/diagnostics.py)
from .diagnostics import (
    logged_base_lin_vel,
    logged_base_ang_vel,
    logged_projected_gravity,
    logged_joint_pos_rel,
    logged_joint_vel_rel,
    logged_last_action,
    logged_generated_commands,  # noqa: F401
)
