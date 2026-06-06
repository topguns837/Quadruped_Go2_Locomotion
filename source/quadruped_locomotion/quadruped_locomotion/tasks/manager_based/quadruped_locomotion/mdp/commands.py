# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended command generator with pitch (lean) command for quadruped locomotion."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from typing import ClassVar

import sys
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg as _UVCCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Step 1: Define the command class first (no forward reference issues)
class UniformVelocityCommandWithPitch(UniformVelocityCommand):
    """Command generator that extends UniformVelocityCommand with a target pitch (lean) angle.

    The command buffer has shape (num_envs, 5) with components:
    [lin_vel_x, lin_vel_y, ang_vel_z, target_pitch, target_lean]
    """

    def __init__(self, cfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # Extend command buffer from (N, 3) to (N, 5)
        new_cmd = torch.zeros(self.num_envs, 5, device=self.device)
        new_cmd[:, :3] = self.vel_command_b
        del self.vel_command_b
        self.vel_command_b = new_cmd
        # Target pitch buffer
        self.target_pitch = torch.zeros(self.num_envs, device=self.device)
        self.target_lean  = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 4).

        Components: [lin_vel_x, lin_vel_y, ang_vel_z, target_pitch, target_lean]
        """
        return self.vel_command_b

    def _resample_command(self, env_ids):
        # Sample base velocities via parent
        super()._resample_command(env_ids)
        # Sample pitch target
        r = torch.empty(len(env_ids), device=self.device)
        if self.cfg.ranges.ang_pos_y is not None:
            pos_range = self.cfg.ranges.ang_pos_y
            self.target_pitch[env_ids] = r.uniform_(*pos_range)
        else:
            self.target_pitch[env_ids] = 0.0
        # Sample lean target
        q = torch.empty(len(env_ids), device=self.device)
        if self.cfg.ranges.ang_pos_x is not None:
            pos_range = self.cfg.ranges.ang_pos_x
            self.target_lean[env_ids] = q.uniform_(*pos_range)
        else:
            self.target_lean[env_ids] = 0.0


    def _update_command(self):
        """Post-processes the velocity and pitch commands."""
        super()._update_command()
        # Copy pitch and lean targets into the command buffer
        self.vel_command_b[:, 3] = self.target_pitch
        self.vel_command_b[:, 4] = self.target_lean
        # Zero pitch and lean for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.target_pitch[standing_env_ids] = 0.0
        self.target_lean[standing_env_ids] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and self.cfg.ranges.ang_pos_y is not None:
            if not hasattr(self, "pitch_goal_visualizer"):
                self.pitch_goal_visualizer = VisualizationMarkers(self.cfg.pitch_visualizer_cfg)
            self.pitch_goal_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self._set_debug_vis_impl(True)
        if self.cfg.ranges.ang_pos_y is None:
            return
        # Visualize pitch targets
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.3
        pitch_rad = self.target_pitch
        pitch_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(pitch_rad), pitch_rad, torch.zeros_like(pitch_rad)
        )
        scale = torch.tensor(
            self.pitch_goal_visualizer.cfg.markers["arrow"].scale, device=self.device
        ).repeat(len(pitch_rad), 1)
        scale[:, 0] *= torch.abs(pitch_rad) * 2.0 + 0.05
        scale[:, 1] *= torch.abs(pitch_rad) * 2.0 + 0.05
        scale[:, 2] *= 0.1
        self.pitch_goal_visualizer.visualize(base_pos_w, pitch_quat, scale)


# Step 2: Now the config can reference the already-defined command class
@configclass
class UniformVelocityCommandCfgWithPitch(_UVCCfg):
    """Configuration for the uniform velocity command generator with pitch and lean angle command.

    The command comprises of:
    - Linear velocity in x and y direction (m/s)
    - Angular velocity around z-axis (rad/s)
    - Target pitch  angle (rad)
    - Target lean angle (rad)

    The robot should lean forward/backward while walking toward the sampled pitch angle.
    """

    class_type: ClassVar = UniformVelocityCommandWithPitch

    @configclass
    class Ranges:
        """Extended uniform distribution ranges for velocity and pitch commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad)."""

        ang_pos_y: tuple[float, float] | None = None
        """Range for the target pitch (lean) angle (in rad). Defaults to None. """

        ang_pos_x: tuple[float, float] | None = None
        """Range for the target pitch (lean) angle (in rad). Defaults to None. """

        """       
        If set, the command generator will sample a target pitch angle for each environment.
        Positive values lean forward, negative values lean backward.
        """

    ranges: Ranges = Ranges()  # type: ignore

    pitch_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pitch_goal"
    )
    """The configuration for the pitch command visualization marker."""


def get_pitch_command(env, command_name: str) -> torch.Tensor:
    """Get the target pitch (lean) command from the command manager.

    Args:
        env: The environment.
        command_name: The name of the command to query.

    Returns:
        The target pitch command tensor of shape (num_envs,).
    """
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, 3]

def get_lean_command(env, command_name: str) -> torch.Tensor:
    """Get the target pitch (lean) command from the command manager.

    Args:
        env: The environment.
        command_name: The name of the command to query.

    Returns:
        The target pitch command tensor of shape (num_envs,).
    """
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, 4]