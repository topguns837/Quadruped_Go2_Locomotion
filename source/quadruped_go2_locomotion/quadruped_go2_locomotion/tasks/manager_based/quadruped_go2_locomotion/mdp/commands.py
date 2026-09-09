# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended command generator with pitch (lean) command for quadruped locomotion."""

from __future__ import annotations

import copy
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from typing import ClassVar

import sys
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg as _UVCCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Step 1: Define the command class first (no forward reference issues)
class UniformVelocityCommandWithPitch(UniformVelocityCommand):
    """Command generator that extends UniformVelocityCommand with target pitch, lean, and height.

    The command buffer has shape (num_envs, 6) with components:
    [lin_vel_x, lin_vel_y, ang_vel_z, target_pitch, target_lean, target_lin_pos_z]
    """

    def __init__(self, cfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # Extend command buffer from (N, 3) to (N, 6)
        new_cmd = torch.zeros(self.num_envs, 6, device=self.device)
        new_cmd[:, :3] = self.vel_command_b
        del self.vel_command_b
        self.vel_command_b = new_cmd
        # Target pitch buffer
        self.target_pitch = torch.zeros(self.num_envs, device=self.device)
        self.target_lean  = torch.zeros(self.num_envs, device=self.device)
        self.target_lin_pos_z = torch.zeros(self.num_envs, device=self.device)
        # Commanded-value metrics: logged/printed the same way as the
        # inherited error_vel_xy/error_vel_yaw metrics (Metrics/base_velocity/*
        # in the RSL-RL console table and TensorBoard), so the raw command
        # vector is visible alongside reward/error during training.
        self.metrics["cmd_lin_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_lin_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_ang_vel_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_lean"] = torch.zeros(self.num_envs, device=self.device)
        # Achieved-velocity metrics: lets a viewer plot commanded vs. actual on the same chart
        # instead of only seeing the tracking-error magnitude (error_vel_xy/error_vel_yaw).
        self.metrics["actual_lin_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["actual_lin_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["actual_ang_vel_z"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 6).

        Components: [lin_vel_x, lin_vel_y, ang_vel_z, target_pitch, target_lean, target_lin_pos_z]
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
        # Sample target lin_pos_z (height command)
        h = torch.empty(len(env_ids), device=self.device)
        if self.cfg.ranges.lin_pos_z is not None:
            pos_range = self.cfg.ranges.lin_pos_z
            self.target_lin_pos_z[env_ids] = h.uniform_(*pos_range)
        else:
            self.target_lin_pos_z[env_ids] = 0.0


    def _update_command(self):
        """Post-processes the velocity and pitch commands."""
        super()._update_command()
        # Copy pitch, lean, and height targets into the command buffer
        self.vel_command_b[:, 3] = self.target_pitch
        self.vel_command_b[:, 4] = self.target_lean
        self.vel_command_b[:, 5] = self.target_lin_pos_z
        # Zero pitch, lean, and height for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.target_pitch[standing_env_ids] = 0.0
        self.target_lean[standing_env_ids] = 0.0
        self.target_lin_pos_z[standing_env_ids] = 0.0

    def _update_metrics(self):
        """Extends the base tracking-error metrics with the raw commanded values themselves."""
        super()._update_metrics()
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["cmd_lin_vel_x"] += self.vel_command_b[:, 0] / max_command_step
        self.metrics["cmd_lin_vel_y"] += self.vel_command_b[:, 1] / max_command_step
        self.metrics["cmd_ang_vel_z"] += self.vel_command_b[:, 2] / max_command_step
        self.metrics["cmd_pitch"] += self.vel_command_b[:, 3] / max_command_step
        self.metrics["cmd_lean"] += self.vel_command_b[:, 4] / max_command_step
        # Achieved velocity, same base frame as vel_command_b, so directly comparable to the cmd_* values.
        self.metrics["actual_lin_vel_x"] += self.robot.data.root_lin_vel_b[:, 0] / max_command_step
        self.metrics["actual_lin_vel_y"] += self.robot.data.root_lin_vel_b[:, 1] / max_command_step
        self.metrics["actual_ang_vel_z"] += self.robot.data.root_ang_vel_b[:, 2] / max_command_step

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and self.cfg.ranges.ang_pos_y is not None:
            if not hasattr(self, "pitch_goal_visualizer"):
                self.pitch_goal_visualizer = VisualizationMarkers(self.cfg.pitch_visualizer_cfg)
            self.pitch_goal_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # This override previously shadowed the base class's own _debug_vis_callback entirely, which is
        # what updates the inherited velocity_goal/velocity_current arrows every frame -- without this,
        # those two markers were only ever posed once at construction (frozen), never following the robot
        # or the live command afterward.
        super()._debug_vis_callback(event)
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
    """Configuration for the uniform velocity command generator with pitch, lean, and height.

    The command comprises of:
    - Linear velocity in x and y direction (m/s)
    - Angular velocity around z-axis (rad/s)
    - Target pitch angle (rad)
    - Target lean angle (rad)
    - Target linear position z / height (m)

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
        """Range for the target pitch angle (in rad). Defaults to None. """

        ang_pos_x: tuple[float, float] | None = None
        """Range for the target lean angle (in rad). Defaults to None. """

        lin_pos_z: tuple[float, float] | None = None
        """Range for the target linear position z / height command (in m). Defaults to None. """

    ranges: Ranges = Ranges()  # type: ignore

    pitch_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pitch_goal"
    )
    """The configuration for the pitch command visualization marker."""
    # .replace() is a plain shallow dataclasses.replace() (confirmed in isaaclab.utils.configclass) -- it
    # does NOT copy the nested `markers` dict, so without this deepcopy, mutating .scale below would
    # mutate the same "arrow" marker object shared by RED_ARROW_X_MARKER_CFG itself and every other
    # config anywhere that also does RED_ARROW_X_MARKER_CFG.replace(...), rather than just this one
    # visualizer.
    pitch_visualizer_cfg.markers = {"arrow": copy.deepcopy(pitch_visualizer_cfg.markers["arrow"])}
    # Was (1.0, 1.0, 1.0) -- reported 2-3x too big relative to the robot once actually visible in the
    # viewport (the earlier fix was for the arrows never updating at all, not their size). Shrunk by ~2.5x.
    # _debug_vis_callback multiplies this base scale by the pitch magnitude every frame, so this stays
    # consistent at every pitch value, not just one snapshot. Colored red (was green, same as the velocity
    # arrows) so it's visually distinct from the commanded/current velocity arrows.
    pitch_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker."""
    goal_vel_visualizer_cfg.markers = {"arrow": copy.deepcopy(goal_vel_visualizer_cfg.markers["arrow"])}
    # Base class default is (0.5, 0.5, 0.5) (isaaclab's UniformVelocityCommandCfg); halved again here per
    # explicit request to shrink the commanded/current velocity arrows by half from their current size.
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.2)

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker."""
    # Overridden here (base class leaves this at its own default of BLUE_ARROW_X_MARKER_CFG, scale
    # (0.5, 0.5, 0.5)) purely to halve its size to match goal_vel_visualizer_cfg above -- same deepcopy
    # requirement as the other two visualizers, to avoid mutating the shared BLUE_ARROW_X_MARKER_CFG.
    current_vel_visualizer_cfg.markers = {"arrow": copy.deepcopy(current_vel_visualizer_cfg.markers["arrow"])}
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.25, 0.25, 0.25)


def get_pitch_command(env, command_name: str) -> torch.Tensor:
    """Get the target pitch command from the command manager.

    Args:
        env: The environment.
        command_name: The name of the command to query.

    Returns:
        The target pitch command tensor of shape (num_envs,).
    """
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, 3]

def get_lean_command(env, command_name: str) -> torch.Tensor:
    """Get the target lean command from the command manager.

    Args:
        env: The environment.
        command_name: The name of the command to query.

    Returns:
        The target lean command tensor of shape (num_envs,).
    """
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, 4]

def get_lin_pos_z_command(env, command_name: str) -> torch.Tensor:
    """Get the target linear position z (height) command from the command manager.

    Args:
        env: The environment.
        command_name: The name of the command to query.

    Returns:
        The target lin_pos_z command tensor of shape (num_envs,).
    """
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, 5]