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
        self.metrics["cmd_height"] = torch.zeros(self.num_envs, device=self.device)
        # Achieved-velocity metrics: lets a viewer plot commanded vs. actual on the same chart
        # instead of only seeing the tracking-error magnitude (error_vel_xy/error_vel_yaw).
        self.metrics["actual_lin_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["actual_lin_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["actual_ang_vel_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["actual_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["actual_lean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["actual_height"] = torch.zeros(self.num_envs, device=self.device)
        # Manual command override (see enable_manual_override/set_manual_command) -- used by play.py's
        # --manual_commands mode to replace automatic resampling with an exact, user-typed command.
        # Off by default: training and the existing random-command play mode are unaffected.
        self.manual_override = False
        self.manual_command = torch.zeros(6, device=self.device)

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


    def enable_manual_override(self):
        """Switch from automatic resampling to a fixed, externally-set command (see set_manual_command).

        Used by play.py's --manual_commands mode. _resample_command may still fire on its usual timer, but
        once manual_override is on, _update_command no longer reads anything it wrote, so it's harmless.
        """
        self.manual_override = True

    def set_manual_command(self, values: list[float]):
        """Set the exact [lin_vel_x, lin_vel_y, ang_vel_z, pitch, lean, height] command for every env."""
        self.manual_command = torch.tensor(values, device=self.device, dtype=torch.float32)

    def _update_command(self):
        """Post-processes the velocity and pitch commands."""
        if self.manual_override:
            self.vel_command_b[:] = self.manual_command
            return
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
        self.metrics["cmd_height"] += self.vel_command_b[:, 5] / max_command_step
        # Achieved velocity, same base frame as vel_command_b, so directly comparable to the cmd_* values.
        self.metrics["actual_lin_vel_x"] += self.robot.data.root_lin_vel_b[:, 0] / max_command_step
        self.metrics["actual_lin_vel_y"] += self.robot.data.root_lin_vel_b[:, 1] / max_command_step
        self.metrics["actual_ang_vel_z"] += self.robot.data.root_ang_vel_b[:, 2] / max_command_step
        self.metrics["actual_height"] += self.robot.data.root_pos_w[:, 2] / max_command_step
        # Same pitch/lean extraction track_pitch_exp/track_lean_exp use in rewards.py -- euler_xyz_from_quat
        # returns (lean, pitch, yaw); each call only keeps the component it needs.
        _, current_pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        current_lean, _, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        self.metrics["actual_pitch"] += current_pitch / max_command_step
        self.metrics["actual_lean"] += current_lean / max_command_step

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and self.cfg.ranges.ang_pos_y is not None:
            if not hasattr(self, "cmd_pitch_visualizer"):
                self.cmd_pitch_visualizer = VisualizationMarkers(self.cfg.cmd_pitch_visualizer_cfg)
                self.actual_pitch_visualizer = VisualizationMarkers(self.cfg.actual_pitch_visualizer_cfg)
            self.cmd_pitch_visualizer.set_visibility(True)
            self.actual_pitch_visualizer.set_visibility(True)
        if debug_vis and self.cfg.ranges.ang_pos_x is not None:
            if not hasattr(self, "cmd_lean_visualizer"):
                self.cmd_lean_visualizer = VisualizationMarkers(self.cfg.cmd_lean_visualizer_cfg)
                self.actual_lean_visualizer = VisualizationMarkers(self.cfg.actual_lean_visualizer_cfg)
            self.cmd_lean_visualizer.set_visibility(True)
            self.actual_lean_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # This override previously shadowed the base class's own _debug_vis_callback entirely, which is
        # what updates the inherited velocity_goal/velocity_current arrows every frame -- without this,
        # those two markers were only ever posed once at construction (frozen), never following the robot
        # or the live command afterward.
        super()._debug_vis_callback(event)
        self._set_debug_vis_impl(True)

        # Commanded values come from vel_command_b (the live active command), not target_pitch/target_lean
        # -- in --manual_commands mode, _update_command's manual branch only ever writes vel_command_b, so
        # target_pitch/target_lean stay frozen at whatever they were before manual override was enabled.
        # Reading vel_command_b here keeps these arrows correct in both auto and manual modes.
        # euler_xyz_from_quat returns (lean, pitch, yaw) -- same extraction _update_metrics already uses
        # for the actual_pitch/actual_lean metrics.
        current_lean, current_pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)

        if self.cfg.ranges.ang_pos_y is not None:
            pitch_pos_w = self.robot.data.root_pos_w.clone()
            pitch_pos_w[:, 2] += 0.3
            self._visualize_angle_arrow(
                self.cmd_pitch_visualizer, pitch_pos_w, self.vel_command_b[:, 3], axis="pitch"
            )
            self._visualize_angle_arrow(self.actual_pitch_visualizer, pitch_pos_w, current_pitch, axis="pitch")

        if self.cfg.ranges.ang_pos_x is not None:
            lean_pos_w = self.robot.data.root_pos_w.clone()
            lean_pos_w[:, 2] += 0.45
            self._visualize_angle_arrow(
                self.cmd_lean_visualizer, lean_pos_w, self.vel_command_b[:, 4], axis="lean"
            )
            self._visualize_angle_arrow(self.actual_lean_visualizer, lean_pos_w, current_lean, axis="lean")

    def _visualize_angle_arrow(
        self, visualizer: VisualizationMarkers, pos_w: torch.Tensor, angle_rad: torch.Tensor, axis: str
    ):
        """Poses `visualizer`'s arrow at `pos_w`, rotated by `angle_rad` about the pitch (Y) or lean/roll
        (X) axis, with arrow length scaled by the angle's magnitude."""
        if axis == "pitch":
            quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(angle_rad), angle_rad, torch.zeros_like(angle_rad)
            )
        else:
            quat = math_utils.quat_from_euler_xyz(
                angle_rad, torch.zeros_like(angle_rad), torch.zeros_like(angle_rad)
            )
        scale = torch.tensor(visualizer.cfg.markers["arrow"].scale, device=self.device).repeat(len(angle_rad), 1)
        scale[:, 0] *= torch.abs(angle_rad) * 2.0 + 0.05
        scale[:, 1] *= torch.abs(angle_rad) * 2.0 + 0.05
        scale[:, 2] *= 0.1
        visualizer.visualize(pos_w, quat, scale)


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

    # Four pitch/lean arrows (commanded + actual, per axis), mirroring the goal/current pattern the base
    # class already uses for linear velocity (green=commanded, blue=actual). .replace() is a plain shallow
    # dataclasses.replace() (confirmed in isaaclab.utils.configclass) -- it does NOT copy the nested
    # `markers` dict, so every one of these deepcopies it before mutating scale/color, or they'd all end up
    # sharing (and stomping) the same underlying "arrow" marker object.
    cmd_pitch_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pitch_cmd"
    )
    """Commanded pitch arrow (red)."""
    cmd_pitch_visualizer_cfg.markers = {"arrow": copy.deepcopy(cmd_pitch_visualizer_cfg.markers["arrow"])}
    cmd_pitch_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

    actual_pitch_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pitch_actual"
    )
    """Actual (measured) pitch arrow (orange)."""
    actual_pitch_visualizer_cfg.markers = {"arrow": copy.deepcopy(actual_pitch_visualizer_cfg.markers["arrow"])}
    actual_pitch_visualizer_cfg.markers["arrow"].visual_material.diffuse_color = (1.0, 0.5, 0.0)
    actual_pitch_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

    cmd_lean_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/lean_cmd"
    )
    """Commanded lean arrow (purple)."""
    cmd_lean_visualizer_cfg.markers = {"arrow": copy.deepcopy(cmd_lean_visualizer_cfg.markers["arrow"])}
    cmd_lean_visualizer_cfg.markers["arrow"].visual_material.diffuse_color = (0.6, 0.0, 1.0)
    cmd_lean_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

    actual_lean_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/lean_actual"
    )
    """Actual (measured) lean arrow (cyan)."""
    actual_lean_visualizer_cfg.markers = {"arrow": copy.deepcopy(actual_lean_visualizer_cfg.markers["arrow"])}
    actual_lean_visualizer_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 1.0, 1.0)
    actual_lean_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

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