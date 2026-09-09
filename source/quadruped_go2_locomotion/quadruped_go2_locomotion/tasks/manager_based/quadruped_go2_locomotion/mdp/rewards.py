# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from . import diagnostics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def track_pitch_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of target pitch (lean) angle using exponential kernel.

    The target pitch angle is commanded by the 4th component of the command buffer
    (index 3). Positive values lean forward, negative values lean backward.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Get target pitch from command (4th component, index 3)
    target_pitch = env.command_manager.get_command(command_name)[:, 3]
    # Extract current pitch angle from quaternion
    import isaaclab.utils.math as math_utils
    current_lean, current_pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # compute the exponential reward
    pitch_error = torch.square(target_pitch - current_pitch)
    reward = torch.exp(-pitch_error / std**2)
    diagnostics.log_step(env, "reward.track_pitch_exp", reward)
    return reward


def track_lean_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of target lean angle using exponential kernel.

    The target lean angle is commanded by the 5th component of the command buffer
    (index 4). Positive values lean right, negative values lean left.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Get target lean from command (5th component, index 4)
    target_lean = env.command_manager.get_command(command_name)[:, 4]
    # Extract current lean angle from quaternion
    import isaaclab.utils.math as math_utils
    current_lean, current_pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # compute the exponential reward
    lean_error = torch.square(target_lean - current_lean)
    reward = torch.exp(-lean_error / std**2)
    diagnostics.log_step(env, "reward.track_lean_exp", reward)
    return reward


def base_height_l2_pitch(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_sensitivity: float = 0.25,
    lean_sensitivity: float = 0.08,
) -> torch.Tensor:
    """Adaptive height penalty that uses the commanded lin_pos_z as the target height.

    The target height is read directly from the command buffer (component index 5).
    An additional offset is applied based on commanded pitch and lean angles so that
    the robot isn't penalized for naturally sitting lower during a lean.

    Args:
        command_name: Name of the command to read the target height from.
        pitch_sensitivity: How much target height increases per radian of commanded pitch.
            A value of 0.15 means ~15cm higher target per radian of pitch.
        lean_sensitivity: How much target height increases per radian of commanded lean.
            A value of 0.08 means ~8cm higher target per radian of lean.
    """

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Get commanded height from command buffer (component index 5)
    cmd = env.command_manager.get_command(command_name)
    cmd_height = cmd[:, 5]
    # Get commanded pitch and lean for adaptive offset
    cmd_pitch = cmd[:, 3]
    cmd_lean = cmd[:, 4]
    # Compute adaptive target height: commanded height + offset from commanded angles
    adaptive_target = cmd_height + pitch_sensitivity * torch.abs(cmd_pitch) + lean_sensitivity * torch.abs(cmd_lean)
    # L2 penalty from adaptive target
    penalty = torch.square(asset.data.root_pos_w[:, 2] - adaptive_target)
    diagnostics.log_step(env, "reward.base_height_l2_pitch", penalty)
    return penalty


def foot_sliding_exp(
    env: ManagerBasedRLEnv,
    std: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_body_names: list[str] | None = None,
    contact_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize foot sliding using an exponential kernel.

    For each foot body in contact with the ground (contact force magnitude
    exceeds ``contact_threshold``), computes the tangential foot velocity in the base
    (robot root) frame. Returns a penalty value between 0 (no sliding) and ~1 (fast sliding).
    Non-contacted feet contribute 0 penalty (no penalty during the swing phase).

    The velocity transformation uses :math:`v_{base} = R_{base}^{world\\, T} \\; v_{world}`,
    i.e. :func:`~isaaclab.utils.math.quat_apply_inverse` with the base orientation
    quaternion.

    Tangential (sliding) component is extracted by removing the projection along the
    projected gravity direction, so normal contact velocity (e.g. foot pressing into ground)
    is not penalised.

    The returned value is a **penalty** (positive = bad). It is multiplied by ``weight < 0``
    in the reward config so the total contribution is negative when sliding.

    Args:
        std: Standard deviation for the exponential kernel. Controls how quickly the
            penalty grows with sliding speed. Use ``sqrt(0.04)`` (= 0.2) as a starting
            point -- a foot sliding at 0.2 m/s gives penalty ``1 - exp(-1)`` ~ 0.63.
        sensor_cfg: Contact sensor configuration. The sensor must monitor all robot
            bodies so that body indices align between the sensor and the articulation.
            Commonly the same ``SceneEntityCfg`` used by ``undesired_contacts``.
        asset_cfg: Articulation configuration for the robot. Used to access body COM
            velocities. Defaults to the "robot" asset.
        feet_body_names: **Required**. List of rigid-body names for the foot end-effectors.
            These names must match the body names in the articulation (case-sensitive).
            Typical Go2 values: ``["FL_foot", "FR_foot", "RL_foot", "RR_foot"]``.
        contact_threshold: Minimum contact force magnitude (N) for a body to be
            considered in contact with the ground.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # --- resolve feet body ids from names ---
    if feet_body_names is None:
        raise ValueError(
            "foot_sliding_exp requires 'feet_body_names'. "
            "Pass the foot rigid-body names from your USD hierarchy, e.g. "
            "[\"FL_foot\", \"FR_foot\", \"RL_foot\", \"RR_foot\"]."
        )
    feet_list, resolved_names = asset.find_bodies(feet_body_names)
    num_art_bodies = asset.data.body_com_vel_w.shape[1]

    # Validate indices are in range
    for fid in feet_list:
        if fid < 0 or fid >= num_art_bodies:
            raise RuntimeError(
                f"Resolved foot body index {fid} is out of bounds [0, {num_art_bodies}). "
                f"feet_body_names={feet_body_names}, resolved_names={resolved_names}, "
                f"asset body_names={[asset.data.body_names[i] for i in range(min(len(asset.data.body_names), 30))]}"
            )

    # --- map sensor body names → feet local sensor indices ---
    # The contact sensor and articulation have independent index spaces; match by name.
    sensor_name_to_idx = {
        name: i for i, name in enumerate(contact_sensor.body_names)
    }
    feet_sensor_local: dict[str, int] = {}  # foot name → local index in contact sensor
    for fname in resolved_names:
        if fname in sensor_name_to_idx:
            feet_sensor_local[fname] = sensor_name_to_idx[fname]

    # --- contact detection (current timestep) ---
    net_forces = contact_sensor.data.net_forces_w
    is_contact_all = torch.norm(net_forces, dim=-1) > contact_threshold  # (N, num_sensor_bodies)

    # --- foot velocities in base frame ---
    # Access body_com_vel_w to trigger timestamped buffer refresh
    _ = asset.data.body_com_vel_w
    foot_vel_w = asset.data.body_com_vel_w[:, feet_list, :]  # (N, n_feet, 6)

    # Transform to base frame.
    # quat_rotate_inverse is TorchScript-compiled: q=(N,4), v=(N,3).
    # Broadcast root_quat_w to (N*n_feet, 4) to handle multiple feet.
    import isaaclab.utils.math as math_utils

    n_envs = foot_vel_w.size(0)
    n_feet = foot_vel_w.size(1)
    foot_vel_w_flat = foot_vel_w[:, :, :3].reshape(n_envs * n_feet, 3)  # (N*n_feet, 3)
    root_quat_broadcast = asset.data.root_quat_w.repeat_interleave(n_feet, dim=0)  # (N*n_feet, 4)
    foot_vel_base_flat = math_utils.quat_apply_inverse(root_quat_broadcast, foot_vel_w_flat)  # (N*n_feet, 3)
    foot_vel_base = foot_vel_base_flat.reshape(n_envs, n_feet, 3)  # (N, n_feet, 3)

    # --- tangential (sliding) component ---
    # projected_gravity_b: (N, 3) in base frame — direction gravity appears to point
    proj_grav = asset.data.projected_gravity_b  # (N, 3) in base frame
    proj_grav = proj_grav / (torch.norm(proj_grav, dim=-1, keepdim=True) + 1e-8)
    tangent_vel = foot_vel_base - proj_grav.unsqueeze(1) * torch.sum(
        foot_vel_base * proj_grav.unsqueeze(1), dim=-1, keepdim=True
    )
    sliding_speed = torch.norm(tangent_vel, dim=-1)  # (N, n_feet)

    # --- penalty per foot ---
    # 1 - exp(-speed/std): 0 when speed=0, increases with speed (approaches 1.0 as speed → ∞).
    # Non-contacted feet contribute 0 (no penalty for being in the swing phase).
    penalty_per_foot = 1.0 - torch.exp(-sliding_speed / std)  # (N, n_feet), range [0, 1)

    # Build a (num_sensor_bodies, n_feet) boolean mapping: which sensor entries correspond to which feet
    foot_to_sensor = torch.zeros(
        len(contact_sensor.body_names), n_feet, dtype=torch.bool, device=is_contact_all.device
    )
    for fname, sensor_local_id in feet_sensor_local.items():
        art_idx = resolved_names.index(fname)
        foot_to_sensor[sensor_local_id, art_idx] = True
    # Matrix multiply: (N, num_sensor_bodies) @ (num_sensor_bodies, n_feet) → (N, n_feet)
    contacted_per_foot = (is_contact_all.float() @ foot_to_sensor.float()).clip(max=1.0)  # (N, n_feet)
    contacted_per_foot = contacted_per_foot.bool()
    # Set non-contacted feet penalty to 0 (no penalty during swing phase)
    penalty_per_foot = torch.where(contacted_per_foot, penalty_per_foot, torch.zeros_like(penalty_per_foot))

    penalty = torch.mean(penalty_per_foot, dim=1)
    diagnostics.log_step(env, "reward.foot_sliding_exp", penalty)
    return penalty


def foot_lift_exp(
    env: ManagerBasedRLEnv,
    std: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_body_names: list[str] | None = None,
    min_height: float = 0.05,
    contact_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward feet that are lifted to at least ``min_height`` when not in contact.

    For each foot body that is **not** in contact with the ground, checks its COM
    height in the world frame. Returns a reward value between 0 (below threshold)
    and ~1 (well above threshold) using an exponential kernel.

    This reward encourages the policy to lift its feet during the swing phase
    rather than dragging them along the ground. It is neutral for contacted feet so
    it does not conflict with the foot sliding penalty.

    Args:
        std: Standard deviation for the exponential kernel. Controls how quickly the
            reward decays as height drops below ``min_height``. Use ``sqrt(0.01)``
            (= 0.1) as a starting point -- reward drops to ``exp(-1)`` ~ 0.37
            when the foot is 10 cm below threshold.
        sensor_cfg: Contact sensor configuration. The sensor must monitor all robot
            bodies so that body indices align between the sensor and the articulation.
            Commonly the same ``SceneEntityCfg`` used by ``undesired_contacts``.
        asset_cfg: Articulation configuration for the robot. Used to access foot
            positions. Defaults to the "robot" asset.
        feet_body_names: **Required**. List of rigid-body names for the foot
            end-effectors. Typical Go2 values:
            ``["FL_foot", "FR_foot", "RL_foot", "RR_foot"]``.
        min_height: Minimum COM height (in meters) above the ground at which a foot
            begins to earn reward. Defaults to 0.05 (5 cm).
        contact_threshold: Minimum contact force magnitude (N) for a body to be
            considered in contact with the ground.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # --- resolve feet body ids from names ---
    if feet_body_names is None:
        raise ValueError(
            "foot_lift_exp requires 'feet_body_names'. "
            "Pass the foot rigid-body names from your USD hierarchy, e.g. "
            "[\"FL_foot\", \"FR_foot\", \"RL_foot\", \"RR_foot\"]."
        )
    feet_list, resolved_names = asset.find_bodies(feet_body_names)
    num_art_bodies = asset.data.body_com_vel_w.shape[1]

    # Validate indices are in range
    for fid in feet_list:
        if fid < 0 or fid >= num_art_bodies:
            raise RuntimeError(
                f"Resolved foot body index {fid} is out of bounds [0, {num_art_bodies}). "
                f"feet_body_names={feet_body_names}, resolved_names={resolved_names}"
            )

    # --- map sensor body names → feet local sensor indices ---
    sensor_name_to_idx = {
        name: i for i, name in enumerate(contact_sensor.body_names)
    }
    feet_sensor_local: dict[str, int] = {}
    for fname in resolved_names:
        if fname in sensor_name_to_idx:
            feet_sensor_local[fname] = sensor_name_to_idx[fname]

    # --- non-contact detection ---
    net_forces = contact_sensor.data.net_forces_w
    is_contact_all = torch.norm(net_forces, dim=-1) > contact_threshold  # (N, num_sensor_bodies)

    # --- foot heights in world frame ---
    # body_com_pose_w: (N, num_bodies, 7) → [:, :, 2] is z-height
    foot_heights_w = asset.data.body_com_pose_w[:, feet_list, 2]  # (N, n_feet)

    # --- height-based reward ---
    # height_above: positive when foot is above min_height
    height_above = foot_heights_w - min_height  # (N, n_feet)
    # exponent clamped to <=0: reward saturates at 1 at/above min_height instead of growing past it, and
    # decays toward 0 (never overflows to inf) the further a foot drags below it, however far that is.
    reward_per_foot = torch.exp(torch.clamp(height_above, max=0.0) / std)  # (N, n_feet), range (0, 1]
    diagnostics.log_step(env, "reward.foot_lift_exp", reward_per_foot)

    # Non-contacted feet (swing phase): reward based on height above threshold
    # Contacted feet (stance phase): reward = 1 (neutral)
    n_envs = is_contact_all.size(0)
    n_feet = len(resolved_names)
    foot_to_sensor = torch.zeros(
        len(contact_sensor.body_names), n_feet, dtype=torch.bool, device=is_contact_all.device
    )
    for fname, sensor_local_id in feet_sensor_local.items():
        art_idx = resolved_names.index(fname)
        foot_to_sensor[sensor_local_id, art_idx] = True

    contacted_per_foot = (is_contact_all.float() @ foot_to_sensor.float()).clip(max=1.0).bool()  # (N, n_feet)
    # Inverted: True for non-contacted (air) feet, False for contacted
    is_air = ~contacted_per_foot
    # Set contacted feet reward to 1.0 (neutral — neither reward nor penalty)
    reward_per_foot = torch.where(is_air, reward_per_foot, torch.ones_like(reward_per_foot))

    return torch.mean(reward_per_foot, dim=1)


def hip_crossing_l2(
    env: ManagerBasedRLEnv,
    joint_ids: list[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.4,
) -> torch.Tensor:
    """Penalize lateral hip rotation that leads to leg crossing the midline.

    Each hip_roll joint controls the lateral swing of one leg. When the hip_roll
    angle is large in magnitude, the leg swings far to the side and risks crossing
    the robot's midline, which degrades gait stability. This reward applies an L2
    penalty scaled so that it only activates once the joint exceeds *threshold*
    radians.

    Args:
        joint_ids: Indices of the hip_roll joints (one per leg). Must match the
            order used by the policy's observation vector.
        threshold: Hip-roll magnitude at which the penalty begins to rise. Below
            this value the reward is zero; above it grows quadratically.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    hip_angles = asset.data.joint_pos[:, joint_ids]
    penalty = torch.clamp(torch.abs(hip_angles) - threshold, min=0.0)
    penalty = torch.sum(torch.square(penalty), dim=1)
    diagnostics.log_step(env, "reward.hip_crossing_l2", penalty)
    return penalty