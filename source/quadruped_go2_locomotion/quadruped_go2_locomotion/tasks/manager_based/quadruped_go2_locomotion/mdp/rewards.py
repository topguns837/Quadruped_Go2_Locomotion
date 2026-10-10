# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
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
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_sensitivity: float = 0.25,
    lean_sensitivity: float = 0.08,
    max_penalty: float = 4.0,
) -> torch.Tensor:
    """Height penalty toward a fixed default stance height, with a small pitch/lean allowance.

    There is no height COMMAND any more (see mdp.commands.UniformVelocityCommandWithPitch's docstring for
    why: no reliable height measurement exists on real hardware). `target_height` is a fixed constant
    instead of being read from the command buffer. The adaptive pitch/lean offset is kept: it still reads
    the current commanded pitch/lean (components [3]/[4], unaffected by the height removal) so the robot
    isn't penalized for naturally sitting slightly lower/higher while pitching or leaning.

    `max_penalty` is defence in depth: `root_pos_w[:, 2]` has no physical bound (unlike `joint_pos`), so an
    unclamped squared error can reach millions if a robot ever leaves the terrain and free-falls (observed:
    the world edge is at +-148 m, robots walked off it, and one fall reached 4e6 here). The primary fix is a
    wider terrain border plus the `terrain_out_of_bounds` termination; this clamp only bounds the damage.
    max_penalty=4.0 saturates at a 2 m height error, far beyond any normal operation.

    Args:
        command_name: Name of the command to read the current pitch/lean from.
        target_height: Fixed default stance height (m) the robot is held near. Should match the default
            joint pose's natural standing height so this reward doesn't fight joint_deviation_l1.
        pitch_sensitivity: How much target height increases per radian of commanded pitch.
            A value of 0.15 means ~15cm higher target per radian of pitch.
        lean_sensitivity: How much target height increases per radian of commanded lean.
            A value of 0.08 means ~8cm higher target per radian of lean.
        max_penalty: Upper bound on the squared error (m^2), see above.
    """

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Get commanded pitch and lean for adaptive offset
    cmd = env.command_manager.get_command(command_name)
    cmd_pitch = cmd[:, 3]
    cmd_lean = cmd[:, 4]
    # Compute adaptive target height: fixed target height + offset from commanded angles
    adaptive_target = target_height + pitch_sensitivity * torch.abs(cmd_pitch) + lean_sensitivity * torch.abs(cmd_lean)
    # L2 penalty from adaptive target, clamped -- see max_penalty above
    penalty = torch.clamp(torch.square(asset.data.root_pos_w[:, 2] - adaptive_target), max=max_penalty)
    diagnostics.log_step(env, "reward.base_height_l2_pitch", penalty)
    return penalty


def foot_sliding_exp(
    env: ManagerBasedRLEnv,
    std: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_body_names: list[str] | None = None,
    contact_threshold: float = 0.5,
    foot_radius: float = 0.022,
) -> torch.Tensor:
    """Penalize foot sliding using an exponential kernel.

    For each foot body in contact with the ground (contact force magnitude
    exceeds ``contact_threshold``), computes the horizontal world-frame velocity of the
    foot's contact point (sphere centre velocity corrected for rolling, radius ``foot_radius``). Returns a penalty value between 0 (no sliding) and ~1 (fast sliding).
    Non-contacted feet contribute 0 penalty (no penalty during the swing phase).

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
        foot_radius: Foot collision-sphere radius (m), 0.022 in Unitree's go2_description.
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

    # --- contact-point velocity (world frame, horizontal) ---
    # Sliding is the velocity of the point touching the ground, not of the foot sphere's centre: in normal
    # stance the centre still moves at w x r as the leg rotates over a stationary foot. The contact point is
    # r below the centre, so v_contact = v_com + w x (-r * z_hat) = v_com - r * (w_y, -w_x, 0).
    _ = asset.data.body_com_vel_w  # trigger timestamped buffer refresh
    foot_vel_w = asset.data.body_com_vel_w[:, feet_list, :]  # (N, n_feet, 6): linear, angular
    lin_xy = foot_vel_w[:, :, 0:2]
    ang = foot_vel_w[:, :, 3:6]
    contact_vel_x = lin_xy[..., 0] - foot_radius * ang[..., 1]
    contact_vel_y = lin_xy[..., 1] + foot_radius * ang[..., 0]
    sliding_speed = torch.sqrt(contact_vel_x**2 + contact_vel_y**2)  # (N, n_feet)
    n_feet = foot_vel_w.size(1)

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

    This only penalises low swing feet: contacting feet score the maximum (1.0), so it
    cannot reward stepping by itself. See GaitDiagonalCoordination for the term that does.
    It is neutral for contacted feet so it does not conflict with the foot sliding penalty.

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

def _motion_blend(env: ManagerBasedRLEnv, command_name: str, cmd_scale: float) -> torch.Tensor:
    """Smooth moving-vs-standing weight w in [0, 1], from the COMMAND only (never the robot's own motion).

    w = clamp(max(|v_xy| / cmd_scale, |w_z| / cmd_scale), 0, 1): exactly 0 for a zero command, 1 once the
    command reaches cmd_scale m/s (or rad/s). Gait terms are scaled by w and stand-still by 1 - w, so the
    robot is never rewarded for trotting when told to stand, while tiny non-zero commands (continuous
    sampling, heading-controller yaw) get a graded rather than all-or-nothing target.
    """
    cmd = env.command_manager.get_command(command_name)
    speed = torch.maximum(torch.norm(cmd[:, :2], dim=-1), torch.abs(cmd[:, 2]))
    return torch.clamp(speed / cmd_scale, 0.0, 1.0)


class GaitDiagonalCoordination(ManagerTermBase):
    """Dense reward for a trot: weight alternating between the FL+RR and FR+RL diagonals.

    With load share ``F_i / sum(F)`` per foot, ``s = share(FL+RR) - share(FR+RL)`` in [-1, 1]:
    ``|s|`` is instant credit (0 standing on four feet, ~0.3 for a 30% weight shift onto a diagonal, 1 for a
    pure trot stance), and ``1 - |EMA(s)|`` is an alternation factor (~1 when both diagonals take turns, ~0
    when always on one). Pace, bound, airborne and standing all score ~0. Because partial weight shifts earn
    partial credit, the policy can discover the gait incrementally. Scaled by the command blend weight
    (see _motion_blend), so it is 0 at a zero command.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        pairs = cfg.params.get("diagonal_pairs", (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")))
        self._idx_a = sensor.find_bodies(list(pairs[0]))[0]
        self._idx_b = sensor.find_bodies(list(pairs[1]))[0]
        if len(self._idx_a) != 2 or len(self._idx_b) != 2:
            raise ValueError(f"diagonal_pairs {pairs} did not resolve to 2 sensor bodies each")
        self._ema = torch.zeros(env.num_envs, device=env.device)
        self._alpha = env.step_dt / cfg.params.get("ema_time_constant", 0.5)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._ema.zero_()
        else:
            self._ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        cmd_scale: float = 0.15,
        min_total_force: float = 1.0,
        diagonal_pairs=(("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
        ema_time_constant: float = 0.5,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        force = torch.norm(sensor.data.net_forces_w, dim=-1)  # (N, num_sensor_bodies)
        fa = force[:, self._idx_a].sum(dim=1)
        fb = force[:, self._idx_b].sum(dim=1)
        total = fa + fb
        airborne = total < min_total_force
        s = torch.where(airborne, torch.zeros_like(total), (fa - fb) / total.clamp(min=min_total_force))
        self._ema += self._alpha * (s - self._ema)
        reward = torch.abs(s) * (1.0 - torch.abs(self._ema)) * _motion_blend(env, command_name, cmd_scale)
        diagnostics.log_step(env, "reward.gait_coordination", reward)
        return reward


def stand_still_default_pose(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    feet_body_names: list[str],
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_scale: float = 0.15,
    contact_threshold: float = 1.0,
    pose_scale: float = 0.5,
    posture_cmd_tol: float = 0.05,
) -> torch.Tensor:
    """Reward standing on all four feet in the default joint pose when the command says stand.

    ``feet_planted * default_pose``: feet_planted is the fraction of feet with contact force above
    ``contact_threshold`` (lifting any foot loses reward), default_pose is exp(-sum_j (q_j - q0_j)^2 /
    pose_scale) over all joints. If a pitch or lean is commanded (> posture_cmd_tol rad) the pose factor is
    1, leaving posture to track_pitch/lean so they never fight. Scaled by 1 - blend weight (see _motion_blend).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    feet_idx = sensor.find_bodies(feet_body_names)[0]
    planted = (torch.norm(sensor.data.net_forces_w[:, feet_idx], dim=-1) > contact_threshold).float().mean(dim=1)
    pose_err = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    default_pose = torch.exp(-pose_err / pose_scale)
    cmd = env.command_manager.get_command(command_name)
    posture_cmd = torch.maximum(torch.abs(cmd[:, 3]), torch.abs(cmd[:, 4])) > posture_cmd_tol
    default_pose = torch.where(posture_cmd, torch.ones_like(default_pose), default_pose)
    reward = planted * default_pose * (1.0 - _motion_blend(env, command_name, cmd_scale))
    diagnostics.log_step(env, "reward.stand_still", reward)
    return reward
