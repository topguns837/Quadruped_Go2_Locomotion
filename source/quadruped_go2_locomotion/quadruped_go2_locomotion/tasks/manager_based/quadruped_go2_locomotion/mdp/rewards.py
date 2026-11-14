# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import rewards as _stock
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


def base_height_l1_pitch(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_sensitivity: float = 0.25,
    lean_sensitivity: float = 0.08,
    max_penalty: float = 2.0,
) -> torch.Tensor:
    """Height penalty toward a fixed default stance height, with a small pitch/lean allowance.

    **L1 (absolute error), not L2.** A quadratic height penalty is flat near zero error -- at -4 weight a
    ~10 cm droop logged only ~0.04/s against ~+14/s of tracking rewards, so the policy knelt (front legs
    folded, base low) nearly for free, and raising the weight enough to bite at 5 cm made normal transients
    (foot-strike, pitch change) spike and destabilise. L1 has a constant gradient: a 5 cm droop always costs
    ``0.05 * |weight|``, a 10 cm droop ``0.10 * |weight|``, with no transient blow-up. See the iter-3 plan.

    There is no height COMMAND any more (see mdp.commands.UniformVelocityCommandWithPitch's docstring for
    why: no reliable height measurement exists on real hardware). `target_height` is a fixed constant
    instead of being read from the command buffer. The adaptive pitch/lean offset is kept: it still reads
    the current commanded pitch/lean (components [3]/[4], unaffected by the height removal) so the robot
    isn't penalized for naturally sitting slightly lower/higher while pitching or leaning.

    `max_penalty` is defence in depth: `root_pos_w[:, 2]` has no physical bound (unlike `joint_pos`), so an
    unclamped absolute error can reach hundreds if a robot ever leaves the terrain and free-falls (observed:
    the world edge is at +-148 m, robots walked off it here). The primary fix is a wider terrain border plus
    the `terrain_out_of_bounds` termination; this clamp only bounds the damage. max_penalty=2.0 saturates at
    a 2 m height error, far beyond any normal operation.

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
    # L1 penalty (absolute height error) from adaptive target, clamped -- see max_penalty above
    penalty = torch.clamp(torch.abs(asset.data.root_pos_w[:, 2] - adaptive_target), max=max_penalty)
    diagnostics.log_step(env, "reward.base_height_l1_pitch", penalty)
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


# Hard command gates (replace the graded _motion_blend for the stand/gait split). The stand reward is a bool
# mask on the WHOLE command (lin_x, lin_y, ang_z, pitch, lean): the instant any term is non-zero the policy must
# track it, not stand. Gait shaping is gated on the velocity terms only (a trot only applies when locomoting).
# 0.01 = the manual slider's resolution. Not 1e-4: the heading controller commands yaw = 0.5 * heading_error, which
# decays toward zero but is never exactly zero, so a tiny threshold would flip a settled env between the stand and
# gait regimes on numerical dust. Anything a human can set on the slider (>= 0.01) still counts as commanded.
_CMD_EPS = 1e-2


def _velocity_commanded(env: ManagerBasedRLEnv, command_name: str, eps: float = _CMD_EPS) -> torch.Tensor:
    """True per env when any of lin_x / lin_y / ang_z is commanded non-zero."""
    cmd = env.command_manager.get_command(command_name)
    return (cmd[:, :3].abs() > eps).any(dim=-1)


def _any_commanded(env: ManagerBasedRLEnv, command_name: str, eps: float = _CMD_EPS) -> torch.Tensor:
    """True per env when any of lin_x / lin_y / ang_z / pitch / lean is commanded non-zero."""
    cmd = env.command_manager.get_command(command_name)
    return (cmd[:, :5].abs() > eps).any(dim=-1)


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
        # Hard-gated on the velocity command (full credit when locomoting, 0 when not); cmd_scale is unused now.
        reward = torch.abs(s) * (1.0 - torch.abs(self._ema)) * _velocity_commanded(env, command_name).float()
        diagnostics.log_step(env, "reward.gait_coordination", reward)
        return reward


def stand_still_default_pose(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    feet_body_names: list[str],
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
    pose_scale: float = 0.5,
    vel_scale: float = 0.1,
) -> torch.Tensor:
    """Reward standing still -- zero velocity, four feet planted, default joint pose -- when the command is zero.

    ``feet_planted * default_pose * zero_vel``: feet_planted is the fraction of feet with contact force above
    ``contact_threshold`` (lifting any foot loses reward), default_pose is exp(-sum_j (q_j - q0_j)^2 /
    pose_scale) over all joints, and zero_vel is exp(-(vx^2 + vy^2 + yaw_rate^2) / vel_scale) so actually holding
    still (not drifting/creeping) is rewarded explicitly -- at vel_scale=0.1 a 0.1 m/s drift pays ~0.90, 0.2 m/s
    ~0.67. **Hard-gated to a fully zero command** (lin_x, lin_y, ang_z, pitch and lean all zero): the instant any
    term is commanded the reward is 0 and the policy must track that command instead (track_lin / track_ang /
    track_pitch / track_lean do the rewarding). There is no longer a pitch/lean exception -- a commanded
    pitch/lean is a tracking target, not a stand.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    feet_idx = sensor.find_bodies(feet_body_names)[0]
    planted = (torch.norm(sensor.data.net_forces_w[:, feet_idx], dim=-1) > contact_threshold).float().mean(dim=1)
    pose_err = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    default_pose = torch.exp(-pose_err / pose_scale)
    # explicit zero-velocity factor: base-frame planar linear velocity + yaw rate
    vel = torch.cat([asset.data.root_lin_vel_b[:, :2], asset.data.root_ang_vel_b[:, 2:3]], dim=1)
    zero_vel = torch.exp(-torch.sum(torch.square(vel), dim=1) / vel_scale)
    reward = planted * default_pose * zero_vel * (~_any_commanded(env, command_name)).float()
    diagnostics.log_step(env, "reward.stand_still", reward)
    return reward


class action_jerk_l2(ManagerTermBase):
    """Penalize the second difference of the policy action (jerk): ``sum_j (a_t - 2 a_{t-1} + a_{t-2})^2``.

    The first difference (``action_rate_l2``) penalizes any motion, so raising it to kill high-frequency chatter
    also flattens a clean 1-3 Hz trot (it was suppressing foot lift). The second difference instead weights high
    frequencies far more heavily: a smooth trot has small jerk, while a Nyquist (every-other-tick) limit cycle --
    the 24.9 Hz buzz seen on hardware 2026-11-04 -- is maximal. So this term targets the buzz specifically and
    lets ``action_rate_l2`` stay low. ``env.action_manager`` only keeps one previous action, so this caches the
    previous first-difference itself.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_rate = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._prev_rate.zero_()
        else:
            self._prev_rate[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, max_value: float | None = None) -> torch.Tensor:
        rate = env.action_manager.action - env.action_manager.prev_action
        jerk = rate - self._prev_rate
        self._prev_rate = rate.clone()
        reward = torch.sum(torch.square(jerk), dim=1)
        if max_value is not None:
            reward = torch.clamp(reward, max=max_value)
        diagnostics.log_step(env, "reward.action_jerk_l2", reward)
        return reward


class joint_torque_rate_l2(ManagerTermBase):
    """Penalize the rate of change of applied joint torque: ``sum_j (tau_t - tau_{t-1})^2`` (Nm^2 per step^2).

    Targets actuator chatter directly. On hardware 2026-11-04 the torque-rate RMS was ~6 Nm/s during a clean gait
    but ~184 Nm/s during the buzz (~30x), so a quadratic penalty separates them strongly. Caches the previous
    applied torque (the articulation does not expose a rate).
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[self._asset_cfg.name]
        self._prev_tau = torch.zeros(env.num_envs, asset.data.applied_torque.shape[1], device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._prev_tau.zero_()
        else:
            self._prev_tau[env_ids] = 0.0

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_value: float | None = None
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque
        rate = tau - self._prev_tau
        self._prev_tau = tau.clone()
        reward = torch.sum(torch.square(rate), dim=1)
        if max_value is not None:
            reward = torch.clamp(reward, max=max_value)
        diagnostics.log_step(env, "reward.joint_torque_rate_l2", reward)
        return reward


def track_lin_vel_xy_exp_rel(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_floor: float = 0.08,
    rel: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear-velocity tracking with a tolerance that scales with the commanded speed.

    ``exp(-|v_cmd - v|^2 / s^2)`` with ``s = max(std_floor, rel * |v_cmd|)``. A fixed std (Isaac Lab's
    ``track_lin_vel_xy_exp``) is lenient at low command: with std 0.5 a robot standing still scores 0.70 of the
    reward at a 0.3 m/s command and 0.96 at 0.1, so low-speed walking has almost no gradient and the policy stood
    still below ~0.3 m/s on hardware. With ``rel=0.5`` the tolerance is half the commanded speed: standing scores
    ~0.02 for any command >= 0.2 m/s, while at speed (>= 1 m/s) ``s`` equals the old std of 0.5, so high-speed
    behaviour is unchanged. ``std_floor`` keeps the kernel from collapsing for tiny commands.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    err = torch.sum(torch.square(cmd - asset.data.root_lin_vel_b[:, :2]), dim=1)
    s = torch.clamp(rel * torch.norm(cmd, dim=1), min=std_floor)
    return torch.exp(-err / s**2)


def track_ang_vel_z_exp_rel(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_floor: float = 0.1,
    rel: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-rate tracking with a tolerance that scales with the commanded rate (see track_lin_vel_xy_exp_rel)."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    err = torch.square(cmd - asset.data.root_ang_vel_b[:, 2])
    s = torch.clamp(rel * torch.abs(cmd), min=std_floor)
    return torch.exp(-err / s**2)


def standing_when_commanded(
    env: ManagerBasedRLEnv,
    command_name: str,
    eps: float = _CMD_EPS,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Scale-free, two-sided penalty for not moving at the commanded linear speed (fixes the low-speed dead zone).

    ``clamp(|(|v_cmd| - v . v_cmd_hat)| / |v_cmd|, 0, 1)``: the speed error along the command direction as a
    fraction of the commanded speed. Standing still scores 1.0 at ANY command size, so the pull off standstill is
    the same at 0.1 m/s as at 1 m/s (the earlier L1 deficit, in m/s, shrank with the command and was negligible at
    0.1 m/s; raising its weight made it huge at high speed and the policy fell and overshot). Overshoot and moving
    the wrong way are penalised too, clamped at 1, so the term is bounded in [0, 1] and needs no extra clamp. A
    robot tracking the command pays ~0. Hard-gated off for a zero linear command (standing envs are owned by
    stand_still_default_pose).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    cmd_mag = torch.norm(cmd_xy, dim=1)
    safe_mag = cmd_mag.clamp(min=1e-6)
    cmd_hat = cmd_xy / safe_mag.unsqueeze(1)
    v_proj = (asset.data.root_lin_vel_b[:, :2] * cmd_hat).sum(dim=1)
    penalty = ((cmd_mag - v_proj).abs() / safe_mag).clamp(max=1.0) * (cmd_mag > eps).float()
    diagnostics.log_step(env, "reward.standing_when_commanded", penalty)
    return penalty


def track_ang_vel_deficit(
    env: ManagerBasedRLEnv,
    command_name: str,
    eps: float = _CMD_EPS,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-sided L1 penalty for under-shooting the commanded yaw rate (rad/s).

    ``relu(|w_cmd| - max(0, w_z * sign(w_cmd)))``: the shortfall of yaw rate in the commanded turn direction.
    Secondary to the linear term (the observed dead zone is mainly linear), but keeps the "move when told"
    gradient symmetric for turning. Hard-gated off for a zero yaw command.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_z = env.command_manager.get_command(command_name)[:, 2]
    w_signed = (asset.data.root_ang_vel_b[:, 2] * torch.sign(cmd_z)).clamp(min=0.0)
    deficit = (cmd_z.abs() - w_signed).clamp(min=0.0) * (cmd_z.abs() > eps).float()
    diagnostics.log_step(env, "reward.track_ang_vel_deficit", deficit)
    return deficit


# ---------------------------------------------------------------------------------------------------------
# Clamped versions of the stock unbounded L2 penalties. Each is sum(x^2) with no upper bound, so a physics edge case
# (a large-but-finite joint velocity/torque that invalid_state does not catch) could produce a huge negative spike and
# destabilise PPO -- the same failure that forced base_height_l2_pitch's max_penalty. ``max_value`` is set well ABOVE
# the value a noisy-but-sane policy produces (so the gradient is untouched in normal operation) and far below a
# blow-up, bounding the worst-case per-step cost.
# ---------------------------------------------------------------------------------------------------------


def action_rate_l2_clamped(env: ManagerBasedRLEnv, max_value: float) -> torch.Tensor:
    return torch.clamp(_stock.action_rate_l2(env), max=max_value)


def joint_vel_l2_clamped(
    env: ManagerBasedRLEnv, max_value: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return torch.clamp(_stock.joint_vel_l2(env, asset_cfg), max=max_value)


def joint_acc_l2_clamped(
    env: ManagerBasedRLEnv, max_value: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return torch.clamp(_stock.joint_acc_l2(env, asset_cfg), max=max_value)


def joint_torques_l2_clamped(
    env: ManagerBasedRLEnv, max_value: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return torch.clamp(_stock.joint_torques_l2(env, asset_cfg), max=max_value)
