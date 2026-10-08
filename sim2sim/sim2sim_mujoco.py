# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2sim: runs an exported Go2 policy (models/*/exported/policy.pt) in MuJoCo.

MuJoCo stands in for the robot: its state is written into deploy_real.py's `LatestRobotState` in SDK motor order,
and the observation is built by deploy_real.py's own `build_observation` / `build_joint_index_maps`. So this checks
both the policy's transfer to a second physics engine and the hardware observation path, with no robot involved.

Actuation reproduces the training actuator (unitree_go2witharm_cfg.py, explicit DCMotorCfg): the policy runs at 50 Hz
and the PD torque is recomputed every 0.005 s physics step (decimation 4), clipped by the DC motor torque-speed curve.

Supports both the current 50-dim policies (5-wide command: lin_x, lin_y, ang_z, pitch, lean -- no height command
or observation, see mdp/commands.py's module docstring for why) and the older committed 51/52-dim policies
(6-wide command, 52-dim additionally observing base_height), via OBS_LAYOUT below. Height is never a command any
more for either: it is tracked only as "actual height vs. a nominal target" in the summary table, matching how
height_penalty regulates it during training.

Interpreter: Isaac Sim's bundled Python (no Kit needed), or any Python with torch, mujoco, numpy, pyyaml:
    /workspace/isaaclab/_isaac_sim/python.sh sim2sim/sim2sim_mujoco.py --manual_commands
    /workspace/isaaclab/_isaac_sim/python.sh sim2sim/sim2sim_mujoco.py --headless \\
        --scenario sim2sim/scenarios/basic.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import mujoco
import numpy as np
import torch
import yaml

_SIM2SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SIM2SIM_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "deploy"))
sys.path.insert(0, _SIM2SIM_DIR)

import deploy_real  # noqa: E402
from go2_model import build_model  # noqa: E402

DECIMATION = 4  # quadruped_go2_locomotion_env_cfg.py
INIT_BASE_HEIGHT = 0.4  # UNITREE_GO2WITHARM_CFG.init_state.pos

# Per-joint-type motor limits as (effort_limit, saturation_effort, velocity_limit).
#
# "hardware" is the real robot, from Unitree's own go2_description.urdf: hip/thigh are 23.7 Nm at 30.1 rad/s,
# but the calf sits behind a 1.9169:1 knee reduction and is 45.43 Nm at 15.70 rad/s. The project USD carries
# exactly these values on its joint drives too. `unitree_go2witharm_cfg.py` now configures these, so
# "hardware" is also what anything trained from here on is trained against -- use it for new policies.
#
# "legacy" is the single 23.5 Nm / 30.0 rad/s DCMotorCfg group that applied to all 12 joints before that fix,
# i.e. what the committed policies (models/9_10_26_vanilla, models/9_13_26_*) were actually trained against.
# Isaac Lab's own stock Go2 config had the same bug; see isaac-sim/IsaacLab PR #7564. Those policies rely on
# the weak calf saturating and destabilise without it (6 falls vs 0, see README), so run them under "legacy"
# to reproduce their training conditions, and under "hardware" to see what they would do on the real robot.
MOTOR_LIMITS = {
    "legacy": {"hip": (23.5, 23.5, 30.0), "thigh": (23.5, 23.5, 30.0), "calf": (23.5, 23.5, 30.0)},
    "hardware": {"hip": (23.7, 23.7, 30.1), "thigh": (23.7, 23.7, 30.1), "calf": (45.43, 45.43, 15.70)},
}

# Joint armature (rotor inertia). Isaac trained with 0: the DCMotorCfg leaves armature=None and the trained USD
# authors no physxJoint:armature, so PhysX used 0. Menagerie's Go2 uses 0.01, and the real robot does have rotor
# inertia (go2_description.urdf models 12 rotor links at 0.089 kg). 0.0 is the faithful-to-training default.
ARMATURE = 0.0
CONTACT_FORCE_THRESHOLD = 1.0  # N, body_contact / body_contact_arm terminations
FALL_MIN_HEIGHT = 0.12
SEGMENT_SETTLE_S = 1.0  # transient after each command change excluded from the tracking error
METRIC_NAMES = ["lin_x", "lin_y", "ang_z", "pitch", "lean", "height"]

# Per-policy observation layout, keyed by the exported policy's input dimension (read off its first Linear
# layer in load_policy, never guessed from the file path). "cmd_width" is how many floats manual_cmd/
# Segment.command must carry for THIS policy; "has_height_obs" is whether a base_height term is inserted
# between projected_gravity and velocity_commands (see deploy_real.build_observation).
#
#   50: current policies -- 5-wide command, no height observation (see mdp/commands.py's module docstring).
#   51: committed vanilla (models/9_10_26_vanilla) -- 6-wide command (height command, no height observation).
#   52: committed Round 3 (models/9_13_26_*) -- 6-wide command AND a base_height observation.
#
# Sim2sim's own canonical command is always 5-wide (see Segment/load_scenario/--manual_commands below); for
# a 51/52-dim policy the extra 6th ("height") slot is appended here from NOMINAL_HEIGHT_M["legacy"], never
# asked of the user/scenario -- there is no height command any more, see the module docstring.
OBS_LAYOUT = {
    50: {"cmd_width": 5, "has_height_obs": False},
    51: {"cmd_width": 6, "has_height_obs": False},
    52: {"cmd_width": 6, "has_height_obs": True},
}

# Fixed height target used only for the summary table's "height" error column and (for has_height_obs
# policies) the base_height observation itself -- NOT a command, see OBS_LAYOUT above.
#   "new" mirrors RewardsCfg.height_penalty's target_height in quadruped_go2_locomotion_env_cfg.py (not
#     imported from there: that file needs Isaac Lab's full env-cfg machinery, which this script deliberately
#     avoids pulling in). Keep these in sync if either changes.
#   "legacy" mirrors the old DEFAULT_HEIGHT_M / lin_pos_z range midpoint the committed policies trained with.
NOMINAL_HEIGHT_M = {"new": 0.337, "legacy": 0.30}


# ---------------------------------------------------------------------------------------------------------
# Config / policy
# ---------------------------------------------------------------------------------------------------------


def load_sim_config(config_path: str, policy_path: str | None) -> deploy_real.DeployConfig:
    """Same fields as deploy_real.load_config, minus the network interface."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return deploy_real.DeployConfig(
        policy_path=policy_path or raw["policy_path"],
        control_dt=float(raw["control_dt"]),
        network_interface=None,
        action_scale=float(raw["action_scale"]),
        kp=float(raw["kp"]),
        kd=float(raw["kd"]),
        effort_limit=float(raw["effort_limit"]),
        isaac_joint_order=list(raw["isaac_joint_order"]),
        sdk_joint_order=list(raw["sdk_joint_order"]),
        default_joint_pos=dict(raw["default_joint_pos"]),
    )


def load_policy(path: str) -> tuple[torch.jit.ScriptModule, int]:
    policy = torch.jit.load(path, map_location="cpu")
    policy.eval()
    first_weight = next(p for name, p in policy.named_parameters() if p.dim() == 2)
    obs_dim = int(first_weight.shape[1])
    if obs_dim not in OBS_LAYOUT:
        raise ValueError(f"{path} expects a {obs_dim}-dim observation; known dims are {sorted(OBS_LAYOUT)}.")
    return policy, obs_dim


# ---------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Segment:
    name: str
    duration: float
    command: list[float]  # lin_x, lin_y, ang_z, pitch, lean -- always 5-wide, see OBS_LAYOUT above


def load_scenario(path: str) -> list[Segment]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    segments = []
    for entry in raw["segments"]:
        cmd = [float(v) for v in entry["command"]]
        if len(cmd) != 5:
            raise ValueError(f"Segment {entry['name']!r}: command must have 5 values, got {len(cmd)}")
        segments.append(Segment(entry["name"], float(entry["duration"]), cmd))
    return segments


# ---------------------------------------------------------------------------------------------------------
# MuJoCo robot
# ---------------------------------------------------------------------------------------------------------


class MujocoGo2:
    def __init__(self, model: mujoco.MjModel, cfg: deploy_real.DeployConfig, motor_limits: str = "hardware"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.cfg = cfg

        # Per-name lookups, so nothing depends on MuJoCo's own joint/actuator ordering.
        mj_joint_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)}
        missing = set(cfg.sdk_joint_order) - mj_joint_names
        if missing:
            raise ValueError(f"MuJoCo model has no joints named {sorted(missing)}")
        self.qpos_adr_sdk = np.array([model.jnt_qposadr[model.joint(n).id] for n in cfg.sdk_joint_order])
        self.dof_adr_sdk = np.array([model.jnt_dofadr[model.joint(n).id] for n in cfg.sdk_joint_order])
        joint_to_actuator = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        self.act_idx_isaac = np.array([joint_to_actuator[model.joint(n).id] for n in cfg.isaac_joint_order])
        self.qpos_adr_isaac = np.array([model.jnt_qposadr[model.joint(n).id] for n in cfg.isaac_joint_order])
        self.dof_adr_isaac = np.array([model.jnt_dofadr[model.joint(n).id] for n in cfg.isaac_joint_order])

        # Per-joint motor limits, in Isaac joint order. The joint type is taken from the name, which the config's
        # own joint lists already guarantee is one of hip/thigh/calf.
        limits = MOTOR_LIMITS[motor_limits]
        self.motor_limits_name = motor_limits
        triples = [limits[n.split("_")[1]] for n in cfg.isaac_joint_order]
        self.effort_limit = np.array([t[0] for t in triples])
        self.saturation_effort = np.array([t[1] for t in triples])
        self.velocity_limit = np.array([t[2] for t in triples])
        # Isaac Lab DCMotor.__init__: the speed at which the torque-speed line crosses effort_limit.
        self.vel_at_effort_lim = self.velocity_limit * (1.0 + self.effort_limit / self.saturation_effort)
        model.actuator_ctrlrange[self.act_idx_isaac, 0] = -self.effort_limit
        model.actuator_ctrlrange[self.act_idx_isaac, 1] = self.effort_limit
        model.dof_armature[self.dof_adr_isaac] = ARMATURE

        self.default_isaac = np.array([cfg.default_joint_pos[n] for n in cfg.isaac_joint_order])
        self.base_id = model.body("base").id
        self.floor_geom = model.geom("floor").id
        # Bodies whose ground contact ends an episode in training: base (+ head, part of base here) and the arm.
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        self.fall_bodies = {self.base_id} | {i for i, name in enumerate(body_names) if name.startswith("arm_")}
        self.peak_torque = 0.0
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, INIT_BASE_HEIGHT]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr_isaac] = self.default_isaac
        mujoco.mj_forward(self.model, self.data)

    # --- state ---------------------------------------------------------------------------------------------

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.data.qpos[3:7])

    @property
    def base_height(self) -> float:
        return float(self.data.qpos[2])

    def lin_vel_b(self) -> tuple[float, float, float]:
        # Free joint: qvel[0:3] is world-frame linear velocity, qvel[3:6] body-frame angular velocity.
        return deploy_real.quat_rotate_inverse_wxyz(self.quat_wxyz, tuple(float(v) for v in self.data.qvel[0:3]))

    def ang_vel_b(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.data.qvel[3:6])

    def to_robot_state(self) -> deploy_real.LatestRobotState:
        """MuJoCo state packed exactly like a LowState/SportModeState message would leave it (SDK motor order)."""
        state = deploy_real.LatestRobotState(len(self.cfg.sdk_joint_order))
        state.joint_pos = [float(v) for v in self.data.qpos[self.qpos_adr_sdk]]
        state.joint_vel = [float(v) for v in self.data.qvel[self.dof_adr_sdk]]
        state.quat_wxyz = self.quat_wxyz
        state.gyro = self.ang_vel_b()
        state.sportmode_velocity = self.lin_vel_b()
        state.ready = True
        state.last_update = time.monotonic()
        return state

    # --- actuation -----------------------------------------------------------------------------------------

    def dc_motor_torque(self, target_isaac: np.ndarray) -> np.ndarray:
        """Isaac Lab IdealPDActuator.compute + DCMotor._clip_effort, for one physics step.

        Mirrors isaaclab/actuators/actuator_pd.py exactly, including two details that are easy to get wrong:
        the joint velocity is pre-clipped to `vel_at_effort_lim`, and only the outer side of each bound is
        clipped, so above `velocity_limit` max_effort goes negative and the motor is forced to brake.
        """
        q = self.data.qpos[self.qpos_adr_isaac]
        qd = self.data.qvel[self.dof_adr_isaac]
        tau = self.cfg.kp * (target_isaac - q) - self.cfg.kd * qd
        vel = np.clip(qd, -self.vel_at_effort_lim, self.vel_at_effort_lim)
        max_effort = np.minimum(self.saturation_effort * (1.0 - vel / self.velocity_limit), self.effort_limit)
        min_effort = np.maximum(self.saturation_effort * (-1.0 - vel / self.velocity_limit), -self.effort_limit)
        return np.clip(tau, min_effort, max_effort)

    def step(self, target_isaac: np.ndarray) -> None:
        for _ in range(DECIMATION):
            tau = self.dc_motor_torque(target_isaac)
            self.peak_torque = max(self.peak_torque, float(np.max(np.abs(tau))))
            self.data.ctrl[self.act_idx_isaac] = tau
            mujoco.mj_step(self.model, self.data)

    # --- termination ---------------------------------------------------------------------------------------

    def fallen(self) -> str | None:
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return "non-finite state"
        if self.base_height < FALL_MIN_HEIGHT:
            return f"base height {self.base_height:.3f} m"
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self.floor_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.floor_geom else contact.geom1
            body = int(self.model.geom_bodyid[other])
            if body in self.fall_bodies:
                mujoco.mj_contactForce(self.model, self.data, i, force)
                if abs(force[0]) > CONTACT_FORCE_THRESHOLD:
                    return f"{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)} touched the ground"
        return None


# ---------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------


@dataclass
class SegmentStats:
    name: str
    sq_err: dict[str, float] = field(default_factory=lambda: dict.fromkeys(METRIC_NAMES, 0.0))
    samples: int = 0
    falls: int = 0
    peak_torque: float = 0.0
    target_sum: np.ndarray = field(default_factory=lambda: np.zeros(12))  # Isaac order, rad

    def add_target(self, target: np.ndarray) -> None:
        self.target_sum += target

    def add(self, errors: dict[str, float]) -> None:
        for k, v in errors.items():
            self.sq_err[k] += v * v
        self.samples += 1

    def rms(self) -> dict[str, float]:
        return {k: math.sqrt(v / self.samples) if self.samples else float("nan") for k, v in self.sq_err.items()}


def print_summary(
    stats: list[SegmentStats], policy_path: str, nominal_height: float,
    default_isaac: np.ndarray, joint_names: list[str],
) -> None:
    print(f"\n[SUMMARY] policy={policy_path} nominal_height={nominal_height:.3f} m")
    print(f"  steady-state RMS error (first {SEGMENT_SETTLE_S:.1f} s of each segment excluded)")
    print("  height error is actual vs. the fixed nominal_height above -- there is no height COMMAND any more")
    print("  max_dev = largest mean joint-target offset from default_joint_pos (deg), worst joint in brackets")
    header = f"  {'segment':<16}" + "".join(f"{n:>9}" for n in METRIC_NAMES) + f"{'falls':>7}{'peak_tau':>10}  max_dev"
    print(header)
    for s in stats:
        rms = s.rms()
        dev = np.degrees(s.target_sum / max(s.samples, 1) - default_isaac)
        worst = int(np.argmax(np.abs(dev)))
        errors = "".join(f"{rms[n]:>9.3f}" for n in METRIC_NAMES)
        print(
            f"  {s.name:<16}{errors}{s.falls:>7d}{s.peak_torque:>10.1f}  {dev[worst]:+6.1f} [{joint_names[worst]}]"
        )
    print(f"  total falls: {sum(s.falls for s in stats)}")


def close_viewer(viewer, viewer_threads: set[threading.Thread], timeout_s: float = 5.0) -> None:
    """Close the passive viewer and wait for its render thread to exit.

    `Handle.close()` only requests an exit; the viewer's daemon thread keeps rendering for a moment and then tears
    down its GL context. Returning straight away lets interpreter shutdown run glfw's atexit `glfw.terminate()`
    while that thread is still inside GLFW, which segfaults (reproduced in the container, faulthandler trace in
    mujoco/viewer.py `_launch_internal` vs glfw `terminate`). Joining the thread first removes the race.
    """
    viewer.close()
    for thread in viewer_threads:
        thread.join(timeout=timeout_s)


# ---------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Run an exported Go2 policy in MuJoCo (sim2sim).")
    parser.add_argument("--config", default=os.path.join(_REPO_ROOT, "deploy", "configs", "go2_locomotion.yaml"))
    parser.add_argument("--policy", default=None, help="Exported policy.pt (default: policy_path in --config).")
    parser.add_argument(
        "--height_source",
        choices=["true", "constant"],
        default="true",
        help="Only affects a 52-dim (Round 3) policy's base_height OBSERVATION input: 'true' = simulated height, "
        "'constant' = the legacy hardware stand-in NOMINAL_HEIGHT_M['legacy']. Ignored for every other policy "
        "(50/51-dim), which have no height observation at all -- see OBS_LAYOUT.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scenario", default=None, help="YAML list of command segments (see sim2sim/scenarios/).")
    source.add_argument("--manual_commands", action="store_true", help="Drive commands from the slider GUI.")
    parser.add_argument("--command_file", default="/tmp/manual_command.json")
    parser.add_argument("--duration", type=float, default=60.0, help="Run length without --scenario (s).")
    parser.add_argument("--headless", action="store_true", help="No viewer; runs as fast as possible.")
    parser.add_argument("--live_plot", action="store_true", help="TensorBoard log to logs/sim2sim_dashboard/.")
    parser.add_argument("--no_reset", action="store_true", help="Stop at the first fall instead of resetting.")
    parser.add_argument(
        "--motor_limits",
        choices=sorted(MOTOR_LIMITS),
        default="hardware",
        help="'hardware' (default) = the real robot's per-joint values, which unitree_go2witharm_cfg.py now also "
        "trains against (calf 45.43 Nm / 15.70 rad/s); 'legacy' = the single 23.5 Nm / 30 rad/s group the "
        "committed pre-fix policies were trained with. See MOTOR_LIMITS.",
    )
    parser.add_argument("--save_xml", default=None, help="Write the compiled MuJoCo model to this XML path and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    model, spec = build_model()
    if args.save_xml:
        with open(args.save_xml, "w") as f:
            f.write(spec.to_xml())
        print(f"[INFO] Wrote {args.save_xml}")
        return

    with open(args.config) as f:
        policy_path = args.policy or yaml.safe_load(f)["policy_path"]
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(_REPO_ROOT, policy_path)
    policy, obs_dim = load_policy(policy_path)
    layout = OBS_LAYOUT[obs_dim]
    is_legacy = obs_dim != 50
    nominal_height = NOMINAL_HEIGHT_M["legacy" if is_legacy else "new"]
    cfg = load_sim_config(args.config, policy_path)
    sdk_to_isaac, _ = deploy_real.build_joint_index_maps(cfg)
    robot = MujocoGo2(model, cfg, motor_limits=args.motor_limits)
    default_isaac = robot.default_isaac.tolist()
    height_source_note = f" (obs source: {args.height_source})" if layout["has_height_obs"] else ""
    print(
        f"[INFO] policy={policy_path} obs_dim={obs_dim} cmd_width={layout['cmd_width']} "
        f"has_height_obs={layout['has_height_obs']}{height_source_note} nominal_height={nominal_height:.3f} m "
        f"mass={float(np.sum(model.body_mass)):.2f} kg dt={model.opt.timestep} decimation={DECIMATION} "
        f"motor_limits={args.motor_limits} (calf {robot.effort_limit[-1]:.2f} Nm @ "
        f"{robot.velocity_limit[-1]:.2f} rad/s) armature={ARMATURE}"
    )

    if args.scenario:
        segments = load_scenario(args.scenario)
    else:
        segments = [Segment("manual" if args.manual_commands else "zero", args.duration, [0, 0, 0, 0, 0])]
    manual_buf = None
    if args.manual_commands:
        manual_buf = deploy_real.ManualCommandBuffer()
        threading.Thread(
            target=deploy_real._manual_command_file_poll_loop, args=(manual_buf, args.command_file), daemon=True
        ).start()
        deploy_real.launch_slider(args.command_file)

    writer = None
    if args.live_plot:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = os.path.join(_REPO_ROOT, "logs", "sim2sim_dashboard", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[INFO] Live plot: {log_dir}")

    viewer = None
    viewer_threads: set[threading.Thread] = set()
    if not args.headless:
        import mujoco.viewer

        threads_before = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(model, robot.data)
        viewer_threads = set(threading.enumerate()) - threads_before
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = robot.base_id
        viewer.cam.distance = 2.0

    try:
        dt = cfg.control_dt
        last_action = [0.0] * 12
        stats: list[SegmentStats] = []
        step = 0
        stop = False
        for seg in segments:
            seg_stats = SegmentStats(seg.name)
            stats.append(seg_stats)
            robot.peak_torque = 0.0
            print(f"[SEGMENT] {seg.name}: {seg.duration:.1f} s, cmd={seg.command}")
            for k in range(round(seg.duration / dt)):
                tick_start = time.monotonic()
                cmd = manual_buf.get() if manual_buf else list(seg.command)
                # Pad to 6-wide with the legacy nominal height for an older (51/52-dim) policy: there is no
                # height command any more (see OBS_LAYOUT/module docstring above), so this is never read from
                # the slider or scenario, only appended here to match what those policies' velocity_commands
                # observation term was trained to expect.
                manual_cmd = cmd + [nominal_height] if layout["cmd_width"] == 6 else cmd
                base_height_obs = None
                if layout["has_height_obs"]:
                    base_height_obs = robot.base_height if args.height_source == "true" else nominal_height
                obs = deploy_real.build_observation(
                    cfg, robot.to_robot_state(), sdk_to_isaac, default_isaac, manual_cmd, last_action,
                    base_height=base_height_obs,
                )
                with torch.inference_mode():
                    action = policy(obs)[0].numpy()
                target = robot.default_isaac + cfg.action_scale * action
                robot.step(target)
                last_action = action.tolist()

                lin_vel = robot.lin_vel_b()
                ang_vel_z = robot.ang_vel_b()[2]
                lean, pitch = deploy_real.quat_to_pitch_lean(robot.quat_wxyz)
                if k * dt >= SEGMENT_SETTLE_S:
                    seg_stats.add(
                        {
                            "lin_x": lin_vel[0] - cmd[0],
                            "lin_y": lin_vel[1] - cmd[1],
                            "ang_z": ang_vel_z - cmd[2],
                            "pitch": pitch - cmd[3],
                            "lean": lean - cmd[4],
                            "height": robot.base_height - nominal_height,
                        }
                    )
                    seg_stats.add_target(target)
                if writer is not None:
                    # deploy_real.log_live_plot covers the 5 velocity/pitch/lean tags; height is logged here
                    # directly (deploy_real's version no longer carries height tags at all -- real hardware
                    # has no height measurement to plot -- but sim2sim, unlike real hardware, CAN measure it).
                    deploy_real.log_live_plot(writer, step, cmd, lin_vel, ang_vel_z, pitch, lean)
                    writer.add_scalar("Metrics/base_velocity/cmd_height", nominal_height, step)
                    writer.add_scalar("Metrics/base_velocity/actual_height", robot.base_height, step)
                step += 1

                reason = robot.fallen()
                if reason:
                    seg_stats.falls += 1
                    print(f"[FALL] {seg.name} t={k * dt:.2f} s: {reason}")
                    if args.no_reset:
                        stop = True
                        break
                    robot.reset()
                    last_action = [0.0] * 12

                if viewer is not None:
                    if not viewer.is_running():
                        stop = True
                        break
                    viewer.sync()
                    time.sleep(max(0.0, dt - (time.monotonic() - tick_start)))
            seg_stats.peak_torque = robot.peak_torque
            if stop:
                break

        print_summary(stats, policy_path, nominal_height, robot.default_isaac, cfg.isaac_joint_order)
    finally:
        if viewer is not None:
            close_viewer(viewer, viewer_threads)
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
