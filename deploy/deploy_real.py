# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-hardware deployment: runs an exported Go2 policy on a physical Unitree Go2 over the SDK (DDS),
reusing this repo's existing manual-command slider and live-dashboard tooling unchanged.

Interpreter: run this with Isaac Sim's bundled Python directly (NOT `isaaclab -p`, which launches the full
Kit app -- unnecessary here, this script never touches pxr/Omniverse/SimulationApp):
    /workspace/isaaclab/_isaac_sim/python.sh deploy/deploy_real.py --network_interface enp3s0 --dry_run

That interpreter already has torch/tensorboard (confirmed importable without launching Kit, same as this
session's TensorBoard log-reading checks). The Unitree SDK itself is NOT on PyPI (confirmed directly) --
install from source, plus its own native-library dependency (see docker/Dockerfile's build block for the
exact commands this was verified with, including the numpy<2 repin the SDK's install otherwise breaks):
    /workspace/isaaclab/_isaac_sim/python.sh -m pip install "git+https://github.com/unitreerobotics/unitree_sdk2_python.git"

See deploy.md for the full staged rollout (hoisted dry-run before anything else, joint-mapping
verification via deploy/dump_joint_order.py, etc.) -- this file implements the control loop, not the
safety procedure around running it.

Frequency note: the policy is stepped at exactly `control_dt` (50 Hz, matching training's
decimation*sim.dt), independent of however fast the robot's own LowState publishes (typically much
faster). The LowState subscriber callback only ever overwrites a "latest state" snapshot. Publishing is
itself two-rate, confirmed against unitree_sdk2py's own reference example
(example/go2/low_level/go2_stand_example.py): the 50 Hz main loop only updates a shared target buffer
(`LowCmdTarget`); a separate ~500 Hz `RecurrentThread` (`publish_low_cmd`) continuously re-publishes
whatever the latest target is, giving the robot's communication watchdog a steady heartbeat independent of
inference timing -- see `LatestRobotState`/`LowCmdTarget`/`publish_low_cmd` and the main loop below.

Before any LowCmd is meaningful, the robot's onboard sport-mode controller must release control
(`release_motion_control`) -- confirmed required via the same reference example (SportClient +
MotionSwitcherClient, looping StandDown()+ReleaseMode() until CheckMode() reports no active mode) and
independently corroborated by docs.quadruped.de's Go2 "Low-Level Control" page.

Height tracking is NOT currently a real estimate -- see `DEFAULT_HEIGHT_M` below; deliberately simplified
to a fixed constant since no reliable real-hardware source has been verified yet (SportModeState's
availability once sport mode is released is unconfirmed).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import torch
import yaml

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread

    _UNITREE_SDK_AVAILABLE = True
except ImportError:
    # Import-time failure is tolerated (not raised) so --help, config-loading, and --dry_run's non-hardware
    # parts (obs assembly, timing, slider/dashboard wiring) can still be exercised/tested without the SDK
    # installed -- e.g. on a dev machine that isn't yet connected to a robot. main() checks this flag and
    # refuses to proceed to actual DDS init if it's False.
    _UNITREE_SDK_AVAILABLE = False

# Confirmed directly against unitree_sdk2py's own reference example
# (example/go2/low_level/go2_stand_example.py + unitree_legged_const.py, fetched and read this session) --
# not shipped as importable constants in the installed package itself, only as a standalone file alongside
# that example, so defined here instead of imported.
_LOWCMD_HEAD = (0xFE, 0xEF)
_LOWLEVEL = 0xFF  # LowCmd.level_flag -- tells the robot to obey LowCmd instead of its onboard controller
_MOTOR_MODE_PMSM = 0x01  # LowCmd.motor_cmd[i].mode -- required on every motor, set once at init

# Height tracking isn't working yet (no reliable real-hardware source verified -- SportModeState's
# real-hardware availability after ReleaseMode() is unconfirmed, see deploy.md Stage 3, and forward
# kinematics isn't implemented). Deliberately simplified to a fixed constant for now rather than guessing:
# the robot's nominal standing height, matching default_joint_pos's implied stance. Revisit once hardware
# is available to test what's actually reliable.
DEFAULT_HEIGHT_M = 0.3


# ---------------------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------------------


@dataclass
class DeployConfig:
    policy_path: str
    control_dt: float
    network_interface: str | None
    action_scale: float
    kp: float
    kd: float
    effort_limit: float
    isaac_joint_order: list[str]
    sdk_joint_order: list[str]
    default_joint_pos: dict[str, float]
    has_height_obs: bool  # True for the experimental (Round 3) policy, False for vanilla -- see obs assembly.


def load_config(path: str, cli_network_interface: str | None) -> DeployConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    network_interface = cli_network_interface or raw["network_interface"]
    if not network_interface:
        raise ValueError(
            "network_interface is not set -- pass --network_interface or set it in the config file. "
            "Refusing to guess a NIC for DDS traffic to the robot."
        )
    return DeployConfig(
        policy_path=raw["policy_path"],
        control_dt=float(raw["control_dt"]),
        network_interface=network_interface,
        action_scale=float(raw["action_scale"]),
        kp=float(raw["kp"]),
        kd=float(raw["kd"]),
        effort_limit=float(raw["effort_limit"]),
        isaac_joint_order=list(raw["isaac_joint_order"]),
        sdk_joint_order=list(raw["sdk_joint_order"]),
        default_joint_pos=dict(raw["default_joint_pos"]),
        has_height_obs="round3" in raw["policy_path"] or "experimental" in raw["policy_path"],
    )


# ---------------------------------------------------------------------------------------------------------
# Joint-order plumbing -- deploy/configs/go2_locomotion.yaml's isaac_joint_order and sdk_joint_order are
# both lists of the SAME 12 joint names, just in each side's own order. Every permutation needed (SDK ->
# Isaac for building observations, Isaac -> SDK for sending commands) is a pure index remap built once at
# startup, never recomputed per-tick.
# ---------------------------------------------------------------------------------------------------------


def build_joint_index_maps(cfg: DeployConfig) -> tuple[list[int], list[int]]:
    """Returns (sdk_to_isaac, isaac_to_sdk): sdk_to_isaac[i] is the SDK motor index whose reading goes into
    Isaac-ordered slot i; isaac_to_sdk[i] is the SDK motor index that Isaac-ordered action slot i should be
    sent to. Both are pure permutations of range(12), built from matching joint *names*, not by assuming any
    numeric relationship between the two orders."""
    if set(cfg.isaac_joint_order) != set(cfg.sdk_joint_order):
        raise ValueError(
            "isaac_joint_order and sdk_joint_order in the config don't contain the same joint names -- "
            f"isaac has {set(cfg.isaac_joint_order) - set(cfg.sdk_joint_order)} extra, "
            f"sdk has {set(cfg.sdk_joint_order) - set(cfg.isaac_joint_order)} extra."
        )
    sdk_index_of = {name: i for i, name in enumerate(cfg.sdk_joint_order)}
    sdk_to_isaac = [sdk_index_of[name] for name in cfg.isaac_joint_order]
    isaac_index_of = {name: i for i, name in enumerate(cfg.isaac_joint_order)}
    isaac_to_sdk = [isaac_index_of[name] for name in cfg.sdk_joint_order]
    return sdk_to_isaac, isaac_to_sdk


# ---------------------------------------------------------------------------------------------------------
# Robot state -- written only by the DDS callback, read only by the main loop. A lock guards the handful of
# plain-float/list fields; there's no tensor math inside the callback itself (kept deliberately trivial, per
# the frequency-matching design: the callback's only job is "remember the latest message").
# ---------------------------------------------------------------------------------------------------------


class LatestRobotState:
    def __init__(self, num_joints: int):
        self._lock = threading.Lock()
        self.joint_pos = [0.0] * num_joints  # SDK order
        self.joint_vel = [0.0] * num_joints  # SDK order
        self.quat_wxyz = (1.0, 0.0, 0.0, 0.0)  # IMU orientation, body-to-world
        self.gyro = (0.0, 0.0, 0.0)  # body-frame angular velocity, rad/s
        # NOTE: same open question as DEFAULT_HEIGHT_M above -- SportModeState's real-hardware availability
        # after ReleaseMode() is unconfirmed (see deploy.md Stage 3), so this may also go stale/frozen once
        # the robot is actually in the state the policy needs it in. Left wired up (unlike height, which
        # was simplified to a constant) since there's no evidence yet it specifically fails, but flagged
        # here so it isn't silently trusted if it turns out not to work either.
        self.sportmode_velocity = (0.0, 0.0, 0.0)  # body-frame linear velocity estimate, m/s
        self.ready = False  # False until at least one LowState message has been received

    def update_from_low_state(self, msg) -> None:
        with self._lock:
            self.joint_pos = [ms.q for ms in msg.motor_state[: len(self.joint_pos)]]
            self.joint_vel = [ms.dq for ms in msg.motor_state[: len(self.joint_vel)]]
            self.quat_wxyz = tuple(msg.imu_state.quaternion)
            self.gyro = tuple(msg.imu_state.gyroscope)
            self.ready = True

    def update_from_sportmode_state(self, msg) -> None:
        with self._lock:
            self.sportmode_velocity = tuple(msg.velocity)

    def snapshot(self) -> "LatestRobotState":
        """Returns a shallow copy safe to read from without holding the lock for the whole control tick."""
        with self._lock:
            snap = LatestRobotState(len(self.joint_pos))
            snap.joint_pos = list(self.joint_pos)
            snap.joint_vel = list(self.joint_vel)
            snap.quat_wxyz = self.quat_wxyz
            snap.gyro = self.gyro
            snap.sportmode_velocity = self.sportmode_velocity
            snap.ready = self.ready
            return snap


def quat_rotate_inverse_wxyz(quat_wxyz: tuple[float, float, float, float], vec: tuple[float, float, float]):
    """Rotates `vec` (world frame) into the body frame described by `quat_wxyz` (w, x, y, z), matching
    Isaac Lab's math_utils.quat_rotate_inverse convention (same one projected_gravity's obs term uses)."""
    w, x, y, z = quat_wxyz
    vx, vy, vz = vec
    # v' = q^-1 * v * q, expanded closed-form (standard quaternion-vector rotation, inverse via conjugate
    # since q is unit-norm).
    qvec = (x, y, z)
    uv = (
        qvec[1] * vz - qvec[2] * vy,
        qvec[2] * vx - qvec[0] * vz,
        qvec[0] * vy - qvec[1] * vx,
    )
    uuv = (
        qvec[1] * uv[2] - qvec[2] * uv[1],
        qvec[2] * uv[0] - qvec[0] * uv[2],
        qvec[0] * uv[1] - qvec[1] * uv[0],
    )
    return (
        vx + 2.0 * (-w * uv[0] + uuv[0]),
        vy + 2.0 * (-w * uv[1] + uuv[1]),
        vz + 2.0 * (-w * uv[2] + uuv[2]),
    )


def quat_to_pitch_lean(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float]:
    """Extracts (lean/roll, pitch) from a body quaternion -- same XYZ-Euler convention this repo's
    mdp/commands.py uses (math_utils.euler_xyz_from_quat returns (lean, pitch, yaw))."""
    w, x, y, z = quat_wxyz
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(torch.tensor(sinr_cosp), torch.tensor(cosr_cosp)).item()
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = torch.asin(torch.tensor(sinp)).item()
    return roll, pitch


# ---------------------------------------------------------------------------------------------------------
# Manual command slider -- identical protocol to scripts/rsl_rl/play.py's --manual_commands (same JSON
# schema, same poll interval, same subprocess launch). No CommandManager exists here, so the poll thread
# writes straight into a plain list guarded by a lock instead of calling a command term's setter.
# ---------------------------------------------------------------------------------------------------------


class ManualCommandBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.3]  # lin_x, lin_y, ang_z, pitch, lean, height

    def set(self, values: list[float]) -> None:
        with self._lock:
            self._values = list(values)

    def get(self) -> list[float]:
        with self._lock:
            return list(self._values)


def _manual_command_file_poll_loop(buf: ManualCommandBuffer, path: str, poll_interval: float = 0.1):
    """Verbatim port of scripts/rsl_rl/play.py's _manual_command_file_poll_loop, minus the CommandManager
    dependency -- see that function's docstring for the tolerated-errors rationale (identical here)."""
    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(path)
            if mtime != last_mtime:
                with open(path) as f:
                    data = json.load(f)
                buf.set(
                    [
                        data["lin_vel_x"],
                        data["lin_vel_y"],
                        data["ang_vel_z"],
                        data["pitch"],
                        data["lean"],
                        data["height"],
                    ]
                )
                last_mtime = mtime
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(poll_interval)


def launch_slider(command_file: str) -> subprocess.Popen:
    """Verbatim port of play.py's slider subprocess launch (see that file for the PYTHONPATH/
    LD_LIBRARY_PATH-stripping rationale -- identical here since this script also runs under Isaac Sim's
    bundled Python, which would otherwise leak its own 3.11 env into the system python3 child)."""
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    slider_script = os.path.join(scripts_dir, "manual_command_slider.py")
    slider_env = os.environ.copy()
    slider_env.pop("PYTHONPATH", None)
    slider_env.pop("LD_LIBRARY_PATH", None)
    proc = subprocess.Popen(["/usr/bin/python3", slider_script, "--command_file", command_file], env=slider_env)
    atexit.register(proc.terminate)
    return proc


# ---------------------------------------------------------------------------------------------------------
# Mode release -- REQUIRED prerequisite, confirmed against unitree_sdk2py's own reference example
# (example/go2/low_level/go2_stand_example.py: SportClient + MotionSwitcherClient, looping
# StandDown()+ReleaseMode() until CheckMode() reports no active mode) and independently corroborated by
# docs.quadruped.de's Go2 "Low-Level Control" page (stand_down -> damp required before low-level control).
# Without this, the robot's onboard sport-mode controller is still active and will fight/ignore LowCmd --
# sending well-formed LowCmd before this step is meaningless, not just ineffective.
# ---------------------------------------------------------------------------------------------------------


def release_motion_control(timeout_s: float = 5.0, max_attempts: int = 10) -> None:
    sc = SportClient()
    sc.SetTimeout(timeout_s)
    sc.Init()
    msc = MotionSwitcherClient()
    msc.SetTimeout(timeout_s)
    msc.Init()

    for attempt in range(max_attempts):
        code, result = msc.CheckMode()
        if code != 0:
            # A failed/timed-out RPC call is NOT the same as "confirmed no active mode" -- it means we
            # don't know the robot's state at all, which could still be dangerously active. Treating this
            # the same as a genuine release (an earlier version of this function did) would let the control
            # loop proceed against a robot we never actually confirmed released. Retry instead; only the
            # final `raise` below is reachable if every attempt fails this way.
            print(f"[WARN] CheckMode() call failed (code={code}) -- cannot confirm robot state "
                  f"(attempt {attempt + 1}/{max_attempts}), retrying...")
            time.sleep(1.0)
            continue
        if not result or not result.get("name"):
            print("[INFO] Onboard motion-control mode released -- LowCmd is now authoritative.")
            return
        print(f"[INFO] Releasing onboard mode {result.get('name')!r} (attempt {attempt + 1}/{max_attempts})...")
        sc.StandDown()
        msc.ReleaseMode()
        time.sleep(1.0)

    raise RuntimeError(
        f"Failed to release the robot's onboard motion-control mode after {max_attempts} attempts -- "
        "check the robot's state manually (e.g. via the app) before proceeding. Sending LowCmd while the "
        "onboard controller is still active is not safe."
    )


# ---------------------------------------------------------------------------------------------------------
# LowCmd construction -- header/level_flag/per-motor mode are set ONCE here (confirmed against the
# reference's InitLowCmd()/LowCmdWrite() split: those fields never change again after init), matching
# unitree_legged_const.py's LOWLEVEL/PMSM-mode values. Only q/dq/kp/kd/tau are mutated per publish tick.
# ---------------------------------------------------------------------------------------------------------


def init_low_cmd(cfg: DeployConfig) -> "LowCmd_":
    low_cmd = unitree_go_msg_dds__LowCmd_()
    low_cmd.head[0] = _LOWCMD_HEAD[0]
    low_cmd.head[1] = _LOWCMD_HEAD[1]
    low_cmd.level_flag = _LOWLEVEL
    for i in range(12):
        low_cmd.motor_cmd[i].mode = _MOTOR_MODE_PMSM
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].kp = cfg.kp
        low_cmd.motor_cmd[i].kd = cfg.kd
        low_cmd.motor_cmd[i].tau = 0.0
    return low_cmd


class LowCmdTarget:
    """Shared buffer between the 50 Hz policy-inference loop (writer, via `set`) and the ~500 Hz publish
    thread (reader, via `get`) -- see the frequency-matching design in the module docstring and
    unitree_sdk2py's own reference example, which uses the same split (a RecurrentThread continuously
    re-publishing whatever the latest target is, decoupled from whatever computes it). Holds only the
    per-tick-varying Isaac-ordered target joint positions; everything else in LowCmd is fixed at init."""

    def __init__(self, initial_target_isaac_order: list[float]):
        self._lock = threading.Lock()
        self._target = list(initial_target_isaac_order)

    def set(self, target_isaac_order: list[float]) -> None:
        with self._lock:
            self._target = list(target_isaac_order)

    def get(self) -> list[float]:
        with self._lock:
            return list(self._target)


def publish_low_cmd(low_cmd, target_buf: LowCmdTarget, isaac_to_sdk: list[int], publisher, crc) -> None:
    """The ~500 Hz RecurrentThread target: reads the latest computed target and republishes it, regardless
    of whether the 50 Hz policy loop has produced a new one since the last tick -- this is what gives the
    robot's communication watchdog a steady heartbeat independent of inference timing."""
    target_isaac_order = target_buf.get()
    for isaac_i, sdk_i in enumerate(isaac_to_sdk):
        low_cmd.motor_cmd[sdk_i].q = target_isaac_order[isaac_i]
    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


# ---------------------------------------------------------------------------------------------------------
# Live dashboard logging -- identical tag names to play.py's _log_live_plot, so scripts/live_dashboard.py
# needs zero changes to render this. `step` increments once per control tick, exactly like play.py's
# per-physics-step counter.
# ---------------------------------------------------------------------------------------------------------


def log_live_plot(writer, step: int, cmd: list[float], actual_lin_vel: tuple, actual_ang_vel_z: float,
                   actual_pitch: float, actual_lean: float, actual_height: float) -> None:
    writer.add_scalar("Metrics/base_velocity/cmd_lin_vel_x", cmd[0], step)
    writer.add_scalar("Metrics/base_velocity/actual_lin_vel_x", actual_lin_vel[0], step)
    writer.add_scalar("Metrics/base_velocity/cmd_lin_vel_y", cmd[1], step)
    writer.add_scalar("Metrics/base_velocity/actual_lin_vel_y", actual_lin_vel[1], step)
    writer.add_scalar("Metrics/base_velocity/cmd_ang_vel_z", cmd[2], step)
    writer.add_scalar("Metrics/base_velocity/actual_ang_vel_z", actual_ang_vel_z, step)
    writer.add_scalar("Metrics/base_velocity/cmd_pitch", cmd[3], step)
    writer.add_scalar("Metrics/base_velocity/actual_pitch", actual_pitch, step)
    writer.add_scalar("Metrics/base_velocity/cmd_lean", cmd[4], step)
    writer.add_scalar("Metrics/base_velocity/actual_lean", actual_lean, step)
    writer.add_scalar("Metrics/base_velocity/cmd_height", cmd[5], step)
    writer.add_scalar("Metrics/base_velocity/actual_height", actual_height, step)
    writer.flush()


# ---------------------------------------------------------------------------------------------------------
# Observation assembly -- must exactly match ObservationsCfg.PolicyCfg's term order in
# quadruped_go2_locomotion_env_cfg.py: base_lin_vel, base_ang_vel, projected_gravity, [base_height if
# has_height_obs], velocity_commands, joint_pos (rel to default), joint_vel, last_action. No noise is added
# (play mode already disables observation corruption -- QuadrupedLocomotionEnvCfg_PLAY sets
# enable_corruption=False -- so hardware inference shouldn't add it either).
# ---------------------------------------------------------------------------------------------------------


def build_observation(
    cfg: DeployConfig,
    state: LatestRobotState,
    sdk_to_isaac: list[int],
    default_joint_pos_isaac_order: list[float],
    manual_cmd: list[float],
    last_action: list[float],
) -> torch.Tensor:
    lean, pitch = quat_to_pitch_lean(state.quat_wxyz)
    gravity_b = quat_rotate_inverse_wxyz(state.quat_wxyz, (0.0, 0.0, -1.0))

    joint_pos_isaac = [state.joint_pos[sdk_i] for sdk_i in sdk_to_isaac]
    joint_vel_isaac = [state.joint_vel[sdk_i] for sdk_i in sdk_to_isaac]
    joint_pos_rel = [q - q0 for q, q0 in zip(joint_pos_isaac, default_joint_pos_isaac_order)]

    parts: list[float] = []
    parts += list(state.sportmode_velocity)  # base_lin_vel (body frame), 3
    parts += list(state.gyro)  # base_ang_vel (body frame), 3
    parts += list(gravity_b)  # projected_gravity, 3
    if cfg.has_height_obs:
        parts += [DEFAULT_HEIGHT_M]  # base_height, 1 -- fixed constant for now, see module docstring
    parts += list(manual_cmd)  # velocity_commands, 6: [lin_x, lin_y, ang_z, pitch, lean, height]
    parts += joint_pos_rel  # joint_pos, 12
    parts += joint_vel_isaac  # joint_vel, 12
    parts += list(last_action)  # actions (previous), 12

    return torch.tensor(parts, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)


# ---------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy a trained Go2 policy on real hardware.")
    parser.add_argument(
        "--config", type=str, default=os.path.join(os.path.dirname(__file__), "configs", "go2_locomotion.yaml")
    )
    parser.add_argument("--network_interface", type=str, default=None, help="Overrides the config file's value.")
    parser.add_argument(
        "--manual_commands",
        action="store_true",
        default=False,
        help="Launch the same slider window play.py uses, writing to --command_file.",
    )
    parser.add_argument("--command_file", type=str, default="/tmp/manual_command.json")
    parser.add_argument(
        "--live_plot",
        action="store_true",
        default=False,
        help="Log per-step commanded-vs-actual data to logs/hardware_dashboard/, same tags as play.py's --live_plot.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help=(
            "Run the full control loop (state read, observation assembly, inference, LowCmd construction, "
            "timing) but never call the DDS publisher's Write() -- prints what would have been sent instead. "
            "Use this before ever connecting to a powered robot -- see deploy.md Stage 4."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.network_interface)

    if not _UNITREE_SDK_AVAILABLE:
        if not args.dry_run:
            raise RuntimeError(
                "unitree_sdk2py is not installed. Install it into this same interpreter "
                "(`_isaac_sim/python.sh -m pip install unitree_sdk2py`) before running without --dry_run."
            )
        print("[WARN] unitree_sdk2py not installed -- --dry_run will skip all DDS init and use zeroed state.")

    sdk_to_isaac, isaac_to_sdk = build_joint_index_maps(cfg)
    default_joint_pos_isaac_order = [cfg.default_joint_pos[name] for name in cfg.isaac_joint_order]
    default_joint_pos_sdk_order = [cfg.default_joint_pos[name] for name in cfg.sdk_joint_order]

    print(f"[INFO] Loading policy from: {cfg.policy_path}")
    policy = torch.jit.load(cfg.policy_path)
    policy.eval()

    state = LatestRobotState(num_joints=12)

    manual_cmd = ManualCommandBuffer()
    if args.manual_commands:
        threading.Thread(
            target=_manual_command_file_poll_loop, args=(manual_cmd, args.command_file), daemon=True
        ).start()
        launch_slider(args.command_file)
        print(f"[INFO] Manual command mode: slider window launched (writing to {args.command_file}).")

    live_plot_writer = None
    if args.live_plot:
        from torch.utils.tensorboard import SummaryWriter

        live_plot_dir = os.path.join("logs", "hardware_dashboard", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        live_plot_writer = SummaryWriter(log_dir=live_plot_dir)
        print(f"[INFO] Live plot logging to: {live_plot_dir}")

    # --- DDS init ---
    low_cmd_publisher = None
    crc = None
    target_buf = None
    publish_thread = None
    if _UNITREE_SDK_AVAILABLE and not args.dry_run:
        ChannelFactoryInitialize(0, cfg.network_interface)
        crc = CRC()

        # MUST happen before LowCmd means anything -- see release_motion_control's docstring/comment above.
        release_motion_control()

        def _on_low_state(msg: LowState_):
            state.update_from_low_state(msg)

        def _on_sportmode_state(msg: SportModeState_):
            state.update_from_sportmode_state(msg)

        low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        low_state_sub.Init(_on_low_state, 10)
        sportmode_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        sportmode_sub.Init(_on_sportmode_state, 10)

        low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        low_cmd_publisher.Init()

        print(f"[INFO] Waiting for first LowState message on interface {cfg.network_interface}...")
        while not state.ready:
            time.sleep(0.05)
        print("[INFO] Robot state stream connected.")

        # Two-rate publish architecture (see module docstring): the 50 Hz loop below only ever updates
        # target_buf; this dedicated ~500 Hz RecurrentThread is what actually calls Write(), matching
        # unitree_sdk2py's own reference example (RecurrentThread(interval=0.002, target=LowCmdWrite)).
        low_cmd = init_low_cmd(cfg)
        target_buf = LowCmdTarget(default_joint_pos_isaac_order)
        publish_thread = RecurrentThread(
            interval=0.002,
            target=publish_low_cmd,
            name="lowcmd_publish",
            args=(low_cmd, target_buf, isaac_to_sdk, low_cmd_publisher, crc),
        )
        publish_thread.Start()
        print("[INFO] LowCmd publish thread started at 500 Hz.")
    else:
        print("[INFO] --dry_run: skipping DDS init, using zeroed/default robot state throughout.")
        state.joint_pos = list(default_joint_pos_sdk_order)
        state.ready = True

    last_action = [0.0] * 12
    step = 0
    control_dt = cfg.control_dt

    print(f"[INFO] Entering control loop at {1.0 / control_dt:.0f} Hz (dry_run={args.dry_run}). Ctrl+C to stop.")
    try:
        while True:
            tick_start = time.perf_counter()
            snap = state.snapshot()
            cmd = manual_cmd.get()

            obs = build_observation(cfg, snap, sdk_to_isaac, default_joint_pos_isaac_order, cmd, last_action)
            with torch.inference_mode():
                action = policy(obs)[0].tolist()
            last_action = action

            # Isaac-ordered target joint positions. PD gains/mode/head are fixed at init (init_low_cmd);
            # only this target changes per policy tick. Actual publishing happens on the separate ~500 Hz
            # thread started above -- this loop only updates the shared buffer at the 50 Hz policy rate.
            target_isaac_order = [
                default_joint_pos_isaac_order[i] + cfg.action_scale * action[i] for i in range(12)
            ]

            if target_buf is not None:
                target_buf.set(target_isaac_order)
            elif args.dry_run and step % 50 == 0:
                print(f"[DRY_RUN] step={step} target_joint_pos(isaac order)={['%.3f' % t for t in target_isaac_order]}")

            if live_plot_writer is not None:
                lean, pitch = quat_to_pitch_lean(snap.quat_wxyz)
                log_live_plot(
                    live_plot_writer,
                    step,
                    cmd,
                    snap.sportmode_velocity,
                    snap.gyro[2],
                    pitch,
                    lean,
                    DEFAULT_HEIGHT_M,
                )

            step += 1
            elapsed = time.perf_counter() - tick_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif step % 50 == 0:
                print(f"[WARN] control loop overran by {-sleep_time * 1000:.1f}ms at step {step}")
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received -- stopping control loop.")
    finally:
        if publish_thread is not None:
            publish_thread.Wait(1.0)
        if low_cmd_publisher is not None and crc is not None:
            print("[INFO] Sending damping command before exit (kp=0, kd small, tau=0 on all joints).")
            damp_cmd = init_low_cmd(cfg)
            for sdk_i in range(12):
                damp_cmd.motor_cmd[sdk_i].q = 0.0
                damp_cmd.motor_cmd[sdk_i].kp = 0.0
                damp_cmd.motor_cmd[sdk_i].kd = 3.0
            damp_cmd.crc = crc.Crc(damp_cmd)
            low_cmd_publisher.Write(damp_cmd)


if __name__ == "__main__":
    main()
