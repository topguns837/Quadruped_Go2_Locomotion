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
import contextlib
import json
import math
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty
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
    """Returns (sdk_to_isaac, isaac_to_sdk), both indexed by ISAAC slot i, both giving the SDK motor index of the
    joint sitting in Isaac slot i: sdk_to_isaac[i] is the motor whose reading goes into Isaac slot i (gather),
    isaac_to_sdk[i] is the motor that Isaac slot i's command is written to (scatter). They are numerically the
    same list -- "which motor is this joint" does not depend on direction -- and are built from matching joint
    *names*, never by assuming a numeric relationship between the two orders.

    HISTORY (real bug, found on hardware day 1): isaac_to_sdk used to be built as the mathematical INVERSE
    permutation (indexed by SDK slot, giving an Isaac index) while publish_low_cmd iterated it as if indexed
    by Isaac slot. Both lists were valid permutations, so a permutation check passed, but 8 of 12 joints were
    sent the WRONG joint's target (e.g. the FL_hip motor received RL_thigh's ~57 deg target). The explicit
    cross-check at the bottom now fails loudly if the lists ever stop naming the right motor."""
    if set(cfg.isaac_joint_order) != set(cfg.sdk_joint_order):
        raise ValueError(
            "isaac_joint_order and sdk_joint_order in the config don't contain the same joint names -- "
            f"isaac has {set(cfg.isaac_joint_order) - set(cfg.sdk_joint_order)} extra, "
            f"sdk has {set(cfg.sdk_joint_order) - set(cfg.isaac_joint_order)} extra."
        )
    sdk_index_of = {name: i for i, name in enumerate(cfg.sdk_joint_order)}
    sdk_to_isaac = [sdk_index_of[name] for name in cfg.isaac_joint_order]
    isaac_to_sdk = [sdk_index_of[name] for name in cfg.isaac_joint_order]
    for i, name in enumerate(cfg.isaac_joint_order):
        if cfg.sdk_joint_order[sdk_to_isaac[i]] != name or cfg.sdk_joint_order[isaac_to_sdk[i]] != name:
            raise ValueError(
                f"Joint index map is wrong for Isaac slot {i} ({name!r}): it points at SDK motor "
                f"{isaac_to_sdk[i]} ({cfg.sdk_joint_order[isaac_to_sdk[i]]!r}). Refusing to continue -- "
                "commands would be sent to the wrong motor."
            )
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
        self.tau_est = [0.0] * num_joints  # SDK order, Nm -- the robot's own estimate of torque it is applying
        self.wireless_remote = bytes(40)  # raw handheld-controller state carried in LowState (see decode_controller_buttons)
        self.quat_wxyz = (1.0, 0.0, 0.0, 0.0)  # IMU orientation, body-to-world
        self.gyro = (0.0, 0.0, 0.0)  # body-frame angular velocity, rad/s
        # NOTE: same open question as DEFAULT_HEIGHT_M above -- SportModeState's real-hardware availability
        # after ReleaseMode() is unconfirmed (see deploy.md Stage 3), so this may also go stale/frozen once
        # the robot is actually in the state the policy needs it in. Left wired up (unlike height, which
        # was simplified to a constant) since there's no evidence yet it specifically fails, but flagged
        # here so it isn't silently trusted if it turns out not to work either.
        self.sportmode_velocity = (0.0, 0.0, 0.0)  # body-frame linear velocity estimate, m/s
        self.ready = False  # False until at least one LowState message has been received
        self.last_update = None  # time.monotonic() of the most recent LowState message

    def age_s(self) -> float:
        """Seconds since the last LowState message (inf if none yet). Used by the link-loss guard."""
        if self.last_update is None:
            return float("inf")
        return time.monotonic() - self.last_update

    def update_from_low_state(self, msg) -> None:
        with self._lock:
            self.last_update = time.monotonic()
            self.joint_pos = [ms.q for ms in msg.motor_state[: len(self.joint_pos)]]
            self.joint_vel = [ms.dq for ms in msg.motor_state[: len(self.joint_vel)]]
            self.tau_est = [ms.tau_est for ms in msg.motor_state[: len(self.tau_est)]]
            self.wireless_remote = bytes(msg.wireless_remote)
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
            snap.tau_est = list(self.tau_est)
            snap.wireless_remote = self.wireless_remote
            snap.quat_wxyz = self.quat_wxyz
            snap.gyro = self.gyro
            snap.sportmode_velocity = self.sportmode_velocity
            snap.ready = self.ready
            snap.last_update = self.last_update
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
# Mode release -- REQUIRED prerequisite. Without this, the robot's onboard sport-mode controller is still
# active and will fight/ignore LowCmd -- sending well-formed LowCmd before this step is meaningless.
#
# Follows Unitree's OFFICIAL C++ Go2 example (unitree_sdk2/example/go2/go2_stand_example.cpp): loop
# ReleaseMode() until CheckMode() reports no active mode, 10 s RPC timeout, 5 s between attempts -- and NO
# StandDown(). That example warns "Make sure the robot is hung up or lying on the ground." The Python
# example (which an earlier version of this function copied) additionally calls StandDown() every
# iteration; on a hoisted robot that makes the sport controller drive the legs to a folded pose in the air,
# the prime suspect for the violent motion seen on first hardware use. StandDown() is therefore opt-in
# (`stand_down=True`, ground only). After ReleaseMode the motors go limp momentarily (confirmed by the
# community write-ups), so the robot must be prone on the ground or already hanging -- never standing.
# ---------------------------------------------------------------------------------------------------------


def release_motion_control(
    timeout_s: float = 10.0,
    retry_sleep_s: float = 5.0,
    max_attempts: int = 10,
    stand_down: bool = False,
    before_release=None,
) -> None:
    """`before_release`, if given, is called exactly once, right before the first StandDown/ReleaseMode --
    i.e. only when an onboard mode is actually active. Used for the operator checklist prompt, so a re-run
    against an already-released robot isn't asked to re-confirm something that isn't about to happen."""
    sc = None
    if stand_down:
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
        if before_release is not None:
            before_release()
            before_release = None
        if sc is not None:
            sc.StandDown()
        release_code, _ = msc.ReleaseMode()
        if release_code != 0:
            print(f"[WARN] ReleaseMode() returned code={release_code}.")
        time.sleep(retry_sleep_s)

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
    """kd is fixed here and never changes (0.5 is already a gentle damping-oriented gain, not the mechanism
    of concern). kp is set here only as the "full" value used for the one-off exit damping command (which
    immediately overrides it to 0 anyway) -- during normal operation, kp is NOT read from here at all:
    publish_low_cmd() overwrites it every tick from LowCmdTarget's own dynamic kp (see ramp_to_default_pose
    and LowCmdTarget's kp ramp-up docstring)."""
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
    # motor_cmd is a fixed 20-slot array (covers arm motors even on a legs-only robot) -- confirmed directly
    # against unitree_sdk2py's own reference example (go2_stand_example.py's InitLowCmd), which sets mode on
    # ALL 20 slots, not just the 12 legs. Default-constructed slots 12-19 are left at mode=0 otherwise
    # (confirmed via unitree_go_msg_dds__LowCmd_()'s own defaults) -- added after real-hardware testing
    # showed tau_est reading back ~0 on every leg motor despite large position errors and kp=25, i.e. the
    # robot was not applying torque to ANY commanded LowCmd at all. kp/kd/tau stay 0 here (unlike the 12 leg
    # slots above) so this introduces no new torque demand on any real motor -- only `mode` changes, matching
    # the verified reference exactly.
    for i in range(12, 20):
        low_cmd.motor_cmd[i].mode = _MOTOR_MODE_PMSM
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 0.0
        low_cmd.motor_cmd[i].tau = 0.0
    return low_cmd


class LowCmdTarget:
    """Shared buffer between the 50 Hz policy-inference loop (writer, via `set`/`set_kp`) and the ~500 Hz
    publish thread (reader, via `get`/`get_kp`) -- see the frequency-matching design in the module
    docstring and unitree_sdk2py's own reference example, which uses the same split (a RecurrentThread
    continuously re-publishing whatever the latest target is, decoupled from whatever computes it).

    Holds both the per-tick-varying Isaac-ordered target joint positions AND a dynamic `kp` -- the latter
    added specifically so `kp` can ramp up from a gentle starting value alongside the position ramp
    (see ramp_to_default_pose), instead of snapping to full stiffness the instant the publish thread starts
    (which it used to: `kp` was fixed once in init_low_cmd and never touched again). `kd` is also dynamic,
    but is not ramped: it is switched between a few fixed phases only (zero-gain heartbeat right after
    release, heavy damping while the robot is being hoisted, then the configured `cfg.kd`)."""

    def __init__(self, initial_target_isaac_order: list[float], initial_kp: float, initial_kd: float):
        self._lock = threading.Lock()
        self._target = list(initial_target_isaac_order)
        self._kp = initial_kp
        self._kd = initial_kd

    def set(self, target_isaac_order: list[float]) -> None:
        with self._lock:
            self._target = list(target_isaac_order)

    def set_kp(self, kp: float) -> None:
        with self._lock:
            self._kp = kp

    def set_kd(self, kd: float) -> None:
        with self._lock:
            self._kd = kd

    def get_kd(self) -> float:
        with self._lock:
            return self._kd

    def get(self) -> list[float]:
        with self._lock:
            return list(self._target)

    def get_kp(self) -> float:
        with self._lock:
            return self._kp


def publish_low_cmd(low_cmd, target_buf: LowCmdTarget, isaac_to_sdk: list[int], publisher, crc) -> None:
    """The ~500 Hz RecurrentThread target: reads the latest computed target (position AND kp) and
    republishes it, regardless of whether the 50 Hz policy loop has produced a new one since the last tick
    -- this is what gives the robot's communication watchdog a steady heartbeat independent of inference
    timing. kp is re-applied here every tick (not just once in init_low_cmd) specifically so the gradual
    kp ramp-up in ramp_to_default_pose actually reaches the robot."""
    target_isaac_order = target_buf.get()
    kp = target_buf.get_kp()
    kd = target_buf.get_kd()
    for isaac_i, sdk_i in enumerate(isaac_to_sdk):
        low_cmd.motor_cmd[sdk_i].q = target_isaac_order[isaac_i]
        low_cmd.motor_cmd[sdk_i].kp = kp
        low_cmd.motor_cmd[sdk_i].kd = kd
    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


# ---------------------------------------------------------------------------------------------------------
# Manual step-confirm mode -- opt-in via --steps_per_confirm. Prints every commanded target in degrees and
# blocks until the space bar is pressed before letting it reach target_buf. Applies uniformly to both
# ramp_to_default_pose() and the main policy loop (see each call site) -- the two places this file ever
# decides on a new target. The ~500 Hz publish thread is untouched either way: it keeps re-publishing
# whatever the last CONFIRMED target was while this blocks, so the robot holds position steadily rather
# than going quiet (and possibly tripping its own comms-timeout fault) while waiting on a keypress.
# ---------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def cbreak_terminal():
    """Puts stdin into cbreak mode (single keypresses readable without waiting for Enter) for the duration
    of the `with` block, restoring the original terminal settings afterward even on exception. Deliberately
    `tty.setcbreak`, not `tty.setraw`: cbreak leaves ISIG enabled, so Ctrl+C still raises KeyboardInterrupt
    normally (confirmed this matters -- `setraw` would silently swallow the existing Ctrl+C-based emergency
    stop, which must keep working while this is active, not just when it isn't)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def wait_for_space(prompt: str, poll=None, flush: bool = True) -> None:
    """Prints `prompt`, then blocks until a space bar press. Enters its own cbreak context (nesting inside
    an outer one is fine) so Ctrl+C keeps working. `poll`, if given, is called ~20x/s while waiting so a
    safety check (e.g. the controller e-stop) is still live during the wait. Reads with os.read, not
    sys.stdin.read, so Python's text-layer buffering can't swallow a queued keypress from select(). `flush`
    discards keys pressed BEFORE this prompt appeared -- required for the safety checklists (a mashed
    space bar from an earlier step must not auto-confirm "the robot is hoisted"), skipped for per-step
    gating where pre-queued presses are intentional."""
    print(prompt, flush=True)
    fd = sys.stdin.fileno()
    with cbreak_terminal():
        if flush:
            termios.tcflush(fd, termios.TCIFLUSH)
        while True:
            if poll is not None:
                poll()
            ready, _, _ = select.select([fd], [], [], 0.05)
            if not ready:
                continue
            ch = os.read(fd, 1)
            if ch == b"":
                raise RuntimeError("stdin closed while waiting for a SPACE confirmation.")
            if ch == b" ":
                return


class StepConfirmGate:
    """Blocks every `steps_per_confirm`-th call to `.wait()` until a space bar press, then lets that many
    calls through before blocking again. `steps_per_confirm=1` (the default when this feature is on at all)
    means every single step waits for its own press. `poll` (see wait_for_space) keeps the controller
    e-stop live while blocked."""

    def __init__(self, steps_per_confirm: int):
        self.steps_per_confirm = max(1, steps_per_confirm)
        self._remaining = 0

    def wait(self, poll=None) -> None:
        if self._remaining <= 0:
            wait_for_space(
                f"[CONFIRM] Press SPACE to send the next {self.steps_per_confirm} command(s)...",
                poll=poll,
                flush=False,
            )
            self._remaining = self.steps_per_confirm
        self._remaining -= 1


# ---------------------------------------------------------------------------------------------------------
# Safety monitoring -- controller e-stop and a torque-response watchdog.
#
# After release_motion_control() the robot's own sport service (which normally handles the handheld
# controller's L2+B damping combo) is gone, so the controller can no longer be assumed to stop the robot.
# The controller's button state is still carried in LowState.wireless_remote, which this process already
# subscribes to, so it is polled here and treated like Ctrl+C. Bit layout is from unitree_sdk2_python's
# example/wireless_controller/wireless_controller.py (parse_botton).
#
# The watchdog exists because a robot that silently isn't applying torque (observed on real hardware:
# tau_est ~0 on every joint with 20-50 degrees of position error at kp~25, while the head LED blinked red)
# looks, from this script's side, exactly like a healthy one -- every print is plausible, nothing moves.
# ---------------------------------------------------------------------------------------------------------

_CONTROLLER_BYTE2_BUTTONS = ("R1", "L1", "Start", "Select", "R2", "L2", "F1", "F3")
_CONTROLLER_BYTE3_BUTTONS = ("A", "B", "X", "Y", "Up", "Right", "Down", "Left")


class EmergencyStop(Exception):
    """Raised when the handheld controller's L2+B e-stop combo is detected."""


class TorqueResponseFault(RuntimeError):
    """Raised when the robot demonstrably is not applying torque to our commands."""


class TrackingFault(RuntimeError):
    """Raised when commands and reality have diverged dangerously: persistent large position error, torque
    near the motors' limit, or a ramp that could not reach its target within the tether."""


def tether_limit(
    target_isaac_order: list[float], actual_isaac_order: list[float], max_dev_rad: float, joint_names: list[str]
) -> tuple[list[float], list[str]]:
    """Limits every commanded joint target to within `max_dev_rad` of that joint's ACTUAL position, returning
    the limited target and the names of joints that were limited.

    Why (real hardware, 2026-10-02): the policy loop's targets ran 70-97 deg away from where the legs actually
    were (robot partly on a winch, feet loaded unevenly), and at kp=25 that gap is torque = kp * error = 32 Nm,
    above the ~23.5 Nm effort limit, after which the robot cut torque and went to a red-LED fault. clamp_step
    only limits change per tick relative to the previous TARGET, so it cannot stop a target that has drifted
    far from reality. This bounds the PD torque at kp * max_dev (e.g. 25 * 15 deg = 6.5 Nm) no matter what the
    policy or ramp asks for -- the software effort limit the unused `effort_limit` config value was meant to be."""
    limited, limited_names = [], []
    for name, tgt, act in zip(joint_names, target_isaac_order, actual_isaac_order):
        delta = tgt - act
        if delta > max_dev_rad:
            limited.append(act + max_dev_rad)
            limited_names.append(name)
        elif delta < -max_dev_rad:
            limited.append(act - max_dev_rad)
            limited_names.append(name)
        else:
            limited.append(tgt)
    return limited, limited_names


class LinkLost(Exception):
    """Raised when LowState stops arriving (cable pulled, robot rebooting/powered off).

    Real incident: a deploy script left running kept streaming full-kp LowCmd at 500 Hz through a cable
    disconnect; when the link returned, the robot had rebooted into normal sport mode and the stale stream
    fought it -- violent ~45 Hz chatter. On link loss the script must STOP publishing and exit, never wait
    for the link to come back."""


LINK_LOST_AFTER_S = 0.5  # LowState normally arrives at ~500 Hz; half a second of silence is a dead link


def decode_controller_buttons(wireless_remote: bytes) -> dict[str, bool]:
    if len(wireless_remote) < 4:
        return {}
    byte2, byte3 = wireless_remote[2], wireless_remote[3]
    buttons = {name: bool((byte2 >> bit) & 1) for bit, name in enumerate(_CONTROLLER_BYTE2_BUTTONS)}
    buttons.update({name: bool((byte3 >> bit) & 1) for bit, name in enumerate(_CONTROLLER_BYTE3_BUTTONS)})
    return buttons


def estop_pressed(wireless_remote: bytes) -> bool:
    buttons = decode_controller_buttons(wireless_remote)
    return buttons.get("L2", False) and buttons.get("B", False)


class TorqueResponseWatchdog:
    """Raises TorqueResponseFault after `consecutive_steps` consecutive updates where stiffness is at least
    `kp_fraction_threshold` of full, the commanded-vs-actual error exceeds `err_threshold_rad`, yet the
    robot reports less than `tau_threshold_nm` of torque on every joint (a stiff PD loop with that much error
    should be producing several Nm). Any update that doesn't meet all three resets the count."""

    def __init__(
        self,
        consecutive_steps: int,
        kp_fraction_threshold: float = 0.5,
        err_threshold_rad: float = math.radians(10.0),
        tau_threshold_nm: float = 0.5,
    ):
        self.consecutive_steps = consecutive_steps
        self.kp_fraction_threshold = kp_fraction_threshold
        self.err_threshold_rad = err_threshold_rad
        self.tau_threshold_nm = tau_threshold_nm
        self._count = 0

    def update(self, kp_fraction: float, max_err_rad: float, max_abs_tau_nm: float) -> None:
        if (
            kp_fraction >= self.kp_fraction_threshold
            and max_err_rad > self.err_threshold_rad
            and max_abs_tau_nm < self.tau_threshold_nm
        ):
            self._count += 1
            if self._count >= self.consecutive_steps:
                raise TorqueResponseFault(
                    f"Robot is not applying torque: for {self._count} consecutive steps at "
                    f">={self.kp_fraction_threshold:.0%} of full kp, joint error up to "
                    f"{math.degrees(max_err_rad):.1f} deg but max |tau_est| only {max_abs_tau_nm:.2f} Nm. "
                    "Check the head LED and the Go app for a fault."
                )
        else:
            self._count = 0


class SafetyMonitor:
    """Per-tick safety checks bundled for ramp_to_default_pose() and the main loop: controller e-stop,
    tracking telemetry (max position error / max |tau_est|), the torque-response watchdog, and optional
    TensorBoard scalars. `poll()` is the e-stop-only variant used while blocked waiting for a keypress."""

    def __init__(self, state: "LatestRobotState", sdk_to_isaac: list[int], watchdog: TorqueResponseWatchdog,
                 joint_names: list[str], max_dev_rad: float, abort_err_rad: float, abort_steps: int,
                 tau_abort_nm: float, writer=None):
        self.state = state
        self.sdk_to_isaac = sdk_to_isaac
        self.watchdog = watchdog
        self.joint_names = joint_names
        self.max_dev_rad = max_dev_rad
        self.abort_err_rad = abort_err_rad
        self.abort_steps = abort_steps
        self.tau_abort_nm = tau_abort_nm
        self.writer = writer
        self._tick = 0
        self._big_err_count = 0

    def actual_isaac_order(self) -> list[float]:
        snap = self.state.snapshot()
        return [snap.joint_pos[sdk_i] for sdk_i in self.sdk_to_isaac]

    def limit_target(self, target_isaac_order: list[float]) -> tuple[list[float], list[str]]:
        """Tether: see tether_limit. Applied to every target before it reaches the publish buffer."""
        return tether_limit(target_isaac_order, self.actual_isaac_order(), self.max_dev_rad, self.joint_names)

    def _raise_if_unsafe(self, snap: "LatestRobotState") -> None:
        age = snap.age_s()
        if age > LINK_LOST_AFTER_S:
            raise LinkLost(
                f"No LowState for {age:.1f} s (cable pulled, or robot rebooting / powered off). Stopping and "
                "NOT resuming: a resumed command stream could fight a robot that came back in sport mode."
            )
        if estop_pressed(snap.wireless_remote):
            raise EmergencyStop("Controller e-stop (L2+B) pressed.")

    def poll(self) -> None:
        self._raise_if_unsafe(self.state.snapshot())

    def check(self, held_target_isaac_order: list[float], kp_fraction: float) -> tuple[float, float]:
        """Returns (max_abs_position_error_rad, max_abs_tau_est_nm) and raises LinkLost / EmergencyStop /
        TorqueResponseFault when warranted. `held_target_isaac_order` is what is currently being published."""
        snap = self.state.snapshot()
        self._raise_if_unsafe(snap)
        actual = [snap.joint_pos[sdk_i] for sdk_i in self.sdk_to_isaac]
        max_err = max(abs(t - a) for t, a in zip(held_target_isaac_order, actual))
        max_tau = max(abs(t) for t in snap.tau_est)
        self._tick += 1
        if self.writer is not None:
            self.writer.add_scalar("Debug/max_pos_error_deg", math.degrees(max_err), self._tick)
            self.writer.add_scalar("Debug/max_abs_tau_est_nm", max_tau, self._tick)
        if max_tau >= self.tau_abort_nm:
            raise TrackingFault(
                f"Joint torque {max_tau:.1f} Nm is at/over {self.tau_abort_nm:.1f} Nm (90% of effort_limit). "
                "Stopping before the robot's own protection trips."
            )
        if max_err > self.abort_err_rad:
            self._big_err_count += 1
            if self._big_err_count >= self.abort_steps:
                raise TrackingFault(
                    f"Commanded vs actual joint error stayed above {math.degrees(self.abort_err_rad):.0f} deg for "
                    f"{self._big_err_count} consecutive steps (now {math.degrees(max_err):.0f} deg): the robot is "
                    "not following its targets (contact, winch, or fault)."
                )
        else:
            self._big_err_count = 0
        self.watchdog.update(kp_fraction, max_err, max_tau)
        return max_err, max_tau


def format_degrees(isaac_order_values: list[float], joint_names: list[str]) -> str:
    return "  ".join(f"{name}={math.degrees(v):+.1f}°" for name, v in zip(joint_names, isaac_order_values))


def clamp_step(
    current_isaac_order: list[float], target_isaac_order: list[float], max_delta_rad: float, joint_names: list[str]
) -> tuple[list[float], list[str]]:
    """Clamps the per-joint change from `current_isaac_order` (whatever target_buf currently holds) to
    `target_isaac_order` (what the policy just computed) to at most `max_delta_rad`, returning the clamped
    target and the names of any joints that were actually clipped.

    Added after a real-hardware incident where the ramp-to-default-pose handoff into the policy loop
    produced a single, unsmoothed ~5-7 degree jump across multiple legs simultaneously: the ramp only
    interpolates the transition TO default_joint_pos -- the first policy inference afterward runs with
    last_action still all-zeros (a genuine reset-like state) and its raw action can legitimately be large,
    but was being written straight to target_buf in one `set()` call with no interpolation, the same as any
    later tick. Applying this clamp on every tick (not just the first) is a general safety net against any
    anomalously large policy output, not a special case for startup alone."""
    clamped = []
    clipped_names = []
    for name, cur, tgt in zip(joint_names, current_isaac_order, target_isaac_order):
        delta = tgt - cur
        if delta > max_delta_rad:
            clamped.append(cur + max_delta_rad)
            clipped_names.append(name)
        elif delta < -max_delta_rad:
            clamped.append(cur - max_delta_rad)
            clipped_names.append(name)
        else:
            clamped.append(tgt)
    return clamped, clipped_names


def ramp_to_default_pose(
    target_buf: LowCmdTarget,
    start_isaac_order: list[float],
    default_isaac_order: list[float],
    duration_s: float,
    control_dt: float,
    isaac_joint_order: list[str],
    start_kp: float,
    full_kp: float,
    gate: StepConfirmGate | None = None,
    monitor: SafetyMonitor | None = None,
) -> None:
    """Required safety step, previously MISSING entirely: linearly interpolates the published target from
    the robot's actual current joint positions to `default_isaac_order` over `duration_s`, instead of
    jumping straight there. Without this, `LowCmdTarget` starts life already set to `default_isaac_order`
    the instant the ~500 Hz publish thread begins (see main()) -- at full `kp` stiffness, with zero regard
    for how far that is from wherever the robot's joints actually are the moment onboard control releases.
    Confirmed directly to cause a sudden, rough snap-to-pose on first real-hardware use (not hypothetical):
    the robot's actual pose right after release can differ from the trained default by tens of degrees on
    some joints (observed up to ~45 degrees during this session's own rig-hung preflight checks), and a
    PD controller commanded to close that gap instantaneously does exactly what you'd expect. The publish
    thread is already running throughout this call (started before this is invoked) -- this only ever
    writes progressively-interpolated values into `target_buf`, same mechanism the policy loop uses after.

    `kp` is ALSO ramped here, from `start_kp` to `full_kp`, synchronized with the same position-ramp alpha
    (not on its own separate schedule) -- added after a real-hardware incident where, even with the position
    ramp already in place, a jerk still happened before the ramp's own ticks ever got reported as sent (the
    *first* gate-confirmed step hadn't even been approved yet). The remaining suspect: `kp` previously
    jumped straight to full stiffness the instant the publish thread started holding "current position" --
    on a joint that may not have been fully settled yet (e.g. residual motion from release_motion_control's
    StandDown(), compounded by a robot not resting normally on the ground), snapping straight to full
    position-holding stiffness can itself produce a jerk, even though the *position* target was already
    correct. Keeping kp and position synchronized on the same alpha means early in the ramp (small position
    error, since start and target are close) stiffness is also at its gentlest, and both tighten together
    as the ramp progresses -- not two independently-timed transitions that could fight each other.

    `gate`, if given (see StepConfirmGate), prints each interpolated step (position and kp) in degrees/Nm
    and blocks for a space bar press before it reaches target_buf -- the ramp becomes operator-paced rather
    than real-time.
    """
    num_steps = max(1, int(round(duration_s / control_dt)))
    print(f"[INFO] Moving smoothly to default pose over {duration_s:.1f}s ({num_steps} steps), kp ramping "
          f"{start_kp:.1f} -> {full_kp:.1f} alongside it -- do not expect the robot to move yet if it's "
          f"already close to default_joint_pos.")
    for step in range(1, num_steps + 1):
        tick_start = time.perf_counter()
        alpha = step / num_steps
        interpolated = [
            start + (default - start) * alpha
            for start, default in zip(start_isaac_order, default_isaac_order)
        ]
        interpolated_kp = start_kp + (full_kp - start_kp) * alpha
        if monitor is not None:
            interpolated, _ = monitor.limit_target(interpolated)  # never lead the actual joints by > tether
        telemetry = ""
        if monitor is not None:
            # Measures the response to the PREVIOUS step's target (still what target_buf holds): in gated
            # mode the robot has had arbitrarily long to respond, in free-run mode one control tick.
            max_err, max_tau = monitor.check(target_buf.get(), kp_fraction=target_buf.get_kp() / full_kp)
            telemetry = f"  | max|err|={math.degrees(max_err):.1f}deg max|tau_est|={max_tau:.2f}Nm"
        if gate is not None:
            print(f"[RAMP {step}/{num_steps}] kp={interpolated_kp:.1f}  "
                  f"{format_degrees(interpolated, isaac_joint_order)}{telemetry}\n")
            gate.wait(poll=monitor.poll if monitor is not None else None)
        target_buf.set_kp(interpolated_kp)
        target_buf.set(interpolated)
        elapsed = time.perf_counter() - tick_start
        sleep_time = control_dt - elapsed
        if sleep_time > 0 and gate is None:
            time.sleep(sleep_time)
    target_buf.set_kp(full_kp)
    final_target = list(default_isaac_order)  # exact, not just alpha=1.0's floating-point approximation
    if monitor is not None:
        final_target, limited = monitor.limit_target(final_target)
        if limited:
            raise TrackingFault(
                f"Ramp finished but {len(limited)} joint(s) are still more than the tether "
                f"({math.degrees(monitor.max_dev_rad):.0f} deg) from default: {', '.join(limited)}. The robot did "
                "not follow the ramp (contact, winch, or fault) -- refusing to hand off to the policy."
            )
    target_buf.set(final_target)
    print("[INFO] Default pose reached, full kp reached. Handing off to the policy control loop.")


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


ZERO_GAIN_HEARTBEAT_S = 1.0  # official examples publish zero-gain frames for ~1 s after release before doing anything
HOLD_DAMPING_KD = 3.0  # same damping gain as the exit damping command; legs hang limp but heavily damped

PRE_RELEASE_CHECKLIST = """
[CHECKLIST] The robot's onboard controller is about to be RELEASED (motors go limp for a moment).
  [ ] Robot is PRONE on the ground (controller L2+A twice) -- not standing, not hoisted.
  [ ] Hoist rope is slack (tether only). Area around the legs is clear.
  [ ] Head LED is steady green (not red).
  [ ] Controller in hand, power button within reach. Ctrl+C here to abort.
Press SPACE to release the onboard controller..."""

HOIST_PAUSE_PROMPT = """
[PAUSE] Onboard controller released. Zero-gain heartbeat done; legs are now held DAMPED (kp=0, kd=3).
  - Check the head LED now. Steady green = good, red = STOP.
  - To hoist: winch up slowly until the feet are a few cm clear. The legs hang limp.
  - To TEST THE CONTROLLER E-STOP: press L2+B now. This script should print [ESTOP] and exit. If nothing
    happens, the controller stop does NOT work after release -- use Ctrl+C / the power button only.
  - Pressed controller buttons are echoed below as they are seen.
Press SPACE when the robot is hoisted (or on the ground, if skipping the hoist) to start the startup ramp..."""


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
    parser.add_argument(
        "--startup_ramp_s", type=float, default=10.0,
        help="Seconds to smoothly interpolate from the robot's actual current joint positions AND from a "
        "gentle starting kp (see --startup_kp_fraction) to default_joint_pos/full kp, instead of jumping "
        "there instantly. Required safety step -- see ramp_to_default_pose(). Increase for extra caution "
        "(e.g. if the robot's starting pose is far from default_joint_pos); do not set to 0.",
    )
    parser.add_argument(
        "--startup_kp_fraction", type=float, default=0.1,
        help="kp during the startup ramp starts at this fraction of the configured kp (see "
        "deploy/configs/go2_locomotion.yaml) and climbs to the full value alongside the position ramp, "
        "instead of the full kp applying instantly the moment the publish thread starts holding 'current "
        "position'. Added after a real-hardware incident where a jerk happened before the ramp's position "
        "interpolation had even sent its first step -- the remaining suspect was kp snapping to full "
        "stiffness on a possibly not-yet-settled joint. 0.1 = 10%% of full kp to start; raise for more "
        "caution, must be > 0 (some holding stiffness from the very first publish is still needed).",
    )
    parser.add_argument(
        "--max_joint_step_deg", type=float, default=3.0,
        help="Per-tick slew-rate limit: no joint's commanded target may move more than this many degrees "
        "from whatever target_buf currently holds, in either direction, applied every control tick (not "
        "just at startup). Added after a real-hardware incident where the ramp-to-default-pose handoff into "
        "the policy loop sent an unsmoothed multi-degree jump in one tick (the ramp only smooths the "
        "transition INTO default_joint_pos; the first policy action afterward, and any later anomalous "
        "action, was previously written straight through with no limit). 3.0 is conservative for this "
        "testing phase -- raise it once continuous full-speed gait is validated and this is confirmed to "
        "bind during normal walking, not just anomalies.",
    )
    parser.add_argument(
        "--steps_per_confirm", type=int, nargs="?", const=1, default=None,
        help="Manual step-confirm mode: print every commanded target in degrees and block for a SPACE bar "
        "press before it's sent, for both the startup ramp and the main policy loop -- control is no longer "
        "real-time while this is set. Pass the bare flag for the default of 1 (confirm every single "
        "command), or a number to let that many commands through per press (e.g. --steps_per_confirm 10). "
        "Omit entirely to run freely as before (the default).",
    )
    parser.add_argument(
        "--max_target_dev_deg", type=float, default=15.0,
        help="Tether: no commanded joint target may be further than this from that joint's ACTUAL position. "
        "Bounds PD torque at kp*dev (25 * 15 deg = 6.5 Nm). Added after a real run where policy targets drifted "
        "70-97 deg from the legs, torque hit 32 Nm (> effort limit) and the robot cut torque with a red LED.",
    )
    parser.add_argument(
        "--abort_err_deg", type=float, default=30.0,
        help="Stop (damping) if commanded-vs-actual error stays above this for 3 consecutive steps.",
    )
    parser.add_argument(
        "--stand_down", action="store_true", default=False,
        help="Also call SportClient.StandDown() while releasing the onboard controller (the Python SDK "
        "example does; the official C++ example does not). Only for a robot STANDING ON THE GROUND that you "
        "want laid down first -- never use it while hoisted (the sport controller will drive the legs to a "
        "folded pose in the air). Default off: lay the robot prone yourself (controller L2+A twice) instead.",
    )
    parser.add_argument(
        "--no_pause_for_hoist", action="store_true", default=False,
        help="Skip the pause after release (where the legs are held damped, kp=0/kd=3, so you can hoist the "
        "robot) and go straight to the startup ramp. Default: pause.",
    )
    parser.add_argument(
        "--skip_release", action="store_true", default=False,
        help="Testing seam: skip the pre-release checklist and release_motion_control(). Only accepted with "
        "--network_interface lo (deploy/fake_robot.py); refused otherwise, since publishing LowCmd to a real "
        "robot whose sport controller is still active is unsafe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.network_interface)

    if args.skip_release and cfg.network_interface != "lo":
        raise ValueError(
            "--skip_release is only accepted with --network_interface lo (the fake robot). Refusing to "
            f"publish LowCmd to '{cfg.network_interface}' without releasing the onboard controller first."
        )

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
    startup_kp = None
    if _UNITREE_SDK_AVAILABLE and not args.dry_run:
        ChannelFactoryInitialize(0, cfg.network_interface)
        crc = CRC()

        # Subscribe BEFORE releasing (same order as Unitree's official C++ example's Init()), so state --
        # including the controller buttons and tau_est -- is already flowing while the release happens.
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

        # MUST happen before LowCmd means anything -- see release_motion_control's comment above. The
        # operator checklist prompt only appears if a release is actually about to happen.
        if args.skip_release:
            print("[WARN] --skip_release: skipping the release checklist and release_motion_control() "
                  "(loopback/fake robot only).")
        else:
            release_motion_control(
                stand_down=args.stand_down,
                before_release=lambda: wait_for_space(PRE_RELEASE_CHECKLIST, flush=True),
            )

        # Two-rate publish architecture (see module docstring): the 50 Hz loop below only ever updates
        # target_buf; this dedicated ~500 Hz RecurrentThread is what actually calls Write(), matching
        # unitree_sdk2py's own reference example (RecurrentThread(interval=0.002, target=LowCmdWrite)).
        #
        # Publishing starts with ZERO gains (kp=kd=0 -> no torque), like Unitree's official examples, whose
        # first ~1 s after release is zero-gain frames. The real starting pose is captured later, after the
        # (optional) hoist pause, so the ramp starts from where the robot actually is by then -- hoisting
        # changes the joint positions by tens of degrees, so a pose captured here would be stale.
        # kp then starts gentle (see --startup_kp_fraction) and ramps alongside the position ramp, not
        # instantly -- see LowCmdTarget's and ramp_to_default_pose's docstrings for why.
        startup_kp = cfg.kp * args.startup_kp_fraction
        initial_snap = state.snapshot()
        hold_isaac_order = [initial_snap.joint_pos[sdk_i] for sdk_i in sdk_to_isaac]

        low_cmd = init_low_cmd(cfg)
        target_buf = LowCmdTarget(hold_isaac_order, initial_kp=0.0, initial_kd=0.0)
        publish_thread = RecurrentThread(
            interval=0.002,
            target=publish_low_cmd,
            name="lowcmd_publish",
            args=(low_cmd, target_buf, isaac_to_sdk, low_cmd_publisher, crc),
        )
        publish_thread.Start()
        print("[INFO] LowCmd publish thread started at 500 Hz with ZERO gains (kp=kd=0 -> no torque).")
    else:
        print("[INFO] --dry_run: skipping DDS init, using zeroed/default robot state throughout.")
        state.joint_pos = list(default_joint_pos_sdk_order)
        state.ready = True

    last_action = [0.0] * 12
    step = 0
    control_dt = cfg.control_dt

    gate = StepConfirmGate(args.steps_per_confirm) if args.steps_per_confirm is not None else None
    if gate is not None:
        print(f"[INFO] Manual step-confirm mode: every target printed in degrees, {gate.steps_per_confirm} "
              f"command(s) sent per SPACE press. Not real-time while this is on.")

    max_joint_step_rad = math.radians(args.max_joint_step_deg)
    print(f"[INFO] Per-tick slew-rate limit: max {args.max_joint_step_deg:.1f} deg/joint/tick.")

    # Gated runs count steps per keypress (each one a deliberate human action), free-running ones count
    # control ticks, so the same "no torque response" evidence needs far fewer gated steps (5) than ticks (25
    # = 0.5 s at 50 Hz).
    monitor = None
    if target_buf is not None:
        monitor = SafetyMonitor(
            state, sdk_to_isaac, TorqueResponseWatchdog(consecutive_steps=5 if gate is not None else 25),
            joint_names=cfg.isaac_joint_order,
            max_dev_rad=math.radians(args.max_target_dev_deg),
            abort_err_rad=math.radians(args.abort_err_deg),
            abort_steps=3,
            tau_abort_nm=0.9 * cfg.effort_limit,
            writer=live_plot_writer,
        )
        print(f"[INFO] Tether: targets limited to {args.max_target_dev_deg:.0f} deg of actual "
              f"(max PD torque ~ kp*dev = {cfg.kp * math.radians(args.max_target_dev_deg):.1f} Nm); abort if error "
              f"> {args.abort_err_deg:.0f} deg for 3 steps or torque >= {0.9 * cfg.effort_limit:.1f} Nm.")

    print(f"[INFO] Entering control loop at {1.0 / control_dt:.0f} Hz (dry_run={args.dry_run}). Ctrl+C to stop.")
    with cbreak_terminal() if gate is not None else contextlib.nullcontext():
        try:
            # Moved inside this try/finally (was previously called before it) so Ctrl+C during the ramp --
            # now a multi-second operation, potentially much longer still when gated on keypresses -- is
            # caught by the same damping-command safety net as the main loop below, not left to crash past
            # it uncaught. target_buf is None only in --dry_run, which never ramps (nothing to interpolate
            # toward on fake data).
            if target_buf is not None:
                # 1) Zero-gain heartbeat (no torque), with the controller e-stop already live.
                heartbeat_end = time.monotonic() + ZERO_GAIN_HEARTBEAT_S
                while time.monotonic() < heartbeat_end:
                    monitor.poll()
                    time.sleep(0.02)

                # 2) Hold damped and pause so the robot can be hoisted while released (no sport controller
                #    exists any more to react to being lifted). Controller buttons are echoed so the e-stop
                #    path can be tested before anything moves.
                if not args.no_pause_for_hoist:
                    target_buf.set_kd(HOLD_DAMPING_KD)
                    seen_buttons: set[str] = set()

                    def _pause_poll() -> None:
                        nonlocal seen_buttons
                        pressed = {
                            n for n, v in decode_controller_buttons(state.snapshot().wireless_remote).items() if v
                        }
                        if pressed != seen_buttons:
                            if pressed:
                                print(f"[CONTROLLER] pressed: {sorted(pressed)}", flush=True)
                            seen_buttons = pressed
                        monitor.poll()

                    wait_for_space(HOIST_PAUSE_PROMPT, poll=_pause_poll, flush=True)

                # 3) Start the ramp from where the robot ACTUALLY is now (hoisting moves the legs), with
                #    target == actual so there is no error at the moment stiffness switches on.
                start_snap = state.snapshot()
                start_isaac_order = [start_snap.joint_pos[sdk_i] for sdk_i in sdk_to_isaac]
                target_buf.set(start_isaac_order)
                target_buf.set_kd(cfg.kd)
                target_buf.set_kp(startup_kp)
                print(f"[INFO] Holding actual pose at gentle kp={startup_kp:.1f}, kd={cfg.kd:.1f} "
                      f"(full kp={cfg.kp:.1f}).")

                ramp_to_default_pose(
                    target_buf, start_isaac_order, default_joint_pos_isaac_order, args.startup_ramp_s,
                    cfg.control_dt, cfg.isaac_joint_order, start_kp=startup_kp, full_kp=cfg.kp, gate=gate,
                    monitor=monitor,
                )

            while True:
                tick_start = time.perf_counter()
                snap = state.snapshot()
                cmd = manual_cmd.get()

                obs = build_observation(cfg, snap, sdk_to_isaac, default_joint_pos_isaac_order, cmd, last_action)
                with torch.inference_mode():
                    action = policy(obs)[0].tolist()

                # Isaac-ordered target joint positions. PD gains/mode/head are fixed at init (init_low_cmd);
                # only this target changes per policy tick. Actual publishing happens on the separate ~500 Hz
                # thread started above -- this loop only updates the shared buffer at the 50 Hz policy rate.
                target_isaac_order = [
                    default_joint_pos_isaac_order[i] + cfg.action_scale * action[i] for i in range(12)
                ]

                if target_buf is not None:
                    clamped_target, clipped_names = clamp_step(
                        target_buf.get(), target_isaac_order, max_joint_step_rad, cfg.isaac_joint_order
                    )
                    if clipped_names:
                        print(f"[WARN] step {step}: slew-rate limit clipped {len(clipped_names)} joint(s) "
                              f"to {args.max_joint_step_deg:.1f} deg/tick: {', '.join(clipped_names)}")
                    max_err, max_tau = monitor.check(target_buf.get(), kp_fraction=1.0)
                    clamped_target, tethered = monitor.limit_target(clamped_target)
                    if tethered:
                        print(f"[WARN] step {step}: tether limited {len(tethered)} joint(s) to within "
                              f"{args.max_target_dev_deg:.0f} deg of actual: {', '.join(tethered)}")
                    if gate is not None:
                        print(f"[STEP {step}] {format_degrees(clamped_target, cfg.isaac_joint_order)}  "
                              f"| max|err|={math.degrees(max_err):.1f}deg max|tau_est|={max_tau:.2f}Nm\n")
                        gate.wait(poll=monitor.poll)
                    target_buf.set(clamped_target)
                    # last_action MUST reflect what was actually sent (clamped_target), not the policy's raw
                    # wish (action) -- target_isaac_order is recomputed fresh from default_joint_pos every
                    # tick (not integrated from the previous target), so if last_action lies about what
                    # really happened, the policy's own feedback observation drifts further from reality
                    # every tick with nothing to ground it. Confirmed directly on real hardware: once the
                    # slew-rate clamp started clipping, commanded targets drifted by a constant +/-3 deg/tick
                    # indefinitely with no sign of settling -- a feedback runaway, not normal policy behavior.
                    last_action = [
                        (clamped_target[i] - default_joint_pos_isaac_order[i]) / cfg.action_scale
                        for i in range(12)
                    ]
                else:
                    last_action = action
                    if args.dry_run and step % 50 == 0:
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
                # When gated, gate.wait() (just above) already blocked far longer than control_dt for the
                # keypress -- the real-time pacing below and its overrun warning are meaningless noise in
                # that mode (would otherwise fire on literally every gated tick) and skipped accordingly.
                if gate is None:
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif step % 50 == 0:
                        print(f"[WARN] control loop overran by {-sleep_time * 1000:.1f}ms at step {step}")
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C received -- stopping control loop.")
        except EmergencyStop as e:
            print(f"\n[ESTOP] {e} -- stopping control loop.")
        except (TorqueResponseFault, TrackingFault) as e:
            print(f"\n[FAULT] {e}")
        except LinkLost as e:
            print(f"\n[LINK] {e}")
        finally:
            # RecurrentThread.Wait() sets the thread's quit flag before joining (confirmed in the SDK source),
            # so the 500 Hz publisher is stopped here and cannot overwrite the damping frames below.
            if publish_thread is not None:
                publish_thread.Wait(1.0)
            if low_cmd_publisher is not None and crc is not None:
                print("[INFO] Sending damping command before exit (kp=0, kd=3, tau=0 on all joints).")
                damp_cmd = init_low_cmd(cfg)
                for sdk_i in range(12):
                    damp_cmd.motor_cmd[sdk_i].q = 0.0
                    damp_cmd.motor_cmd[sdk_i].kp = 0.0
                    damp_cmd.motor_cmd[sdk_i].kd = HOLD_DAMPING_KD
                damp_cmd.crc = crc.Crc(damp_cmd)
                # Repeated for ~0.5 s rather than sent once: a single dropped frame must not leave the robot
                # without its damping command, and what the robot does when LowCmd simply stops is unverified.
                for _ in range(100):
                    low_cmd_publisher.Write(damp_cmd)
                    time.sleep(0.005)


if __name__ == "__main__":
    main()
