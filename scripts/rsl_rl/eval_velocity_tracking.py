# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-vs-MuJoCo velocity/pitch/lean-tracking comparison: one checkpoint, one short scripted command set, both sims.

Runs a short, fixed set of commanded [lin_vel_x, lin_vel_y, ang_vel_z, pitch, lean] segments (the same values
as `sim2sim/scenarios/basic.yaml`'s velocity and pitch/lean rows, minus its stand_end repeat) through the SAME
policy weights on Isaac Lab (the training engine) and MuJoCo (`sim2sim/sim2sim_mujoco.py`'s engine, run
in-process as a library, not a subprocess), and prints one RMS-tracking-error table comparing the two.

Pitch and lean error are measured on EVERY segment, not only the dedicated pitch_up/pitch_down/lean_left/
lean_right ones -- unwanted pitch/lean drift while only a velocity is commanded (e.g. nosing down under hard
acceleration) is itself a real stability signal, not just tracking of a deliberate pitch/lean command.

A single process, because the comparison is only meaningful if both sims load literally the same weights:
this script loads the checkpoint through RSL-RL's own `OnPolicyRunner` (exactly as `play.py` does -- correct
network architecture resolved from the run's own config, not hand-reconstructed), exports it once via Isaac
Lab's own `export_policy_as_jit`, and feeds that single exported file to the MuJoCo side. Needs the full Kit
runtime for the Isaac half (`isaaclab -p`), so the normally Kit-free `sim2sim/` module is imported as a
library here rather than run as its own process -- confirmed this session that mujoco/torch/numpy all import
fine under Isaac Sim's own interpreter.

Both sides use nominal (non-randomized, non-delayed) physics for a fair comparison: Isaac via
`QuadrupedLocomotionEnvCfg_PLAY` (which already turns off all of `EventCfg`'s sim-to-real randomization and
the action-latency term, see CLAUDE.md "Real-hardware parity") plus a deterministic reset override (same
trick as `play.py --trace_deterministic_reset`); MuJoCo via `sim2sim_mujoco.ARMATURE`'s nominal default,
`motor_limits="hardware"`, no `--joint_speed_cap`, no action delay -- i.e. the same "as close to the real
robot as either sim gets" baseline, not an adversarial worst case.

Usage (inside the container):
    isaaclab -p scripts/rsl_rl/eval_velocity_tracking.py --headless
    isaaclab -p scripts/rsl_rl/eval_velocity_tracking.py --headless --checkpoint path/to/model_N.pt

With no --checkpoint, resolves the latest run/checkpoint the same way play.py's own `--resume` does (RSL-RL's
`get_checkpoint_path` with the default `load_run: '.*'` / `load_checkpoint: 'model_.*.pt'` -- most recently
modified run folder, highest-iteration checkpoint in it), so no separate "find the latest" logic is needed.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Compare Isaac vs MuJoCo velocity tracking for one checkpoint.")
parser.add_argument(
    "--task", type=str, default="Quadruped-Locomotion-Go2-Play", help="Isaac task to load the checkpoint against."
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--mujoco_config", type=str, default="deploy/configs/go2_locomotion.yaml",
    help="deploy/ config the MuJoCo side reads joint order/kp/kd/action_scale from (NOT the policy path -- "
    "this script always points MuJoCo at the same exported checkpoint Isaac just loaded).",
)
parser.add_argument(
    "--real_obs_log", type=str, default=None,
    help="A .npz from deploy/collect_still_observations.py (real, standing-still hardware recording). When "
    "given, prints an additional table comparing the real robot's recorded standing POSE (absolute joint "
    "angles, pitch/lean, reported velocity) against what Isaac and MuJoCo's own 'stand' segment (zero "
    "command) converge to -- i.e. is the real robot's natural resting pose different from what the policy "
    "thinks standing looks like.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import gymnasium as gym

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import quadruped_go2_locomotion.tasks  # noqa: F401

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sim2sim"))
import sim2sim_mujoco as s2s  # noqa: E402

# Same segments and values as sim2sim/scenarios/basic.yaml (minus its stand_end repeat), for direct
# comparability with any earlier basic.yaml-based sim2sim runs.
SEGMENTS = [
    s2s.Segment("stand", 5.0, [0.0, 0.0, 0.0, 0.0, 0.0]),
    s2s.Segment("forward_slow", 5.0, [0.3, 0.0, 0.0, 0.0, 0.0]),
    s2s.Segment("forward_fast", 5.0, [0.8, 0.0, 0.0, 0.0, 0.0]),
    s2s.Segment("backward", 5.0, [-0.5, 0.0, 0.0, 0.0, 0.0]),
    s2s.Segment("lateral", 5.0, [0.0, 0.5, 0.0, 0.0, 0.0]),
    s2s.Segment("turn", 5.0, [0.0, 0.0, 0.8, 0.0, 0.0]),
    s2s.Segment("forward_turn", 5.0, [0.5, 0.0, 0.5, 0.0, 0.0]),
    s2s.Segment("pitch_up", 5.0, [0.0, 0.0, 0.0, -0.2, 0.0]),
    s2s.Segment("pitch_down", 5.0, [0.0, 0.0, 0.0, 0.2, 0.0]),
    s2s.Segment("lean_left", 5.0, [0.0, 0.0, 0.0, 0.0, -0.2]),
    s2s.Segment("lean_right", 5.0, [0.0, 0.0, 0.0, 0.0, 0.2]),
]
AXES = ("lin_x", "lin_y", "ang_z", "pitch", "lean")
SETTLE_S = s2s.SEGMENT_SETTLE_S  # exclude the same post-command-change transient both sims already agree on


def _rms(errors: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")


def run_isaac(env, policy, dt: float) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Steps `env` (already created, already reset) through SEGMENTS, returns ({segment: {axis: rms_error}},
    {segment: mean_joint_pos_isaac_order}) -- the latter is absolute joint angles (radians, NOT relative to
    default), averaged over the same settled window, for comparing standing POSE against real hardware
    (see --real_obs_log), not just velocity-tracking error."""
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_term.enable_manual_override()
    robot = env.unwrapped.scene["robot"]
    obs = env.get_observations()

    results: dict[str, dict[str, float]] = {}
    mean_joint_pos: dict[str, np.ndarray] = {}
    for seg in SEGMENTS:
        cmd_term.set_manual_command(list(seg.command))
        errs = {a: [] for a in AXES}
        joint_pos_samples = []
        num_steps = round(seg.duration / dt)
        for k in range(num_steps):
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if k * dt >= SETTLE_S:
                lin = robot.data.root_lin_vel_b[0]
                ang_z = robot.data.root_ang_vel_b[0, 2]
                # Same XYZ-Euler convention as deploy_real.quat_to_pitch_lean (confirmed deliberately matching,
                # see that function's docstring): euler_xyz_from_quat returns (lean, pitch, yaw).
                lean, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[0:1])
                errs["lin_x"].append((lin[0] - seg.command[0]).item())
                errs["lin_y"].append((lin[1] - seg.command[1]).item())
                errs["ang_z"].append((ang_z - seg.command[2]).item())
                errs["pitch"].append((pitch[0] - seg.command[3]).item())
                errs["lean"].append((lean[0] - seg.command[4]).item())
                # Isaac's own scene robot joint order already matches cfg.isaac_joint_order -- confirmed via
                # deploy/dump_joint_order.py, see deploy/configs/go2_locomotion.yaml's isaac_joint_order comment.
                joint_pos_samples.append(robot.data.joint_pos[0].cpu().numpy().copy())
        results[seg.name] = {k: _rms(v) for k, v in errs.items()}
        mean_joint_pos[seg.name] = np.mean(joint_pos_samples, axis=0)
    return results, mean_joint_pos


def run_mujoco(policy_path: str, config_path: str) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Same SEGMENTS, same metrics (tracking error + mean joint pose), MuJoCo's own physics (nominal: hardware
    motor limits, no speed cap, no action delay, armature = s2s.ARMATURE's current default -- see module
    docstring)."""
    model, _ = s2s.build_model()
    cfg = s2s.load_sim_config(config_path, policy_path)
    sdk_to_isaac, _ = s2s.deploy_real.build_joint_index_maps(cfg)
    robot = s2s.MujocoGo2(model, cfg, motor_limits="hardware")
    default_isaac = robot.default_isaac
    policy, obs_dim = s2s.load_policy(policy_path)
    if obs_dim != 50:
        raise ValueError(f"Expected a 50-dim (current) policy, got {obs_dim}-dim -- can't compare against Isaac.")

    results: dict[str, dict[str, float]] = {}
    mean_joint_pos: dict[str, np.ndarray] = {}
    last_action = [0.0] * 12
    for seg in SEGMENTS:
        errs = {a: [] for a in AXES}
        joint_pos_samples = []
        num_steps = round(seg.duration / cfg.control_dt)
        for k in range(num_steps):
            obs = s2s.deploy_real.build_observation(
                cfg, robot.to_robot_state(), sdk_to_isaac, default_isaac, list(seg.command), last_action
            )
            with torch.inference_mode():
                action = policy(obs)[0].numpy()
            target = default_isaac + cfg.action_scale * action
            robot.step(target)
            last_action = action.tolist()
            if k * cfg.control_dt >= SETTLE_S:
                lin = robot.lin_vel_com_b()  # COM-based, matching Isaac's root_lin_vel_b (see module docstring)
                ang_z = robot.ang_vel_b()[2]
                lean, pitch = s2s.deploy_real.quat_to_pitch_lean(robot.quat_wxyz)
                errs["lin_x"].append(lin[0] - seg.command[0])
                errs["lin_y"].append(lin[1] - seg.command[1])
                errs["ang_z"].append(ang_z - seg.command[2])
                errs["pitch"].append(pitch - seg.command[3])
                errs["lean"].append(lean - seg.command[4])
                # qpos_adr_isaac already gives isaac-ordered joint angles -- same indexing replay_actions.py uses.
                joint_pos_samples.append(robot.data.qpos[robot.qpos_adr_isaac].copy())
        results[seg.name] = {k: _rms(v) for k, v in errs.items()}
        mean_joint_pos[seg.name] = np.mean(joint_pos_samples, axis=0)
    return results, mean_joint_pos


def load_real_standing_state(path: str, cfg, sdk_to_isaac: list[int]) -> dict[str, object]:
    """Loads a collect_still_observations.py recording and reduces it to the same steady-state summary
    computed for the "stand" segment above: mean joint_pos (isaac order, absolute radians), pitch, lean,
    ang_z, lin_x, lin_y -- so it can be compared directly against Isaac/MuJoCo's own idea of "standing"."""
    log = np.load(path)
    joint_pos_sdk = log["joint_pos"]  # (N, 12), SDK order
    joint_pos_isaac = joint_pos_sdk[:, sdk_to_isaac]  # (N, 12), isaac order
    quat_wxyz = log["quat_wxyz"]
    gyro = log["gyro"]
    sportmode_velocity = log["sportmode_velocity"]

    leans, pitches = [], []
    for q in quat_wxyz:
        lean, pitch = s2s.deploy_real.quat_to_pitch_lean(tuple(q.tolist()))
        leans.append(lean)
        pitches.append(pitch)

    return {
        "joint_pos": joint_pos_isaac.mean(axis=0),
        "pitch": float(np.mean(pitches)),
        "lean": float(np.mean(leans)),
        "ang_z": float(gyro[:, 2].mean()),
        "lin_x": float(sportmode_velocity[:, 0].mean()),
        "lin_y": float(sportmode_velocity[:, 1].mean()),
        "num_samples": joint_pos_sdk.shape[0],
    }


def print_real_vs_sim_pose_report(
    real: dict[str, object], isaac_stand_joint_pos: np.ndarray, mujoco_stand_joint_pos: np.ndarray,
    isaac_joint_order: list[str],
) -> None:
    """Compares the real robot's recorded standing pose against what Isaac/MuJoCo converge to for the same
    zero-command "stand" segment -- this is the "is the real robot's natural resting pose different from what
    the policy thinks standing looks like" check, independent of velocity tracking."""
    print("\n" + "=" * 70)
    print(f"REAL vs ISAAC vs MUJOCO -- standing pose ({real['num_samples']} real samples)")
    print("=" * 70)
    print(f"{'joint':16s}{'real(deg)':>12s}{'isaac(deg)':>12s}{'mujoco(deg)':>12s}{'isaac-real':>12s}{'mujoco-real':>12s}")
    for j, name in enumerate(isaac_joint_order):
        r, i, m = real["joint_pos"][j], isaac_stand_joint_pos[j], mujoco_stand_joint_pos[j]
        print(f"{name:16s}{np.degrees(r):12.1f}{np.degrees(i):12.1f}{np.degrees(m):12.1f}"
              f"{np.degrees(i - r):12.1f}{np.degrees(m - r):12.1f}")
    print("-" * 70)
    print("Other standing-state quantities (pitch/lean in deg, velocities in m/s or rad/s):")
    print(f"{'quantity':16s}{'real':>12s}")
    for key, unit, scale in (("pitch", "deg", 180 / np.pi), ("lean", "deg", 180 / np.pi),
                              ("ang_z", "rad/s", 1.0), ("lin_x", "m/s", 1.0), ("lin_y", "m/s", 1.0)):
        print(f"  {key:14s}{real[key] * scale:12.3f} {unit}")
    print("=" * 70)
    print("Real hardware has no commanded velocity here (collect_still_observations.py records an all-zero "
          "command) and no velocity SENSOR of its own beyond SportModeState -- lin_x/lin_y/ang_z above are "
          "what the real robot's own state estimate reports while standing, not a tracking-error metric.")


def print_report(isaac: dict, mujoco: dict, checkpoint_path: str) -> None:
    width = 24 + 10 * len(AXES)
    print("\n" + "=" * width)
    print(f"VELOCITY / PITCH / LEAN TRACKING: Isaac vs MuJoCo -- {checkpoint_path}")
    print("=" * width)
    print(f"{'segment':16s}{'cmd':8s}" + "".join(f"{a + ' RMS':>20s}" for a in AXES))
    print(f"{'':16s}{'':8s}" + "".join(f"{'isaac':>10s}{'mujoco':>10s}" for _ in AXES))
    for seg in SEGMENTS:
        cmd_str = "/".join(f"{c:+.1f}" for c in seg.command)
        row = f"{seg.name:16s}{cmd_str:8s}"
        for axis in AXES:
            row += f"{isaac[seg.name][axis]:10.4f}{mujoco[seg.name][axis]:10.4f}"
        print(row)
    print("-" * width)
    overall = f"{'MEAN (all segments)':24s}"
    for axis in AXES:
        isaac_mean = float(np.mean([isaac[s.name][axis] for s in SEGMENTS]))
        mujoco_mean = float(np.mean([mujoco[s.name][axis] for s in SEGMENTS]))
        overall += f"{isaac_mean:10.4f}{mujoco_mean:10.4f}"
    print(overall)
    print("=" * width)
    print("RMS tracking error (m/s for lin_x/lin_y, rad for pitch/lean, rad/s for ang_z), first "
          f"{SETTLE_S:.1f}s of each segment excluded. Lower is better; 0 = perfect tracking. cmd column shown "
          "truncated (lin_x/lin_y/ang_z/pitch/lean); full values are in the SEGMENTS list above.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1  # one clean env: deterministic, matches MuJoCo's own single simulated robot
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Deterministic reset (same override play.py's --trace_deterministic_reset uses): a repeatable report
    # needs a repeatable start. Does not prevent a later mid-eval reset after a fall from being randomized
    # (EventCfg.reset_base/reset_robot_joints are unconditional reset-mode events) -- acceptable for a ~35s
    # eval where falls are the exception, not block on for this first version.
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(policy_nn, "student_obs_normalizer", None)

    # Export ONCE, feed the identical weights to MuJoCo -- the comparison is only meaningful if both sims run
    # the exact same network, not a hand-reconstructed approximation of it.
    export_dir = "/tmp/eval_velocity_tracking_export"
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    exported_policy_path = os.path.join(export_dir, "policy.pt")

    dt = env.unwrapped.step_dt
    print(f"[INFO] Running {len(SEGMENTS)} segments on Isaac ({sum(s.duration for s in SEGMENTS):.0f}s sim time)...")
    isaac_results, isaac_joint_pos = run_isaac(env, policy, dt)
    env.close()

    print(f"[INFO] Running the same {len(SEGMENTS)} segments on MuJoCo...")
    mujoco_results, mujoco_joint_pos = run_mujoco(exported_policy_path, args_cli.mujoco_config)

    print_report(isaac_results, mujoco_results, resume_path)

    if args_cli.real_obs_log:
        mujoco_cfg = s2s.load_sim_config(args_cli.mujoco_config, exported_policy_path)
        sdk_to_isaac, _ = s2s.deploy_real.build_joint_index_maps(mujoco_cfg)
        real = load_real_standing_state(args_cli.real_obs_log, mujoco_cfg, sdk_to_isaac)
        print_real_vs_sim_pose_report(
            real, isaac_joint_pos["stand"], mujoco_joint_pos["stand"], mujoco_cfg.isaac_joint_order
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
