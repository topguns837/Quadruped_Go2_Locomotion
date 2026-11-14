# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--manual_commands",
    action="store_true",
    default=False,
    help=(
        "Set an exact [lin_vel_x, lin_vel_y, ang_vel_z, pitch, lean] command via a slider window "
        "(auto-launched) instead of the policy's usual random commands, without closing the Isaac Sim window."
    ),
)
parser.add_argument(
    "--command_file",
    type=str,
    default="/tmp/manual_command.json",
    help="Path the manual-command slider window and this script agree on (see --manual_commands).",
)
parser.add_argument(
    "--trace_out",
    type=str,
    default=None,
    help=(
        "Diagnostic: write a per-step .npz trace (policy obs, raw action, root/joint state, torque, foot "
        "forces) of env 0 at a fixed ZERO command (no slider needed) and exit after --trace_steps steps. "
        "Same fields as sim2sim_mujoco.py --trace_out; compare with sim2sim/compare_traces.py. Off by default."
    ),
)
parser.add_argument("--trace_steps", type=int, default=1100, help="Control steps to record with --trace_out.")
parser.add_argument(
    "--scenario",
    type=str,
    default=None,
    help=(
        "With --trace_out: replay a scenario YAML (sim2sim/scenarios/*.yaml, or a schedule written by "
        "sim2sim/random_commands.py / sim2sim_mujoco.py --save_scenario) as fixed-command segments instead of a "
        "zero command, so the trace can be compared with the MuJoCo run of the same file. The run lasts exactly "
        "the scenario's length (--trace_steps is ignored) and episode timeouts are disabled, so only falls reset."
    ),
)
parser.add_argument(
    "--trace_deterministic_reset",
    action="store_true",
    default=False,
    help=(
        "With --trace_out: start from the exact default pose with zero velocity (disables the reset-time "
        "random base pose/velocity and x0.5-1.5 joint scaling), matching how sim2sim starts."
    ),
)
parser.add_argument(
    "--live_plot",
    action="store_true",
    default=False,
    help=(
        "Log per-step commanded-vs-actual velocity/pitch/lean, plus actual-vs-default-stance height, to a "
        "TensorBoard file under logs/play_dashboard/, viewable live with "
        "scripts/live_dashboard.py --logdir logs/play_dashboard."
    ),
)
parser.add_argument(
    "--no_depth_view",
    action="store_true",
    default=False,
    help=(
        "Disable the live depth-camera viewer window. The viewer is shown automatically whenever the task "
        "has a 'depth_camera' sensor and the GUI is up (not --headless). It renders the actual sensor tensor "
        "the policy consumes, not the viewport RGB render the Camera Inspector shows."
    ),
)
parser.add_argument(
    "--depth_view_every",
    type=int,
    default=2,
    help="Refresh the depth viewer every N control steps (higher = cheaper). Only affects the viewer.",
)
parser.add_argument(
    "--depth_view_env",
    type=int,
    default=0,
    help="Which environment's depth image the viewer/--save_depth_dir shows.",
)
parser.add_argument(
    "--save_depth_dir",
    type=str,
    default=None,
    help=(
        "Also save a colorized depth PNG every --depth_view_every steps to this directory (works headless; "
        "a fallback for when the live omni.ui window isn't available)."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# The stairs task carries a depth camera in its observations, so the RTX sensor pipeline must be on. Force
# it for any -Stairs task so --enable_cameras never has to be passed by hand.
if args_cli.task is not None and "Stairs" in args_cli.task:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable the Camera Inspector tool (Tools -> Sensors -> Camera Inspector) by default, so it doesn't need
# to be manually turned on via Window -> Extensions every session. Harmless to enable even headless (it's
# a GUI panel; it just won't be visible with no window), so no args_cli.headless gate needed.
import omni.kit.app  # noqa: E402

if not args_cli.headless:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.util.camera_inspector", True
    )

"""Rest everything follows."""

import atexit
import json
import os
import subprocess
import threading
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import quadruped_go2_locomotion.tasks  # noqa: F401


def _manual_command_file_poll_loop(cmd_term, path: str, poll_interval: float = 0.1):
    """Polls `path` (written by scripts/manual_command_slider.py, a separate process) for changes and
    applies them to cmd_term, without blocking the main thread's simulation/render loop.

    Tolerates the file not existing yet (slider window hasn't started/written its first value yet) and
    transient malformed reads (a torn read racing the slider's write, despite it writing atomically via
    temp-file + os.replace) by simply retrying next poll -- never raises, never stops the loop.
    """
    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(path)
            if mtime != last_mtime:
                with open(path) as f:
                    data = json.load(f)
                values = [
                    data["lin_vel_x"],
                    data["lin_vel_y"],
                    data["ang_vel_z"],
                    data["pitch"],
                    data["lean"],
                ]
                cmd_term.set_manual_command(values)
                last_mtime = mtime
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(poll_interval)


def _log_live_plot(writer, env, target_height: float):
    """Logs one step's commanded vs. actual [lin_vel_x, lin_vel_y, ang_vel_z, pitch, lean] for env 0 to
    `writer`, using the exact tag names scripts/live_dashboard.py's PANELS already expects
    (Metrics/base_velocity/{cmd,actual}_*) -- no dashboard changes needed to read this.

    Height has no command component any more (see mdp/commands.py's module docstring): `target_height` is
    the fixed value RewardsCfg.height_penalty holds the robot near (passed in by the caller, which reads it
    from env_cfg, rather than hard-coding it here a second time), logged as `cmd_height` purely so the
    existing "Height: commanded vs actual" dashboard panel stays meaningful.

    Unlike mdp/commands.py's training-time metrics, these are instantaneous per-step values, not
    accumulated over a resample cycle: play is "watch one robot," so there's no per-episode aggregation to
    get right or wrong here, just the current value at each step.
    """
    cmd = env.unwrapped.command_manager.get_command("base_velocity")[0]
    robot = env.unwrapped.scene["robot"]
    lean, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[0:1])
    step = env.unwrapped.common_step_counter

    writer.add_scalar("Metrics/base_velocity/cmd_lin_vel_x", cmd[0].item(), step)
    writer.add_scalar("Metrics/base_velocity/actual_lin_vel_x", robot.data.root_lin_vel_b[0, 0].item(), step)
    writer.add_scalar("Metrics/base_velocity/cmd_lin_vel_y", cmd[1].item(), step)
    writer.add_scalar("Metrics/base_velocity/actual_lin_vel_y", robot.data.root_lin_vel_b[0, 1].item(), step)
    writer.add_scalar("Metrics/base_velocity/cmd_ang_vel_z", cmd[2].item(), step)
    writer.add_scalar("Metrics/base_velocity/actual_ang_vel_z", robot.data.root_ang_vel_b[0, 2].item(), step)
    writer.add_scalar("Metrics/base_velocity/cmd_pitch", cmd[3].item(), step)
    writer.add_scalar("Metrics/base_velocity/actual_pitch", pitch.item(), step)
    writer.add_scalar("Metrics/base_velocity/cmd_lean", cmd[4].item(), step)
    writer.add_scalar("Metrics/base_velocity/actual_lean", lean.item(), step)
    writer.add_scalar("Metrics/base_velocity/cmd_height", target_height, step)
    writer.add_scalar("Metrics/base_velocity/actual_height", robot.data.root_pos_w[0, 2].item(), step)
    writer.flush()


class _DepthViewer:
    """Live view of the depth sensor's actual output tensor -- the thing the vision policy will consume --
    not the viewport RGB render the Camera Inspector shows. Depth is colorized (near = red ... far = blue,
    invalid/no-return pixels = magenta) so step edges and sensor holes are obvious at a glance; the window
    title carries min/max/mean depth and the invalid-pixel fraction as a numeric cross-check.

    Reads the sensor's data type and clipping range straight off the camera cfg, so it works both before and
    after the planned switch of data_types (distance_to_camera -> distance_to_image_plane) with no change
    here. Degrades gracefully: any omni.ui issue disables the live window but leaves play running (and
    --save_depth_dir still writes PNGs); with --save_depth_dir alone it works headless.
    """

    # near -> far color ramp (red, orange, yellow, green, blue), RGB in [0, 1].
    _STOPS = [
        (1.00, 0.16, 0.16),
        (1.00, 0.63, 0.00),
        (0.90, 0.90, 0.00),
        (0.00, 0.82, 0.00),
        (0.16, 0.31, 1.00),
    ]

    def __init__(self, camera, data_type: str, near: float, far: float, env_idx: int, save_dir: str | None,
                 show_window: bool):
        self._camera = camera
        self._data_type = data_type
        self._near = float(near)
        self._far = float(far)
        self._env_idx = env_idx
        self._save_dir = save_dir
        self._stops = None  # built lazily on the tensor's device
        self._save_count = 0
        self._window = None
        self._provider = None
        self._label = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        if show_window:
            self._try_build_window()

    def _try_build_window(self):
        try:
            import omni.ui as ui

            w, h = int(self._camera.cfg.width), int(self._camera.cfg.height)
            scale = max(1, int(round(480 / max(w, h))))
            self._provider = ui.ByteImageProvider()
            self._window = ui.Window("Depth Camera (policy input)", width=w * scale + 20, height=h * scale + 60)
            with self._window.frame:
                with ui.VStack(spacing=4):
                    self._label = ui.Label("depth: waiting for first frame", height=18)
                    ui.ImageWithProvider(self._provider, width=w * scale, height=h * scale)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] depth viewer: could not create omni.ui window ({exc}); live view disabled.")
            self._window = None
            self._provider = None

    def _colorize(self, depth: torch.Tensor) -> torch.Tensor:
        """depth: (H, W) float tensor -> (H, W, 4) uint8 RGBA on the same device."""
        if self._stops is None:
            self._stops = torch.tensor(self._STOPS, device=depth.device, dtype=torch.float32)
        invalid = ~torch.isfinite(depth) | (depth <= 0.0)
        t = ((depth - self._near) / max(self._far - self._near, 1e-6)).clamp(0.0, 1.0)
        n = self._stops.shape[0] - 1
        scaled = (t * n).nan_to_num(0.0)
        idx = scaled.floor().clamp(0, n - 1).long()
        frac = (scaled - idx).unsqueeze(-1)
        rgb = self._stops[idx] * (1.0 - frac) + self._stops[idx + 1] * frac
        rgb[invalid] = torch.tensor([1.0, 0.0, 1.0], device=depth.device)
        rgba = torch.cat([rgb, torch.ones_like(rgb[..., :1])], dim=-1)
        return (rgba * 255.0).to(torch.uint8)

    def update(self, step: int):
        if self._window is None and self._save_dir is None:
            return
        out = self._camera.data.output.get(self._data_type)
        if out is None or out.shape[0] <= self._env_idx:
            return
        depth = out[self._env_idx, ..., 0]
        finite = depth[torch.isfinite(depth) & (depth > 0.0)]
        invalid_frac = 1.0 - (finite.numel() / depth.numel())
        stats = (
            f"depth [{self._data_type}]  min {finite.min().item():.2f}  max {finite.max().item():.2f}  "
            f"mean {finite.mean().item():.2f} m  invalid {invalid_frac * 100:.1f}%"
            if finite.numel() > 0
            else f"depth [{self._data_type}]  all pixels invalid"
        )
        rgba = self._colorize(depth).cpu().numpy()
        if self._label is not None:
            self._label.text = stats
        if self._provider is not None:
            h, w = rgba.shape[:2]
            try:
                self._provider.set_bytes_data(rgba.reshape(-1).tolist(), [w, h])
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] depth viewer: set_bytes_data failed ({exc}); disabling live view.")
                self._provider = None
                self._window = None
        if self._save_dir is not None:
            self._save_png(rgba, step)

    def _save_png(self, rgba, step: int):
        path = os.path.join(self._save_dir, f"depth_{self._save_count:05d}_step{step}.png")
        self._save_count += 1
        try:
            import matplotlib.image as mpimg

            mpimg.imsave(path, rgba)
        except Exception:  # noqa: BLE001
            import numpy as np

            np.save(path.replace(".png", ".npy"), rgba)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    if args_cli.trace_out and args_cli.trace_deterministic_reset:
        env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        env_cfg.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        print("[INFO] trace: deterministic reset (default pose, zero velocity).")

    if args_cli.scenario:
        if not args_cli.trace_out:
            raise SystemExit("--scenario needs --trace_out (it replays the scenario while recording the trace).")
        # A 40 s episode timeout would reset the robot mid-schedule and be mistaken for a fall: only falls reset.
        env_cfg.episode_length_s = 1.0e6

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    if args_cli.manual_commands:
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        cmd_term.enable_manual_override()
        threading.Thread(
            target=_manual_command_file_poll_loop, args=(cmd_term, args_cli.command_file), daemon=True
        ).start()
        # Slider window is a separate process on the plain system python3 (never isaaclab -p) -- Isaac
        # Sim's bundled Python has no tkinter. See docker/Dockerfile's note next to the python3-tk install.
        # Absolute path, not just "python3": this container's own ~/.bashrc aliases the bare name to Isaac
        # Sim's python for interactive shells (see docker/Dockerfile's `alias python3=...` line). subprocess
        # bypasses shell aliases either way (no shell involved), but using the absolute path here keeps
        # this call and the manual-relaunch instructions below consistent with each other.
        scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        slider_script = os.path.join(scripts_dir, "manual_command_slider.py")
        # subprocess.Popen inherits the full parent environment by default, including PYTHONPATH/
        # LD_LIBRARY_PATH set by /isaac-sim/setup_python_env.sh to point at Isaac Sim's *own* Python 3.11
        # stdlib/site-packages/libs. Left in place, the system python3 spawned below picks up that 3.11
        # PYTHONPATH ahead of its own stdlib and crashes on a compiled-extension ABI mismatch
        # ("AssertionError: SRE module mismatch", confirmed directly). Stripping these two vars lets it use
        # its own, correct sys.path and shared libraries instead.
        slider_env = os.environ.copy()
        slider_env.pop("PYTHONPATH", None)
        slider_env.pop("LD_LIBRARY_PATH", None)
        slider_proc = subprocess.Popen(
            ["/usr/bin/python3", slider_script, "--command_file", args_cli.command_file], env=slider_env
        )
        atexit.register(slider_proc.terminate)
        print(f"[INFO] Manual command mode: slider window launched (writing to {args_cli.command_file}).")

    trace = None
    if args_cli.trace_out:
        import numpy as np

        if not args_cli.manual_commands:
            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            cmd_term.enable_manual_override()
            cmd_term.set_manual_command([0.0, 0.0, 0.0, 0.0, 0.0])
        trace = {k: [] for k in ("obs", "action", "target", "joint_pos", "joint_vel", "torque", "root_pos",
                                 "root_quat", "lin_vel_origin_b", "lin_vel_com_b", "ang_vel_b", "foot_force",
                                 "command", "done", "fell")}
        scenario_cmds = None
        if args_cli.scenario:
            import yaml

            _dt = env.unwrapped.step_dt
            scenario_cmds = []
            for _seg in yaml.safe_load(open(args_cli.scenario))["segments"]:
                _cmd = [float(v) for v in _seg["command"]]
                if len(_cmd) != 5:
                    raise ValueError(f"scenario segment {_seg['name']!r}: command must have 5 values")
                scenario_cmds += [_cmd] * round(float(_seg["duration"]) / _dt)
            args_cli.trace_steps = len(scenario_cmds)
            cmd_term.set_manual_command(scenario_cmds[0])
            print(f"[INFO] trace: replaying scenario {args_cli.scenario} ({len(scenario_cmds)} steps)")
        _robot = env.unwrapped.scene["robot"]
        _sensor = env.unwrapped.scene["contact_forces"]
        _foot_ids = [_sensor.body_names.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
        _view = _robot.root_physx_view
        _phys = {
            "phys_dof_max_velocity": _view.get_dof_max_velocities()[0].cpu().numpy(),
            "phys_dof_armature": _view.get_dof_armatures()[0].cpu().numpy(),
            "phys_dof_friction": _view.get_dof_friction_coefficients()[0].cpu().numpy(),
            "phys_dof_stiffness": _view.get_dof_stiffnesses()[0].cpu().numpy(),
            "phys_dof_damping": _view.get_dof_dampings()[0].cpu().numpy(),
            "phys_dof_max_force": _view.get_dof_max_forces()[0].cpu().numpy(),
            "phys_dof_limits": _view.get_dof_limits()[0].cpu().numpy(),
            "phys_body_masses": _view.get_masses()[0].cpu().numpy(),
        }
        print("[INFO] trace: PhysX dof max velocity (rad/s):", _phys["phys_dof_max_velocity"])
        print("[INFO] trace: PhysX dof armature:", _phys["phys_dof_armature"])
        print(f"[INFO] trace: recording {args_cli.trace_steps} steps at zero command to {args_cli.trace_out}")

    live_plot_writer = None
    if args_cli.live_plot:
        from torch.utils.tensorboard import SummaryWriter

        live_plot_dir = os.path.join("logs", "play_dashboard", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        live_plot_writer = SummaryWriter(log_dir=live_plot_dir)
        print(f"[INFO] Live plot logging to: {live_plot_dir}")
        # Read once, not re-read every step: the fixed target height height_penalty holds the robot near
        # (see RewardsCfg.height_penalty in quadruped_go2_locomotion_env_cfg.py).
        live_plot_target_height = env_cfg.rewards.height_penalty.params["target_height"]

    # Debug visualization: draw an RGB coordinate-frame gizmo at the depth camera's reported world pose
    # each step, if this task has one (only the stairs tasks do). Lets you see exactly where the sensor
    # thinks it is/which way it thinks it's facing, directly in the viewport -- useful for diagnosing
    # camera mount-pose issues without reading raw tensors.
    depth_camera_marker = None
    if "depth_camera" in env.unwrapped.scene.keys():
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.markers.config import FRAME_MARKER_CFG

        # Single-prototype config, not a copy of FRAME_MARKER_CFG directly -- that one bundles a SECOND
        # "connecting_line" prototype (meant for a two-point line-plus-frame visualization), which left
        # marker_indices ambiguous for our single-point use and is the likely reason nothing appeared.
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/DepthCameraFrame",
            markers={"frame": FRAME_MARKER_CFG.markers["frame"]},
        )
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # small, the real D435f housing is ~90mm
        depth_camera_marker = VisualizationMarkers(marker_cfg)
        depth_camera_marker.set_visibility(True)

    # Live depth viewer: shows the sensor tensor the policy consumes (see _DepthViewer). Auto-on when the
    # task has a depth camera; needs --enable_cameras for the sensor to produce output. --no_depth_view to
    # suppress, --save_depth_dir to also dump PNGs (works headless).
    depth_viewer = None
    if "depth_camera" in env.unwrapped.scene.keys() and (not args_cli.no_depth_view or args_cli.save_depth_dir):
        camera = env.unwrapped.scene["depth_camera"]
        data_type = camera.cfg.data_types[0]
        near, far = camera.cfg.spawn.clipping_range
        depth_viewer = _DepthViewer(
            camera=camera,
            data_type=data_type,
            near=near,
            far=far,
            env_idx=args_cli.depth_view_env,
            save_dir=args_cli.save_depth_dir,
            show_window=not args_cli.headless and not args_cli.no_depth_view,
        )

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if trace is not None and scenario_cmds is not None:
                _i = len(trace["obs"])
                if _i == 0 or scenario_cmds[_i] != scenario_cmds[_i - 1]:
                    cmd_term.set_manual_command(scenario_cmds[_i])
                    # the observation computed at the end of the last step still holds the previous command
                    obs = env.get_observations()
            # agent stepping
            actions = policy(obs)
            if trace is not None:
                _obs_in = obs["policy"][0].detach().cpu().numpy().copy()
                _act = actions[0].detach().cpu().numpy().copy()
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if trace is not None:
                _d = _robot.data
                trace["obs"].append(_obs_in)
                trace["action"].append(_act)
                trace["target"].append(_d.joint_pos_target[0].cpu().numpy().copy())
                trace["joint_pos"].append(_d.joint_pos[0].cpu().numpy().copy())
                trace["joint_vel"].append(_d.joint_vel[0].cpu().numpy().copy())
                trace["torque"].append(_d.applied_torque[0].cpu().numpy().copy())
                trace["root_pos"].append(_d.root_pos_w[0].cpu().numpy().copy())
                trace["root_quat"].append(_d.root_quat_w[0].cpu().numpy().copy())
                trace["lin_vel_origin_b"].append(_d.root_link_lin_vel_b[0].cpu().numpy().copy())
                trace["lin_vel_com_b"].append(_d.root_com_lin_vel_b[0].cpu().numpy().copy())
                trace["ang_vel_b"].append(_d.root_ang_vel_b[0].cpu().numpy().copy())
                _f = _sensor.data.net_forces_w[0, _foot_ids].norm(dim=-1)
                trace["foot_force"].append(_f.cpu().numpy().copy())
                trace["command"].append(
                    env.unwrapped.command_manager.get_command("base_velocity")[0, :5].cpu().numpy().copy()
                )
                trace["done"].append(float(dones[0].item()))
                # a real fall (base/arm touched the ground), as opposed to a timeout reset
                trace["fell"].append(float(env.unwrapped.termination_manager.terminated[0].item()))
                if len(trace["obs"]) >= args_cli.trace_steps:
                    break
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if live_plot_writer is not None:
            _log_live_plot(live_plot_writer, env, live_plot_target_height)
        if depth_camera_marker is not None:
            camera = env.unwrapped.scene["depth_camera"]
            depth_camera_marker.visualize(
                translations=camera.data.pos_w,
                orientations=camera.data.quat_w_world,
                marker_indices=[0] * camera.data.pos_w.shape[0],
            )
        if depth_viewer is not None:
            _vstep = env.unwrapped.common_step_counter
            if _vstep % args_cli.depth_view_every == 0:
                depth_viewer.update(_vstep)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if trace is not None and trace["obs"]:
        np.savez_compressed(
            args_cli.trace_out,
            dt=dt,
            joint_names=np.array(list(env.unwrapped.scene["robot"].data.joint_names), dtype=str),
            **_phys,
            **{k: np.stack(v) for k, v in trace.items()},
        )
        print(f"[INFO] trace: wrote {len(trace['obs'])} steps to {args_cli.trace_out}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
