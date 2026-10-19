# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop dynamics test: replay another simulator's recorded policy actions in MuJoCo.

Takes an Isaac trace (`scripts/rsl_rl/play.py --trace_out`), applies its raw action sequence to the MuJoCo Go2
from the same default start pose, and writes a MuJoCo trace in the same format. The policy is NOT run, so the
control inputs are identical in both simulators and any difference in joint velocities, base motion or contact
is purely physics (actuator model, joint limits and velocity caps, inertias, contact), not policy feedback.

Compare the result with the source trace:
    python3 sim2sim/compare_traces.py isaac_trace.npz replay_trace.npz

Usage (inside the project container, or any Python with torch, mujoco, numpy<2, pyyaml):
    python3 sim2sim/replay_actions.py --isaac_trace isaac_zero_det.npz --out replay.npz
    [--armature 0.01] [--foot_friction 0.8] [--joint_speed_cap 30.1 30.1 15.7]   # one change at a time, see sim2sim README
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import mujoco
import numpy as np

_SIM2SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SIM2SIM_DIR)
import sim2sim_mujoco as s2s  # noqa: E402

deploy_real = s2s.deploy_real
build_model = s2s.build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isaac_trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=os.path.join(s2s._REPO_ROOT, "deploy", "configs", "go2_locomotion.yaml"))
    parser.add_argument("--motor_limits", choices=sorted(s2s.MOTOR_LIMITS), default="hardware")
    parser.add_argument("--armature", type=float, default=None, help="Override joint armature (default: sim2sim_mujoco.ARMATURE, 0.01).")
    parser.add_argument("--foot_friction", type=float, default=None, help="Override foot sliding friction.")
    parser.add_argument("--floor_friction", type=float, default=None, help="Override floor sliding friction.")
    parser.add_argument(
        "--joint_speed_cap", type=float, nargs=3, default=None, metavar=("HIP", "THIGH", "CALF"),
        help="Hard-clip joint speeds (rad/s) per joint type each physics step, emulating PhysX's maxJointVelocity.",
    )
    parser.add_argument(
        "--viewer", action="store_true", default=False,
        help="Open the same live GLFW viewer sim2sim_mujoco.py uses (mujoco.viewer.launch_passive), paced to "
        "real time, so the replay can be watched instead of only written to --out. Off by default (runs as "
        "fast as possible, headless).",
    )
    args = parser.parse_args()

    ref = {k: v for k, v in np.load(args.isaac_trace, allow_pickle=False).items() if v.dtype != object}
    actions = ref["action"]
    steps = len(actions)

    model, _ = build_model()
    cfg = deploy_real.load_config(args.config, "lo")
    if args.armature is not None:
        s2s.ARMATURE = args.armature
    robot = s2s.MujocoGo2(model, cfg, motor_limits=args.motor_limits)
    if args.foot_friction is not None:
        for g in robot.foot_geoms:
            model.geom_friction[g, 0] = args.foot_friction
    if args.floor_friction is not None:
        model.geom_friction[robot.floor_geom, 0] = args.floor_friction
    sdk_to_isaac, _ = deploy_real.build_joint_index_maps(cfg)
    default_isaac = robot.default_isaac
    cap = None
    if args.joint_speed_cap is not None:
        per_type = dict(zip(("hip", "thigh", "calf"), args.joint_speed_cap))
        cap = np.array([per_type[n.split("_")[1]] for n in cfg.isaac_joint_order])

    viewer = None
    viewer_threads: set[threading.Thread] = set()
    if args.viewer:
        import mujoco.viewer

        threads_before = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(model, robot.data)
        viewer_threads = set(threading.enumerate()) - threads_before
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = robot.base_id
        viewer.cam.distance = 2.0

    names = ("obs", "action", "target", "joint_pos", "joint_vel", "torque", "root_pos", "root_quat",
             "lin_vel_origin_b", "lin_vel_com_b", "ang_vel_b", "foot_force", "command")
    tr = {k: [] for k in names}
    last_action = [0.0] * 12
    first_fall = None
    try:
        for k in range(steps):
            tick_start = time.monotonic()
            obs = deploy_real.build_observation(
                cfg, robot.to_robot_state(), sdk_to_isaac, default_isaac, [0.0] * 5, last_action
            )
            action = actions[k].astype(np.float64)
            target = default_isaac + cfg.action_scale * action
            if cap is None:
                robot.step(target)
            else:
                for _ in range(s2s.DECIMATION):
                    tau = robot.dc_motor_torque(target)
                    robot.last_tau = tau
                    robot.data.ctrl[robot.act_idx_isaac] = tau
                    mujoco.mj_step(robot.model, robot.data)
                    qd = robot.data.qvel[robot.dof_adr_isaac]
                    robot.data.qvel[robot.dof_adr_isaac] = np.clip(qd, -cap, cap)
            last_action = action.tolist()
            tr["obs"].append(obs[0].numpy().copy())
            tr["action"].append(action)
            tr["target"].append(target)
            tr["joint_pos"].append(robot.data.qpos[robot.qpos_adr_isaac].copy())
            tr["joint_vel"].append(robot.data.qvel[robot.dof_adr_isaac].copy())
            tr["torque"].append(robot.last_tau.copy())
            tr["root_pos"].append(robot.data.qpos[0:3].copy())
            tr["root_quat"].append(robot.data.qpos[3:7].copy())
            tr["lin_vel_origin_b"].append(np.array(robot.lin_vel_b()))
            tr["lin_vel_com_b"].append(np.array(robot.lin_vel_com_b()))
            tr["ang_vel_b"].append(np.array(robot.ang_vel_b()))
            tr["foot_force"].append(robot.foot_normal_forces())
            tr["command"].append(np.zeros(5))
            reason = robot.fallen()
            if reason and first_fall is None:
                first_fall = k
                print(f"[REPLAY] 'fall' criterion first met at step {k} ({reason}); continuing (open loop, no reset)")

            if viewer is not None:
                if not viewer.is_running():
                    print("[REPLAY] viewer window closed -- stopping early.")
                    break
                viewer.sync()
                time.sleep(max(0.0, cfg.control_dt - (time.monotonic() - tick_start)))
    finally:
        if viewer is not None:
            s2s.close_viewer(viewer, viewer_threads)

    n = len(tr["obs"])
    np.savez_compressed(args.out, dt=cfg.control_dt, **{k: np.stack(v) for k, v in tr.items()})
    print(f"[REPLAY] wrote {n}/{steps} steps to {args.out}")

    # Open-loop divergence from the source trace (same inputs): how long do the two physics stay together?
    # Only meaningful when the source trace has its own joint_pos/joint_vel to compare against -- e.g.
    # play.py's --trace_out, or deploy/replay_policy_offline.py's --out when it was given a recording that
    # itself stored joint_pos (not every source trace does; a source with just 'obs'/'action', like a bare
    # offline-policy replay, has nothing of its own to diverge from -- this block is skipped in that case,
    # not an error).
    if "joint_pos" in ref and "joint_vel" in ref:
        m = min(n, steps)
        err = np.abs(np.stack(tr["joint_pos"])[:m] - ref["joint_pos"][:m]).max(axis=1)
        for thr in (0.05, 0.2, 0.5):
            idx = np.nonzero(err > thr)[0]
            print(f"[REPLAY] max joint-position gap first exceeds {thr:.2f} rad at step "
                  f"{int(idx[0]) if len(idx) else 'never'} (dt={cfg.control_dt}s)")
        print(f"[REPLAY] joint_vel RMS  source={np.sqrt((ref['joint_vel'][:m] ** 2).mean()):.3f}  "
              f"replay={np.sqrt((np.stack(tr['joint_vel'])[:m] ** 2).mean()):.3f} rad/s")
    else:
        print("[REPLAY] source trace has no joint_pos/joint_vel of its own (only obs/action) -- skipping the "
              "divergence-from-source comparison; the replay trace above was still written in full.")


if __name__ == "__main__":
    main()
