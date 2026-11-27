# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the observation-stacking layout the deploy/sim2sim mirror must reproduce.

Why: training stacks the last HISTORY_LENGTH frames via Isaac Lab's group ``history_length``. The exact flat
layout (TERM-major vs FRAME-major, and newest sample FIRST vs LAST) is an internal Isaac Lab detail. deploy_real.py
reproduces it with ObsHistory under two flags (OBS_HISTORY_TERM_MAJOR, OBS_HISTORY_NEWEST_LAST). If those flags are
wrong, the policy sees scrambled history on hardware. This script runs the REAL env and auto-discovers the true
layout, so you can confirm the flags (or learn exactly what to set them to) before training or deploying.

How: it runs the Play env twice, each pass in its OWN process (Isaac Lab cannot build a second env in the process
that already ran one: the second build hangs), with the SAME seed, zero actions and deterministic resets, so the two
rollouts are identical step-for-step:
  pass 1: history_length = 1  -> record each step's single-frame policy obs      (T, base)
  pass 2: history_length = H  -> record each step's stacked policy obs           (T, base*H)
Then, offline, it rebuilds the stacked vector from the single frames under all four layout combos and reports which
one matches the env. The winning combo is what deploy_real's two flags must equal.

Run (inside the container, from the repo root):
    ../IsaacLab/isaaclab.sh -p src/play/verify_obs_stacking.py --task Quadruped-Locomotion-Go2-Play \
        --num_envs 1 --history_length 6 --steps 20 --headless

It also checks deploy_real.ObsHistory directly (same frames in -> does it reproduce the env vector) and prints a
clear PASS/FAIL plus the flag values to use.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the observation-stacking layout against the real env.")
parser.add_argument("--task", type=str, default="Quadruped-Locomotion-Go2-Play")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--history_length", type=int, default=6, help="H to verify (must match env_cfg.HISTORY_LENGTH).")
parser.add_argument("--steps", type=int, default=20, help="Rollout length; must be > H.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
# Internal: set by the orchestrator when it starts one rollout in a fresh process.
parser.add_argument("--worker_history", type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument("--worker_out", type=str, default=None, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
IS_WORKER = args_cli.worker_history is not None

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

if IS_WORKER:
    # Only a worker starts Isaac Sim; the orchestrator below just launches workers and compares their output.
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import quadruped_go2_locomotion  # noqa: F401

# Reuse the SAME ObsHistory / frame_term_widths the robot uses (deploy/ is not a package; add it to the path like
# sim2sim does).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy"))
import deploy_real  # noqa: E402


def rollout(history_length: int) -> np.ndarray:
    """Zero-action rollout of the Play env with the given group history_length; returns (steps, obs_dim) for env 0."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = 0
    # Override the policy group's history so pass 1 is single-frame and pass 2 is stacked, everything else identical.
    env_cfg.observations.policy.history_length = history_length
    env_cfg.observations.policy.flatten_history_dim = True
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset(seed=0)
    frames = []
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            frames.append(policy_obs[0].detach().cpu().numpy().copy())
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, _, _, _, _ = env.step(actions)
    env.close()
    return np.array(frames)


def build_stack(frames: np.ndarray, H: int, widths: list[int], term_major: bool, newest_last: bool) -> np.ndarray:
    """Rebuild the stacked vector at each step from single frames, under one layout combo. Startup pads by
    repeating the first frame (Isaac Lab fills a reset CircularBuffer with the current obs)."""
    T, base = frames.shape
    out = np.empty((T, base * H), dtype=frames.dtype)
    for t in range(T):
        window = [frames[max(0, t - H + 1 + k)] for k in range(H)]  # oldest..newest, padded with frames[0]
        ordered = window if newest_last else window[::-1]
        if term_major:
            cols, a = [], 0
            for w in widths:
                cols.extend(f[a:a + w] for f in ordered)
                a += w
            out[t] = np.concatenate(cols)
        else:
            out[t] = np.concatenate(ordered)
    return out


def run_pass_in_new_process(history_length: int, out_path: str) -> np.ndarray:
    """One rollout in a fresh Isaac Sim process (see the module docstring for why); returns its (steps, obs_dim)."""
    cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + [
        "--worker_history", str(history_length), "--worker_out", out_path]
    result = subprocess.run(cmd)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise SystemExit(f"[verify] FAIL: the history_length={history_length} rollout process exited with "
                         f"{result.returncode}; cannot verify the layout.")
    return np.load(out_path)


def main() -> None:
    H = args_cli.history_length
    assert args_cli.steps > H, "need more steps than H"
    with tempfile.TemporaryDirectory() as tmp:
        print("[verify] pass 1: single-frame rollout (history_length=1), own process")
        single = run_pass_in_new_process(1, os.path.join(tmp, "single.npy"))
        print(f"[verify] pass 2: stacked rollout (history_length={H}), own process")
        stacked = run_pass_in_new_process(H, os.path.join(tmp, "stacked.npy"))
    base = single.shape[1]
    assert stacked.shape[1] == base * H, (stacked.shape, base, H)
    layout = deploy_real.OBS_LAYOUT[base]
    widths = deploy_real.frame_term_widths(base, layout)
    print(f"[verify] base frame dim = {base}, term widths = {widths}")

    print("\n[verify] which layout combo reproduces the env's stacked obs?")
    winner = None
    for term_major in (True, False):
        for newest_last in (True, False):
            rebuilt = build_stack(single, H, widths, term_major, newest_last)
            max_err = float(np.abs(rebuilt - stacked).max())
            tag = f"term_major={term_major}  newest_last={newest_last}"
            match = max_err < 1e-4
            print(f"  {tag:40s} max|err|={max_err:.3e}  {'<== MATCH' if match else ''}")
            if match:
                winner = (term_major, newest_last)

    print()
    if winner is None:
        print("[verify] FAIL: no simple combo matched. The env may use a different history layout; inspect "
              "env.observation_manager before deploying. Do NOT run on hardware.")
    else:
        tm, nl = winner
        ok = (tm == deploy_real.OBS_HISTORY_TERM_MAJOR and nl == deploy_real.OBS_HISTORY_NEWEST_LAST)
        print(f"[verify] env layout is term_major={tm}, newest_last={nl}.")
        print(f"[verify] deploy_real.py currently has term_major={deploy_real.OBS_HISTORY_TERM_MAJOR}, "
              f"newest_last={deploy_real.OBS_HISTORY_NEWEST_LAST}.")
        print(f"[verify] {'PASS: flags already correct.' if ok else 'ACTION: set the two OBS_HISTORY_* flags in deploy_real.py to the env values above.'}")

        # Direct check of the exact ObsHistory the robot uses, with the discovered flags applied.
        deploy_real.OBS_HISTORY_TERM_MAJOR, deploy_real.OBS_HISTORY_NEWEST_LAST = tm, nl
        hist = deploy_real.ObsHistory(H, widths)
        rebuilt = np.array([hist.push_and_stack(torch.tensor(f).unsqueeze(0))[0].numpy() for f in single])
        err = float(np.abs(rebuilt - stacked).max())
        print(f"[verify] deploy_real.ObsHistory reproduces the env stacked obs: max|err|={err:.3e} "
              f"-> {'PASS' if err < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    if IS_WORKER:
        np.save(args_cli.worker_out, rollout(args_cli.worker_history))
        simulation_app.close()
    else:
        main()
