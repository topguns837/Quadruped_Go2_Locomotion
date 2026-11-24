# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning project for training a Unitree Go2 quadruped robot in NVIDIA Isaac Lab. The core environment is `Quadruped-Locomotion-Go2`, a ManagerBasedRLEnv that tracks velocity commands (linear x/y, angular z) with extended pitch/lean angle control. Uses RSL-RL (PPO) for training.

## Key Dependencies

- Isaac Sim 5.1.0 + Isaac Lab v2.3.2
- RSL-RL 3.0.1 (PPO runner)
- Python 3.11

## Repository Structure

```
source/quadruped_go2_locomotion/          # Isaac Lab extension package
  quadruped_go2_locomotion/
    tasks/manager_based/quadruped_go2_locomotion/
      quadruped_go2_locomotion_env_cfg.py  # Main env config (scene, MDP, rewards, terminations)
      __init__.py                      # Gym registration (Go2, Go2-Play)
      agents/rsl_rl_ppo_cfg.py        # PPO hyperparameters
      mdp/
        commands.py                    # UniformVelocityCommandCfgWithPitch (extends isaac lab velocity commands with pitch/lean)
        rewards.py                     # Custom rewards: track_pitch_exp, track_lean_exp, base_height_l1_pitch, hip_crossing_l2
        observations.py               # Depth observation term
        curriculums.py                # terrain_levels_vel curriculum
scripts/
  rsl_rl/train.py                     # RSL-RL training entry point
  rsl_rl/play.py                      # RSL-RL playback entry point
  zero_agent.py                       # Zero-action validation agent
  random_agent.py                     # Random-action validation agent
  list_envs.py                        # List available environments
continualTraining.sh                  # Script for resuming training
logs/rsl_rl/<experiment_name>/        # Training checkpoints and logs (go2_with_pitch, unitree_go2_flat)
deploy/                               # Real-hardware deployment over unitree_sdk2py (see deploy.md, deploy/Checklist.md)
sim2sim/                              # MuJoCo sim2sim runner for exported policies (see sim2sim/README.md)
```

## Running Python Scripts

All Python scripts in this repo require the Isaac Sim environment. Use Isaac Lab's launcher:

```bash
# From repo root, use the isaaclab.sh -p wrapper:
../IsaacLab/isaaclab.sh -p scripts/<script>.py [args...]
```

The `-p` flag tells `isaaclab.sh` to use the Isaac Sim Python interpreter (`_isaac_sim/python.sh`).

## Testing Code Snippets

To quickly test small code snippets without the full simulator, use Isaac Sim's Python binary directly. It sources its own env so no conda/uv setup is needed:

```bash
# One-liner — no alias or env setup needed:
$ISAACLAB_ROOT/_isaac_sim/python.sh -c "
import torch
import importlib.util as iu
spec = iu.spec_from_file_location(
    'math_utils',
    '$ISAACLAB_ROOT/source/isaaclab/isaaclab/utils/math.py')
math_utils = iu.module_from_spec(spec)
spec.loader.exec_module(math_utils)

q = math_utils.quat_from_euler_xyz(
    torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0))
print(q)
"
```

Or add a shell alias to `~/.bashrc` / `~/.zshrc`:

```bash
alias isaaclab-python='$ISAACLAB_ROOT/_isaac_sim/python.sh'
```

Then use the same pattern: `isaaclab-python -c "..."`

> **Note:** `pxr` (USD) requires the Omniverse Kit runtime and cannot be imported without the full simulator (`isaaclab.sh -p` with an active sim). For pure-Python modules like `torch` and `numpy`, `_isaac_sim/python.sh -c "..."` works instantly. For `isaaclab` utilities, use `iu.spec_from_file_location` to load the module file directly — `import isaaclab.utils.math` triggers `isaaclab.__init__` which pulls in `pxr`.

## Common Commands

```bash
# Install the extension (requires Isaac Lab Python env)
../IsaacLab/isaaclab.sh -p -m pip install -e source/quadruped_go2_locomotion

# List available environments
../IsaacLab/isaaclab.sh -p scripts/list_envs.py

# Train with RSL-RL (PPO)
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --headless

# Resume training
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --resume --headless

# Batched training: 2000-iteration chunks, each in a fresh process so host RAM and VRAM are released between chunks.
# Run inside the container; resumes from the latest checkpoint and writes to ONE folder (train.py --run_dir).
BATCH=2000 TARGET=18000 EXTRA="--headless --num_envs 2048" bash scripts/train_batched.sh

# Play a trained checkpoint
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1

# Validate environment wiring (zero action)
../IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Quadruped-Locomotion-Go2

# Validate environment wiring (random action)
../IsaacLab/isaaclab.sh -p scripts/random_agent.py --task=Quadruped-Locomotion-Go2

# Sim2sim: run an exported policy in MuJoCo (no Kit needed; fetch the model once first)
./sim2sim/fetch_go2_model.sh
../IsaacLab/_isaac_sim/python.sh sim2sim/sim2sim_mujoco.py --headless --scenario sim2sim/scenarios/basic.yaml
```

## Training Configuration

- **Environment**: `QuadrupedLocomotionEnvCfg` (1024 envs, 40s episodes, decimation=4, dt=0.005s)
- **Robot**: Unitree GO2WITHARM (from `isaaclab_assets`)
- **Terrain**: 8x8m generator with flat (20%) + random_rough (20%), 9x21 grid
- **Observations**: base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, last_action (actor, with additive uniform noise; base_lin_vel is critic-only). **Observation stacking (obs_stacking_with_height, 2026-11-15)**: both the policy and critic groups set `history_length=HISTORY_LENGTH` (=6) + `flatten_history_dim=True`, so with the 6-dim command (height re-added, below) the actor input is 48*6=288 and the critic 51*6=306. Isaac Lab pushes group history down to each term, so the flat layout is **term-major** ([ang_vel×6, gravity×6, command×6, joint_pos×6, joint_vel×6, last_action×6], newest sample last); `deploy/deploy_real.py`'s `ObsHistory` reproduces this (flags `OBS_HISTORY_TERM_MAJOR`/`OBS_HISTORY_NEWEST_LAST`), and `DeployConfig.history_length` / sim2sim `--history_length` must match the policy (**deploy defaults to `history_length: 1`, so the current single-frame kp40 policy is unaffected**; the new policy uses `deploy/profiles/go2_obs_stacking.yaml`). **Changing HISTORY_LENGTH changes the input dim, so a stacked policy cannot resume a single-frame checkpoint — it trains from scratch.** Before any hardware deploy, run `scripts/verify_obs_stacking.py` in the container to confirm the layout flags against the real env. **joint_vel noise + filter (obs_stacking_with_height):** the `joint_vel` obs term is `mdp.JointVelFiltered` — hardware-realistic uniform noise `noise_amp` (0.3 rad/s; the real floor measured ~0.06, the old 1.5 was adversarial and trained the ~25 Hz buzz in) added BEFORE a first-order EMA low-pass (`cutoff_hz`, 10 Hz in the obs_stacking profile), the same order deploy uses (`DeployConfig.joint_vel_filter_hz` + `ObsEmaFilter`, `mdp/obs_filters.py`), so the policy sees the same filtered joint_vel in sim and on hardware. Noise is internal to the term (`noise=None` on the ObsTerm); PLAY zeroes `noise_amp`.
- **Commands** (6-dim: lin_x, lin_y, ang_z, pitch, lean, height): lin_vel_x [-1,1], lin_vel_y [-1,1], ang_vel_z [-1,1], heading [-pi,pi], pitch (`ang_pos_y`) [-0.6,0.6], lean (`ang_pos_x`) [-0.3,0.3], height [0.28,0.38]m. **Height re-added as a command (obs_stacking_with_height):** it is a *command and a reward target only* — the policy observes the COMMANDED height (known on the robot) and is rewarded against the simulator's ground-truth base height (`root_pos_w[:,2]`), but actual height is **never** an observation (no reliable hardware height sensor), so it stays deployable and open-loop. Height is NOT zeroed for standing envs (a standing robot still holds a stance height); the hard stand-still gate (`rewards._any_commanded`) reads only command columns 0-4, so a non-zero height can't un-gate it. See `mdp/commands.py` and `mdp/height_command.py`.
- **Actions**: joint position control (scale=0.25, default offset), no clipping. Each applied target lags the policy output by 0-2 control steps (random per env per episode, `mdp/actions.py` `DelayedJointPositionAction`); `last_action` stays the policy's own undelayed output
- **Real-hardware parity (2026-10-13)**: PhysX's hard joint-speed clamp is DISABLED (`PHYSX_JOINT_SPEED_LIMIT=500` in `unitree_go2witharm_cfg.py`; before this every policy was trained with a hidden clamp of 30.1 rad/s hip/thigh and 15.7 calf that the real robot and MuJoCo do not have, so **all policies trained before 2026-10-13 are not valid for hardware**). Nominal joint armature 0.01 (`JOINT_ARMATURE`, an assumption until measured) plus startup randomisation in `EventCfg`: friction static 0.5-1.2 / dynamic 0.4-1.0, base mass -1..+2 kg, PD gains x0.9-1.1, armature x0.5-2.0, joint friction 0-0.1 Nm, pushes +/-0.5 m/s every 8-15 s. iter4 realistic (2026-11-08): disturbance magnitudes reduced to realistic sim-to-real levels (external force +-40->+-15 N, torque +-5->+-2 Nm, base mass +-5->+-2 kg, COM/armature tightened), difficulty step 0.05->0.025 (slower ramp), and posture weights raised (height_penalty -15, stand_still 5.0, fall penalty -750); push and action delay scale with the difficulty curriculum. `QuadrupedLocomotionEnvCfg_PLAY` turns all of that off (nominal physics, no delay) so play and sim2sim compare like for like. No safety layer is in training by design; any robot-side capping is added in deploy, not here. **obs_stacking_with_height (2026-11): kp/kd raised 40/0.8 -> 50/1.0** (matches the Glowing-Torch reference stand gain; the higher kd damps the buzz), and the actuator-gain DR band widened x0.9-1.1 -> x0.8-1.2 so a kd~1.0 deploy is in-distribution. Deploy gains must match the policy in use — the current kp40 policy keeps `deploy/profiles/go2_locomotion.yaml` (kp40), the new kp50 policy uses `go2_obs_stacking.yaml` (kp50/kd1.0).
- **Rewards**: track_lin_vel_xy_exp(10.0), track_ang_vel_z_exp(2.0), track_pitch_exp(3.0, std 0.2), track_lean_exp(3.0, std 0.141), lin_vel_z_l2(-2.0), ang_vel_xy_l2(-0.05), dof_torques_l2(-1e-5), dof_acc_l2(-2.5e-7), action_rate_l2(-0.01), undesired_contacts(-1.0), undesired_contacts_arm(-1.0), height_penalty(-15.0 from iter4 realistic, L1 abs-error, target_height=0.337), hip_crossing(-0.5), foot_sliding(-1.0, on contact-point velocity), foot_lift(0.1), gait_coordination(1.0, dense trot shaping, scaled by command magnitude), stand_still(5.0 from iter4 realistic, zero velocity + four feet planted in default pose, gated to a fully zero command), dof_pos_limits(-1.0), joint_deviation(-0.005). `patches/round3_weights_and_height.patch` no longer applies (its height hunks conflict with the height-command removal); kept as a historical artifact only
- **Terminations**: time_out (40s), body_contact (base touch >1.0N), terrain_out_of_bounds (a truncation; the terrain border is 70m so it should never fire)
- **Standing envs**: rel_standing_envs=0.12 (zero command on every component)
- **PPO**: 10000 max iterations, 24 steps/env, lr=1e-3, gamma=0.99, hidden=[256,512,128], elu, checkpoint every 100 iterations

## Architecture

### Environment Config (`quadruped_go2_locomotion_env_cfg.py`)
- `QuadrupedLocomotionSceneCfg` — scene entities (terrain, robot, contact sensors, sky light)
- `QuadrupedLocomotionEnvCfg` — main training config, inherits `ManagerBasedRLEnvCfg`, sets scene/observations/actions/commands/rewards/terminations/events/curriculum
- `QuadrupedLocomotionEnvCfg_PLAY` — play config: 50 envs, no observation corruption, closer viewer

### MDP Module (`mdp/`)
- `commands.py` — `UniformVelocityCommandCfgWithPitch` extends isaac-lab's `UniformVelocityCommand` to include pitch/lean angle components (command buffer shape: [N, 6])
- `rewards.py` — custom Isaac Lab MDP reward terms for pitch tracking, lean tracking, adaptive height penalty (accounts for pitch/lean COM shift), and hip-crossing penalty
- `observations.py` — depth observation term (sampled from tiled camera output)
- `curriculums.py` — terrain level progression based on commanded vs. actual distance

### Agent Config (`agents/rsl_rl_ppo_cfg.py`)
- PPO actor-critic network configs and algorithm hyperparameters
- All standard RSL-RL settings (clip, entropy, learning rate schedule, etc.)

### Gym Registration (`tasks/manager_based/quadruped_go2_locomotion/__init__.py`)
- Registers `Quadruped-Locomotion-Go2` (training) and `Quadruped-Locomotion-Go2-Play` (playback) via `gym.register()`
- Both point to `ManagerBasedRLEnv` as the entry point

## Code Style

- Line length: 120 (ruff)
- Python target: 3.10+
- Ruff config: ruff linter (E, W, F, I, UP, C92, SIM, RET) + pyright type checking
- Pre-commit: ruff (fix + format), codespell, license headers, trailing-whitespace
- Import order: future -> standard-library -> third-party -> omniverse-extensions -> isaaclab -> isaaclab-contrib -> isaaclab-rl -> isaaclab-tasks -> first-party -> local-folder
- Google-style docstrings
- All files carry BSD-3-Clause Isaac Lab license header

## Working with Rewards

Reward terms in `mdp/rewards.py` follow the Isaac Lab MDP pattern: `def reward_term(env: ManagerBasedRLEnv, ...) -> torch.Tensor`. Each returns a tensor of shape `(num_envs,)`. To add or modify rewards, register the term in `RewardsCfg` within `quadruped_go2_locomotion_env_cfg.py`.

## Working with Commands

The pitch/lean command extension in `mdp/commands.py` adds components [3] (pitch) and [4] (lean) to the base command buffer (5-dim total). There is no height component: see that file's module docstring for why, and `mdp/rewards.py`'s `base_height_l1_pitch` for how height is regulated instead (a fixed `target_height`, not a command). The robot should lean forward/backward while walking toward the sampled pitch angle. When modifying commands, ensure observation terms (e.g., `velocity_commands`) correctly feed the command values to the policy.

## Training Artifacts

- Checkpoints saved to `logs/rsl_rl/<experiment_name>/<timestamp>/model_<N>.pt`
- Exported policy (JIT + ONNX) in `logs/rsl_rl/<experiment_name>/<timestamp>/exported/policy.pt`
- Config dumps in `logs/rsl_rl/<experiment_name>/<timestamp>/params/env.yaml` and `params/agent.yaml`
- `curriculum_<N>.pt` next to every `model_<N>.pt` (written by `train.py`): the domain-randomisation `difficulty_scale` and the per-env terrain levels. They live in the env process, not in the RSL-RL checkpoint, so restoring them on `--resume` is what keeps a restarted process (batched training) from silently restarting the curriculum at difficulty 0. A checkpoint without one (older runs) prints a warning and the curriculum restarts
- Batched runs (`scripts/train_batched.sh`): RSL-RL names a batch's last checkpoint `model_{end-1}` (`model_1999`, `model_3999`, ...) and a resume restarts at the loaded checkpoint's number, so one iteration repeats per batch. The adaptive learning rate and the logged episode-reward buffers also restart each batch (a short transient in `Train/mean_reward` right after each seam, not a regression)

## Git updates

- Git adds / commits and pushes are only to be implemented after any code changes are tested and confirmed