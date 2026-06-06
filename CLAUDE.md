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
source/quadruped_locomotion/          # Isaac Lab extension package
  quadruped_locomotion/
    tasks/manager_based/quadruped_locomotion/
      quadruped_locomotion_env_cfg.py  # Main env config (scene, MDP, rewards, terminations)
      __init__.py                      # Gym registration (Go2, Go2-Play)
      agents/rsl_rl_ppo_cfg.py        # PPO hyperparameters
      mdp/
        commands.py                    # UniformVelocityCommandCfgWithPitch (extends isaac lab velocity commands with pitch/lean)
        rewards.py                     # Custom rewards: track_pitch_exp, track_lean_exp, base_height_l2_pitch, hip_crossing_l2
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
/home/gavin/IsaacLab/_isaac_sim/python.sh -c "
import torch
import importlib.util as iu
spec = iu.spec_from_file_location(
    'math_utils',
    '/home/gavin/IsaacLab/source/isaaclab/isaaclab/utils/math.py')
math_utils = iu.module_from_spec(spec)
spec.loader.exec_module(math_utils)

q = math_utils.quat_from_euler_xyz(
    torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0))
print(q)
"
```

Or add a shell alias to `~/.bashrc` / `~/.zshrc`:

```bash
alias isaaclab-python='/home/gavin/IsaacLab/_isaac_sim/python.sh'
```

Then use the same pattern: `isaaclab-python -c "..."`

> **Note:** `pxr` (USD) requires the Omniverse Kit runtime and cannot be imported without the full simulator (`isaaclab.sh -p` with an active sim). For pure-Python modules like `torch` and `numpy`, `_isaac_sim/python.sh -c "..."` works instantly. For `isaaclab` utilities, use `iu.spec_from_file_location` to load the module file directly — `import isaaclab.utils.math` triggers `isaaclab.__init__` which pulls in `pxr`.

## Common Commands

```bash
# Install the extension (requires Isaac Lab Python env)
../IsaacLab/isaaclab.sh -p -m pip install -e source/quadruped_locomotion

# List available environments
../IsaacLab/isaaclab.sh -p scripts/list_envs.py

# Train with RSL-RL (PPO)
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --headless

# Resume training
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --resume --headless

# Play a trained checkpoint
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1

# Validate environment wiring (zero action)
../IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Quadruped-Locomotion-Go2

# Validate environment wiring (random action)
../IsaacLab/isaaclab.sh -p scripts/random_agent.py --task=Quadruped-Locomotion-Go2
```

## Training Configuration

- **Environment**: `QuadrupedLocomotionEnvCfg` (1024 envs, 40s episodes, decimation=4, dt=0.005s)
- **Robot**: Unitree GO2WITHARM (from `isaaclab_assets`)
- **Terrain**: 8x8m generator with flat (20%) + random_rough (20%), 9x21 grid
- **Observations**: base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, last_action (with additive uniform noise)
- **Commands**: lin_vel_x [-1,1], lin_vel_y [-1,1], ang_vel_z [-1,1], heading [-pi,pi], pitch [-0.3,0.3], lean [-0.6,0.6]
- **Actions**: joint position control (scale=0.25, default offset)
- **Rewards**: track_lin_vel_xy_exp(1.0), track_ang_vel_z_exp(0.5), track_pitch_exp(0.5), track_lean_exp(0.3), lin_vel_z_l2(-2.0), ang_vel_xy_l2(-0.05), dof_torques_l2(-1e-5), dof_acc_l2(-2.5e-7), action_rate_l2(-0.01), undesired_contacts(-1.0), height_penalty(-1.65), hip_crossing_l2(-0.5), joint_deviation_l1(-0.005), joint_pos_limits(-1.0)
- **Terminations**: time_out (40s), body_contact (base touch >1.0N)
- **PPO**: 30000 max iterations, 24 steps/env, lr=1e-3, gamma=0.99, hidden=[256,256,128], elu

## Architecture

### Environment Config (`quadruped_locomotion_env_cfg.py`)
- `QuadrupedLocomotionSceneCfg` — scene entities (terrain, robot, contact sensors, sky light)
- `QuadrupedLocomotionEnvCfg` — main training config, inherits `ManagerBasedRLEnvCfg`, sets scene/observations/actions/commands/rewards/terminations/events/curriculum
- `QuadrupedLocomotionEnvCfg_PLAY` — play config: 50 envs, no observation corruption, closer viewer

### MDP Module (`mdp/`)
- `commands.py` — `UniformVelocityCommandCfgWithPitch` extends isaac-lab's `UniformVelocityCommand` to include pitch/lean angle components (command buffer shape: [N, 5])
- `rewards.py` — custom Isaac Lab MDP reward terms for pitch tracking, lean tracking, adaptive height penalty (accounts for pitch/lean COM shift), and hip-crossing penalty
- `observations.py` — depth observation term (sampled from tiled camera output)
- `curriculums.py` — terrain level progression based on commanded vs. actual distance

### Agent Config (`agents/rsl_rl_ppo_cfg.py`)
- PPO actor-critic network configs and algorithm hyperparameters
- All standard RSL-RL settings (clip, entropy, learning rate schedule, etc.)

### Gym Registration (`tasks/manager_based/quadruped_locomotion/__init__.py`)
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

Reward terms in `mdp/rewards.py` follow the Isaac Lab MDP pattern: `def reward_term(env: ManagerBasedRLEnv, ...) -> torch.Tensor`. Each returns a tensor of shape `(num_envs,)`. To add or modify rewards, register the term in `RewardsCfg` within `quadruped_locomotion_env_cfg.py`.

## Working with Commands

The pitch/lean command extension in `mdp/commands.py` adds components [3] and [4] to the base command buffer. The robot should lean forward/backward while walking toward the sampled pitch angle. When modifying commands, ensure observation terms (e.g., `velocity_commands`) correctly feed the command values to the policy.

## Training Artifacts

- Checkpoints saved to `logs/rsl_rl/<experiment_name>/<timestamp>/model_<N>.pt`
- Exported policy (JIT + ONNX) in `logs/rsl_rl/<experiment_name>/<timestamp>/exported/policy.pt`
- Config dumps in `logs/rsl_rl/<experiment_name>/<timestamp>/params/env.yaml` and `params/agent.yaml`
