# Quadruped Go2 Locomotion

[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-76b900.svg)](https://developer.nvidia.com/isaac/sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v2.3.2-76b900.svg)](https://isaac-sim.github.io/IsaacLab/v2.3.2/)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-3.1.2-orange.svg)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)

Reinforcement-learning locomotion policies for the **Unitree Go2** (with a static OpenManipulator-X mounted on its
head), trained with **RSL-RL's PPO** in **NVIDIA Isaac Lab**, cross-checked in **MuJoCo**, and deployable to the real
robot.

<!-- Add a demo GIF here, e.g.: ![Go2 walking with pitch/lean/height commands](docs/media/demo.gif) -->

The policy takes a velocity command (forward, sideways, yaw rate) together with a **body pitch, lean and height**
command, and tracks all of them while walking.

## What this project is (and is not)

- **The learning algorithm is not ours.** Training uses the stock **PPO implementation from
  [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (`rsl-rl-lib` 3.1.2)**, unmodified: `OnPolicyRunner` with a
  feed-forward actor-critic MLP, driven through Isaac Lab's `isaaclab_rl` RSL-RL wrapper. This repository does not
  change the algorithm. RSL-RL's extras (RND, symmetry augmentation, recurrent policies, student-teacher
  distillation) are **not used**.
- **What this repository provides** is everything around it:
  - the **`Quadruped-Locomotion-Go2` task**: the Go2 + arm robot model, commands, observations, terminations,
    domain randomization and a difficulty curriculum
  - **custom reward terms** (table below)
  - **profile-driven, phased training**: one command, with the number of phases and everything else set in a YAML profile (the latest recipe is a "clean" warm-up followed by "realistic" disturbances), run in resumable chunks
  - a **MuJoCo sim2sim** runner to check policies outside Isaac
  - **real-robot deployment** with layered safety checks
  - a Docker workflow and curated, documented policies

## Features

- Velocity **plus pitch, lean and height** command tracking (6-dimensional command).
- **Custom rewards** for command tracking, posture, gait, standing and smoothness.
- **Configurable phased training**: a profile declares one phase or several. The latest recipe forms a stable gait
  under mild disturbances first, then introduces the full realistic disturbance set. Curriculum state survives restarts.
- **6-frame observation history** (actor input 48 x 6 = 288), inspired by
  [Glowing-Torch's Go2 deployment project](https://github.com/Glowing-Torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot),
  and hardware-realistic `joint_vel` noise with a 10 Hz [low-pass filter](#what-is-the-low-pass-filter), the same
  filter used on the robot.
- **Sim-to-real parity**: real per-joint torque/speed limits, DC-motor actuator model, no hidden joint-speed clamp,
  randomized friction, mass, PD gains, armature, pushes and action delay.
- **Cross-simulator check**: replay the same random command schedule in Isaac Sim and MuJoCo and compare.
- **Deployment** over `unitree_sdk2py` (wired or relayed over WiFi), with a staged safety checklist.

## What is the low-pass filter?

A low-pass filter smooths a signal: slow changes pass through, fast jitter is removed. Real joint-velocity readings are
noisy, and a network that sees raw noise can amplify it into visible shaking of the legs.

This project filters the **joint-velocity observation** with a first-order exponential moving average:

```text
y = y + alpha * (x - y),    alpha = 1 - exp(-2 * pi * f_c * dt)
```

Here `x` is the new reading, `y` the filtered value, `f_c` the cutoff frequency (10 Hz) and `dt` the control period
(0.02 s, i.e. 50 Hz). Lower cutoffs smooth more. Training adds realistic noise to the velocity and then applies this
filter, and the deployment code applies the same filter to the real readings, so the policy sees the same kind of
signal in simulation and on the robot.

The deploy script also has a separate, optional filter on the policy's **actions**, enabled with `--low_pass` (5 Hz by
default). It was a workaround for a ~25 Hz buzz seen with the single-frame policy. It is off by default and not needed
for the stacked policy.

## Status

| Policy (`models/`) | Isaac lin_x / lin_y / ang_z RMS | MuJoCo lin_x / lin_y / ang_z RMS | Falls (of 60 segments, Isaac / MuJoCo) | Hardware |
|---|---|---|---|---|
| [`11_18_26_iter4_realistic`](models/11_18_26_iter4_realistic/SUMMARY.md) (single frame, kp 40) | 0.049 / 0.049 / 0.195 | 0.051 / 0.045 / 0.222 | 0 / 0 | **Walks.** A ~25 Hz buzz was mitigated with a deploy-side action low-pass. |
| [`11_24_26_obs_stacking_with_height`](models/11_24_26_obs_stacking_with_height/SUMMARY.md) (6-frame, kp 50) | 0.044 / 0.043 / 0.133 | 0.071 / 0.079 / 0.181 | 1 / 1 | **Not tested.** Yaw improved, but MuJoCo linear tracking is worse than Isaac, the height command does not reach 0.38 m, and the observation-layout parity check has not passed. |

RMS tracking errors are in m/s (linear) and rad/s (yaw) from `src/sim2sim/run_sim2sim_eval.sh` (3 seeds x 20 random
command segments, nominal physics). Older policies and per-policy details are in [`models/`](models/README.md).

> **Research code.** Nothing here is a certified safety system. Read
> [`src/deploy/Checklist.md`](src/deploy/Checklist.md) before running anything on a real robot.

## Requirements

Setup is Docker-based: you only need the host side below. Everything else is built into the image.

- Linux (developed on Ubuntu 24.04) with an NVIDIA RTX GPU and a recent NVIDIA driver (developed on an RTX A4500, 20 GB)
- Docker and the NVIDIA Container Toolkit
- A free NVIDIA NGC account (to pull the Isaac Sim base image)

**Included in the Docker image** (nothing to install yourself): Isaac Sim 5.1.0, Isaac Lab v2.3.2, Python 3.11,
`rsl-rl-lib` 3.1.2, MuJoCo 3.15.0 (sim2sim), and `unitree_sdk2py` with CycloneDDS 0.10.x (real-robot deployment).

## Installation

The image contains Isaac Sim, Isaac Lab and the deployment dependencies, and this repository is bind-mounted into the
container, so editing code never needs a rebuild. Troubleshooting for every step is in
[`docker/README.md`](docker/README.md).

### 1. Get the code

```bash
git clone https://github.com/topguns837/Quadruped_Go2_Locomotion.git
cd Quadruped_Go2_Locomotion
```

### 2. Log in to NVIDIA NGC (once)

The base image is pulled from NVIDIA's registry. Create a free account at [ngc.nvidia.com](https://ngc.nvidia.com), open the
[Isaac Sim container page](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim) and click "Get Container" once
to accept the license, then generate an API key (profile menu, Setup, Generate API Key) and log in:

```bash
docker login nvcr.io -u '$oauthtoken' -p '<YOUR_NGC_API_KEY>'
```

### 3. Build the Docker image

Start the menu and choose **2) Build docker image**:

```bash
./start.sh
```

The same build can be run directly from the repository root:

```bash
docker compose build
```

The first build downloads several gigabytes and takes roughly 15 to 30 minutes. It installs Isaac Sim 5.1.0, Isaac Lab
v2.3.2, RSL-RL, MuJoCo and the Unitree SDK. You only need to build again when the Isaac Sim or Isaac Lab version pins
in `docker-compose.yml` change, not when you edit project code.

### 4. Open the Isaac Lab shell

Run the menu again and choose **1) Open Stack**:

```bash
./start.sh
```

This starts the container and opens a tmux session with one window per task (`list-envs`, `train`, `play`, `hardware`,
`sim2sim`), each with its command typed in but not run. Choosing a menu option recreates the container, so a job running
in an earlier session is stopped.

Everything in the Usage section below is run **inside this container, from the repository root**
(`/workspace/quadruped_go2_locomotion`). The alias `isaaclab` means `/workspace/isaaclab/isaaclab.sh -p`, `python` means
the Isaac Sim Python, and the `quadruped_go2_locomotion` extension is already installed.

## Usage

All commands run inside the Docker container, from the repository root (see [Installation](#installation)).
`isaaclab -p <script>` runs a script under Isaac Lab's Python. `python` is the plain Isaac Sim Python used by the MuJoCo
and deployment tools, which do not need the Isaac Sim application.

### 1. Check the setup

> **tmux pane:** go to the `list-envs` window. `list_envs.py` is already typed in; press Enter to run it. The zero and
> random agent commands below are not pre-typed, so paste them into the same window.

List the registered tasks. Both `Quadruped-Locomotion-Go2` and `Quadruped-Locomotion-Go2-Play` should appear:

```bash
isaaclab -p src/env/tools/list_envs.py
```

Step the environment with zero actions, then with random actions, to confirm it loads and runs:

```bash
isaaclab -p src/env/tools/zero_agent.py --task=Quadruped-Locomotion-Go2
```

```bash
isaaclab -p src/env/tools/random_agent.py --task=Quadruped-Locomotion-Go2
```

### 2. Train a policy

> **tmux pane:** go to the `train` window. The training command is already typed in; press Enter to run it. It is
> `train.sh --profile obs_stacking_with_height` (the two-phase recipe) on a capable GPU. On a weak GPU the window
> instead has the single-phase profile with `--gui --num_envs 1 --rendering_mode performance` and a small cached
> terrain, so you can watch one robot. Edit the line before pressing Enter to change the profile or add any option
> from the table below. All profiles are in [`src/training/profiles/`](src/training/profiles/).

`src/training/train.sh` is the only command you need. It reads a **training profile**, a YAML file in
[`src/training/profiles/`](src/training/profiles/) that sets everything about the run: how many phases, each phase's
iterations and disturbance settings, reward weights, PPO settings, the number of environments, and whether to show the
Isaac Sim window.

#### Choose a profile

| Profile | Phases | Use |
|---|---:|---|
| [`obs_stacking_with_height.yaml`](src/training/profiles/obs_stacking_with_height.yaml) | 2 | the latest recipe: a `clean` warm-up (10k iterations), then `realistic` (15k) |
| [`obs_stacking_with_height_single_phase.yaml`](src/training/profiles/obs_stacking_with_height_single_phase.yaml) | 1 | the same recipe as one `realistic` phase (25k) |
| [`iter4_realistic.yaml`](src/training/profiles/iter4_realistic.yaml) | 2 | the single-frame recipe behind the hardware-tested policy |
| [`example.yaml`](src/training/profiles/example.yaml) | 1 | annotated template; copy it to start your own recipe |
| [`aggressive.yaml`](src/training/profiles/aggressive.yaml) | 2 | the earlier high-disturbance recipe (40 N pushes), kept for reproducibility; not recommended for hardware |

#### Preview the commands

A dry run prints the command for each phase and launches nothing:

```bash
bash src/training/train.sh --profile obs_stacking_with_height --dry_run
```

#### Start training

```bash
bash src/training/train.sh --profile obs_stacking_with_height --num_envs 2048
```

Phases run back to back in 2000-iteration chunks, each in a fresh process so memory is released between them. All
phases write to one run folder, `logs/rsl_rl/<experiment_name>/<run_dir>/`, which holds the checkpoints
(`model_<N>.pt`), TensorBoard logs and config dumps (`params/`). A live training graph opens when a display is available.
If a run stops, run the same command again: it continues from the latest checkpoint in that folder.

#### Single-phase and multi-phase training

Every profile starts with `num_phases`, followed by one entry per phase under `phases:`. The two must agree, or the profile
is rejected with an error naming both numbers.

```yaml
num_phases: 2
phases:
  - name: clean
    iterations: 10000
  - name: realistic
    iterations: 15000
```

For a single-phase run, use a profile with `num_phases: 1`, such as the single-phase profile above. To add a phase to
your own profile, raise `num_phases` by one and append an entry. Phase k resumes phase k-1 in the same run folder, so the
curriculum and optimiser carry over. To run only the first K phases of a profile without editing it:

```bash
bash src/training/train.sh --profile obs_stacking_with_height --num_phases 1
```

A stacked-observation policy cannot resume a single-frame checkpoint, because the actor input size differs. Changing
`HISTORY_LENGTH` means training from scratch.

#### Options

| Flag | Meaning |
|---|---|
| `--num_envs N` | parallel environments for all phases (profile default: 2048) |
| `--seed N` | random seed for all phases |
| `--num_phases K` | run only the first K phases |
| `--start_phase N` / `--only_phase N` | start at, or run only, phase N (1-based) |
| `--gui` | open the Isaac Sim window instead of running headless |
| `--rendering_mode M` | `performance`, `balanced` or `quality` |
| `--livestream N` | `1` or `2`: stream the viewport |
| `--no_dashboard`, `--no_viewer` | do not start the live graphs |
| `env.<path>=<value>`, `agent.<path>=<value>` | trailing Hydra overrides applied to every phase |

Each of these, except the overrides, can also be set in the profile. See
[`src/training/README.md`](src/training/README.md) for the full list of profile keys.

Watch a short run in a window with one environment:

```bash
bash src/training/train.sh --profile obs_stacking_with_height_single_phase --gui --num_envs 1 --rendering_mode performance
```

Change a reward weight for one run without editing the profile:

```bash
bash src/training/train.sh --profile obs_stacking_with_height env.rewards.track_ang_vel_z_exp.weight=3.0
```

Reward terms, the observation layout and the PPO settings are defined in code, not in the profile:
[`env_cfg.py`](src/env/quadruped_go2_locomotion/env_cfg.py),
[`mdp/rewards.py`](src/env/quadruped_go2_locomotion/mdp/rewards.py) and
[`agents/rsl_rl_ppo_cfg.py`](src/env/quadruped_go2_locomotion/agents/rsl_rl_ppo_cfg.py).

### 3. Play and evaluate a policy in Isaac Sim

> **tmux pane:** go to the `play` window. `play.py` is already typed in with `--manual_commands` (a slider window for
> velocity, pitch, lean and height) and `--live_plot` (graphs). Add `--checkpoint <path>` before pressing Enter to pick
> a policy. The panes beside it show the live dashboard and its viewer.

Watch a checkpoint, with a slider window to command velocity, pitch, lean and height:

```bash
isaaclab -p src/play/play.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1 \
  --checkpoint models/11_24_26_obs_stacking_with_height/model_24999.pt --manual_commands --live_plot
```

`play.py` also writes `exported/policy.pt` and `policy.onnx` next to the checkpoint it loads; `git checkout` them
afterwards if you pointed it at a committed `models/` checkpoint.

To compare Isaac and MuJoCo tracking for a checkpoint on a seeded random command schedule, use the sim2sim eval in
step 4 (`run_sim2sim_eval.sh`).

### 4. Cross-check in MuJoCo (sim2sim)

> **tmux pane:** go to the `sim2sim` window. The top-left pane has the model download plus `sim2sim_mujoco.py` already
> typed in, using the deploy profile [`go2_obs_stacking.yaml`](src/deploy/profiles/go2_obs_stacking.yaml) via
> `--config` (all deploy profiles are in [`src/deploy/profiles/`](src/deploy/profiles/)). Another pane holds the Isaac
> eval command, and the right panes show the live dashboard.

Download the MuJoCo Go2 model once (it is not baked into the image), then run a scripted scenario with an exported
policy:

```bash
./src/sim2sim/fetch_go2_model.sh

python src/sim2sim/sim2sim_mujoco.py --headless \
  --config src/deploy/profiles/go2_obs_stacking.yaml \
  --scenario src/sim2sim/scenarios/basic.yaml
```

`--config` is a [deploy profile](src/deploy/profiles/) (it supplies the policy path, PD gains, filter and stack
depth); `--scenario` is a command schedule from [`src/sim2sim/scenarios/`](src/sim2sim/scenarios/) (`basic.yaml`, and
`height_check.yaml` for the height command). Use `--manual_commands` instead of `--scenario` for a slider window.

To compare both simulators on the **same seeded random command schedule** (writes a side-by-side `report.md` under
`logs/sim2sim_eval_<N>/random/`):

```bash
bash src/sim2sim/run_sim2sim_eval.sh \
  --checkpoint models/11_24_26_obs_stacking_with_height/model_24999.pt \
  --config src/deploy/profiles/go2_obs_stacking.yaml
```

Options: `--seeds "0 1 2"`, `--segments 20`, `--segment_s 4`, `--scale 0.5` (fraction of the training command range).
Details: [`src/sim2sim/README.md`](src/sim2sim/README.md).

### 5. Deploy to the real robot

> **tmux pane:** go to the `hardware` window. `deploy_real.py` is already typed in with `--deploy_profile
> go2_obs_stacking`, `--safety_profile confident` and **`--dry_run`**, so pressing Enter does not move the motors. The
> pane below it holds the pre-flight check. Remove `--dry_run` only after following the checklist below. The profile
> files are described under the table.

> **Warning.** A policy that works in simulation can still fall or damage the robot. Read
> [`src/deploy/README.md`](src/deploy/README.md) and follow [`src/deploy/Checklist.md`](src/deploy/Checklist.md):
> suspend the robot first, dry-run first, keep an operator on the controller's stop.

Two independent profiles choose what runs:

| Flag | Chooses | Files |
|---|---|---|
| `--deploy_profile <name>` | the **policy recipe**: policy file, PD gains, joint-velocity filter and the **observation stack depth** (1 = single frame, 6 = stacked) | [`go2_locomotion.yaml`](src/deploy/profiles/go2_locomotion.yaml) (single frame, kp 40), [`go2_obs_stacking.yaml`](src/deploy/profiles/go2_obs_stacking.yaml) (6 frames, kp 50) |
| `--safety_profile <name>` | **how cautious the session is**: command slew, tethers, abort thresholds | [`safety_profiles.yaml`](src/deploy/configs/safety_profiles.yaml): `cautionary`, `confident`, `no_limits`, `iter4_realistic`, `wireless` |

Dry run (no motors; `lo` is the loopback network interface, so nothing reaches a robot):

```bash
python src/deploy/deploy_real.py --deploy_profile go2_obs_stacking --safety_profile cautionary \
  --network_interface lo --skip_release --dry_run
```

On the robot, replace `lo` with the host network interface connected to it, drop `--skip_release`, and start from
`--safety_profile cautionary`. Add `--manual_commands` for a slider window, and `--live_plot --live_dashboard` for
live graphs:

```bash
python src/deploy/deploy_real.py --deploy_profile go2_obs_stacking --safety_profile cautionary \
  --network_interface <nic> --manual_commands --live_plot --live_dashboard
```

A **wireless** setup (policy on the desktop, actions relayed through a Jetson on the robot) is described in
[`src/deploy/wireless/README.md`](src/deploy/wireless/README.md).

### 6. Use the tmux workspace

`./start.sh` -> "Open Isaac Lab shell" opens a tmux session with five windows (`list-envs`, `train`, `play`,
`hardware`, `sim2sim`). Each has its command pre-typed but **not executed**, as described at the top of steps 1 to 5
above; press Enter to run one, or edit it first. The commands are the `*_CMD` variables in
[`docker/isaaclab-shell.sh`](docker/isaaclab-shell.sh).

### Configuration files

| What you want to change | File |
|---|---|
| Training recipe: reward weights, disturbances, PPO overrides, phases | [`src/training/profiles/`](src/training/profiles/) |
| Reward functions, observations, commands, curriculum, terminations | [`src/env/quadruped_go2_locomotion/env_cfg.py`](src/env/quadruped_go2_locomotion/env_cfg.py), [`mdp/`](src/env/quadruped_go2_locomotion/mdp/) |
| Motor limits, kp/kd, USD asset | [`src/env/quadruped_go2_locomotion/robot_cfg.py`](src/env/quadruped_go2_locomotion/robot_cfg.py) |
| PPO algorithm and network settings | [`src/env/quadruped_go2_locomotion/agents/rsl_rl_ppo_cfg.py`](src/env/quadruped_go2_locomotion/agents/rsl_rl_ppo_cfg.py) |
| Which policy / stack depth the robot or MuJoCo runs | [`src/deploy/profiles/`](src/deploy/profiles/) |
| Hardware caution levels | [`src/deploy/configs/safety_profiles.yaml`](src/deploy/configs/safety_profiles.yaml) |
| sim2sim command schedules | [`src/sim2sim/scenarios/`](src/sim2sim/scenarios/) |
| The tmux workspace commands | [`docker/isaaclab-shell.sh`](docker/isaaclab-shell.sh) |

## Repository layout

All code lives in `src/`, one folder per stage. A file lives in the stage that runs it; if two or more stages run it,
it lives in `src/common/`.

| Folder | What it is | Change this when you want to... |
|---|---|---|
| `src/env/` | The robot + task as an Isaac Lab extension (`quadruped_go2_locomotion/`: `env_cfg.py`, `robot_cfg.py`, `mdp/`, `agents/`, `resources/`) | change rewards, observations, commands, curriculum, kp/kd, PPO settings |
| `src/training/` | `train.sh` (the entry point), its drivers, and `profiles/` (training recipes) | start or tune a training run |
| `src/play/` | `play.py` (plus `eval_velocity_tracking.py` and `verify_obs_stacking.py`, see [`src/play/README.md`](src/play/README.md)) | watch/evaluate a policy in Isaac Sim |
| `src/sim2sim/` | MuJoCo runner, random-command eval, `scenarios/` | cross-check a policy outside Isaac |
| `src/deploy/` | Real-robot runner, `profiles/` (per-policy recipe), `configs/safety_profiles.yaml`, `wireless/` | run a policy on the Go2 |
| `src/common/` | `cli_args.py`, `live_dashboard.py`, `manual_command_slider.py` | shared UI/CLI helpers |
| `models/` | Curated policies, each with a `SUMMARY.md` | keep a finished policy |
| `docker/`, `start.sh` | Container workflow | build/open the dev container |
| `docs/` | Design notes; `docs/archive/` holds historical artifacts | read the reasoning behind changes |

## Method

### Task

| | |
|---|---|
| Control | 50 Hz (sim dt 0.005 s, decimation 4); 40 s episodes |
| Command (6-dim) | lin_x, lin_y in [-1, 1] m/s; ang_z in [-1, 1] rad/s; pitch in [-0.6, 0.6] rad; lean in [-0.3, 0.3] rad; **height** in [0.28, 0.38] m. 20 % of environments get an all-zero (standing) command. |
| Actor observation | per frame (48): base angular velocity 3, projected gravity 3, command 6, joint position 12, joint velocity 12, last action 12. **6 frames stacked = 288.** The commanded height is observed; the actual height never is (no reliable hardware sensor), so the policy runs open-loop in height. The 6-frame history length follows [Glowing-Torch's Go2 deployment project](https://github.com/Glowing-Torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot); the stacking itself uses Isaac Lab's observation-group history. |
| Critic observation | actor terms plus privileged base linear velocity (51 per frame, 306 stacked); discarded at deployment (asymmetric actor-critic) |
| Action | 12 joint-position offsets from the default pose (scale 0.25), applied with a random 0-2 step delay |
| Terminations | time-out (40 s), base contact, out of terrain bounds |

### Robot

Go2 with a static OpenManipulator-X mounted on its head (the USD is committed at
`src/env/quadruped_go2_locomotion/resources/go2withArm/`). The calf is a different motor from the hip and thigh, so the
actuators are two DC-motor groups (`src/env/quadruped_go2_locomotion/robot_cfg.py`):

| Group | Joints | Peak torque | Max speed | kp | kd |
|---|---|---|---|---|---|
| `hip_thigh` | hip, thigh | 23.7 Nm | 30.1 rad/s | 50 | 1.0 |
| `calves` | calf (1.9169:1 knee reduction) | 45.43 Nm | 15.70 rad/s | 50 | 1.0 |

PhysX's own joint-speed clamp is disabled (it would impose a limit the real robot and MuJoCo do not have), and joint
armature is 0.01.

### Reward terms

Weights are those of the latest training profile
(`src/training/profiles/obs_stacking_with_height.yaml`), which overrides the defaults in `env_cfg.py`. "Custom" terms
are defined in [`src/env/quadruped_go2_locomotion/mdp/rewards.py`](src/env/quadruped_go2_locomotion/mdp/rewards.py);
the rest come from Isaac Lab.

| Group | Term | Weight | Source | Intent |
|---|---|---:|---|---|
| Tracking | `track_lin_vel_xy_exp` | 6.0 | custom | tight xy-velocity tracking; kernel width scales with commanded speed |
| | `track_lin_vel_xy_broad` | 4.0 | Isaac Lab | wide xy-velocity kernel, keeps a gradient far from the target |
| | `track_ang_vel_z_exp` | 3.2 | custom | tight yaw-rate tracking (speed-scaled) |
| | `track_ang_vel_z_broad` | 1.6 | Isaac Lab | wide yaw-rate kernel |
| | `track_pitch_exp` / `track_lean_exp` | 3.0 / 3.0 | custom | body pitch and lean tracking |
| Posture | `height_penalty` | -15.0 | custom | L1 error to the commanded height, compensated for pitch/lean |
| | `joint_deviation` | -0.005 | Isaac Lab | stay near the default pose |
| | `hip_crossing` | -0.5 | custom | penalise large hip-roll angles beyond a threshold (legs crossing the midline) |
| Anti-stall | `standing_when_commanded` | -4.0 | custom | penalise standing still while commanded to move |
| | `track_ang_vel_deficit` | -1.0 | custom | penalise under-rotation |
| Standing | `stand_still` | 5.0 | custom | zero velocity, four feet planted, default pose (only when the whole command is zero) |
| Gait | `gait_coordination` | 1.0 | custom | dense diagonal-trot shaping, scaled by command magnitude |
| | `foot_lift` | 0.5 | custom | swing feet must clear a minimum height (stance feet score the maximum) |
| | `foot_sliding` | -1.0 | custom | no slip of planted feet (contact-point velocity) |
| Stability | `lin_vel_z_l2` / `ang_vel_xy_l2` | -2.0 / -0.05 | Isaac Lab | damp vertical and roll/pitch-rate motion |
| Smoothness | `action_rate_l2` | -0.03 | custom | clamped action-rate penalty |
| | `action_jerk_l2` | -0.03 | custom | action jerk |
| | `joint_torque_rate_l2` | -0.001 | custom | torque rate |
| | `dof_torques_l2` / `dof_acc_l2` / `dof_vel_l2` | -3e-5 / -5e-7 / -1e-4 | custom | clamped so one outlier environment cannot dominate a batch |
| Limits | `dof_pos_limits` | -1.0 | Isaac Lab | joint position limits |
| Safety | `undesired_contacts` / `undesired_contacts_arm` | -1.0 / -1.0 | Isaac Lab | body and arm contact |
| | `is_alive` / `termination_penalty` | 2.0 / -750.0 | Isaac Lab | survive; large penalty for falling |

### Training schedule

The latest recipe, [`obs_stacking_with_height.yaml`](src/training/profiles/obs_stacking_with_height.yaml), has two
phases (the single-phase profile runs only the second, for 25 000 iterations). `train.sh` runs them through
`train_profile.py` and `train_batched.sh`:

| Phase | Iterations | Push force / torque | Difficulty step | Purpose |
|---|---:|---|---:|---|
| 1 `clean` | 10 000 | 5 N / 1 Nm | 0.05 | a stable trot forms under mild disturbances |
| 2 `realistic` | 15 000 (same run folder) | 15 N / 2 Nm | 0.025 | full disturbances, slower curriculum ramp |

Phases run in 2000-iteration chunks. The difficulty scale and per-environment terrain levels live in the
environment process, not in the RSL-RL checkpoint, so `train.py` saves them next to every checkpoint
(`curriculum_<N>.pt`) and restores them on resume. The curriculum therefore does not restart at a chunk or phase
boundary. The difficulty scale also scales the pushes, external forces, base-mass offset and action delay.

Domain randomization at full difficulty: friction (static 0.5-1.2, dynamic 0.4-1.0), base mass +-2 kg, centre-of-mass
offset (0.06, 0.05, 0.08) m, PD gains x0.8-1.2, armature x0.75-1.5, joint friction 0-0.1 Nm, velocity pushes of 0.5 m/s
and external force/torque pushes.

### PPO (RSL-RL) settings

From [`agents/rsl_rl_ppo_cfg.py`](src/env/quadruped_go2_locomotion/agents/rsl_rl_ppo_cfg.py):

| Setting | Value |
|---|---|
| Actor / critic network | MLP [256, 512, 128], ELU; initial action noise std 1.0 (log parameterization) |
| Observation normalization | running mean/std (`empirical_normalization=True`; deprecated alias in RSL-RL 3.x for actor and critic observation normalization) |
| Rollout | 24 steps per environment per iteration; 2048 environments in the default profile |
| Optimization | 5 epochs x 4 minibatches; learning rate 1e-3 with the adaptive KL schedule (target KL 0.01); max grad norm 1.0 |
| PPO | clip 0.2 (clipped value loss), value-loss coefficient 0.5, entropy coefficient 0.0075, gamma 0.99, lambda 0.95 |
| Checkpoints | every 100 iterations |

## Pretrained models

Each directory under [`models/`](models/README.md) holds a final checkpoint, JIT/ONNX exports, the config dumps, the
training profile that produced it (when one existed), and a `SUMMARY.md` with sim2sim, sim2real and training metrics.
Several older runs predate the current reward set and the real-hardware-parity changes; their summaries say so.

## Development

Run the unit tests, which sit next to the code they test in `src/<stage>/tests/`:

```bash
isaaclab -p -m pytest
```

Install the git hooks once per clone, then run them on every file when you like:

```bash
pip install pre-commit && pre-commit install
```

```bash
pre-commit run --all-files
```

The hooks run ruff (lint only), whitespace/YAML/TOML checks, codespell, and refuse to commit secrets such as the
access-point config or relay key. `CLAUDE.md` documents the project conventions for AI-assisted development.

### Health check

Before merging, run the repo health check from the host. It starts a separate throwaway container
(`quadruped_go2_locomotion_testing`, your main container is not touched) and runs the basics of every stage: training
profiles (single, multi-phase and edge cases), a tiny real training run, export, play, MuJoCo sim2sim, the Isaac-vs-MuJoCo
eval, and deploy against a synthetic robot. It ends with a pass/fail report and a code-coverage report:

```bash
./healthcheck.sh
```

Test one specific commit without touching your working tree:

```bash
./healthcheck.sh --ref <commit>
```

The report and the coverage report (`htmlcov/index.html`) are written to `logs/healthcheck/<timestamp>_<commit>/`. See
[`src/healthcheck/README.md`](src/healthcheck/README.md) for the groups, the edge-case profiles, and how to add a case.

For VS Code, run the workspace task `setup_python_env` to improve Pylance resolution of Isaac Sim packages.

## Troubleshooting

- **Pylance cannot resolve Isaac/Omniverse modules.** Add `"python.analysis.extraPaths": ["<path-to-repo>/src/env"]`
  to `.vscode/settings.json`. If indexing is too heavy, remove unused Omniverse extension groups from that list.
- **Files in `src/` owned by root after using the container.** The container runs as root against the bind mount.
  Fix with `sudo chown -R "$USER:$USER" src/`.
- **Missing arm payload / asset errors.** See the USD asset section of [`docker/README.md`](docker/README.md).
- **`play.py` modified a committed model's `exported/` files.** It re-exports next to the checkpoint it loads;
  `git checkout -- models/` restores them.
- **A trained stacked policy fails with "not divisible by history_length".** The deploy profile's `history_length`
  must match the policy (6 for the stacked one, 1 otherwise); use `--deploy_profile` so they travel together.

## Citing

If you use this repository, please also cite the work it builds on:

```bibtex
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}

@misc{schulman2017proximal,
  title         = {Proximal Policy Optimization Algorithms},
  author        = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  year          = {2017},
  eprint        = {1707.06347},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mittal, Mayank and others},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}

@misc{rudin2021learning,
  title         = {Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning},
  author        = {Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  year          = {2021},
  eprint        = {2109.11978},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

The Isaac Lab entry above abbreviates the author list; the complete entry is in the
[Isaac Lab README](https://github.com/isaac-sim/IsaacLab#citation).

## Acknowledgements

Built on [NVIDIA Isaac Sim / Isaac Lab](https://github.com/isaac-sim/IsaacLab) (this repository started from the Isaac
Lab extension template), [RSL-RL](https://github.com/leggedrobotics/rsl_rl) from ETH Zurich's Robotic Systems Lab,
[Unitree](https://github.com/unitreerobotics)'s Go2 description and `unitree_sdk2_python`, the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) Go2 model, and ROBOTIS' OpenManipulator-X.

The decision to stack **6 observation frames**, and the **kp 50 / kd 1.0** PD gains used by the latest policy, were taken
from [Glowing-Torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot](https://github.com/Glowing-Torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot),
a Go2 deployment project that runs those settings and does not report the ~25 Hz joint buzz we saw with a single frame.
Only the settings were borrowed; the implementation here (Isaac Lab group history, the matching deploy-side
`ObsHistory`) is our own.

## Authors and license

Udula Ranasinghe (original author), Gavin Kane, and Arjun K Haridas (maintainer).

Released under the [BSD 3-Clause License](LICENSE). Files derived from the Isaac Lab template keep their
"Isaac Lab Project Developers" headers, under the same license.
