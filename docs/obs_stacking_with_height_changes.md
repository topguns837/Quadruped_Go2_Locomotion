# Iteration change list: `iter4_realistic` → `obs_stacking_with_height`

Written for: whoever reviews or runs this retrain.

- **Previous iteration (trained & deployed):** `go2_kp40_iter4_realistic` — `models/11_18_26_iter4_realistic/`
  (`model_24999`). Single-frame observation, kp/kd 40/0.8, 5-dim command (no height).
- **Going-to-be iteration:** `go2_obs_stacking_with_height` — profile
  `src/training/profiles/obs_stacking_with_height.yaml`, run dir `run_obs_stacking_with_height`. 6-frame observation
  history, kp/kd 50/1.0, 6-dim command (height re-added), yaw reward rebalance. **Trains from scratch** (the actor
  input dim changes, so it cannot resume a kp40 checkpoint).

Source of truth: a diff of the two resolved `params/env.yaml` + `params/agent.yaml` dumps, plus the robot actuator
config. Everything not listed under "Changed" is **carried over unchanged** from `iter4_realistic` (see the bottom).

Short paths below:
- `env_cfg.py` = `src/env/quadruped_go2_locomotion/env_cfg.py`
- `robot_cfg.py` = `.../manager_based/quadruped_go2_locomotion/robot_cfg.py`
- `ppo_cfg.py` = `.../manager_based/quadruped_go2_locomotion/agents/rsl_rl_ppo_cfg.py`
- `profile` = `src/training/profiles/obs_stacking_with_height.yaml`

## Changed parameters

| # | Parameter | File(s) | Old (iter4_realistic) | New (obs_stacking_with_height) | Why |
|---|---|---|---|---|---|
| 1 | `observations.policy.history_length` | `env_cfg.py` (`HISTORY_LENGTH`, `PolicyCfg.__post_init__`) | `None` (single frame) | `6` + `flatten_history_dim=True` | Feed the policy the last 6 frames (~120 ms at 50 Hz) so it can read trends/latency and implicitly system-identify the randomized dynamics. Main sim-to-real mechanism and the principled fix for the ~25 Hz buzz (replaces the deploy `--low_pass` band-aid). |
| 2 | `observations.critic.history_length` | `env_cfg.py` (`CriticCfg.__post_init__`) | `None` | `6` + `flatten_history_dim=True` | Critic sees the same time window as the actor (privileged, discarded at deploy). |
| 3 | Actor input dim | consequence of 1+2+5 | `47` (3 ang_vel + 3 gravity + 5 cmd + 12 pos + 12 vel + 12 act) | `288` = `48 × 6` | 6-dim command (height) + 6-frame stack. |
| 4 | Critic input dim | consequence | `50` | `306` = `51 × 6` | Same, plus the 3-dim `base_lin_vel` the critic alone sees. |
| 5 | Command: `commands.base_velocity.ranges.height` | `env_cfg.py` `CommandsCfg`, `mdp/commands.py`, `mdp/height_command.py` | *absent* (5-dim command) | `[0.28, 0.38]` m (6-dim command) | Height re-added as a **command + reward target only** — the policy observes the *commanded* height (known on-robot) and is scored vs sim ground-truth height; actual height is never observed (no hardware sensor), so it stays deployable/open-loop. |
| 6 | `joint_vel` observation noise model | `env_cfg.py` (`joint_vel` ObsTerm), `mdp/diagnostics.py:JointVelFiltered`, `mdp/obs_filters.py` | manager `Unoise(±1.5 rad/s)`, no filter | `mdp.JointVelFiltered`, `noise=None`, `noise_amp=0.3`, **`cutoff_hz=10.0`** (noise added *before* a 10 Hz EMA) | ±1.5 was adversarial and trained the buzz in; 0.3 matches the measured hardware floor (~0.06). The 10 Hz EMA mirrors deploy's `joint_vel_filter_hz=10` (same noise-then-filter order), so the policy sees the same filtered signal in sim and on the robot. |
| 7 | Actuator `kp` (stiffness) | `robot_cfg.py` (DCMotorCfg) | `40.0` | `50.0` | Matches the Glowing-Torch reference stand gain; stiffer hold. Deploy must match (`go2_obs_stacking.yaml` kp50). |
| 8 | Actuator `kd` (damping) | `robot_cfg.py` | `0.8` | `1.0` | Higher damping helps suppress the buzz. Deploy kd 1.0. |
| 9 | `events.actuator_gains` DR band (stiffness & damping) | `env_cfg.py` `EventCfg.actuator_gains` (+ profile override) | `×[0.9, 1.1]` | `×[0.8, 1.2]` | Wider PD-gain randomization keeps a real-robot kd that differs from the nominal (and the deploy kd≈1.0) inside the training distribution. |
| 10 | Reward `track_ang_vel_z_exp.weight` (tight yaw term) | `profile` (override; `env_cfg.py` default stays 1.2) | `1.2` | `3.2` | **Yaw rebalance.** sim2sim random eval of the kp40 policy (`logs/sim2sim_eval_24999/random/report.md`) showed yaw RMS **0.195 rad/s** vs ~0.049 for lin_xy — yaw carried ~5× less reward pressure (~2.9 vs ~14 units) and under-rotated on hard commands. |
| 11 | Reward `track_ang_vel_z_broad.weight` (broad yaw term) | `profile` | `0.8` | `1.6` | Same rebalance; keeps the 2:1 tight:broad ratio. Yaw tight+broad total `2.0 → 4.8` (≈ half the linear 10, above pitch/lean 3.0). |
| 12 | Reward `track_ang_vel_deficit.weight` (yaw under-rotation penalty) | `profile` | `-0.5` | `-1.0` | Doubled; the yaw analogue of `standing_when_commanded`, directly targeting the observed under-rotation. |
| 13 | `agent.experiment_name` | `ppo_cfg.py`, `profile` | `go2_kp40_iter4_realistic` | `go2_obs_stacking_with_height` | New experiment dir; renamed from the working label `go2_kp50_iter5_hist6`. |
| 14 | Training schedule | `profile` (phased) | one continuous run (`iter4_realistic`) | phase 1 `clean` 10k iters → phase 2 `realistic` 15k iters, 2k-iteration batched chunks | Clean warm-up forms a stable gait before the full realistic disturbances ramp back in. |

## Pipeline fixes made while validating (code bugs, not iteration tuning)

These were needed for the phased pipeline to run at all in the container; found by a bounded smoke test.

| Parameter / code | File | Change | Why |
|---|---|---|---|
| Interpreter for `train_profile.py` | `src/training/train.sh` | bare `python3` → prefer a yaml-capable `python3`, else the Isaac bundled Python (`TRAIN_PROFILE_PYTHON` to override) | The container's `/usr/bin/python3` has no PyYAML (it ships only in Isaac's Python); the wrapper crashed on `import yaml`. |
| `events.robot_mass.params.max_abs_kg` | `profile` | override key `mass_distribution_params` → `max_abs_kg` (value `2.0`, i.e. ±2 kg) | This branch's `robot_mass` uses the custom `randomize_base_mass_curriculum` (param `max_abs_kg`); the stock `mass_distribution_params` key does not exist on it → Hydra "Key not in struct". |
| `joint_vel.params.cutoff_hz` default | `env_cfg.py` | `None` → `10.0` | Isaac Lab's Hydra config updater refuses a `None → float` type change, so a `None` default could not be overridden to 10 Hz by the profile. A float default fixes it and matches deploy. |

## Carried over unchanged from `iter4_realistic`

All other reward weights (`track_lin_vel_xy_exp` 6.0 / `_broad` 4.0, `track_pitch_exp`/`track_lean_exp` 3.0,
`standing_when_commanded` −4.0, `height_penalty` −15.0, `stand_still` 5.0, `is_alive` 2.0, `termination_penalty`
−750, the smoothness trio, `foot_lift` 0.5, `gait_coordination` 1.0, etc.); all disturbance magnitudes
(`push_robot_force` 15 N / 2 N·m, `push_robot` 0.5 m/s, COM, frictions, armature DR); the difficulty curriculum
(`step` 0.025, thresholds, dwell/settle/min_episodes); joint armature 0.01; the disabled PhysX joint-speed clamp
(`velocity_limit_sim=500`); and the PPO hyperparameters + network (hidden `[256,512,128]`, elu, 24 steps/env,
lr 1e-3 adaptive, γ 0.99, λ 0.95). So `obs_stacking_with_height` = `iter4_realistic` **plus** rows 1–14 above.
