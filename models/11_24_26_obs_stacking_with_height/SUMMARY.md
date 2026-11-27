# 11_24_26_obs_stacking_with_height

Status: **trained 2026-11-23 (25k iters) · sim2sim-evaluated · NOT hardware-tested.** Supersedes `iter4_realistic`
as the obs-stacking *candidate* — it fixes the buzz at the source (6-frame history + lower joint_vel noise) instead
of the deploy `--low_pass` band-aid, and improves yaw tracking — but it has **open regressions** (see Caveats). Not
ready for hardware.

- Training profile: `src/training/profiles/obs_stacking_with_height.yaml` (frozen snapshot: `training_profile.yaml` in this dir); resolved config in `params/env.yaml` + `params/agent.yaml`.
- Source run: `logs/rsl_rl/go2_obs_stacking_with_height/run_obs_stacking_with_height`, `model_24999.pt` (final
  checkpoint of the 25k-iteration phased batched run: phase-1 "clean" 10k → phase-2 "realistic" 15k).
- Observation: **48-dim per frame**, **6-wide command** (lin_x, lin_y, ang_z, pitch, lean, **height**), **6-frame
  history stacked** (`HISTORY_LENGTH=6`, `flatten_history_dim`), so the actor input is **48×6 = 288** (critic
  51×6 = 306). Term-major, newest-last flat layout. No `base_lin_vel` in the actor.
- Height is a **command + reward target only** — the policy observes the *commanded* height (known on-robot) and is
  scored against the simulator's ground-truth base height; actual height is never observed (no hardware sensor), so
  the policy stays open-loop and deployable.
- Training deltas vs iter4: kp/kd **40/0.8 → 50/1.0**; joint_vel obs gets hardware-realistic noise (0.3, was the
  adversarial 1.5) **then a 10 Hz EMA low-pass**, the same noise-then-filter order deploy uses; actuator-gain DR
  band widened ×0.9–1.1 → ×0.8–1.2; **yaw reward rebalance** (`track_ang_vel_z_exp` 1.6→3.2, `_broad` 0.8→1.6,
  `track_ang_vel_deficit` −0.5→−1.0); phased clean→realistic schedule. See `docs/obs_stacking_with_height_changes.md`.
- Config dump: `params/env.yaml`, `params/agent.yaml`. Export: `exported/policy.pt` (JIT, 288-dim in / 12 out) +
  `exported/policy.onnx`.
- Deploy: `src/deploy/profiles/go2_obs_stacking.yaml` (kp50/kd1.0, `history_length: 6`, `joint_vel_filter_hz: 10.0`).
  Its `policy_path` is **not yet repointed** to this export — hardware isn't ready (see Caveats).

## Metrics

Values are per-policy snapshots; the training-metric table is **not** cross-policy comparable where reward weights
differ (e.g. the yaw-term weight changed 1.6→3.2 vs iter4).

### 1. Sim2sim (nominal physics, Isaac vs MuJoCo)

`bash src/sim2sim/run_sim2sim_eval.sh --checkpoint model_24999.pt --config src/deploy/profiles/go2_obs_stacking.yaml`
— 3 seeds × 20 segments (60 segments, 240 s/sim), pooled RMS. Source:
`logs/sim2sim_eval_obs_stacking_24999/random/report.md`.

| metric | Isaac | MuJoCo |
|---|---|---|
| lin_x RMS (m/s) | 0.044 | 0.071 |
| lin_y RMS (m/s) | 0.043 | 0.079 |
| ang_z RMS (rad/s) | 0.133 | 0.181 |
| pitch RMS (rad) | 0.048 | 0.053 |
| lean RMS (rad) | 0.021 | 0.021 |
| height RMS (m) | 0.059 | 0.069 |
| segments with a fall (/60) | 1 | 1 |
| peak joint torque (Nm) | 42.2 | 45.4 |

### 2. Sim2real (hardware)

Filled after a hardware session; quantitative tracking is not captured on hardware yet.

| metric | value |
|---|---|
| lin_x RMS (m/s) | — |
| lin_y RMS (m/s) | — |
| ang_z RMS (rad/s) | — |
| pitch RMS (rad) | — |
| lean RMS (rad) | — |
| height RMS (m) | — |
| walks? | — (not tested) |
| buzz | — |
| control-loop timing | — |
| notes | — |

### 3. Final training metrics (mean of last 300 iterations)

| metric | value |
|---|---|
| total iterations | 25000 |
| mean_reward | 580.1 |
| mean_episode_length | 1852.5 |
| difficulty_scale | 1.0 |
| error_vel_xy | 0.462 |
| error_vel_yaw | 0.960 |
| track_lin_vel_xy_exp | 3.631 |
| track_ang_vel_z_exp | 0.826 |
| track_pitch_exp | 2.509 |
| track_lean_exp | 2.524 |
| height_penalty | −0.441 |
| body_contact (term rate) | 0.120 |
| entropy | 3.462 |

## Caveats / open issues (why this is a candidate, not a release)

- **Yaw improved** vs iter4: sim2sim ang_z RMS 0.195→0.133 Isaac (−32%), 0.222→0.181 MuJoCo (−18%). The rebalance
  worked, but yaw is still ~3× the linear error in Isaac.
- **MuJoCo linear tracking regressed:** lin_x/lin_y MuJoCo errors are ~1.6–1.9× the Isaac errors (iter4 was ~1.0×).
  Could be sim-specific overfitting or the kp50/kd1.0 gains behaving differently in MuJoCo — unresolved.
- **Height command saturates:** in a MuJoCo height sweep the robot reached 0.28–0.33 m targets (err ≤0.04 m) but
  sat near ~0.30 m when commanded 0.38 m (err 0.07–0.10 m). Commands work downward, not upward.
- **Falls:** 1 fall segment in each sim (iter4 had 0/0); the MuJoCo one reset 4× on one hard command.
- **Layout-parity check not passed:** `src/play/verify_obs_stacking.py` hung and was killed — the deploy
  `ObsHistory` layout flags are not yet confirmed against the real env. This is a hard gate before any hardware use.
