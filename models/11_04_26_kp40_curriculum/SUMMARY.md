# 11_04_26_kp40_curriculum

Status: **sim2sim only. Not yet tried on hardware.**

- Training profile: No training-profile file — this run predates the `src/training/profiles/` system. Authoritative recipe: `params/env.yaml` + `params/agent.yaml` (full auto-generated config dump).
- Source run: `logs/rsl_rl/go2_kp40_curriculum_ppo/2026-11-03_22-14-59`, `model_2699.pt` (final checkpoint of the
  resume that completed the 18k-iteration run; first leg is `2026-11-02_14-39-14`, up to `model_15300.pt`).
- Observation: **47-dim** (no `base_lin_vel`), 5-wide command (lin_x, lin_y, ang_z, pitch, lean), no height.
- Training: kp 40 / kd 0.8, action scale 0.25, no PhysX joint-speed clamp, armature 0.01, DR curriculum
  (mass, COM, push force scaled by `Curriculum/difficulty`), hardened smoothness penalties.
- Config dump: `params/env.yaml`, `params/agent.yaml` (these are from the resumed run).
- Deploy: `src/deploy/profiles/go2_locomotion.yaml` `policy_path` points at `exported/policy.pt`. Deploy detects the
  47-dim layout and omits `base_lin_vel` automatically; kp/kd/action_scale in that config already match.

## Metrics

Values are per-policy snapshots; the training-metric table is **not** cross-policy comparable where reward weights
differ — this run used an earlier reward schema (e.g. `track_lin_vel_xy_exp` weight ~10, hence the larger raw value
below) and only reached difficulty 0.33 over its 2.7k-iteration final leg.

### 1. Sim2sim (nominal physics, Isaac vs MuJoCo)

No random-command `run_sim2sim_eval.sh` report was retained for this policy.

| metric | Isaac | MuJoCo |
|---|---|---|
| lin_x RMS (m/s) · lin_y RMS (m/s) · ang_z RMS (rad/s) · pitch RMS (rad) · lean RMS (rad) · height RMS (m) · segments with a fall (/60) · peak joint torque (Nm) | n/a — not retained | n/a — not retained |

See "Sim results (nominal physics)" below for the `eval_velocity_tracking.py` numbers that do exist (a different,
non-random eval).

### 2. Sim2real (hardware)

Hardware-tested 2026-11-04 (details in "Hardware results" below); quantitative tracking not captured.

| metric | value |
|---|---|
| lin_x RMS (m/s) | — |
| lin_y RMS (m/s) | — |
| ang_z RMS (rad/s) | — |
| pitch RMS (rad) | — |
| lean RMS (rad) | — |
| height RMS (m) | — |
| walks? | partial — stood/handed over cleanly; gait-like motion only on −X; stands below ~0.3 m/s |
| buzz | 24.9 Hz joint oscillation at ~51.8 s after a pitch command; did not recover |
| control-loop timing | — |
| notes | pitch/lean tracked (corr 0.93/0.97); no abort fired (peak τ 13.9 Nm, speed 4.1 rad/s) |

### 3. Final training metrics (mean of last 300 iterations, final leg `2026-11-03_22-14-59`)

| metric | value |
|---|---|
| total iterations | 2700 (final leg; 18k across both legs) |
| mean_reward | 561.2 |
| mean_episode_length | 1882.7 |
| difficulty_scale | 0.333 |
| error_vel_xy | 0.677 |
| error_vel_yaw | 1.331 |
| track_lin_vel_xy_exp | 8.075 |
| track_ang_vel_z_exp | 1.192 |
| track_pitch_exp | 2.631 |
| track_lean_exp | 2.561 |
| height_penalty | −0.002 |
| body_contact (term rate) | 0.078 |
| entropy | 8.873 |

## Sim results (nominal physics)

- `src/play/eval_velocity_tracking.py`: Isaac and MuJoCo agree (mean RMS lin_x 0.033/0.029, ang_z
  0.130/0.135, pitch 0.022/0.023, lean 0.009/0.009). Pure in-place turn at 0.8 rad/s barely rotates.
- `sim2sim_mujoco.py --action_delay 2`: 0 falls over 12 segments; yaw drift roughly doubles with delay.

## Expected on hardware

Lower, kneeling stance and foot skim/drag (see `docs/gait_realism_diagnosis.md`). Weak yaw. Previous
hardware-tested policy: `models/10_14_26_real_hardware_parity_tested/` (50-dim).

## Deploy-time flags for sim parity (on-ground policy loop only)

`--max_joint_speed_rad_s 1000 --policy_max_target_dev_deg 1000` so the slew cap and the policy-loop tether never
reshape the targets (sim had neither). Do NOT use `--max_target_dev_deg 1000`: that also removes the tether from the
startup ramp, and a robot rising from a crouch (especially on a mat) lags the ramp target until the 30 deg error abort
trips (2026-11-04 run, fault at ~7 s).
Leave `--speed_fault_cap_rad_s 30.1 30.1 15.7` (= sim's motor speed limit), the torque abort and
`--abort_err_deg 30` on.

## Startup lift from a crouch (rear knees down)

Add `--hold_kp 60 --lift_calf_tether_deg 25`. This is the recipe that lifted the robot on 2026-10-30 (hold_kp was 60
then; commit 44bb1b9 changed the yaml to 40). Calf torque ceiling in lift mode is hold_kp x lift tether: 60 x 25 =
26 Nm works (the rear calf needed ~18.6 Nm), 40 x 25 = 17.5 Nm stalls (2026-11-04 15:06 run: error pinned at 25 deg,
tau flat at 18.2 Nm, RL_calf 38.6 deg short). Fallback at hold_kp 40: `--lift_calf_tether_deg 40` (~28 Nm).

Full command:

    src/deploy/deploy_real.py --network_interface enp2s0 --manual_commands --live_plot \
      --hold_kp 60 --lift_calf_tether_deg 25 \
      --max_joint_speed_rad_s 1000 --policy_max_target_dev_deg 1000

## Test profiles (replace the long flag string)

`--test_profile cautionary | confident | no_limits` loads the matching block of `src/deploy/configs/safety_profiles.yaml`
(slew, tethers, abort, speed-fault, startup-lift values). The file is re-read at every launch, so edits apply from
the next run; an explicit flag still overrides one value. The startup banner and `cli_args.txt` record the resolved
values. `no_limits` is what the 2026-11-04 15:39 run used:

    src/deploy/deploy_real.py --network_interface enp2s0 --manual_commands --live_plot --test_profile no_limits

## Hardware results, 2026-11-04

- Startup lift, handover and stand were clean; pitch/lean tracked (corr 0.93/0.97); the commands that were sent
  (forward <= 0.22, lateral 0.29) did not produce stepping because the policy stands below ~0.3 m/s, in MuJoCo too.
- 15:39 run: gait-like motion on -X (-0.19 to -0.27 m/s), then at ~51.8 s, after a pitch command, a 24.9 Hz
  (control Nyquist) joint oscillation that did not recover; stopped with L2+B. Peak torque 13.9 Nm and joint speed
  4.1 rad/s, so no abort could fire. Commanded joint rate 6 rad/s RMS vs 0.1 in clean gait (about 50x).
