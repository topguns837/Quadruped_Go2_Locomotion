# 11_18_26_iter4_realistic

Status: **hardware-tested 2026-11-18 — walks; a ~25 Hz buzz was diagnosed and mitigated with a deploy-side action low-pass.**

- Training profile: No training-profile file — this run predates the `src/training/profiles/` system. Authoritative recipe: `params/env.yaml` + `params/agent.yaml` (full auto-generated config dump).
- Source run: `logs/rsl_rl/go2_kp40_iter4_realistic/run_iter4_realistic`, `model_24999.pt` (final checkpoint of the
  25k-iteration batched run; phase-2 "realistic" re-ramp from the clean phase-1 `model_9999`).
- Observation: **47-dim** (no `base_lin_vel`), 5-wide command (lin_x, lin_y, ang_z, pitch, lean), no height,
  **single frame** (no history stacking).
- Training: kp 40 / kd 0.8, action scale 0.25, no PhysX joint-speed clamp, armature 0.01. Realistic
  domain-randomisation magnitudes (push ±15 N/±2 Nm, base mass ±2 kg, tightened COM/armature), difficulty step
  0.025 (slow ramp), posture weights raised (height_penalty -15, stand_still 5.0, fall -750); push and action
  delay scale with the difficulty curriculum. Reached difficulty 1.0 by iter ~21,250 and held.
- Config dump: `params/env.yaml`, `params/agent.yaml`.
- Deploy: `src/deploy/profiles/go2_locomotion.yaml` `policy_path` points at `exported/policy.pt`. Deploy detects the
  47-dim layout and omits `base_lin_vel` automatically; kp/kd/action_scale in that config already match training.

## Metrics

Values are per-policy snapshots; the training-metric table is **not** cross-policy comparable where reward weights
differ (e.g. the yaw-term weight changed 1.6→3.2 in the later obs-stacking run).

### 1. Sim2sim (nominal physics, Isaac vs MuJoCo)

`bash src/sim2sim/run_sim2sim_eval.sh --checkpoint model_24999.pt` — 3 seeds × 20 segments (60 segments, 240 s/sim),
pooled RMS. Source: `logs/sim2sim_eval_24999/random/report.md`.

| metric | Isaac | MuJoCo |
|---|---|---|
| lin_x RMS (m/s) | 0.049 | 0.051 |
| lin_y RMS (m/s) | 0.049 | 0.045 |
| ang_z RMS (rad/s) | 0.195 | 0.222 |
| pitch RMS (rad) | 0.045 | 0.040 |
| lean RMS (rad) | 0.032 | 0.026 |
| height RMS (m) | 0.057 | 0.070 |
| segments with a fall (/60) | 0 | 0 |
| peak joint torque (Nm) | 44.8 | 45.4 |

### 2. Sim2real (hardware)

Three `--deploy_profile iter4_realistic` runs, 2026-11-23, `logs/hardware_dashboard/2026-11-23_{13-37-59,13-44-32,14-00-31}`
(`cli_args.txt` + `events.out.tfevents.*` + `policy_trace.npz` in each). Metrics below computed directly from those
files (RMS over the full run; torque/joint-vel/autocorr over ticks at the settled trained gain kp=40, excluding the
startup-ramp handover at kp up to 59).

| metric | 13:37 (59 s) | 13:44 (126 s) | 14:00 (115 s) | sim2sim MuJoCo (model_24999) |
|---|---|---|---|---|
| lin_x RMS (m/s) | 0.122† | 0.131† | 0.131† | 0.051 |
| lin_y RMS (m/s) | 0.078† | 0.109† | 0.111† | 0.045 |
| ang_z RMS (rad/s) | 0.160 | 0.164 | 0.182 | 0.222 |
| pitch RMS (rad) | 0.041 | 0.064 | 0.040 | 0.040 |
| lean RMS (rad) | 0.024 | 0.028 | 0.028 | 0.026 |
| height RMS (m) | n/a (no height input/estimate on this policy) | | | 0.070 |
| peak \|tau_est\| (Nm) | 22.9 | 24.7 | 22.7 | 45.4 (calf limit, not a fault) |
| mean \|tau_est\| (Nm) | 4.50 | 4.52 | 4.60 | — |
| joint_vel RMS (rad/s) | 0.48 | 0.65 | 0.62 | — |
| gyro RMS (rad/s) | 0.141 | 0.200 | 0.173 | — |
| sent-target lag-1 autocorr | +0.99 | +0.99 | +0.99 | — |
| tether / slew-clip ticks | 0 / 0 | 0 / 0 | 0 / 0 | — |
| falls / aborts | 0 | 0 | 0 | 0 |

† **Not a tracking error — do not read as sim2real velocity transfer.** `deploy_real.py` only subscribes to
`rt/sportmodestate` when the policy's observation includes `base_lin_vel` (`include_base_lin_vel`); this 47-dim
policy doesn't, so the subscription is never opened and `LatestRobotState.sportmode_velocity` stays at its
initializer, `(0.0, 0.0, 0.0)`, for the whole run (confirmed directly: `actual_lin_vel_x/y` in all three
`events.out.tfevents.*` files are exactly 0.000000 at every step). The "RMS" above is therefore just `RMS(cmd - 0)
= RMS(cmd)`, i.e. a measure of how large the commands were, not of tracking. **There is currently no linear-velocity
ground truth on hardware for this policy** — ang_z/pitch/lean above are real (IMU-derived), lin_x/lin_y are not.
To get a real number, either log (not feed) `rt/sportmodestate` regardless of `include_base_lin_vel`, or use an
external reference (mocap/vision).

Other results: walks on all three runs; **zero** faults, aborts, tether engagements or slew-rate clips; peak torque
well inside the 40.9 Nm calf / 21.15 Nm global abort. The 5 Hz low-pass keeps the sent-target lag-1 autocorrelation
at +0.99 (smooth) vs the −0.70 reported unfiltered on 2026-11-18, consistent with the buzz mitigation holding here too.
Loop ran at a mean ~22–23 ms/tick (vs the 20 ms / 50 Hz design point) on this laptop's CPU; did not destabilize
anything but is slower than the 2026-11-18 run's steady 20.1 ms.

### 3. Final training metrics (mean of last 300 iterations)

| metric | value |
|---|---|
| total iterations | 25000 |
| mean_reward | 539.2 |
| mean_episode_length | 1885.1 |
| difficulty_scale | 1.0 |
| error_vel_xy | 0.546 |
| error_vel_yaw | 1.129 |
| track_lin_vel_xy_exp | 3.369 |
| track_ang_vel_z_exp | 0.282 |
| track_pitch_exp | 2.648 |
| track_lean_exp | 2.571 |
| height_penalty | −0.367 |
| body_contact (term rate) | 0.083 |
| entropy | 4.682 |

## Sim results (nominal physics)

- Random-command Isaac-vs-MuJoCo eval (`src/sim2sim/run_sim2sim_eval.sh`, 3 seeds x 20 segments): **0 falls** in both
  sims; pooled RMS errors Isaac/MuJoCo lin_x 0.049/0.051, lin_y 0.049/0.045, ang_z 0.195/0.222, pitch 0.045/0.040,
  lean 0.032/0.026, height 0.057/0.070 m; peak torque ~45 Nm (calf limit). Per-segment errors correlate strongly
  between the two sims. Weakest axis: **yaw** (~0.2 rad/s).

## Hardware results, 2026-11-18

- Walked; pitch/lean/velocity commands tracked and looked visibly fine with the low-pass on.
- **~24.9 Hz (control-Nyquist) "buzz"**, worst during dynamic commands (forward walk, pitch). Diagnosed as the
  single-frame policy **amplifying joint-velocity observation noise** (NOT latency): in MuJoCo it reproduces the
  instant training-level obs noise is injected (`sim2sim_mujoco.py --obs_noise`), but not from command replay +
  `--obs_delay` up to 40 ms. Control loop timing was a steady 20.1 ms; the startup ramp was clean (peak 18.4 Nm).
- **Mitigation that worked:** `src/deploy/deploy_real.py --low_pass` (5 Hz EMA on the action). Filtered vs unfiltered
  run: raw-action lag-1 autocorr -0.70 -> +0.40, body gyro RMS -28%, joint-vel RMS -40%, with *more* aggressive
  commands. It also breaks the loop at the source (the smoothed sent target feeds back as `last_action`).

## Deploy command (hardware)

Use the `confident` profile (gives hold_kp 60, which the startup lift needs — hold_kp 40 stalls) plus the low-pass:

    src/deploy/deploy_real.py --network_interface enp2s0 --manual_commands --live_plot \
      --test_profile confident --low_pass

Drop `--low_pass` for an unfiltered comparison; raise/lower the cutoff with `--low_pass 8` / `--low_pass 3`.

## For the next training (the real fix, see docs/obs_stacking_notes.md / memory project_go2_buzz_fix)

The low-pass is a deploy band-aid. The root fix is **observation stacking (H=6)** plus **lowering the joint_vel
observation noise** (from ±1.5; in sim, less noise reduced the buzz AND improved tracking). Also consider matching
the no-buzz reference deploy's gains (**kp 50 / kd 1.0**, it stacks 6 frames) and widening the PD-gain DR band so a
higher deploy kd is in-distribution.
