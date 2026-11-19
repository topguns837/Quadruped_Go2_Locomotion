# 11_18_26_iter4_realistic

Status: **hardware-tested 2026-11-18 — walks; a ~25 Hz buzz was diagnosed and mitigated with a deploy-side action low-pass.**

- Source run: `logs/rsl_rl/go2_kp40_iter4_realistic/run_iter4_realistic`, `model_24999.pt` (final checkpoint of the
  25k-iteration batched run; phase-2 "realistic" re-ramp from the clean phase-1 `model_9999`).
- Observation: **47-dim** (no `base_lin_vel`), 5-wide command (lin_x, lin_y, ang_z, pitch, lean), no height,
  **single frame** (no history stacking).
- Training: kp 40 / kd 0.8, action scale 0.25, no PhysX joint-speed clamp, armature 0.01. Realistic
  domain-randomisation magnitudes (push ±15 N/±2 Nm, base mass ±2 kg, tightened COM/armature), difficulty step
  0.025 (slow ramp), posture weights raised (height_penalty -15, stand_still 5.0, fall -750); push and action
  delay scale with the difficulty curriculum. Reached difficulty 1.0 by iter ~21,250 and held.
- Config dump: `params/env.yaml`, `params/agent.yaml`.
- Deploy: `deploy/configs/go2_locomotion.yaml` `policy_path` points at `exported/policy.pt`. Deploy detects the
  47-dim layout and omits `base_lin_vel` automatically; kp/kd/action_scale in that config already match training.

## Sim results (nominal physics)

- Random-command Isaac-vs-MuJoCo eval (`sim2sim/run_random_eval.sh`, 3 seeds x 20 segments): **0 falls** in both
  sims; pooled RMS errors Isaac/MuJoCo lin_x 0.049/0.051, lin_y 0.049/0.045, ang_z 0.195/0.222, pitch 0.045/0.040,
  lean 0.032/0.026, height 0.057/0.070 m; peak torque ~45 Nm (calf limit). Per-segment errors correlate strongly
  between the two sims. Weakest axis: **yaw** (~0.2 rad/s).

## Hardware results, 2026-11-18

- Walked; pitch/lean/velocity commands tracked and looked visibly fine with the low-pass on.
- **~24.9 Hz (control-Nyquist) "buzz"**, worst during dynamic commands (forward walk, pitch). Diagnosed as the
  single-frame policy **amplifying joint-velocity observation noise** (NOT latency): in MuJoCo it reproduces the
  instant training-level obs noise is injected (`sim2sim_mujoco.py --obs_noise`), but not from command replay +
  `--obs_delay` up to 40 ms. Control loop timing was a steady 20.1 ms; the startup ramp was clean (peak 18.4 Nm).
- **Mitigation that worked:** `deploy/deploy_real.py --low_pass` (5 Hz EMA on the action). Filtered vs unfiltered
  run: raw-action lag-1 autocorr -0.70 -> +0.40, body gyro RMS -28%, joint-vel RMS -40%, with *more* aggressive
  commands. It also breaks the loop at the source (the smoothed sent target feeds back as `last_action`).

## Deploy command (hardware)

Use the `confident` profile (gives hold_kp 60, which the startup lift needs — hold_kp 40 stalls) plus the low-pass:

    deploy/deploy_real.py --network_interface enp2s0 --manual_commands --live_plot \
      --test_profile confident --low_pass

Drop `--low_pass` for an unfiltered comparison; raise/lower the cutoff with `--low_pass 8` / `--low_pass 3`.

## For the next training (the real fix, see docs/obs_stacking_notes.md / memory project_go2_buzz_fix)

The low-pass is a deploy band-aid. The root fix is **observation stacking (H=6)** plus **lowering the joint_vel
observation noise** (from ±1.5; in sim, less noise reduced the buzz AND improved tracking). Also consider matching
the no-buzz reference deploy's gains (**kp 50 / kd 1.0**, it stacks 6 frames) and widening the PD-gain DR band so a
higher deploy kd is in-distribution.
