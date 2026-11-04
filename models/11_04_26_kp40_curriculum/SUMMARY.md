# 11_04_26_kp40_curriculum

Status: **sim2sim only. Not yet tried on hardware.**

- Source run: `logs/rsl_rl/go2_kp40_curriculum_ppo/2026-11-03_22-14-59`, `model_2699.pt` (final checkpoint of the
  resume that completed the 18k-iteration run; first leg is `2026-11-02_14-39-14`, up to `model_15300.pt`).
- Observation: **47-dim** (no `base_lin_vel`), 5-wide command (lin_x, lin_y, ang_z, pitch, lean), no height.
- Training: kp 40 / kd 0.8, action scale 0.25, no PhysX joint-speed clamp, armature 0.01, DR curriculum
  (mass, COM, push force scaled by `Curriculum/difficulty`), hardened smoothness penalties.
- Config dump: `params/env.yaml`, `params/agent.yaml` (these are from the resumed run).
- Deploy: `deploy/configs/go2_locomotion.yaml` `policy_path` points at `exported/policy.pt`. Deploy detects the
  47-dim layout and omits `base_lin_vel` automatically; kp/kd/action_scale in that config already match.

## Sim results (nominal physics)

- `scripts/rsl_rl/eval_velocity_tracking.py`: Isaac and MuJoCo agree (mean RMS lin_x 0.033/0.029, ang_z
  0.130/0.135, pitch 0.022/0.023, lean 0.009/0.009). Pure in-place turn at 0.8 rad/s barely rotates.
- `sim2sim_mujoco.py --action_delay 2`: 0 falls over 12 segments; yaw drift roughly doubles with delay.

## Expected on hardware

Lower, kneeling stance and foot skim/drag (see `gait_realism_diagnosis.md`). Weak yaw. Previous
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

    deploy/deploy_real.py --network_interface enp2s0 --manual_commands --live_plot \
      --hold_kp 60 --lift_calf_tether_deg 25 \
      --max_joint_speed_rad_s 1000 --policy_max_target_dev_deg 1000
