# 10_14_26_real_hardware_parity_tested

**What this is:** the first checkpoint trained under the real-hardware-parity config (see CLAUDE.md
"Real-hardware parity (2026-10-13)") -- PhysX's hard joint-speed clamp disabled
(`PHYSX_JOINT_SPEED_LIMIT=500`, so only the real DC-motor torque-speed curve shapes dynamics, same as the
real robot and MuJoCo), nominal joint armature 0.01, and training-time domain randomization (friction,
mass, PD gains, armature, joint friction, random 0-2 step action delay, periodic pushes). No action
clipping. `base_lin_vel` is kept in the observation (sourced from the robot's own SportModeState estimate
on real hardware, not a ground-truth sensor). Every checkpoint trained before 2026-10-13 is NOT valid for
hardware -- this is the first one that is.

**Training:** run `2026-10-14_15-26-46`, 10000 max iterations, **complete, no crash** (resumed once after
a host freeze at iteration 3842, see project history -- the completed run itself has no instability).
Final checkpoint `model_9999.pt`. Full config: `params/agent.yaml` (PPO/runner) and `params/env.yaml`
(environment).

**Measured via TensorBoard:** final `Train/mean_reward` ≈613.7, mean episode length ≈1932.7/2000 steps
(87% full-episode completion rate, not timing out from falls).

**Isaac vs MuJoCo velocity/pitch/lean tracking** (`scripts/rsl_rl/eval_velocity_tracking.py`, nominal
non-randomized physics both sides): RMS tracking error stays under ~0.03 m/s (lin_x/lin_y) and under
~0.03 rad (pitch/lean) across stand/forward/backward/lateral/turn/pitch/lean segments, with Isaac and
MuJoCo agreeing closely with each other throughout. The one reproducible weakness found: pure in-place
turning (`ang_z` only) has a visibly higher RMS (~0.74 rad/s) than every other segment, in both engines.

**Real-hardware test (this session):** deployed via `deploy/deploy_real.py` with `--max_target_dev_deg 25`
(loosened from the 15° default -- see caveat below), hoisted first per `deploy/Checklist.md`, then handed
off to the policy loop. Ramp and policy loop ran without any `TrackingFault`/`TorqueResponseFault`/
`JointSpeedFault`/`LinkLost` abort; the one joint that needed the loosened tether (`FR_calf_joint`)
stabilized within a handful of steps and stopped triggering clamp warnings, i.e. the robot reached a
stable standing equilibrium rather than continuing to strain against the limit. **Operator-confirmed: the
policy "works good" on real hardware** for standing/low-speed testing under this configuration.

**Known caveat -- real vs. simulated standing pose differs by up to ~20-26° on a few joints.** A real,
standing-still observation recording (`deploy/collect_still_observations.py`) compared against Isaac's and
MuJoCo's own zero-command "stand" segment (`eval_velocity_tracking.py --real_obs_log`) showed:

| joint | isaac − real | mujoco − real |
|---|---:|---:|
| RR_calf | -24.3° | -26.0° |
| RL_hip | -14.6° | -15.7° |
| RR_hip | +13.7° | +11.4° |
| all four thighs | +12 to +23° | +16 to +23° |

Pitch/lean were both near zero (<1°), so this is specifically a leg-joint-configuration gap, not an
overall balance/tilt issue. Root cause not yet identified (candidates: real vs. simulated arm/manipulator
mass distribution, a joint zero-offset/calibration difference, or the real robot's own stable equilibrium
under its true weight distribution). This is why the real-hardware test above used a loosened
`--max_target_dev_deg 25` instead of the 15° default -- at 15°, several joints (RR_hip, RR_calf, RL_hip)
would be continuously clamped rather than reaching the policy's actual target. Worth investigating before
extended/unattended real-hardware sessions; not investigated further this session.

**To play or deploy this model:**
```bash
isaaclab -p scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1 \
  --checkpoint models/10_14_26_real_hardware_parity_tested/model_9999.pt
```
For real hardware, point `deploy/configs/go2_locomotion.yaml`'s `policy_path` at
`models/10_14_26_real_hardware_parity_tested/exported/policy.pt` and follow `deploy/Checklist.md` in full
-- do not skip the hoisted phases even though this checkpoint has already been tested, since the checklist
protects against cable/mapping/release issues independent of which policy is loaded.
