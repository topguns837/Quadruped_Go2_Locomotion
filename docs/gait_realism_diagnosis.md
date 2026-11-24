# Diagnosis: why the latest policy kneels and drags its feet

Status: findings only. No config changes, no retrain — rectification is deferred (see last section).
Policy under study: experiment `go2_kp40_curriculum_ppo`, run `2026-11-03_22-14-59`, `model_2699.pt`, obs 47-dim.

## Context

The latest policy moves far more like the real Go2 than earlier policies: it sits low / kneels and skims
(drags) its feet rather than stepping cleanly and springily. This is visible in both Isaac and MuJoCo. The new
realism is useful for sim2real; this document records *why* it happens so training parameters can later be
retuned to bring gait coordination, foot clearance and stance height back within limits.

## What changed in this retrain (commit `44bb1b9`, vs prior tip `8d26178`)

Confirmed by `git diff 8d26178 44bb1b9` on `quadruped_go2_locomotion_env_cfg.py` and `unitree_go2witharm_cfg.py`:

- **Smoothness penalties hardened** in `RewardsCfg`:
  - `dof_torques_l2`  −1.0e-5 → **−3.0e-5** (×3)
  - `dof_acc_l2`  −2.5e-7 → **−5.0e-7** (×2)
  - `action_rate_l2`  −0.01 → **−0.03** (×3)
  - `dof_vel_l2`  **new, −1.0e-4** (direct joint-speed penalty, did not exist before)
- **`foot_lift` reward left at weight 0.1** (unchanged, tiny) against **`foot_sliding` −1.0**.
- **PhysX hard joint-speed clamp removed** in the earlier commit `e1ad833`
  (`PHYSX_JOINT_SPEED_LIMIT = 500`, `unitree_go2witharm_cfg.py:30`) — real-hardware parity.
- **kp/kd raised 25/0.5 → 40/0.8** (`unitree_go2witharm_cfg.py:95-100,110-111`) so the stance can be held
  passively; deploy gains in `deploy/profiles/go2_locomotion.yaml` must match.
- Added the mass/COM/push **DR curriculum** (`mdp/events.py`, `mdp/curriculums.py`).

## Evidence: the reward balance the policy actually optimized

Per-term `Episode_Reward/*` from the run's tfevents, mean over the last 30% of training:

| term | contribution |
|---|---|
| track_lin_vel_xy_exp | +8.11 |
| track_pitch_exp | +2.63 |
| track_lean_exp | +2.56 |
| track_ang_vel_z_exp | +1.19 |
| gait_coordination | +0.51 |
| action_rate_l2 | **−0.29** |
| dof_acc_l2 | **−0.28** |
| foot_sliding | −0.24 |
| ang_vel_xy_l2 | −0.13 |
| stand_still | +0.09 |
| **foot_lift** | **+0.083** |
| lin_vel_z_l2 | −0.08 |
| dof_torques_l2 | −0.02 |
| hip_crossing | −0.019 |
| dof_vel_l2 | −0.015 |
| joint_deviation | −0.015 |
| dof_pos_limits | −0.006 |
| **height_penalty** | **−0.0019** |
| undesired_contacts(_arm) | ~0 |

## Attribution

### 1. Foot-dragging — a reward-balance artifact (confident)

- `foot_lift` earns only **+0.083**, while `foot_sliding` costs **−0.24** and the smoothness trio
  (`action_rate` −0.29 + `dof_acc` −0.28 + `dof_vel` −0.015) costs **−0.59 combined — more than the whole
  `gait_coordination` reward (+0.51)**.
- Mechanics (`mdp/rewards.py`): `foot_sliding_exp` only penalizes feet **in contact** (swing feet contribute 0);
  `foot_lift_exp` only rewards feet **not in contact and above `min_height = 0.05 m`**. So the policy's cheapest
  path is to **skim the feet a few cm off the ground** — high enough to leave contact (dodging the sliding
  penalty) but below 5 cm (so `foot_lift` stays ≈0) — while the tripled smoothness penalties suppress the
  vigorous joint motion a clean step needs. That skim is exactly the drag/shuffle observed.
- This retrain tipped the balance: tripling `action_rate_l2`, doubling `dof_acc_l2` and adding `dof_vel_l2`
  while `foot_lift` stayed at 0.1.

### 2. Kneeling / low stance — mechanism is config-solid; magnitude needs an Isaac-side trace

- **Mechanism (solid, Isaac config/log fact):** `height_penalty` logs only **−0.0019** (bottom of the table).
  `base_height_l2_pitch` (`mdp/rewards.py:82`) returns `(root_z − adaptive_target)²` at weight −1.65, so the
  term exerts **almost no gradient**: sitting a few cm low costs the policy essentially nothing, while standing
  tall would cost torque/joint-speed (the dominant penalties). Height is barely regulated, so the policy is
  free to crouch. `adaptive_target` additionally **forgives** height loss during commanded pitch/lean
  (`pitch_sensitivity=0.25`, `lean_sensitivity=0.08`, `rewards.py:123`).
- **Magnitude (corroborating, not yet Isaac-true):** the 0.05–0.12 m droop (incl. 0.052 m on a plain `stand`)
  came from a **MuJoCo-only** run. Per the parity check below, MuJoCo matches Isaac's actuator/gains/limits but
  **not** its contact solver, and uses a rigid-arm approximation, so a few cm of that could be MuJoCo contact
  behaviour rather than the Isaac policy. The droop direction is consistent with the weak-penalty mechanism,
  but to quote an Isaac-true stance height, record one with `scripts/rsl_rl/play.py --trace_out` (it logs root
  height) and read it back.
- kp40 lets the legs *hold* whatever posture the policy picks; it does not force the posture up. The reward,
  not the gains, chooses the low stance.

### 2a. Does MuJoCo reproduce Isaac? (parity check — why the two claims have different confidence)

Verified against the trained run's own dump (`…/2026-11-02_14-39-14/params/env.yaml`),
`unitree_go2witharm_cfg.py`, `deploy/profiles/go2_locomotion.yaml`, `sim2sim/go2_model.py` and
`sim2sim/sim2sim_mujoco.py`.

**Matches exactly (so the actuator/robot the policy drives is the same in both):** PD gains kp40/kd0.8;
`action_scale` 0.25 with default offset; physics dt 0.005, decimation 4, control 50 Hz; **armature 0.01** (both
actuator groups, confirmed in `params/env.yaml` — the `go2_model.py:19` "USD value unknown" comment is stale);
motor limits hip/thigh 23.7 Nm @ 30.1, calf 45.43 Nm @ 15.70 (saturation = effort); the DCMotor torque-speed
curve (sim2sim reimplements Isaac's `IdealPDActuator.compute` + `DCMotor._clip_effort`, `sim2sim_mujoco.py:282`);
PhysX speed clamp disabled on both (`velocity_limit_sim 500` / `joint_speed_cap off`); passive joint damping and
frictionloss zeroed (Isaac does PD in software); default pose (hip ±0.1, thigh 0.8/1.0, calf −1.5) and init
height 0.4; floor friction 1.0 = training terrain material; gravity −9.81; no compliant/soft contact.

**Differs (by design or approximation) — the reason the kneeling magnitude is held loosely:**
- **Contact solver:** PhysX (TGS) vs MuJoCo default (Newton, pyramidal cone, condim 3, default solref/solimp).
  Fundamental and intentional — the whole purpose of a second engine. Affects foot-ground detail and can shift
  absolute stance height by a few cm.
- **Arm:** rigid fixed child bodies of `base` in MuJoCo vs GO2WITHARM with arm joints held at zero by stiff USD
  drives in Isaac; masses/COM/inertia taken from the USD, arm inertia approximated (`go2_model.py:11-19`).
- **Mass:** MuJoCo total ~15.81 kg (Menagerie Go2 15.206 + arm) vs Isaac Go2 USD 15.019 + arm — small delta.
- **Physics regime:** MuJoCo is nominal — no mass/COM/push DR, no startup gain/armature/friction randomization,
  no action delay. So **sim2sim ≈ Isaac _play_, not Isaac _training_**; it does not exercise the DR curriculum.

**Consequence:** foot-dragging is seen with the *same policy in both engines* and is backed by Isaac's own
training-reward logs, so it is engine-independent (confident). The kneeling *mechanism* is an Isaac config fact
(weak `height_penalty`), but its *magnitude* rests on a MuJoCo-only number and should be confirmed with an
Isaac-side height trace before being quoted.

### 3. Why it now looks like the real robot (context, not a defect)

Removing the fictitious PhysX speed clamp (`e1ad833`) stripped away the artificial "springiness" earlier
policies learned to exploit (they produced bang-bang joint targets the clamp silently tamed; see
`sim2sim/README.md`). Without it, Isaac, MuJoCo and hardware share the same torque-speed limits, so the gait
the policy settles into under the hardened smoothness penalties is a genuine low-energy gait — which is why it
reads as realistic in both sims and is expected to match the robot.

## Deferred: rectification (later)

Not changing anything now. When retuning, the knobs implicated above, to bring gait coordination, foot
clearance and stance height within limits together:

- **Foot clearance:** raise `foot_lift` weight and/or lower its `min_height` toward the skim height; and/or ease
  the smoothness trio (`action_rate_l2`, `dof_acc_l2`, `dof_vel_l2`) that is currently out-weighing the gait
  reward. Balance against keeping the realistic, hardware-safe energy level.
- **Stance height:** strengthen `height_penalty` (or narrow its allowance/clamp) so a few-cm droop actually
  costs gradient — it currently does not.
- **Gait coordination:** `gait_coordination` (+0.51) is nearly cancelled by the −0.59 smoothness pressure;
  re-weight so the trot shaping is not swamped.
- Any retune is a **from-scratch 18k run** (the action→torque mapping is tied to kp40 and these weights), with
  the mem guard (`scripts/train_mem_guard.sh`) alongside, then re-checked in both engines.

## How to reproduce these findings (all read-only)

1. **Config diff:** `git diff 8d26178 44bb1b9 -- …/quadruped_go2_locomotion_env_cfg.py`; read `unitree_go2witharm_cfg.py`.
2. **Reward balance:** `EventAccumulator` over the run's tfevents, mean of each `Episode_Reward/*` over the last
   30% of steps. Run via the container's `/workspace/isaaclab/_isaac_sim/python.sh`.
3. **Height reconciliation:** read `base_height_l2_pitch` (`mdp/rewards.py:82`) against the sim2sim per-segment
   height column.
4. **MuJoCo↔Isaac parity:** cross-read `params/env.yaml`, `unitree_go2witharm_cfg.py`,
   `deploy/profiles/go2_locomotion.yaml`, `sim2sim/go2_model.py`, `sim2sim/sim2sim_mujoco.py` (see §2a).
5. **Behaviour cross-check:** `sim2sim/sim2sim_mujoco.py --headless --scenario sim2sim/scenarios/basic.yaml
   --policy …/2026-11-03_22-14-59/exported/policy.pt` shows the low stance (0.05–0.12 m, MuJoCo) and 0 falls.
   For the Isaac-true stance height, `scripts/rsl_rl/play.py --trace_out …` then read the root-height channel;
   `sim2sim/compare_traces.py` quantifies per-foot clearance and joint-target offsets across the two engines.

## Correction (2026-11-04, after the hardware run)

The reward-balance table above is real, but its reading of the smoothness terms was wrong in one respect: those
penalties are computed on the *sampled* actions, and the trained policy's exploration noise std was 0.51-0.66
(`Policy/mean_noise_std`). At that noise the logged `action_rate_l2` (about 9.8 per step) is almost entirely the noise
itself; the deterministic policy on hardware has an action-rate value of about 0.001 in clean gait. So the
smoothness trio was mainly a tax on exploration noise, not evidence that the gait was being flattened by jerk
penalties. `foot_lift` being tiny next to `foot_sliding`, and `height_penalty` having almost no gradient, still stand.

Hardware (15:12 run) also showed the kneeling was milder than MuJoCo suggested: about 2 cm below the default stance by
a joint-angle estimate, against the 5 cm MuJoCo figure.
