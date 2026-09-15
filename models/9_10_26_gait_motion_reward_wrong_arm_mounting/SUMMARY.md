# 9_10_26_gait_motion_reward_wrong_arm_mounting

**What this is:** Round 2 reward-engineering policy ("Policy B" in `go2_round2_reward_engineering_plan.md`'s
terms) — trained from scratch (not fine-tuned from the baseline), adding a trot-like diagonal gait
coordination reward, a stricter swing-foot clearance target, and higher angular-velocity/pitch/lean tracking
weights on top of the otherwise-unchanged baseline reward system. Compare against the sibling run
`9_10_26_vanilla_wrong_arm_mounting` (same PPO config, same 10000 max iterations, reward system is the only
deliberate difference).

**What changed vs. the vanilla baseline:**

| Term | Vanilla | This run | Notes |
|---|---:|---:|---|
| `track_lin_vel_xy_exp` | 10.0 | 10.0 | unchanged, still the dominant term |
| `track_ang_vel_z_exp` | 0.5 | **5.0** | |
| `track_pitch_exp` | 0.5 | **2.0** | |
| `track_lean_exp` | 0.3 | **2.0** | |
| `foot_lift` swing-clearance target | 0.05m | **0.10m** | `min_height` param, weight unchanged (0.1) |
| `gait_coordination` | — | **new, weight 0.30** | diagonal sync + alternating-support reward, gated on commanded speed > 0.15 m/s |

**Training:** 10000 max iterations, final checkpoint `model_9999.pt`. Full config: `params/agent.yaml`
(PPO/runner) and `params/env.yaml` (environment).

**Known issue:** the OpenManipulatorX arm's USD offset/orientation was incorrect (wrong mounting) during
this training run — the policy learned around a physically incorrect arm placement. Keep this in mind when
evaluating behavior involving the arm.
