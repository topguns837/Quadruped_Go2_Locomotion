# 9_10_26_vanilla_wrong_arm_mounting

**What this is:** baseline policy ("Policy A" in `go2_round2_reward_engineering_plan.md`'s terms) — original
reward system, trained from scratch, no Round 2 changes applied. Used as the control to compare Round 2's
gait-coordination/clearance/tracking-weight changes against (see the sibling run
`9_10_26_gait_motion_reward_wrong_arm_mounting`).

**Reward weights (relevant to the Round 2 comparison):**

| Term | Weight | Notes |
|---|---:|---|
| `track_lin_vel_xy_exp` | 10.0 | dominant term, unchanged in both runs |
| `track_ang_vel_z_exp` | 0.5 | original value |
| `track_pitch_exp` | 0.5 | original value |
| `track_lean_exp` | 0.3 | original value |
| `foot_lift` (`min_height`) | 0.1 weight, 0.05m target | original swing-clearance target |
| `gait_coordination` | — | not present in this run |

**Training:** 10000 max iterations, final checkpoint `model_9999.pt`. Full config: `params/agent.yaml`
(PPO/runner) and `params/env.yaml` (environment).

**Known issue:** the OpenManipulatorX arm's USD offset/orientation was incorrect (wrong mounting) during
this training run — the policy learned around a physically incorrect arm placement. Keep this in mind when
evaluating behavior involving the arm.
