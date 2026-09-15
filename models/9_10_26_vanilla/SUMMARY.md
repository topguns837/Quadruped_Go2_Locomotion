# 9_10_26_vanilla_corrected_arm_mounting

**What this is:** baseline policy, original reward weights, trained from scratch on the **corrected**
Go2 + OpenManipulatorX asset (`go2withOpenXstatic.usd`, arm USD offset/orientation fixed — unlike the
earlier `9_10_26_vanilla_wrong_arm_mounting` run, which used the incorrectly-mounted arm). This is the
control/baseline for comparing against runs trained with the Round 2/3 reward changes on the same corrected
asset.

**Reward weights (all original defaults, unchanged from the very first baseline):**

| Term | Weight |
|---|---:|
| `track_lin_vel_xy_exp` | 10.0 |
| `track_ang_vel_z_exp` | 0.5 |
| `track_pitch_exp` | 0.5 |
| `track_lean_exp` | 0.3 |
| `foot_lift` (`min_height`) | 0.1 weight, 0.05m target |
| `gait_coordination` | — not present |

**Training:** 10000 max iterations (complete, not interrupted), final checkpoint `model_9999.pt`. Full
config: `params/agent.yaml` (PPO/runner) and `params/env.yaml` (environment).

**Measured this session** (via TensorBoard, `Train/mean_reward` trend over the last 200 iterations):
final reward ≈335.5, and the run had **plateaued** by iteration 10000 — reward was flat/slightly declining
(Δ −3.1 over the last 200 iterations), noise std barely moving. Unlike the experimental run from the same
day, 10000 iterations appears to have been enough for this simpler reward config; extending it further is
unlikely to change much.

**No known mounting/asset issues** — this is the first vanilla run on the corrected asset, so (unlike the
two earlier `_wrong_arm_mounting` runs) there's no arm-offset caveat to flag here.
