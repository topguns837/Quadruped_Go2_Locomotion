# 9_13_26_round3_pitch_lean_height_weights

**What this is:** the Round 3 reward-engineering experiment (see `go2_round2_reward_engineering_plan.md`
and `patches/round3_weights_and_height.patch`), trained on the corrected Go2 + OpenManipulatorX asset.
Adds a `base_height` (root world-Z) observation term on top of the vanilla baseline's observation space
(9 obs terms vs. vanilla's 8 — **not** checkpoint-compatible with vanilla runs), a `gait_diagonal_coordination`
reward for trot-like diagonal foot sync, a raised `foot_lift` `min_height` (0.05m → 0.10m), and bumped
tracking/height weights. This is the best-performing, fully-complete run to date.

**Reward weights (vs. vanilla baseline):**

| Term | Vanilla | Round 3 |
|---|---:|---:|
| `track_ang_vel_z_exp` | 0.5 | 5.0 |
| `track_pitch_exp` | 0.5 | 3.0 |
| `track_lean_exp` | 0.3 | 3.0 |
| `height_penalty` | -1.65 | -2.5 |
| `foot_lift` (`min_height`) | 0.05m | 0.10m |
| `gait_coordination` | — not present | 0.30 |
| `rel_standing_envs` | 0.02 | 0.12 |

**Training:** started fresh at iteration 0 (not resumed from vanilla or Round 2 — those use a different
observation space / reward set). 17999 max iterations, **complete, no crash**. Final checkpoint
`model_17999.pt`. Full config: `params/agent.yaml` (PPO/runner) and `params/env.yaml` (environment).

**Measured this session** (via TensorBoard, `Train/mean_reward`): final reward ≈636.6, mean episode length
≈1914 — the highest final reward of any run so far (vanilla ≈335.5; the earlier Round 2 attempt peaked
≈584 before diverging, see caveat below).

**Relationship to Round 2:** Round 2 (`patches` history, not committed to `models/`) used the same
`base_height` observation addition and `gait_coordination` reward, but with `rel_standing_envs=0.02` (not
raised yet) and smaller tracking-weight bumps (`track_pitch_exp=2.0`, `track_lean_exp=2.0`,
`height_penalty=-1.65`). That run was resumed past 10k iterations and was tracking *even higher* reward
(~584 at iteration 17660) than this Round 3 run reached at the same point, but **diverged catastrophically**
at iteration 17670 (`Train/mean_reward` collapsed from ~395 to ~-3.6e8 in a single logged step — a NaN/Inf-
style blowup) and was never resumed further. Round 3 was trained fresh rather than continuing from Round 2's
last-safe checkpoint (`model_17500.pt`, not committed here). Worth keeping in mind: Round 2's raw reward
trajectory before the crash suggests there may be more headroom above this run's 636.6 with more stable
hyperparameters (e.g. gradient/value clipping) — this run is the safe, complete result, not necessarily the
final word on this reward configuration's ceiling.

**No known mounting/asset issues** — trained on the corrected `go2withOpenXstatic.usd` asset.
