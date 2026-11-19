# Plan: penalize standing still when a linear velocity is commanded

Status: plan only. No code changed for this yet. Date: 2026-11-04.

## Context

The iteration-2 reward set (alive +2, fall -500, clamped penalties, two-scale velocity tracking, hard stand gate)
trains stably, but 400-iteration calibration shows the policy still stands still at small linear commands
(0.1-0.2 m/s) and barely moves at 0.3. The change below replaces the current speed-deficit term with a scale-free,
two-sided penalty for standing still while a linear velocity is commanded.

Why the current `track_lin_vel_deficit` fails: it is `relu(|cmd| - v_proj)` in m/s, so its pull shrinks with the
command. Standing at a 0.1 m/s command costs 1.5 x 0.1 = 0.15/s, against roughly 5 of 10 velocity points that a
standing robot still banks from the broad tracking kernel. Raising its weight (ablation B) fixed low speed but then
costs 4.5 x 0.8 = 3.6/s of standing at 0.8 m/s, which is where B's falls and 1.6-1.9x overshoot come from. The
term is also one-sided, so overshoot is free.

## Evidence (held-command probe, 400-iteration checkpoints, one seed each)

Forward speed divided by command, held for 10 s in the Play env (nominal physics):

| Run | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 | 0.8 | Falls at 0.5 / 0.8 |
|---|---|---|---|---|---|---|---|
| No deficit, eased trio (earlier run) | stands | 0.32x | 0.75x | 1.32x | 1.26x | 1.06x | 0 / 0 |
| Main: deficit -1.5, heavy trio | stands | stands | stands | 0.04x | 1.22x | 1.56x | 0 / 1 |
| A: deficit -1.5, eased trio | stands | stands | 0.10x | 0.76x | 1.31x | 0.77x | 0 / 1 |
| B: deficit -4.5, heavy trio | 1.60x | 1.83x | 1.91x | 1.75x | 1.38x | 1.09x | 27 / 57 |

Training side (mean of the last 20 iterations): main reward 226, arm-contact 0.21; A 238 and 0.20; B 165 and 0.26.
Backward walking (-0.2 / -0.3) is weak in every new config (B reaches 0.70x at -0.3).

## Design: `standing_when_commanded`

Add to `mdp/rewards.py`, replacing `track_lin_vel_deficit`:

    cmd_mag = ||cmd_xy||
    cmd_hat = cmd_xy / max(cmd_mag, 1e-6)
    v_proj  = dot(root_lin_vel_b[:, :2], cmd_hat)          # signed speed along the commanded direction
    penalty = clamp(|cmd_mag - v_proj| / cmd_mag, 0, 1) * (cmd_mag > _CMD_EPS)

- Scale-free: standing at any commanded speed gives 1.0, so the pull off standstill is the same at 0.1 m/s as at
  1.0 m/s. A robot that tracks well pays about 0.
- Two-sided: overshoot and backward motion are penalized too (clamped at 1), which removes B's free overshoot.
- Bounded in [0, 1], so it needs no extra clamp. Gated off for a zero linear command, where
  `stand_still_default_pose` owns the reward (same `_CMD_EPS` = 0.01 gate as the other terms).
- Weight: standing costs the weight itself, so high-speed pressure is much lower than B's. Calibrate two candidates:
  -2.0 and -4.0.
- Unchanged: `track_ang_vel_deficit`, the heavy smoothness trio (-0.03 / -5e-7 / -3e-5, the user's decision),
  and everything else in the reward set. Only the linear L1 deficit is removed.

Not chosen:
- An indicator penalty for "all four feet planted while commanded": discontinuous, and gives no credit for partial
  progress.
- Scaling up the tracking weights alone: the ratio of standing credit to moving credit would not change. Kept as
  ablation C (linear 10 -> 16, yaw 2 -> 3.2) if this design fails.

## Files to change

- `source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/mdp/rewards.py`:
  add `standing_when_commanded` (same pattern as `track_lin_vel_deficit`, logged via `diagnostics.log_step`);
  remove `track_lin_vel_deficit`.
- `.../mdp/__init__.py`: swap the export.
- `.../quadruped_go2_locomotion_env_cfg.py` (`RewardsCfg`): replace the `track_lin_vel_deficit` term with
  `standing_when_commanded`, weight chosen by calibration, with a comment explaining scale-free and two-sided.

## Verification

1. `ast` parse, plus a pure-Python truth table of the formula: standing at 0.1 / 0.3 / 1.0 -> 1.0; perfect
   tracking -> 0; half speed -> 0.5; 2x overshoot -> 1.0 (clamped); backward -> 1.0; command < 0.01 -> 0;
   diagonal command with aligned velocity -> 0.
2. Re-run the ledger and stand-gate probes in an isolated `--network none` container. Net reward per second must
   stay positive in every scenario, including noise sigma = 1.0, and `stand_still` must pay only for an all-zero
   command.
3. Two 400-iteration calibration runs concurrently (weights -2 and -4, scratch trees, 1024 envs), then the
   held-command probe (0.1-0.8 m/s forward, -0.2 / -0.3 backward).
   Pass criteria: steps at 0.1-0.2 m/s (at least about 0.5x command), overshoot at most about 1.3x, falls at
   0.5 / 0.8 near 0 (like main and A, not B's 27 / 57), training mean reward and arm-contact fraction no worse than
   main (about 226 and 0.21).
4. Only if a weight passes: commit the iteration-2 changes (stage `rsl_rl_ppo_cfg.py` by hunk; keep the stairs WIP
   and `deploy/wireless/hostapd.conf` out), then launch the 18k run. If neither weight passes, fall back to
   ablation C or the earlier no-deficit config.

## Results (2026-11-04, 400 iterations, one seed each)

Implemented as designed. Truth table, ledger (net reward positive in every scenario) and stand-gate probes pass.

| Run | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 | 0.8 | -0.2 | -0.3 | Falls at 0.8 | Reward | Vel. error | Arm contact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W2 (weight -2) | 2.00x | 1.87x | 1.89x | 0.33x | 0.29x | 0.23x | 1.83x | 1.66x | 7 | 167 | 1.32 | 0.19 |
| W4 (weight -4) | stands | stands | 0.85x | 1.33x | 1.63x | 1.57x | 0.76x | 1.11x | 0 | 193 | 0.91 | 0.23 |

Neither met every pass criterion. The weights ranked non-monotonically (-2 moved at 0.1 m/s, -4 did not), so
single-seed variance at 400 iterations is at least as large as the effect. Both policies hold roughly one preferred
speed regardless of command, i.e. command-proportional tracking is not learned yet at 400 iterations.

Decision: -4.0 goes into the 18k run (best tracking, no falls). Check low-speed stepping (0.1-0.2 m/s) and overshoot
on the 2k and 5k checkpoints with the held-command probe; if it still stands below 0.2 m/s, revisit.

## Caveats

- One seed per config, so some of the gaps in the table may be noise.
- 400 iterations cannot certify the 18k-iteration outcome; low-speed stepping must be re-checked on later
  checkpoints and on hardware.
