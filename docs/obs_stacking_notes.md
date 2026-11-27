# Observation stacking (history of frames) — implementation notes

Status: code written, NOT committed, NOT yet verified in the container. Date: 2026-11-18.

## Goal
Feed the policy the last **H = 6** observation frames instead of only the most recent one. At the 50 Hz control
rate (dt 0.005 × decimation 4 = 20 ms/step) that is a 120 ms window. A short history lets the MLP read trends and
latency and, more importantly for this project, implicitly system-identify the randomised dynamics (friction, mass,
PD gains, 0–2-step action delay) that `EventCfg` applies — the main sim-to-real benefit. Matches the 6-frame stack
used by the referenced `Glowing-Torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot` project.

Decisions: **H = 6**, stack **both actor and critic**, **whole group incl. the command**.
Resulting input dims: **actor 47·6 = 282**, **critic 50·6 = 300**.

## How to decide H (the number of stacks)
H is a hyperparameter; choose it by covering the physics you need, then sweep:
- **Window must cover latency + a slice of dynamics.** Action delay here is 0–2 steps (0–40 ms) plus sensor/comms
  lag; ~100–120 ms (H = 5–6) covers that and a sub-stride of the ~300–500 ms trot without memorising a whole gait.
- **Diminishing returns vs cost.** Each frame adds 47 dims (H = 6 → 282-dim actor input), a bigger first layer, more
  params, slower rollouts, more overfitting and more to normalise. For proprioceptive locomotion gains flatten past
  ~5–8 frames. Legged-gym / Go2 sim-to-real work uses either 1 (rely on velocity terms) or a 5–10 frame history for
  history-based system-id; the reference's 6 sits in that band.
- **Decide empirically.** Pick a principled default, sweep a few values (e.g. {1, 5, 6}) for a few k iters each, and
  compare on the existing random-command sim2sim eval (`src/sim2sim/run_sim2sim_eval.sh`: tracking RMS + falls), ideally
  confirmed on hardware. Pick the **smallest** H that captures the gain — cheaper and less overfit.

## Changes made

### Training
- `source/.../env_cfg.py`
  - New module constant `HISTORY_LENGTH = 6`.
  - `PolicyCfg.__post_init__` and `CriticCfg.__post_init__`: set `self.history_length = HISTORY_LENGTH` and
    `self.flatten_history_dim = True`. Isaac Lab's `ObservationManager` maintains the per-env ring buffer and clears
    it on reset — no custom env code.
- `source/.../agents/rsl_rl_ppo_cfg.py`
  - `experiment_name = "go2_kp40_iter5_hist6"`. `empirical_normalization=True` and hidden dims `[256,512,128]`
    unchanged (normalises the full 282/300 vector transparently).

### Deploy mirror (`src/deploy/deploy_real.py`)
- `DeployConfig.history_length` (default 1), read from the YAML config in `load_config`.
- `policy_obs_layout(policy, history_length)`: reads the net's first Linear input (the STACKED width, e.g. 282),
  asserts it divides by `history_length`, and that the per-frame dim (47) is a known layout.
- `frame_term_widths(base_dim, layout)`: per-term widths within one frame, in `build_observation` order.
- `ObsHistory`: per-step ring buffer, startup-padded by repeating the first frame H times (mirrors Isaac Lab's
  CircularBuffer reset fill), outputs the stacked vector in the layout set by two flags (see below).
- Main loop: build the single 47-dim frame as before, then `obs = obs_history.push_and_stack(frame)`.
- `src/deploy/deploy_real_wireless.py`: same wiring (and it now correctly sets `include_base_lin_vel`, which it
  previously left at the `True` default — already wrong for 47-dim policies).

### sim2sim mirror (`src/sim2sim/sim2sim_mujoco.py`)
- `--history_length` CLI (default: config's `history_length`, else 1).
- `load_policy(path, history_length)` returns the per-frame base dim; the loop stacks via `deploy_real.ObsHistory`
  and resets it (and the obs-delay buffer) on a fall. Frame **delay** (`--obs_delay`, which frame) and frame
  **history** (stack the last H) are orthogonal and compose.
- `src/sim2sim/compare_traces.py`: `newest_frame()` reduces a stacked trace to its newest single frame (term-major
  aware) so per-term stats don't mis-slice or crash on the 282-dim width.

## THE LAYOUT RISK — must verify in the container before trusting it
Isaac Lab applies a **group** `history_length` by pushing it down to each **term** and concatenating the per-term
histories (`concatenate_terms=True`). So the flat layout is **term-major**, not frame-major:

    [ang_vel(t-5..t), gravity(t-5..t), command(t-5..t), joint_pos(t-5..t), joint_vel(t-5..t), last_action(t-5..t)]

with each term's samples newest-**last**. `deploy_real.py` reproduces this under two flags:
`OBS_HISTORY_TERM_MAJOR = True`, `OBS_HISTORY_NEWEST_LAST = True`. These are my best reading of Isaac Lab's
internals but were **not** verified here (Isaac Lab source lives in the container, not on this host). If the flags
are wrong, the policy sees scrambled history on hardware.

**Gate before training/deploy:** run the auto-discovery check in the container, from the repo root:

    ../IsaacLab/isaaclab.sh -p src/play/verify_obs_stacking.py --task Quadruped-Locomotion-Go2-Play \
        --num_envs 1 --history_length 6 --steps 20 --headless

It runs the real env twice (H=1 and H=6, same seed, zero actions, deterministic reset), rebuilds the stacked vector
from the single frames under all four layout combos, reports which matches the env, and checks that
`deploy_real.ObsHistory` reproduces the env vector bit-for-bit. `PASS` → flags correct. Otherwise it prints the two
flag values to set.

## Forced implications
- **From-scratch retrain.** The actor input dim changes 47 → 282, so the first Linear layer changes shape — no
  resume from `model_24999` or any single-frame checkpoint.
- Deploy config and sim2sim must set `history_length = 6` to match the policy.

## Verified on the host (without Isaac Lab)
- All six edited files + the new script parse (`ast.parse`).
- Term-major stacking and its `newest_frame` inverse are correct (numpy simulation).
- `ObsHistory` padding/ordering correct; the four layout combos are distinguishable (auto-discovery is sound).

## Not yet done
- Container run of `src/play/verify_obs_stacking.py` (the layout gate).
- Training smoke test showing actor input = 282 / critic = 300 and a `policy.pt` export of width 282.
- `sim2sim_mujoco.py --history_length 6` zero-mismatch check against a short Isaac trace.
- Nothing committed.

## Hardware buzz finding (2026-11-18) — why this retrain matters, and gains to change
The real robot shook with a ~24.9 Hz every-tick ("Nyquist") oscillation. Investigation (sim2sim + hardware logs):
- **Cause = the single-frame policy amplifying observation noise**, dominated by the large joint-velocity obs noise
  (±1.5 rad/s) it was trained with. In MuJoCo the buzz did NOT reproduce from command replay + latency (obs_delay up
  to 40 ms stayed smooth), but DID reproduce the moment training-level observation noise was injected (action lag-1
  autocorr flipped +0.36 → −0.57, dose-responsive). MuJoCo has clean analytic joint velocities, which is why latency
  alone never triggered it.
- **A 5 Hz action low-pass on hardware clearly helped (not chance):** comparing the unfiltered run
  (`logs/hardware_dashboard/2026-11-18_14-19-26`) to the filtered one (`...15-17-03`, `--low_pass`), raw-action lag-1
  went −0.70 → **+0.40**, near-Nyquist power 2% → **0%**, body gyro RMS 0.187 → **0.135 (−28%)**, joint-vel RMS
  0.322 → **0.192 (−40%)** — and the filtered run used *more* aggressive commands. Mechanism: the smoothed sent
  target feeds back as `last_action`, so the policy's own feedback is clean and the amplification loop breaks at the
  source (the filter is in `src/deploy/deploy_real.py` `--low_pass`, default 5 Hz; validated by
  `sim2sim_mujoco.py --action_lpf_hz`).
- **Bearing on this retrain:** observation stacking (H=6) + a smaller joint-velocity obs noise is the *root* fix (the
  LPF is a deploy band-aid that masks it). In sim, lowering joint_vel noise reduced the buzz AND improved tracking —
  pure win. So for the next training run, besides the stacking already configured, **reduce the joint_vel observation
  noise** (from ±1.5) and consider an action-smoothness (2nd-order) penalty; see `logs/buzz_repro/` and the sim2sim
  knobs `--obs_noise`, `--jv_noise_scale`, `--action_lpf_hz`, `--kp/--kd`.

### Gains: the no-buzz reference deploy uses different kp/kd — change for the next training (DO NOT FORGET)
The `Glowing-Torch/Deploy-an-RL-policy-...` Go2 deploy (which does not report buzz) runs **kp = 50, kd = 1.0** and
stacks **6 frames**. We train and deploy at **kp = 40, kd = 0.8, single frame**. Higher kd damps the physical
response (sim: kd 1.5 brought joint-velocity/body shake back to clean levels, at a tracking cost; kd does NOT fix the
command chatter — only noise-robustness does). **Action item for the next training session: consider matching
kp≈50 / kd≈1.0** so sim, MuJoCo and the real robot agree, and **widen the PD-gain domain-randomisation band**
(currently ×0.9–1.1 of kp40/kd0.8 = kd 0.72–0.88, which does not even cover 1.0) so a kd≈1.0 deploy is in-distribution.

## Files touched
- `source/.../env_cfg.py`
- `source/.../agents/rsl_rl_ppo_cfg.py`
- `src/deploy/deploy_real.py`, `src/deploy/deploy_real_wireless.py`
- `src/sim2sim/sim2sim_mujoco.py`, `src/sim2sim/compare_traces.py`
- `src/play/verify_obs_stacking.py` (new)
- `CLAUDE.md` (Observations bullet updated)
- Buzz investigation (separate, deploy-side): `src/deploy/deploy_real.py` (`--low_pass`), `docker/isaaclab-shell.sh`
  (tmux default), `src/sim2sim/sim2sim_mujoco.py` (`--obs_noise`/`--jv_noise_scale`/`--action_lpf_hz`/`--kp`/`--kd`),
  `src/sim2sim/analyze_hw_trace.py`, `src/sim2sim/scenario_from_trace.py` (new); traces in `logs/buzz_repro/`.
