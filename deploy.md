# Real-hardware deployment pipeline for trained Go2 policies

## Context

The goal is to test trained policies (`models/*/exported/policy.pt` / `policy.onnx`) on the physical
Unitree Go2. This repo currently contains **zero** hardware-deployment code — confirmed via a full repo
search (no `unitree_sdk`, no `deploy/`, no ROS nodes, no onnxruntime/pyserial dependency, nothing in
`pyproject.toml`). It's purely an Isaac Lab sim training pipeline today. This doc answers three questions:
whether deployment code exists (no), how to use the exported `.pt`/`.onnx` files on hardware, and whether
control runs from the dev machine via SDK or onboard the robot.

**Answer to the SDK-vs-onboard question**: use `unitree_sdk2_python` (DDS-based) from a **development
machine connected to the robot over a direct Ethernet cable**. This is the standard, documented pattern used
by Unitree's own reference deployments (e.g. `unitreerobotics/unitree_rl_gym`) — DDS communication works
over the wire the same whether the process runs on the dev machine or Go2's onboard Jetson, so nothing is
gained by running onboard except not needing a cable; the dev machine is simpler to iterate on (no
cross-compiling/pushing code to the robot) and is the recommended starting point. Onboard deployment (SDK on
the robot's own Jetson) can be revisited later purely as a "cut the cable" convenience once the dev-machine
pipeline is proven safe and correct — it changes nothing about the control logic itself.

## Sim2real gaps specific to this repo (resolved)

- **Depth camera observation**: NOT a real gap — `mdp.observations.depth_array` and the `depth_camera`
  `TiledCameraCfg` in `quadruped_go2_locomotion_env_cfg.py` are dead code (the camera cfg is wrapped in a
  docstring, and `depth_array` is never referenced in `ObservationsCfg.PolicyCfg`). Ignore entirely.
- **`base_pos_z` (height) observation**: real gap, **currently unresolved, deliberately deferred**. Original
  plan was to read it from `SportModeState`'s fused body-position estimate, falling back to forward
  kinematics if that proved unreliable — but `SportModeState`'s real-hardware availability *after* releasing
  the onboard motion-control service (a required step for `LowCmd` to work at all, see Stage 2.5) is
  unconfirmed (web research suggests it may stop publishing once released; not yet checked against Unitree's
  own docs or hardware). Rather than build either path on that uncertain footing, `deploy_real.py` currently
  sends a **fixed constant** (`DEFAULT_HEIGHT_M = 0.3`) for this observation. Revisit once hardware is
  available — see Stage 3.
- **Joint order mapping (the classic sim2real bug)**: Isaac Lab's articulation joint order (USD-defined, via
  `unitree_go2witharm_cfg.py`) will NOT match the Go2 SDK's `LowCmd`/`LowState` motor index order. The SDK
  side is now **confirmed** (not just documented convention) — fetched and read `unitree_sdk2py`'s own
  `example/go2/low_level/unitree_legged_const.py` directly: FR(0,1,2), FL(3,4,5), RR(6,7,8), RL(9,10,11).
  The Isaac Lab side is still a placeholder guess pending `deploy/dump_joint_order.py` — see Stage 2. Same
  applies to action scale (0.25) + default joint offsets, which must be reproduced exactly (done, copied
  directly from `unitree_go2witharm_cfg.py`/`ActionsCfg`, not generic hardware defaults).
- **Command vector shape**: this repo's commands are 6-dim (lin_x, lin_y, ang_z, pitch, lean, height) vs.
  stock Go2's 3-dim joystick (lin_x, lin_y, yaw rate). The existing `scripts/manual_command_slider.py` GUI
  (communicates via a JSON file) is directly reusable as the real-robot command source instead of parsing
  the Unitree remote's joystick channels — it already produces exactly this 6-dim vector, and
  `deploy/deploy_real.py` polls it the same way `scripts/rsl_rl/play.py` does (verbatim-ported poll loop).
- **Control frequency mismatch (robot's native publish rate vs. the policy's trained rate)**: the policy
  must be stepped at exactly 50 Hz (`decimation=4` × `sim.dt=0.005` in
  `quadruped_go2_locomotion_env_cfg.py`), independent of whatever rate the robot's own `LowState` publishes
  at (typically much faster). Resolved by decoupling the two: the DDS subscriber callback only ever
  overwrites a "latest state" snapshot (no processing), and the main control loop runs on its own
  fixed-rate 20ms timer, sampling that snapshot once per tick — see `deploy_real.py`'s
  `LatestRobotState`/main loop. The robot's own onboard joint controller handles fine-grained PD servoing
  between our updates using the `kp`/`kd`/`q` sent each tick, the same conceptual relationship as sim's
  `decimation`.
- **Live dashboard + slider, reused unchanged**: `deploy_real.py --live_plot` writes the exact same 12
  TensorBoard tag names (`Metrics/base_velocity/{cmd,actual}_*`) that `scripts/live_dashboard.py`'s `PANELS`
  already expect (just pointed at `logs/hardware_dashboard/` instead of `logs/play_dashboard/`), and
  `--manual_commands` launches the identical `scripts/manual_command_slider.py` subprocess play.py does —
  zero changes needed to either script.
- **Exported policy format**: `models/*/exported/policy.pt` is a TorchScript export (RSL-RL/Isaac Lab
  standard) — loadable with plain `torch.jit.load(...)` and no Isaac Sim/Kit runtime required, confirmed by
  how these were exported. This is what real-time inference on the dev machine will load directly; ONNX is
  available as a fallback if `onnxruntime` inference is preferred over LibTorch.

## Proposed pipeline (staged, safety-first)

### Stage 0 — Physical safety setup (no code)
Robot hoisted off the ground for all initial tests. Learn the controller's debug-mode sequence
(`L2+R2` → damping mode) and emergency stop (`select` button / `Ctrl+C` → damping) before running anything.
Direct Ethernet cable from dev machine to Go2; identify the network interface (`ifconfig` after connecting).

### Stage 1 — `deploy/` directory — **done, and now checked against the real SDK, not just documentation**
- `deploy/configs/go2_locomotion.yaml` — joint order/index mapping (both `isaac_joint_order` and
  `sdk_joint_order`, remapped via `build_joint_index_maps`), PD gains (`kp=25.0`, `kd=0.5`, copied from
  `unitree_go2witharm_cfg.py`'s actual training-time actuator config, not generic hardware defaults),
  `action_scale=0.25`, `default_joint_pos` per joint, `control_dt=0.02`, `policy_path` (points at the
  **exported** `policy.pt`, not the raw RSL-RL checkpoint — see that file's comment for why).
- `deploy/deploy_real.py` — the control loop, rewritten this session after fetching and reading
  `unitree_sdk2py`'s own `example/go2/low_level/go2_stand_example.py` reference example line-by-line and
  comparing against what was here before (written from general knowledge, never checked). Real gaps found
  and fixed: `LowCmd.head`/`level_flag`/`motor_cmd[i].mode` were entirely missing (required for the robot to
  accept the command at all, set once at init via `init_low_cmd`); the mode-release prerequisite was missing
  entirely (see Stage 2.5); publishing is now two-rate, not one — a `RecurrentThread` at ~500 Hz
  (`publish_low_cmd`) continuously republishes whatever's in a shared `LowCmdTarget` buffer, decoupled from
  the 50 Hz policy-inference loop that only updates that buffer (matches the reference exactly). Height
  tracking is a fixed constant for now (`DEFAULT_HEIGHT_M`), not a real estimate — see Stage 3.
- `deploy/dump_joint_order.py` — prints Isaac Lab's real articulation joint order (needs the sim, no
  hardware) so `isaac_joint_order` in the config can be filled with ground truth instead of a guess.
- Dependency: `unitree_sdk2py` — **now actually installed** in the running container and baked into
  `docker/Dockerfile` (pinned to commit `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`) so it persists across
  container recreations. Not on PyPI at all — installed from source, which required two non-obvious fixes,
  both now in the Dockerfile with full explanations: (1) its `cyclonedds==0.10.2` dependency needs the
  native CycloneDDS C library built from source first (no PyPI wheel, no apt package for this platform);
  (2) unitree_sdk2py's install silently upgraded `numpy` to 2.x, breaking `isaaclab`/`isaaclab-rl`/
  `isaaclab-tasks` (all pin `numpy<2`) — caught and re-pinned immediately. `deploy_real.py` runs under Isaac
  Sim's bundled Python directly (not the `isaaclab -p` wrapper, since it never touches `pxr`/`SimulationApp`).
- New `hardware` tmux window in `docker/isaaclab-shell.sh`, mirroring `train`/`play`'s 3-pane layout
  (control loop | dashboard-gen | `feh` viewer). The pre-typed command defaults to `--dry_run` with a
  literal `REPLACE_WITH_NIC_NAME` placeholder (deliberately not `<FILL_IN_NIC>` — angle brackets are shell
  redirect operators and were confirmed to silently corrupt the command line) — not ready to run against a
  real robot without editing first.
- **Verified this session, now against the real installed SDK** (still no physical robot): `ast.parse`
  syntax check; real module import; `build_joint_index_maps` produces a valid permutation;
  `build_observation` produces exactly the right dimension (52 experimental / 51 vanilla) **with a real
  forward pass through both actual exported policies**; `--dry_run` paces at ~50 Hz. Additionally, with the
  SDK actually installed: every import path resolves correctly (`ChannelFactoryInitialize`, `LowCmd_`/
  `LowState_`/`SportModeState_`, `CRC`, `RecurrentThread`, `SportClient`, `MotionSwitcherClient`); on the
  loopback interface (`lo`, no robot, no network traffic leaves the machine) — `SportClient`/
  `MotionSwitcherClient` construct and `Init()` successfully, `CRC().Crc(...)` computes a real checksum
  (after fixing a missing native `.so` file, see the Dockerfile), `ChannelPublisher`/`ChannelSubscriber`
  construct and a `Write()` call doesn't crash, and the new `RecurrentThread`-based 500 Hz publish loop runs
  for real without error.
- **Still not verified** (needs actual hardware): whether `release_motion_control`'s `StandDown()`+
  `ReleaseMode()` sequence actually releases a real robot's onboard controller as expected; whether
  `SportModeState` keeps publishing after that release (see Stage 3); `sdk_joint_order`'s convention is now
  sourced from Unitree's own example rather than guessed, but still not cross-checked against this specific
  unit's firmware.

### Stage 2 — Joint mapping verification (before any torque) — **SDK side confirmed, Isaac side still pending**
`sdk_joint_order` in the config is now sourced directly from `unitree_sdk2py`'s own
`unitree_legged_const.py` (fetched and read this session), not just "standard convention" — still worth a
final cross-check against your specific unit's firmware docs, but no longer a guess. `isaac_joint_order` is
still a **placeholder guess** (alphabetical FL/FR/RL/RR × hip/thigh/calf) — run `deploy/dump_joint_order.py`
(needs the sim, no robot, no other sim session running concurrently) to get the real order and replace it
before any hardware use.

### Stage 2.5 — Release the robot's onboard motion control — **implemented, unverified against hardware**
New, previously-missing prerequisite, found by comparing against the reference example: `LowCmd` does
nothing useful while the robot's onboard sport-mode controller is still active — it'll fight or ignore
custom commands. `deploy_real.py`'s `release_motion_control()` now runs this before anything else touches
`LowCmd`: `SportClient`/`MotionSwitcherClient` init, then loop `StandDown()` + `ReleaseMode()` until
`CheckMode()` reports no active mode. Independently corroborated by `docs.quadruped.de`'s Go2
"Low-Level Control" page (stand_down → damp sequence, same idea from a second source) — that page also
carries a direct warning: *"The GO2 in low-level mode can easily be damaged if used incorrectly."* The
object construction and method calls are verified working (on loopback, no robot); whether it actually
releases control on a real unit is not.

### Stage 3 — Height-estimate wiring — **deferred to a fixed constant, not implemented for real**
Height tracking isn't working yet, and rather than guess at a fix, `deploy_real.py` now always sends
`DEFAULT_HEIGHT_M = 0.3` (the robot's nominal standing height) for this observation — no `SportModeState`
subscription for height, no forward-kinematics stub pretending to be a real implementation. The open
question this was blocked on remains open: `SportModeState`'s real-hardware availability *after*
`ReleaseMode()` (Stage 2.5) is unconfirmed — web research suggests it may stop publishing once the onboard
controller releases, which would make it useless for exactly the state the policy needs it in, but this
hasn't been checked against Unitree's own docs or a real robot. `state.sportmode_velocity` (used for
`base_lin_vel`) is left wired up to the same subscription despite this same open question — flagged in code
comments, not silently trusted, but not simplified to a constant since there's no evidence yet it
specifically fails. Revisit both once hardware is available to test directly.

### Stage 4 — Dry run (hoisted, zero command)
`deploy_real.py --dry_run` already exists (Stage 1) and has been smoke-tested off-hardware. With the robot
connected and hoisted: run the same flag, slider at all-zero, confirm the printed target-joint-position
logs look sane and the loop holds ~50 Hz against real `LowState` data (not the zeroed stand-in state used
in this session's non-hardware test) before ever removing `--dry_run`.

### Stage 5 — Hoisted policy test, then grounded
With the robot still hoisted, enable actual command publishing with small/zero commands, watch leg motion
matches sim expectations. Only once that looks correct, lower the robot gradually per the standard Unitree
procedure (`start` → default pose → lower hoist → `A` → walk).

## How to run a trained `.pt` policy on real hardware (step-by-step)

This walks through actually using `deploy/deploy_real.py` — from "I have an exported `policy.pt`" to "the
robot is executing it." Read the safety notes at each step before running the command below it; this file
sends real torque to real motors once you're past the dry-run stages.

### 0. One-time setup (per machine, not per run)

`unitree_sdk2py` is already installed in this project's container image (baked into `docker/Dockerfile`,
pinned to a specific commit) — nothing to install yourself if you're using `docker/isaaclab-shell.sh`'s
container. If you're on a fresh container that predates this, rebuild the image
(`docker/build.sh` or equivalent) so the install actually lands.

Confirm it's there:
```bash
/workspace/isaaclab/_isaac_sim/python.sh -c "import unitree_sdk2py; print('ok')"
```

### 1. Pick which policy to run

Both committed models work with this pipeline — pick based on what you're testing:
- **Vanilla**: `models/9_10_26_vanilla/exported/policy.pt` — simpler baseline, no pitch/lean/height weight
  bumps, no `base_height` observation (51-dim input).
- **Experimental (Round 3)**: `models/9_13_26_pitch_lean_height_weights_experimental/exported/policy.pt` —
  the best-performing trained policy so far (52-dim input, includes `base_height`, but see the height-gap
  caveat below — it's fed a constant, not a real estimate, on hardware right now).

Set `policy_path` in `deploy/configs/go2_locomotion.yaml` to whichever `exported/policy.pt` you're using —
**not** the raw `model_<N>.pt` checkpoint (that's the full RSL-RL training state, not a standalone inference
module; see the config's own comment). If you switch policies later, remember `has_height_obs` is inferred
from the path containing `"round3"` or `"experimental"` — don't rename the file to something that breaks
that check, or the observation vector will be the wrong size and `torch.jit.load`'s forward pass will fail
loudly (a real, if unlikely, footgun worth knowing about).

### 2. Get the joint order right (offline, no robot needed, do this once per asset)

```bash
# GPU-safe: check nvidia-smi first, don't run this while another sim session has the GPU.
isaaclab -p deploy/dump_joint_order.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1 --headless
```
Copy the printed order into `isaac_joint_order` in `deploy/configs/go2_locomotion.yaml`, replacing the
current placeholder guess. `sdk_joint_order` is already filled in correctly (confirmed against Unitree's own
SDK source this session) and shouldn't need touching unless Unitree changes their convention.

### 3. Test everything except real hardware first

Two tools exist specifically so you can exercise the whole pipeline with zero robot risk:

- **`--dry_run`**: runs the full loop (state read → observation → inference → target computation → timing)
  but never calls the DDS publisher. Works with zeroed stand-in state, no DDS at all:
  ```bash
  /workspace/isaaclab/_isaac_sim/python.sh deploy/deploy_real.py --network_interface lo --dry_run
  ```
- **`deploy/fake_robot.py`**: a synthetic robot that publishes real `LowState`/`SportModeState` DDS messages
  and logs whatever `LowCmd` it receives back — this exercises the *entire* real message-passing path (real
  DDS pub/sub, real observation assembly from received data, real policy inference, real `LowCmd`
  construction/CRC/publish) with nothing simulated except the robot itself. Run it in one terminal:
  ```bash
  /workspace/isaaclab/_isaac_sim/python.sh deploy/fake_robot.py --network_interface lo
  ```
  It will also demonstrate `release_motion_control()`'s fail-safe behavior if you try running
  `deploy_real.py` for real (not `--dry_run`) against it: since `fake_robot.py` doesn't implement the
  `SportClient`/`MotionSwitcherClient` RPC service side, `release_motion_control()` will correctly retry and
  then raise `RuntimeError` rather than silently proceeding — that's the intended behavior when the robot's
  state genuinely can't be confirmed, not a bug in either tool.
  
  Both tools were used together this session to verify the full pipeline works end-to-end (1000+ real
  `LowCmd` messages round-tripped correctly, joint targets converged to sane values) without any physical
  robot — see git history / session notes for the exact test if you want to reproduce it.

### 4. First real connection (robot hoisted, per Stage 0)

1. Ethernet cable from your dev machine to the robot; power the robot on.
2. Identify your NIC: `ifconfig` (look for the interface that came up when you plugged in).
3. Edit `deploy/configs/go2_locomotion.yaml`'s `network_interface`, or pass `--network_interface` on the
   command line (the CLI flag overrides the config file).
4. Run with `--dry_run` still on, but now against the real robot:
   ```bash
   /workspace/isaaclab/_isaac_sim/python.sh deploy/deploy_real.py --network_interface enp3s0 --dry_run
   ```
   This *does* call `release_motion_control()` for real (it's only the final `Write()` that's skipped under
   `--dry_run`) — confirm you see `"[INFO] Onboard motion-control mode released"` and the robot visibly
   relaxes/stops resisting manual movement, and that the printed `[DRY_RUN]` target joint positions look
   physically sane (close to the robot's actual current pose, not wildly different) before proceeding.

### 5. Go live (hoisted)

```bash
/workspace/isaaclab/_isaac_sim/python.sh deploy/deploy_real.py --network_interface enp3s0 --manual_commands --live_plot
```
- `--manual_commands` opens the same slider GUI used in sim play sessions — **leave every slider at zero**
  for this first run.
- `--live_plot` logs to `logs/hardware_dashboard/<timestamp>/`; view it with
  `isaaclab -p scripts/live_dashboard.py --logdir logs/hardware_dashboard --output /tmp/hardware_dashboard.png`
  (+ `feh --reload 3 /tmp/hardware_dashboard.png` for a live view), or use the pre-wired `hardware` tmux
  window from `docker/isaaclab-shell.sh`, which does exactly this in a 3-pane split.
- Watch the legs. At zero command the robot should hold a stable standing pose close to
  `default_joint_pos`, not twitch, drift, or fight the hoist. `Ctrl+C` sends a damping command
  (`kp=0, kd=3.0`) and exits cleanly — use it the moment anything looks wrong.

### 6. Ground contact

Only after a clean hoisted run: lower the robot per the standard Unitree procedure, keeping a hand near
`Ctrl+C` the whole time. Start with small nonzero commands on the slider once grounded, not full-speed
walking.

### Known limitation to keep in mind throughout

`base_height` (fed to the experimental policy only) is currently a **fixed 0.3m constant**, not a real
measurement (see Stage 3) — the policy will behave as if the robot is always at exactly that height
regardless of reality. This is a known, deliberate simplification, not a bug to chase if height-dependent
behavior looks off during testing.

## Verification

- Stage 2's Isaac-side joint-mapping check is a pure Python unit test, no hardware needed — do this first
  and get it right before anything else. `deploy/dump_joint_order.py` gives that side; the SDK side is now
  sourced from Unitree's own example (Stage 1), still worth a firmware-doc cross-check.
- Everything code-level that *could* be verified without hardware, was, this session — including, now that
  the SDK is actually installed, real import/construction/API-call checks (not just documentation-based
  guesses) for every symbol `deploy_real.py` uses: `ChannelFactoryInitialize`/`ChannelPublisher`/
  `ChannelSubscriber`, `CRC` (after fixing a missing native library), `RecurrentThread`, `SportClient`,
  `MotionSwitcherClient`, all exercised on the loopback interface with no robot involved. See Stage 1's
  verification notes above for the full list of what was and wasn't checked.
- What's left is inherently hardware-in-the-loop and not simulable: whether `release_motion_control`
  actually releases a real robot's onboard controller (Stage 2.5), whether `SportModeState` keeps publishing
  afterward (Stage 3 — currently sidestepped via a fixed height constant, but `sportmode_velocity` still
  depends on this), and `sdk_joint_order`/`isaac_joint_order` ground-truth confirmation (Stage 2). Proceed
  stage-by-stage, do not skip the hoisted dry-run (Stage 4).
- Keep the robot hoisted for every test until Stage 5 explicitly says otherwise.
