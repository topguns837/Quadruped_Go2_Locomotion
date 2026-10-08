# Sim2sim (MuJoCo)

Runs an exported policy (`models/*/exported/policy.pt`) in MuJoCo, a second physics engine with a different
contact solver from PhysX. A policy that walks here is much more likely to survive the real robot, and a
policy that fails here will almost certainly fail on hardware. No robot, no Isaac Sim, no GPU needed.

It also exercises the hardware code path: MuJoCo's state is packed into `deploy_real.LatestRobotState` in
SDK motor order, and the observation comes from `deploy_real.build_observation` / `build_joint_index_maps`,
the same functions `deploy/deploy_real.py` uses on the robot.

## Files

| File | What it is |
|---|---|
| `fetch_go2_model.sh` | Fetches MuJoCo Menagerie's `unitree_go2` (pinned commit) into `mujoco/menagerie/` (gitignored) |
| `go2_model.py` | Builds the MuJoCo model: Menagerie Go2 + rigid OpenManipulator-X + floor (via `mujoco.MjSpec`) |
| `sim2sim_mujoco.py` | The runner: policy loop, DC motor model, fall check, metrics, viewer, slider, dashboard |
| `scenarios/basic.yaml` | Scripted command segments (stand, walk, turn, pitch, lean, height) |

## Running

Inside the container (`./startScript.sh` -> Isaac Lab shell; there is a pre-typed `sim2sim` tmux window):

```bash
./sim2sim/fetch_go2_model.sh   # once

# Interactive: viewer + slider GUI + live dashboard (logs/sim2sim_dashboard/)
/workspace/isaaclab/_isaac_sim/python.sh sim2sim/sim2sim_mujoco.py --manual_commands --live_plot

# Repeatable eval, prints a per-segment table
/workspace/isaaclab/_isaac_sim/python.sh sim2sim/sim2sim_mujoco.py --headless \
  --scenario sim2sim/scenarios/basic.yaml --policy models/9_10_26_vanilla/exported/policy.pt
```

Any Python with `torch`, `mujoco`, `numpy<2`, `pyyaml` (and `tensorboard` for `--live_plot`) also works.

Useful flags:
- `--policy`: any exported `policy.pt`. The observation size (51 vanilla, 52 Round 3) is read from the
  policy's first layer, so no file naming convention is needed.
- `--height_source true|constant`: Round 3's `base_height` input. `constant` feeds `DEFAULT_HEIGHT_M=0.3`,
  exactly what the robot currently gets (deploy.md Stage 3).
- `--no_reset`: stop at the first fall instead of resetting.
- `--motor_limits training|hardware`: which motor limits to simulate. See the next section; this is the single
  most important knob in this harness.
- `--save_xml out.xml`: dump the compiled MuJoCo model for inspection.

## Parameter verification against official Unitree sources (2026-10-08)

Checked against Unitree's own `go2_description.urdf` (unitreerobotics/unitree_ros), the Go2 spec sheet
(docs.quadruped.de), and the trained asset `go2withOpenXstatic.usd` itself.

| Parameter | Official Unitree | Trained USD | Training config (`unitree_go2witharm_cfg.py`) | MuJoCo (Menagerie) |
|---|---|---|---|---|
| hip range | +/-1.0472 rad (+/-60 deg) | same | (not overridden) | same |
| thigh range, front | -1.5708 to 3.4907 | same | (not overridden) | same |
| thigh range, rear | -0.5236 to 4.5379 | same | (not overridden) | same |
| calf range | -2.7227 to -0.83776 | same | (not overridden) | same |
| hip/thigh torque | 23.7 Nm | 23.7 Nm | **23.5 Nm** | 23.7 Nm |
| hip/thigh max speed | 30.1 rad/s | 30.1 rad/s | **30.0 rad/s** | n/a |
| **calf torque** | **45.43 Nm** | **45.43 Nm** | **23.5 Nm** | 45.43 Nm |
| **calf max speed** | **15.70 rad/s** | **15.70 rad/s** | **30.0 rad/s** | n/a |
| Go2 mass | ~15 kg spec; URDF links sum to 16.087 | 15.019 kg | - | 15.206 kg |

Joint ranges agree everywhere, so there is no mirrored-joint or limit mismatch. Two real discrepancies:

**1. The calf actuator limits are wrong in the training config (significant).** The Go2's calf sits behind a
1.9169:1 knee reduction, so it is a 45.43 Nm / 15.70 rad/s joint, not 23.5 Nm / 30.0 rad/s. The URDF says so,
and the trained USD carries the correct values (`drive:angular:physics:maxForce = 45.43`,
`physxJoint:maxJointVelocity = 899.54 deg/s = 15.70 rad/s`). They are discarded because
`unitree_go2witharm_cfg.py` puts all 12 joints in one `DCMotorCfg` group with a single scalar
`effort_limit`/`saturation_effort`/`velocity_limit`. Isaac Lab's own stock Go2 config had the identical bug;
see [isaac-sim/IsaacLab PR #7564](https://github.com/isaac-sim/IsaacLab/pull/7564), which fixed it by letting
`saturation_effort` take a joint-name-pattern dict. So **both committed policies were trained with a calf at
0.52x its real torque, and a torque-speed curve rolling off at 1.9x the real speed.**

**2. Mass.** Isaac's USD drops the URDF's 12 rotor links (0.089 kg each, 1.068 kg total); Menagerie folds the
calf rotor in, hence 15.206 vs 15.019 kg. The Isaac-to-MuJoCo gap is 1.2% and not worth chasing. The real
robot also has that rotor inertia as joint armature, which neither the trained USD nor the training config
models (`ARMATURE = 0.0` here, matching training; Menagerie's own default is 0.01).

Secondary note: the spec sheet quotes the hip range as +/-48 deg while every URDF/USD/MJCF says +/-60 deg. All
three sims agree with each other, so it is not a sim2sim issue, but the real robot may refuse hip angles the
policy is willing to command.

## What matches training, and what doesn't

Matches `unitree_go2witharm_cfg.py` / `quadruped_go2_locomotion_env_cfg.py`:
- 0.005 s physics step, policy every 4 steps (50 Hz), spawn at z=0.4 in the default pose.
- Explicit DC motor, a line-by-line port of Isaac Lab's `IdealPDActuator.compute` + `DCMotor._clip_effort`
  (including the joint-velocity pre-clip to `vel_at_effort_lim`, and clipping only the outer side of each
  bound so `max_effort` can go negative above `velocity_limit`). Joint damping, friction and armature are
  zero, as in Isaac.
- Ground friction 1.0. Arm mass, COM and inertia from the arm's physics USD (0.60 kg), mounted on the head at
  (0.219, 0, 0.106) m from the base origin, as in the trained asset `go2withOpenXstatic.usd`. Note:
  `scripts/compose_go2_with_arm.py` mounts it at (0, 0, 0.109) instead; that script did not build the trained asset.
- No deploy safety code is in the loop (no tether, step clamp, ramp or watchdog): targets are exactly
  `default + 0.25 * action`, and the only clip is the DC motor torque-speed limit.
- Falls: base or arm contact with the ground above 1 N (the training terminations), or base below 0.12 m.

Does not match / unknown:
- Go2 link masses and inertias are Menagerie's, not Isaac's USD (see the table above, 1.2% heavier).
- The arm is rigid. In Isaac its joints are held at zero by stiff drives, so it can flex slightly.
- Flat ground only; training also had random rough terrain.
- Contact solver: MuJoCo elliptic cone with `impratio=100` vs PhysX. Not reconcilable by configuration.

## Results (2026-10-08, `scenarios/basic.yaml`)

### With `--motor_limits training` (what the policies were trained against)

Both policies complete all 14 segments with **zero falls** and track velocity, yaw, pitch and lean closely
(Round 3 better than vanilla, e.g. turn yaw RMS 0.29 vs 0.75 rad/s). **The policies are not unstable in
MuJoCo.** Behavior here is consistent with Isaac, which is the expected result.

The odd posture is real but benign under these limits: with zero command, both policies send one calf a target
26 to 81 deg beyond its joint limit, and that motor sits at 23.5 Nm saturation 14% (vanilla) to 25% (Round 3)
of the time. Round 3's joint velocities while standing are 0.0 rad/s: it stands by bracing the calf against
its mechanical stop at constant saturated torque, not by balancing. The saturation is acting as the
controller, and it works because the limit is low.

### With `--motor_limits hardware` (the real robot)

Giving the calf its real 45.43 Nm, with no other change, breaks it:

| | mean abs torque | joint accel | max joint speed | falls (14 segments) |
|---|---|---|---|---|
| vanilla, training | 4.3 Nm | 73 rad/s^2 | 0.1 rad/s | 0 |
| vanilla, hardware | 4.8 Nm | 160 rad/s^2 | 28.9 rad/s | 0 |
| Round 3, training | 5.1 Nm | 128 rad/s^2 | 0.0 rad/s | 0 |
| **Round 3, hardware** | **12.0 Nm** | **310 rad/s^2** | **93.4 rad/s** | **6** |

Round 3, the policy `deploy/configs/go2_locomotion.yaml` currently points at, **falls over while trying to
stand still**, and its tracking error grows 5 to 15x. Joint acceleration roughly doubles for both policies.
This matches the hardware incident already recorded in `deploy/Checklist.md` ("targets drifted 70-97 deg from
the legs, torque hit 32 Nm, robot cut torque, red LED") -- 32 Nm is between the sim's 23.5 and the real
45.43, i.e. the real calf delivered torque the policy had never experienced.

So the abrupt behavior is **not** an Isaac-versus-MuJoCo gap. It is a sim-versus-real gap, present identically
in Isaac, and it is caused by the calf actuator limits above.

### Recommended follow-up

1. Fix the calf limits in `unitree_go2witharm_cfg.py` (per-joint `saturation_effort`/`effort_limit`/
   `velocity_limit`, per Isaac Lab PR #7564) and **retrain**. Both committed policies depend on the wrong
   limits and cannot be patched after the fact.
2. Until then, do not run Round 3 on hardware. Vanilla survives the correct limits in MuJoCo but gets twice as
   jerky, so it is not safe either without the `deploy_real.py` tether.
3. Separately: the height command is ignored by both policies (they stand at ~0.39 m whatever is asked for
   0.25 to 0.35), and Round 3 barely uses its `base_height` input, so the hardware constant
   `DEFAULT_HEIGHT_M=0.3` is low risk.
