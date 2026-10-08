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
| `scenarios/basic.yaml` | Scripted command segments (stand, walk, turn, pitch, lean) |

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
- `--policy`: any exported `policy.pt`. The observation size (50 current, 51 committed vanilla, 52 committed
  Round 3) is read from the policy's first layer, so no file naming convention is needed -- see `OBS_LAYOUT`
  in `sim2sim_mujoco.py`. A 50-dim policy's command is 5-wide (no height); the two older ones are 6-wide
  (height command, no height observation for vanilla, both for Round 3) -- there is no height command for
  any policy any more (see `mdp/commands.py`'s module docstring), so for the 6-wide ones the 6th slot is
  padded in automatically from a fixed legacy nominal height, never asked of the user or a scenario.
- `--height_source true|constant`: only affects the committed Round 3 (52-dim) policy's `base_height`
  OBSERVATION. `constant` feeds the legacy nominal height stand-in; ignored for every other policy, which
  have no height observation at all.
- `--no_reset`: stop at the first fall instead of resetting.
- `--motor_limits hardware|legacy`: which motor limits to simulate. `hardware` (the default) is the real robot,
  which `unitree_go2witharm_cfg.py` now also trains against. `legacy` is the pre-fix single 23.5 Nm group the
  committed policies were trained with. See the next section; this is the single most important knob here.
- `--save_xml out.xml`: dump the compiled MuJoCo model for inspection.

## Parameter verification against official Unitree sources (2026-10-08)

Checked against Unitree's own `go2_description.urdf` (unitreerobotics/unitree_ros), the Go2 spec sheet
(docs.quadruped.de), and the trained asset `go2withOpenXstatic.usd` itself.

| Parameter | Official Unitree | Project USD | Config now | Config before fix | MuJoCo (Menagerie) |
|---|---|---|---|---|---|
| hip range | +/-1.0472 rad (+/-60 deg) | same | (not overridden) | (not overridden) | same |
| thigh range, front | -1.5708 to 3.4907 | same | (not overridden) | (not overridden) | same |
| thigh range, rear | -0.5236 to 4.5379 | same | (not overridden) | (not overridden) | same |
| calf range | -2.7227 to -0.83776 | same | (not overridden) | (not overridden) | same |
| hip/thigh torque | 23.7 Nm | 23.7 Nm | 23.7 Nm | 23.5 Nm | 23.7 Nm |
| hip/thigh max speed | 30.1 rad/s | 30.1 rad/s | 30.1 rad/s | 30.0 rad/s | n/a |
| **calf torque** | **45.43 Nm** | **45.43 Nm** | **45.43 Nm** | **23.5 Nm** | 45.43 Nm |
| **calf max speed** | **15.70 rad/s** | **15.70 rad/s** | **15.70 rad/s** | **30.0 rad/s** | n/a |
| Go2 mass | ~15 kg spec; URDF links sum to 16.087 | 15.019 kg authored, **17.019 loaded (before fix)** | 15.019 kg loaded (after fix, see below) | 17.019 kg loaded | 15.206 kg |

Joint ranges agree everywhere, so there is no mirrored-joint or limit mismatch. Two findings:

**1. The calf actuator limits were wrong in the training config (significant, now fixed).** The Go2's calf
sits behind a 1.9169:1 knee reduction, so it is a 45.43 Nm / 15.70 rad/s joint, not 23.5 Nm / 30.0 rad/s. The
URDF says so, and the project USD carries the correct values (`drive:angular:physics:maxForce = 45.43`,
`physxJoint:maxJointVelocity = 899.54 deg/s = 15.70 rad/s`). They were discarded because
`unitree_go2witharm_cfg.py` put all 12 joints in one `DCMotorCfg` group with a single scalar
`effort_limit`/`saturation_effort`/`velocity_limit`. Isaac Lab's own stock Go2 config had the identical bug;
see [isaac-sim/IsaacLab PR #7564](https://github.com/isaac-sim/IsaacLab/pull/7564).

`unitree_go2witharm_cfg.py` now splits the actuators into a `hip_thigh` group and a `calves` group with the
real per-joint values. Two groups rather than one with per-joint dicts, because `DCMotorCfg.saturation_effort`
is a scalar `float` in Isaac Lab v2.3.2 (PR #7564 added dict support upstream).

**Consequence: both committed policies are obsolete.** They were trained with a calf at 0.52x its real torque
and must be retrained from scratch, not resumed -- see the results below for why resuming is not an option.

**2. Mass: Isaac loaded the robot 2.0 kg heavier than it should be (significant, NOW FIXED).** Measured in a
live env, not read off the file: `root_physx_view.get_masses()` gave Go2 bodies **17.019 kg** + arm 0.604 kg
= **17.623 kg total**, against a ~15 kg spec. The entire 2.0 kg discrepancy was two bodies:

| body | URDF mass | USD authored | Isaac loaded (before fix) | Isaac loads (after fix) | position vs base |
|---|---|---|---|---|---|
| `imu` | 0.0 | 0.0 | **1.0000 kg** | 0.001 kg | (-0.026, 0, +0.042) |
| `radar` | 0.0 | 0.0 | **1.0000 kg** | 0.001 kg | (+0.289, 0, -0.047) |

Both are massless placeholders in Unitree's URDF and are authored `physics:mass = 0.0` in the USD. A rigid
body cannot have zero mass in PhysX, so it substituted its **1.0 kg default** for each -- 13% excess mass,
with 1 kg sitting 29 cm forward of the base origin at the radar, shifting the centre of mass and the pitch
inertia. This was a URDF-to-USD import artifact, silent, and it affected every policy trained before the
fix.

Fixed in `quadruped_go2_locomotion_env_cfg.py`'s `EventCfg` with a `startup`-mode
`mdp.randomize_rigid_body_mass` term (`fix_massless_sensor_bodies`) that sets `imu` and `radar` to 1 g each
via `operation="abs"`, confirmed live: total robot mass is now **15.6251 kg**, within 1.2% of the project
USD's own authored total (15.019 kg + 0.604 kg arm = 15.623 kg) and of MuJoCo's 15.206 kg. The small
remaining gap to MuJoCo is Menagerie's link masses differing slightly from Isaac's USD, not worth chasing.

1 g rather than 0 (PhysX clamps at `min_mass=1e-6` anyway) also makes `radar` dynamically absent, matching
this project's hardware, which has no radar fitted.

**3. Armature and joint friction are both 0 (NOT fixed).** Confirmed live: `get_dof_armatures()` and
`get_dof_friction_coefficients()` are all zero for the 12 leg DOFs. The real robot has rotor inertia -- the
URDF models 12 rotor links at 0.089 kg with 1.118e-4 kg*m^2 of spin inertia each, reflected at the joint by
the gear ratio squared -- and real joints have stiction. Isaac Lab's own stock `UNITREE_GO2_CFG` also leaves
both at zero, so this is an inherited default rather than a project-specific mistake. Not corrected here
because a principled armature value needs the gear ratio, which the URDF does not state; Menagerie assumes
0.01 for the Go2. `ARMATURE = 0.0` in `sim2sim_mujoco.py` deliberately matches Isaac rather than Menagerie.

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
- Go2 link masses and inertias are Menagerie's, not Isaac's USD (see the table above; about 1.2% apart,
  now that the imu/radar phantom mass is fixed on the Isaac side).
- The arm is rigid. In Isaac its joints are held at zero by stiff drives, so it can flex slightly.
- Flat ground only; training also had random rough terrain.
- Contact solver: MuJoCo elliptic cone with `impratio=100` vs PhysX. Not reconcilable by configuration.

## Results (2026-10-08, `scenarios/basic.yaml`)

**These specific numbers predate the mass and height fixes below** (finding 2 above, and the height-command
removal) and were measured to isolate the calf-torque question, so they are not re-quoted here after those
later fixes. They remain valid for what they demonstrate: the calf-torque gap is a sim-versus-real issue,
not a MuJoCo-versus-Isaac one. Re-run `--scenario sim2sim/scenarios/basic.yaml` against a retrained policy
for current numbers.

### With `--motor_limits legacy` (what the committed policies were trained against)

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
| vanilla, legacy | 4.3 Nm | 73 rad/s^2 | 0.1 rad/s | 0 |
| vanilla, hardware | 4.8 Nm | 160 rad/s^2 | 28.9 rad/s | 0 |
| Round 3, legacy | 5.1 Nm | 128 rad/s^2 | 0.0 rad/s | 0 |
| **Round 3, hardware** | **12.0 Nm** | **310 rad/s^2** | **93.4 rad/s** | **6** |

Round 3, the policy `deploy/configs/go2_locomotion.yaml` currently points at, **falls over while trying to
stand still**, and its tracking error grows 5 to 15x. Joint acceleration roughly doubles for both policies.
This matches the hardware incident already recorded in `deploy/Checklist.md` ("targets drifted 70-97 deg from
the legs, torque hit 32 Nm, robot cut torque, red LED") -- 32 Nm is between the sim's 23.5 and the real
45.43, i.e. the real calf delivered torque the policy had never experienced.

So the abrupt behavior is **not** an Isaac-versus-MuJoCo gap. It is a sim-versus-real gap, present identically
in Isaac, and it is caused by the calf actuator limits above.

### Recommended follow-up (status as of the mass/height fixes)

1. ~~Fix the calf limits~~ **DONE.** `unitree_go2witharm_cfg.py` now splits `hip_thigh`/`calves` with the
   real per-joint `saturation_effort`/`effort_limit`/`velocity_limit` (Isaac Lab PR #7564's approach).
2. ~~Fix the imu/radar phantom mass~~ **DONE** (finding 2 above).
3. ~~Height command/observation~~ **REMOVED.** There is no height command for any new policy, and no
   `base_height` observation -- see `mdp/commands.py`'s module docstring. The robot is held near a fixed
   default stance height (0.337m) purely by `height_penalty`, a training-side reward, not a command.
4. **Retrain is required.** Both committed policies depend on the old (wrong) calf limits, the old (wrong,
   heavier) mass, and the old 6-dim/`base_height` observation -- none of which the current config
   reproduces, and none of which can be patched into an already-trained policy. Do not resume from either
   committed checkpoint.
5. Once retrained, re-run this scenario against the new 50-dim policy to replace the predates-the-fixes
   numbers above, and re-check whether vanilla's extra jerkiness under `--motor_limits hardware` (finding 1)
   persists now that the mass is also correct.
