# Go2 Hardware Test Checklist

Work top to bottom. Do not start a phase until the previous phase's **Go / No-Go** box is ticked.
Phases D-H use the guardrails in `deploy_real.py`, `mode_probe.py` and `single_joint_test.py`.
Background and sources: `deploy.md` (Stage 0) and the approved plan.

## Rules for the whole session
- **Never lift or hoist the robot while it is powered in sport mode.** (Unitree manual: "Do not lift the robot after it is powered up.") Release control on the ground first, then hoist.
- **Never run `--stand_down` while hoisted.**
- **Stops, in this order:** controller `L2+B`, then `Ctrl+C` in the deploy terminal, then the robot's power button.
  After release the controller stop only works if the Phase 2/3 test below passed. If it did not, treat `Ctrl+C` and the power button as the only stops.
- **Stop immediately if:** the head LED turns red, `tau_est` stays near 0 while joint error is large, any joint moves faster than expected, or anything touches the rig or cable.
- **Never run the policy loop hoisted or half-supported.** Hoisted testing ends at the ramp reaching default pose.
- **One change at a time.** If anything is ambiguous, stop and note it. Do not push on.
- **A script that exits with `[LINK]`** lost the robot's state stream (cable pulled, robot rebooting). It deliberately does not resume. Re-check the robot's state and LED, then start fresh. Never leave a script waiting through a cable or power change.

## A. Setup (before touching the robot)
- [ ] 2 m clear radius. Nobody within reach of the legs.
- [ ] Rig and winch rated for the robot's weight and tested. Harness on the body, not the legs.
- [ ] Ethernet cable routed with slack, away from legs and rig.
- [ ] Battery above ~50%.
- [ ] Handheld controller on and paired.
- [ ] Second person present, or the power button within reach.
- [ ] **No stale scripts anywhere.** On the host and inside the container, `ps aux | grep -E 'deploy_real|single_joint'` shows nothing. A deploy script left running streams `LowCmd` at 500 Hz and drives the robot the moment the cable or robot comes back (this caused the 15:57 incident, see "Known incidents").
- [ ] **Cable and power rule:** plug in the LAN cable and power the robot **before** starting any script. Never connect, disconnect or power-cycle while a script is running.
- [ ] Any `ssh` session to the robot's computer (192.168.123.18) is closed, or you know nothing is running in it.
- [ ] **Joint mapping check passes** (no hardware, no network needed):
  ```
  /workspace/isaaclab/_isaac_sim/python.sh deploy/verify_joint_mapping.py
  ```
  It must print `RESULT: 12/12 correct (read and write)`. Anything else: do not proceed.
- [ ] Container running. Dashboard running (also serves as the read-only baseline):
  ```
  /workspace/isaaclab/_isaac_sim/python.sh deploy/preflight_check.py --network_interface enp2s0 --live_dashboard
  ```

## B. Phase 0 - clear the red LED (no code)
- [ ] Check the Unitree Go app for a fault message. Write it down: `__________`
- [ ] Put the robot in damping (`L2+B`). Winch down slowly until belly pad and feet are on the ground.
- [ ] Fold the legs into the manual's start posture: belly pad flat, lower legs fully retracted, thighs and calves not pinned under the body, all four feet and joints flat on the ground.
- [ ] Power off: short press, then hold 2 s. Wait ~10 s. Power on: short press, then hold 2 s.
- [ ] LED sequence: green flash -> blue slow (calibrating, **hands off**) -> steady green (~2 min). Robot then stands on its own.
- [ ] LED is steady green and the robot is standing normally.
- [ ] If red again at boot: recheck placement once and retry. If it persists, **stop software testing** and email support@unitree.cc with the list of what we ran (ReleaseMode + LowCmd, hoisting while powered).
- **Go / No-Go:** [ ] steady green after a clean boot

## C. Phase 1 - controller drill (on the ground, standing)
- [ ] `L2+B` puts the robot in damping. Recover with the controller.
- [ ] `L2+A` locks the stand; pressing again goes prone.
- [ ] Controller stays in hand for every remaining phase.
- **Go / No-Go:** [ ] both keys work, robot back in a normal state

## D. Phase 2 - read-only baseline (sport mode still on, on the ground)
- [ ] `preflight_check.py` (section A) shows `rt/lowstate` OK (~500 Hz), 12/12 joints, gravity PASS.
- [ ] Head LED stays steady green while it runs.
- [ ] Probe mode and controller (read-only, cannot move the robot):
  ```
  /workspace/isaaclab/_isaac_sim/python.sh deploy/mode_probe.py --network_interface enp2s0
  ```
- [ ] Active mode name recorded: `__________` (expected `normal`, `ai` or `advanced`)
- [ ] Press `L2+B` on the controller while the probe runs: the probe prints it as pressed (proves the controller-stop data path exists).
- **Go / No-Go:** [ ] LED green, mode recorded, controller buttons decoded

## E. Phase 3 - release on the ground
- [ ] Robot **prone** on the ground (`L2+A` twice). Rope slack, used only as a tether.
- [ ] Legs and surroundings clear. Controller in hand. Power button in reach.
- [ ] Run (gated, one command per SPACE):
  ```
  /workspace/isaaclab/_isaac_sim/python.sh deploy/deploy_real.py --network_interface enp2s0 --manual_commands --live_plot --steps_per_confirm 1
  ```
- [ ] Answer the pre-release checklist prompt with SPACE only if every line is true.
- [ ] Output shows `Onboard motion-control mode released` (no `StandDown` was called).
- [ ] Zero-gain heartbeat runs ~1 s, then the **hoist pause** prompt appears (legs held damped).
- [ ] Head LED after release: `__________` (steady green = good)
- [ ] `tau_est` should be ~0 here (damped, not commanded). Note anything else: `__________`
- [ ] **Controller stop test:** press `L2+B`. The script must print `[ESTOP]` and exit with a damping command.
  - Worked -> controller stop is usable. Re-run from the first command of this phase.
  - Did nothing -> controller stop does **not** work after release. Use `Ctrl+C` and the power button only. Note this.
- **Go / No-Go:** [ ] LED still green after release  [ ] stop test result recorded
- If the LED turned red at release: stop. It is the "release triggers red" branch. Record it and email support.

## F. Phase 4 - hoist while released
- [ ] Re-run the same command and reach the hoist pause again (robot is already released; this is fast).
- [ ] Winch up **slowly** until the feet are a few cm clear. Legs hang limp and damped.
- [ ] Nothing contacts the rig or cable. Legs are free to swing.
- [ ] Press SPACE. The script re-reads the actual (hanging) joint positions and starts the ramp, kp starting at 10%.
- **Go / No-Go:** [ ] ramp prints `max|err|` and `max|tau_est|` each step

## G. Phase 4b - single-joint command-path test (hoisted)
Do this before trusting the full-body ramp. One hip joint at kp=10, kd=1 (Unitree's own safe test gains).
- [ ] Run:
  ```
  /workspace/isaaclab/_isaac_sim/python.sh deploy/single_joint_test.py --network_interface enp2s0 --joint RL_hip_joint
  ```
- [ ] SPACE-step the +/-5 deg nudges. The joint visibly follows.
- [ ] `tau_est` for that joint rises with the nudge (not stuck near 0).
- **Go / No-Go:** [ ] joint moved and `tau_est` responded
- If `tau_est` stays ~0 and the joint does not move: **robot-side problem** (fault, protection). Stop. Check LED and the app. Email support.

## H. Phase 5 - hoisted ramp to default pose (stop here; no policy while hoisted)
The policy was trained with the feet carrying the body. Hoisted or half-supported, its inputs (tilt, contact, base velocity) are out of distribution and its targets run away (this tripped the robot's torque protection, see "Known incidents"). So the hoisted stage ends at "ramp reaches default and holds".
- [ ] Run the Phase 3 command, reach the hoist pause (already hoisted, feet clear), press SPACE.
- [ ] Startup line shows the tether: `Tether: targets limited to 15 deg of actual (max PD torque ~ 6.5 Nm)`.
- [ ] Watch each `[RAMP n/500]` line: as `kp` rises, `max|tau_est|` rises (but stays well under 21 Nm) and `max|err|` stays under ~15 deg.
- [ ] No `[FAULT]`. The script aborts to damping by itself on: no torque with large error, error above 30 deg for 3 steps, torque at 21 Nm, or a ramp that could not reach default within the tether.
- [ ] Ramp ends with `Default pose reached`. **Ctrl+C here** (do not let the policy loop run hoisted). Legs held at default with no twitching for 1+ minute first if you like.
- **Go / No-Go:** [ ] ramp reached full kp and default pose  [ ] max|tau_est| stayed under ~10 Nm  [ ] LED still green

## I. Phase 6 - ground, policy at zero command
Only after Phase 5 passed and the LED is steady green.
- [ ] Stop everything, lower the robot onto the ground with the winch. Rope **slack**, tether only, not carrying weight. All four feet flat.
- [ ] Re-run the Phase 3 command (robot is already released). At the hoist pause, press SPACE only when the robot is fully on the ground.
- [ ] Standing needs real torque at default pose (several Nm per leg). Expect `max|tau_est|` to rise at the ramp's end; that is normal if it stays well under 21 Nm.
- [ ] Keep `--steps_per_confirm 1` for the first minute of policy steps. Hand on `Ctrl+C`.
- [ ] Watch for `[WARN] tether limited ...` lines. Occasional ones are fine. **Many joints limited every step = the targets are running away: stop.**
- [ ] `[STEP n]` targets stay near default (hips about +/-6, thighs about 46-57, calves about -86 deg), not drifting. If a joint drifts steadily, stop.
- [ ] Sliders at zero. Stable stand, no twitching, LED green.
- [ ] Base velocity input is still a stand-in (`sportmode_velocity`, see deploy.md), so the first ground run may be imperfect. (Height is no longer an input at all -- removed, not a stand-in -- so there is nothing to tape-measure here any more.)
- [ ] Only then: small slider commands (e.g. 0.1 m/s), increasing gradually.
- **Go / No-Go:** [ ] stable stand for 1+ minute at zero command  [ ] no tether spam  [ ] LED green

## J. Shutdown
- [ ] `Ctrl+C` the deploy script (it sends a damping command). Wait for it to exit. **Confirm no script is still running** (`ps aux | grep deploy_real`) **before** touching the cable or the robot's power.
- [ ] Lower to the ground, let the robot settle prone.
- [ ] Power off per the manual (short press, then hold 2 s). Never power off while it hangs or stands: it drops heavily.
- [ ] Fold legs into the start posture for the next boot.
- [ ] Note logs: `logs/preflight_dashboard/<timestamp>/`, `logs/hardware_dashboard/<timestamp>/`

## LED reference (head light)
| Pattern | Meaning |
|---|---|
| Green flash | Switching on |
| Blue, slow flash | Motor and IMU calibration in progress (hands off) |
| Green, steady | Powered on |
| Yellow, slow flash | Low battery, will crouch within 10 min |
| Red, fast flash | Motor and IMU calibration failed |
| Red, slow flash (~1/s) | System abnormality, boot or hardware failure -> contact Unitree support |

## Controller reference (Go2 handheld)
| Keys | Action |
|---|---|
| `L2` (hold) + `B` | Damping (soft emergency stop) |
| `L2` (hold) + `A` | Lock stand; press again for prone |
| `L2` (hold) + `X` | Recover standing after a fall |
| `SELECT` | "Make a pose" (**not** an e-stop) |

## If the LED turns red: where it happened tells you why
| Red first appears | Likely meaning | Action |
|---|---|---|
| At power-on | Placement or boot fault | Recheck posture, retry once, then support |
| At release (Phase 3) | Releasing the motion service shows as a fault | Record, email support before continuing |
| Only once LowCmd flows (Phase 4b/5) | Command path or protection trip | Stop, email support |
| Never | The earlier red came from lifting the live robot | Continue |

Support: support@unitree.cc

## Known incidents (why some items above exist)
| When | What happened | Cause | Fix |
|---|---|---|---|
| First live run | Very rough, jerky motion | `publish_low_cmd` sent **8 of 12 joints the wrong joint's target** (index list was a valid permutation in the wrong direction) | Fixed in `build_joint_index_maps`, with a loud cross-check and `verify_joint_mapping.py` |
| 15:57 cable reconnect | ~34 s of violent ~45 Hz chatter, up to 170 deg on one thigh | A deploy script left running (500 Hz, kp=25) streamed `LowCmd` into a robot that had rebooted into normal sport mode; the two controllers fought | Scripts now stop on link loss and never resume; checklist rules on stale scripts and cable/power order |
| Earlier | Legs hung and moved when lifted while powered | Sport-mode balance controller reacting (manual: do not lift a powered robot) | Release on the ground first, then hoist |
| Later run, policy steps | Targets drifted 70-97 deg from the legs, torque hit 32 Nm (> 23.5 Nm limit), robot cut torque, red LED | Policy ran with the robot half on a winch (out-of-distribution inputs), PD torque = kp x error | Tether (15 deg of actual), abort rules, no policy while hoisted |
| Earlier | `tau_est` ~0 on every joint, slow red LED | Unresolved; robot-side fault suspected | Torque-response watchdog and the single-joint test isolate it |
