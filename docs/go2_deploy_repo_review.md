# Review: Glowing-Torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot

Status: documentation only. No code changed for this. Date: 2026-11-06.

Repo: https://github.com/Glowing-Torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot (MIT; ROS 2, MuJoCo, Pinocchio).
The question: how does it deploy an RL policy to a Go2, and does it use robot base velocity?

Method and limits: the sources were read through raw.githubusercontent.com (nothing cloned). Not read: the C++
headers (`*.hpp`), so noise covariances, contact thresholds and some publish rates are unknown, and the `.pt` policy
files, so what each policy was trained on is inferred from file names and config.

## How it operates

Three cooperating ROS 2 nodes. One flag (`is_simulation`) switches between MuJoCo and the real robot; the only
difference is the topic names (`/mujoco/lowstate`, `/mujoco/lowcmd` versus `/lowstate`, `/lowcmd`).

| Node | Rate | Role |
|---|---|---|
| `mujoco_simulator.py` (sim only) | sim step | Publishes `/mujoco/lowstate`, takes `/mujoco/lowcmd` |
| `low_level_ctrl` (C++) | 200 Hz wall timer | Joystick state machine, builds and sends `LowCmd` |
| `rl_policy.py` | 50 Hz | Runs the TorchScript policy, publishes joint targets on `/rl/target_pos` |

### low_level_ctrl
- State machine: lie down -> stand -> RL. Button B stands up, A lies down, LB+RB hands control to the policy,
  both triggers at -1 shut it down.
- `LowCmd` per joint: mode 0x01, position target `q`, `dq = 0`, `tau = 0`, kp/kd set per source, CRC computed
  before every publish.
- Stand and lie-down: a 2 s linear interpolation from the measured pose, with kp 50 / kd 1.
- RL: hard-coded kp 30 / kd 0.75, and no blend-in; the first target is applied directly.
- A `clip_val()` limit function exists (+-35.55 on every third joint, +-23.7 on the others) but is never called.
- The README says the Unitree sport-mode service must be turned off before deploying.

### rl_policy.py
- Subscribes to `/lowstate` only (plus the Xbox input node); publishes 12 targets on `/rl/target_pos`.
- Target = `default_angles + 0.25 * action`. The action clip is commented out and nothing is filtered.
- Command: zero unless LB and RB are both held, then `0.7 * (left stick x, left stick y, right stick x)`.
- Observation, 45 values per frame:

| Slice | Term |
|---|---|
| 0:3 | command * [2.0, 2.0, 0.3] |
| 3:6 | projected gravity (from the IMU quaternion) |
| 6:9 | gyro * 0.3 |
| 9:21 | (joint position - default) * 1.0 |
| 21:33 | joint velocity * 0.05 |
| 33:45 | last action |

  Values are clipped to +-100. A FIFO of 6 frames gives 270 inputs (`num_obs: 270`). On the real robot the joints
  are reordered from SDK order to policy order with `[3,4,5,0,1,2,9,10,11,6,7,8]`.
- Default pose: `[0.1,0.8,-1.5, -0.1,0.8,-1.5, 0.1,1.0,-1.5, -0.1,1.0,-1.5]`, which is the same as this project's.
- Policies shipped in `resources/policies/`: `kp_kd_random_policy.pt` (the default in the config), `HIM_policy.pt`,
  `policy_with_45_obs.pt`.
- Transport is ROS 2 DDS with the `unitree_ros2` message types; this project uses `unitree_sdk2py` instead.

## Does it use robot base velocity?

**No, not in the deployed policy.**

- The observation above has no linear-velocity term. The deploy package's complete topic list is `/lowstate`,
  `/lowcmd`, `/rl/target_pos`, `/pos`, `/joy` and the `/mujoco/*` topics. Nothing in it subscribes to a velocity topic.
- A separate `base_velocity_estimator` package exists but nothing consumes its output:
  - kinematics node: publishes `/obs_velocity`. Per contact leg it computes `v = -omega x p - J * qdot` with
    Pinocchio, and averages over the feet whose z-force exceeds a threshold.
  - IMU node: publishes `/imu_velocity`, from integrating bias-corrected acceleration with a zero-velocity reset
    when all four feet are force-stable.
  - EKF: a 9-state filter publishing `/base_velocity` (Twist, 50 Hz). It fuses IMU acceleration with `/obs_velocity`
    and adds its own zero-velocity update.
  - Its own simulator builds a 45-value observation without velocity, and publishes `/true_velocities` only as
    ground truth for judging the estimator.
  - The README says the estimator is for policies that take base velocity as an observation.
- Weaknesses seen in the estimator, so treat it as a demo:
  - The accelerometer bias calibration absorbs gravity, and nothing rotates acceleration into the world frame, so it
    is only valid near level.
  - In the IMU node, `small_acc_` tests the bias vector rather than live acceleration.
  - The EKF mean update and its covariance model are inconsistent, and its zero-velocity reset changes the state
    without touching the covariance.
  - Whether the Pinocchio model is fixed-base or floating-base is not clear from the code, and the URDF path is
    hard-coded.
- Likely reason no velocity is needed: the policies are history-based (6 frames, and `HIM_policy.pt` suggests the
  HIMLoco-style implicit estimator), which infers velocity from the observation history. That is inferred, not
  confirmed.

## Comparison with this project

| | This project | That repo |
|---|---|---|
| Actor sees base velocity | No (the critic does, as privileged information) | No |
| Actor observation | 47 values, single frame (5-dim command incl. pitch/lean) | 45 values x 6 frames = 270 |
| Gains on the robot | Training uses kp 40 / kd 0.8 | Deploys with kp 30 / kd 0.75 (stand: 50 / 1) |
| Blend into the policy | Handled in this project's own deploy code | None on the RL path; a 2 s blend for stand/lie-down |
| Action clipping | None in training; any capping is added in deploy | Commented out; the limit function is never called |
| DDS interface | `unitree_sdk2py` | ROS 2 with `unitree_ros2` messages |
| Default pose | Same | Same |

One thing to watch if their config is ever reused: its yaw scales are 0.3 in the deploy config but 0.25 in the
estimator package's simulator config. I cannot tell which one matches their training.

Nothing in that repo is needed to deploy this project's policy.

## Deferred: history-stacked observation (a later iteration, not started)

Idea taken from the repo: stack N past observation frames (they use 6 x 45 = 270) so that a policy can infer base
velocity and contact state without an estimator.

- For this project the actor input would go from 47 to 47 * N. That needs a from-scratch retrain and a change in the
  deploy code to keep the frame buffer (zero-filled at the start, and reset on the lie-down and stand transitions).
- Open questions for that iteration:
  - the value of N (they use 6);
  - whether `last_action` stays inside each frame;
  - whether it helps low-speed stepping and the kneeling;
  - a frame-stacking helper at train time versus a custom observation term;
  - the critic stays privileged (it keeps `base_lin_vel`).
- It is separate from the iter-3 run (reward fixes and batched training) and should not be mixed into it.
