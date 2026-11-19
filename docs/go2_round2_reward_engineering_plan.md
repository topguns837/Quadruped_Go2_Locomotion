# Go2 Locomotion — Round 2 Reward-System Engineering Plan

## 1. Purpose

This document defines the engineering plan for the second training round of the Unitree Go2 locomotion project.

The goal is to create a **gait-aware locomotion policy** while preserving the existing task, observation space, action space, PPO configuration, terrain, command distribution, and baseline reward terms as much as possible.

Round 2 adds four behavioral objectives:

1. Encourage coordinated, trot-like diagonal gait movement.
2. Require approximately **10 cm minimum swing-foot clearance**.
3. Apply a **small penalty** when the Go2 calf/knee region contacts the ground.
4. Apply a **small penalty** when the head-mounted Open Manipulator X contacts the ground.

The experiment must remain controlled: Policy B should be trained **from scratch**, using the same PPO configuration as the vanilla baseline, so that the main experimental difference is the reward formulation.

---

## 2. Repository and Current Configuration

Repository:

`https://github.com/SMEAC/Quadruped_Go2_Locomotion`

Project description: Isaac Lab task for Unitree Go2 locomotion with extended pitch/lean control and a static Open Manipulator X mounted on the robot head.

Relevant files:

- `source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/quadruped_go2_locomotion_env_cfg.py`
- `source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/mdp/rewards.py`
- `source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/mdp/__init__.py`
- `source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/agents/rsl_rl_ppo_cfg.py`
- `scripts/rsl_rl/train.py`

---

## 3. Experimental Structure

### Policy A — Vanilla Baseline

Train from scratch using the current reward system.

No changes to:

- observation space
- action space
- PPO configuration
- command ranges
- terrain configuration
- episode length
- simulation timestep
- decimation
- current reward terms

Purpose:

> Establish what locomotion behavior naturally emerges from the current reward system.

Policy A is the control/baseline policy.

### Policy B — Gait-Aware Policy

Train **from scratch**, not by fine-tuning Policy A.

Keep everything above identical and add/refactor only the reward objectives required for:

- gait coordination
- 10 cm foot clearance
- calf/knee contact avoidance
- arm contact avoidance

This isolates reward shaping as the main experimental variable.

---

## 4. Current Simulation and RL Timing

Current configuration:

```python
sim.dt = 0.005
decimation = 4
```

Physics therefore runs at:

\[
1 / 0.005 = 200\text{ Hz}
\]

One policy action is held for four physics steps:

\[
4 \times 0.005 = 0.020\text{ s}
\]

Therefore policy frequency is approximately:

\[
50\text{ Hz}
\]

A policy/environment step is therefore approximately 20 ms, while a physics step is 5 ms.

Current episode duration:

```python
episode_length_s = 40.0
```

At 50 Hz, a maximum-length episode contains approximately:

\[
40 \times 50 = 2000
\]

policy steps.

An episode can terminate earlier because of configured termination conditions.

---

## 5. Rollout Configuration

Current PPO runner:

```python
num_envs = 1024
num_steps_per_env = 24
```

Each PPO rollout therefore collects:

\[
1024 \times 24 = 24,576
\]

environment transitions.

Important distinction:

- A **step** is one policy/environment interaction.
- An **episode** is one trajectory from reset until termination.
- A **rollout** is a fixed chunk of experience collected for PPO.
- A rollout is **not** necessarily an episode.
- An episode can span multiple rollouts.
- A rollout can contain portions of multiple episodes because environments reset independently.

Current PPO optimization:

```python
num_mini_batches = 4
num_learning_epochs = 5
```

Approximate mini-batch size:

\[
24,576 / 4 = 6,144
\]

samples.

Five optimization epochs over four mini-batches give approximately:

\[
5 \times 4 = 20
\]

mini-batch optimizer updates per rollout, subject to the exact RSL-RL implementation.

---

## 6. Existing Observation and Action Space

Round 2 should initially keep the observation and action spaces unchanged.

Current policy observations include:

- base linear velocity
- base angular velocity
- projected gravity
- velocity commands
- relative joint positions
- relative joint velocities
- previous action

Observation corruption/noise is enabled during training.

Current action configuration:

```python
JointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*"],
    scale=0.25,
    use_default_offset=True,
)
```

The policy therefore produces joint-position actions that are scaled around the default joint pose. Conceptually:

\[
q_{\text{target}}
\approx
q_{\text{default}} + 0.25a
\]

The actuator/PD layer then converts the target position into joint torque.

Do not add contact state or gait phase to the policy observation for the initial experiment. The reward may use simulator contact information without exposing that information to the actor.

This is deliberate: it keeps

\[
O_A = O_B
\]

so the primary experimental change is reward design.

---

# 7. Existing Contact-Sensor Infrastructure

The current environment already has contact sensors.

The Go2 contact sensor tracks:

- robot links
- contact history
- air time

The arm has a separate contact sensor.

Both are configured to update at the simulation timestep.

This is sufficient for the proposed gait and contact rewards.

Existing foot identifiers include:

```text
FL_foot
FR_foot
RL_foot
RR_foot
```

Use these same identifiers rather than introducing duplicate foot definitions.

---

# 8. Existing Reward System

The current reward system contains, among others:

### Positive objectives

```text
track_lin_vel_xy_exp     +10.0
track_ang_vel_z_exp       +0.5
track_pitch_exp           +0.5
track_lean_exp            +0.3
foot_lift                 +0.1
```

### Negative objectives

```text
lin_vel_z_l2              -2.0
ang_vel_xy_l2             -0.05
dof_torques_l2            -1e-5
dof_acc_l2                -2.5e-7
action_rate_l2             -0.01
undesired_contacts         -1.0
undesired_contacts_arm     -1.0
height_penalty             -1.65
hip_crossing               -0.5
foot_sliding               -1.0
dof_pos_limits             -1.0
joint_deviation            -0.005
```

The main velocity-tracking reward has weight 10.0 and should remain the dominant locomotion objective.

---

# 9. Important Existing Foot-Lift Issue

The current configuration contains approximately:

```python
foot_lift = RewTerm(
    func=mdp.foot_lift_exp,
    weight=0.1,
    params={
        "std": math.sqrt(0.01),
        "min_height": 0.05,
    },
)
```

The implementation uses a quantity equivalent to:

\[
h_{\text{above}} = h_{\text{foot}} - h_{\min}
\]

and:

\[
r_{\text{lift}}
=
\exp(-h_{\text{above}}/\text{std})
\]

The implementation therefore does **not** behave like a conventional minimum-clearance reward if the intended interpretation is "higher than the threshold is better":

- at the threshold, reward is 1;
- above the threshold, the exponential decreases;
- below the threshold, the exponential increases.

Therefore:

> Do not simply change `min_height = 0.05` to `0.10` and assume the desired behavior has been implemented.

For Round 2, replace/refactor the existing foot-lift term into a mathematically correct clearance objective.

---

# 10. Round 2 Objective 1 — 10 cm Swing-Foot Clearance

## 10.1 Definition

Interpret "minimum leg height" as **minimum swing-foot clearance above the terrain/ground**, not knee height.

Target:

\[
h_{\min}=0.10\text{ m}
\]

The desired constraint is:

\[
h_{\text{foot}} \ge 0.10\text{ m}
\]

during swing.

Do not reward arbitrarily high feet. The objective is minimum clearance, not maximum foot height.

---

## 10.2 Recommended formulation

For each foot \(i\), when it is in swing:

\[
d_i = \max(0, 0.10-h_i)
\]

where \(h_i\) is the foot height above the relevant ground reference.

Then use a soft penalty:

\[
R_{\text{clearance}}
=
-\frac{1}{N_{\text{swing}}}
\sum_i
\left(
\frac{d_i}{0.05}
\right)^2
\]

The scale of 0.05 m is an initial shaping scale, not a physical requirement.

Properties:

- 5 cm clearance → meaningful penalty.
- 8 cm → smaller penalty.
- 10 cm → zero penalty.
- 15 cm → zero additional penalty.
- 20 cm → zero additional penalty.

This prevents the policy from learning that "higher is always better."

---

## 10.3 Alternative bounded target formulation

A bounded target reward may also be used:

\[
R_{\text{clearance}}
=
\exp
\left(
-\frac{(h_i-0.10)^2}{\sigma_h^2}
\right)
\]

However, this formulation rewards heights around exactly 10 cm and can discourage legitimate higher clearance.

For the stated requirement of a **minimum** 10 cm, the deficit-only penalty is preferred.

---

## 10.4 Swing gating

Clearance should only be evaluated for feet that are actually swinging.

A stance foot should not be punished for remaining close to the ground.

Use contact information to create:

\[
I_{\text{swing},i} =
1-I_{\text{contact},i}
\]

with appropriate contact thresholding/hysteresis if necessary.

---

# 11. Round 2 Objective 2 — Gait Coordination

## 11.1 Desired gait prior

The first gait-aware experiment should encourage a **trot-like diagonal coordination pattern**, rather than prescribing exact joint trajectories.

Quadruped feet:

```text
             FRONT

        FL          FR

        RL          RR

             REAR
```

Diagonal pairs:

\[
D_1=(FL,RR)
\]

\[
D_2=(FR,RL)
\]

A trot-like pattern has approximately:

```text
Phase A:
FL + RR = stance
FR + RL = swing

Phase B:
FL + RR = swing
FR + RL = stance
```

The policy should be encouraged toward this structure, but not forced to follow exact joint trajectories.

---

# 12. Why Not Prescribe Joint Trajectories?

Do not initially specify:

```text
joint angle at t=0
joint angle at t=0.1
joint angle at t=0.2
...
```

That would turn RL into trajectory tracking and could reduce robustness.

Instead reward gait properties:

- diagonal coordination
- alternating support
- swing-foot clearance
- low stance-foot slip
- periodic stepping

This allows the policy to discover its own joint-level motion.

---

# 13. Contact Representation

A hard binary contact signal can flicker around a contact threshold.

Let:

\[
F_i
\]

be the contact force magnitude for foot \(i\).

A soft contact value can be constructed:

\[
c_i =
\sigma
\left(
\frac{F_i-F_{\text{threshold}}}{s_F}
\right)
\]

where:

- \(c_i \approx 0\): swing
- \(c_i \approx 1\): stance/contact

This smooth representation is preferable for reward shaping if the sensor data supports it.

If the existing contact API provides a reliable boolean contact signal with sufficient filtering, it may be used instead. Avoid unnecessary complexity if the raw contact state is already stable.

---

# 14. Diagonal Synchronization Reward

Define:

\[
D_1=\frac{c_{FL}+c_{RR}}{2}
\]

and:

\[
D_2=\frac{c_{FR}+c_{RL}}{2}
\]

Reward diagonal agreement:

\[
R_{\text{sync}}
=
1-
\frac{
|c_{FL}-c_{RR}|
+
|c_{FR}-c_{RL}|
}{2}
\]

This is high when:

\[
c_{FL}\approx c_{RR}
\]

and:

\[
c_{FR}\approx c_{RL}
\]

---

# 15. Alternating Support Reward

Diagonal synchronization alone is insufficient.

All four feet could remain on the ground:

\[
[1,1,1,1]
\]

and still satisfy diagonal synchronization.

Therefore require approximately one diagonal pair to be supporting at a time.

Use:

\[
R_{\text{support}}
=
1-|D_1+D_2-1|
\]

This favors:

\[
D_1+D_2\approx1
\]

meaning approximately one diagonal pair is in stance while the other is in swing.

---

# 16. Combined Gait Reward

Initial proposed form:

\[
R_{\text{gait}}
=
R_{\text{sync}}
\cdot
R_{\text{support}}
\]

This is preferable to a simple "reward alternating feet" term because it combines:

1. diagonal synchronization
2. alternating support

The gait reward should be a **secondary objective**.

Initial weight:

\[
\boxed{w_{\text{gait}}=0.30}
\]

This is deliberately small relative to the primary linear velocity tracking reward:

\[
w_{\text{velocity}}=10.0
\]

The intended hierarchy is:

```text
Velocity tracking        primary
Stability                primary
Gait coordination        secondary
Foot clearance            secondary
Contact avoidance         safety/shaping
```

---

# 17. Condition the Gait Reward on Locomotion

Do not apply a trot coordination reward equally during standing.

The command system allows very low/zero velocity commands.

If the gait reward is always active, PPO could learn to move the feet while the robot is supposed to stand.

Use a command-speed gate such as:

\[
v_{\text{cmd}}
=
\sqrt{
v_{x,\text{cmd}}^2+
v_{y,\text{cmd}}^2
}
\]

and activate the gait reward when:

\[
v_{\text{cmd}}>0.15\text{ m/s}
\]

The 0.15 m/s threshold is an initial engineering value and should be exposed as a parameter.

---

# 18. Turning Behavior

A rigid trot prior may conflict with turning.

During large yaw commands, asymmetric foot timing can be legitimate.

Therefore gait shaping should be reduced or gated for strong turning commands.

For example:

\[
g_{\omega}
=
f(|\omega_{z,\text{cmd}}|)
\]

where \(f\) approaches zero for sufficiently large yaw commands.

Do not initially impose a strict numerical turning cutoff without measuring the baseline. Start with a simple speed gate and evaluate whether the gait reward interferes with turning.

If turning performance degrades, introduce a yaw-dependent gate.

---

# 19. Round 2 Objective 3 — Calf/Knee Ground Contact

## 19.1 Existing implementation

The current `undesired_contacts` reward already includes:

```text
.*calf
Head_lower
Head_upper
```

with weight approximately:

\[
-1.0
\]

Therefore calf contacts are already penalized.

Do **not** blindly add another calf penalty, because that would double-count the same event.

---

## 19.2 Recommended restructuring

Separate the existing contact penalty into conceptually distinct terms:

\[
R_{\text{head-contact}}
=
-w_{\text{head}}C_{\text{head}}
\]

and:

\[
R_{\text{calf-contact}}
=
-w_{\text{calf}}C_{\text{calf}}
\]

Initial calf weight:

\[
\boxed{w_{\text{calf}}=0.25}
\]

Thus:

\[
R_{\text{calf-contact}}
=
-0.25C_{\text{calf}}
\]

This is intentionally much smaller than the main locomotion reward.

The objective is:

> Calf/knee contact is undesirable, but it should not dominate recovery behavior.

---

# 20. Why Calf Contact Should Be a Soft Penalty

A quadruped can briefly contact a calf during:

- recovery from a disturbance
- rough terrain
- aggressive pitch/lean commands
- transitional states

If every calf contact receives a huge penalty, PPO may become overly conservative.

The desired behavior is:

```text
normal foot-ground contact
    → allowed

occasional calf contact
    → small penalty

repeated/long calf contact
    → increasingly undesirable
```

Do not initially terminate an episode for calf contact.

---

# 21. Round 2 Objective 4 — Arm Ground Contact

## 21.1 Existing implementation

The arm currently has its own undesired-contact reward covering:

```text
link1
link2
link3
link4
link5
gripper_left_link
gripper_right_link
```

with a penalty around:

\[
-1.0
\]

The environment also has a separate arm-contact termination condition.

This is stronger than the requested "small penalty."

---

# 22. Recommended Arm Treatment

For Round 2, treat arm-ground contact primarily as a **soft penalty**.

Initial weight:

\[
\boxed{w_{\text{arm}}=0.25}
\]

Therefore:

\[
R_{\text{arm-contact}}
=
-0.25C_{\text{arm}}
\]

Initially disable the arm contact as an automatic episode termination if the research objective is specifically to learn recovery from occasional arm contact.

Keep hard termination for genuinely catastrophic body/head contacts.

The resulting hierarchy is:

```text
Base/head contact
    → hard failure / termination

Calf/knee contact
    → small penalty

Arm contact
    → small penalty
```

Before removing arm termination, measure the baseline arm-contact frequency. If arm contact is extremely rare, the termination change may have little practical effect.

---

# 23. Reward Weight Summary

Initial Round 2 proposal:

| Reward component | Weight | Purpose |
|---|---:|---|
| Existing linear velocity tracking | +10.0 | Primary locomotion objective |
| Existing yaw velocity tracking | +0.5 | Turning |
| Existing pitch tracking | +0.5 | Pitch command tracking |
| Existing lean tracking | +0.3 | Lean command tracking |
| Existing foot sliding | -1.0 | Stance stability |
| Existing stability/smoothness terms | unchanged | Preserve baseline behavior |
| Gait coordination | **+0.30** | Diagonal gait structure |
| 10 cm clearance deficit | **negative, initial scale ~0.25** | Prevent low swing feet |
| Calf/knee contact | **-0.25** | Discourage non-foot ground contact |
| Arm contact | **-0.25** | Protect head-mounted manipulator |

The proposed weights are starting hypotheses, not final optimal values.

They must be evaluated against observed reward magnitudes and behavior.

---

# 24. Important Reward-Scale Principle

Do not compare reward weights in isolation.

A term with weight 0.25 can still dominate if its unweighted output is large or is activated very frequently.

For every new reward term, log:

1. raw reward value
2. weighted reward contribution
3. activation frequency
4. mean contribution
5. maximum contribution
6. contribution relative to total reward

For example:

```text
gait raw mean           = ...
gait weighted mean      = ...
clearance raw mean      = ...
clearance weighted mean = ...
calf penalty mean       = ...
arm penalty mean        = ...
```

The final engineering decision should be based on **actual contribution**, not just configured weight.

---

# 25. Do Not Modify PPO Yet

Round 2 should initially retain:

```text
actor hidden dims = [256, 512, 128]
critic hidden dims = [256, 512, 128]

learning_rate = 1e-3
gamma = 0.99
lambda = 0.95
clip_param = 0.2
entropy_coef = 0.0075
num_learning_epochs = 5
num_mini_batches = 4
desired_kl = 0.01
max_grad_norm = 1.0
```

The purpose of this experiment is to test reward shaping.

Changing PPO simultaneously would make the experiment difficult to interpret.

---

# 26. Do Not Modify the Observation Space Yet

Keep the current observation vector unchanged.

Do not initially add:

- contact state
- gait phase
- foot height
- desired gait state

The reward can use simulator information that the actor does not observe.

This makes the comparison:

```text
Policy A:
same observations + old reward

Policy B:
same observations + new reward
```

which is experimentally cleaner.

---

# 27. Implementation Plan

## File: `mdp/rewards.py`

Implement/refactor the following.

### Function 1 — `gait_diagonal_coordination`

Inputs should include:

- contact sensor
- four foot body IDs
- optional contact threshold
- optional speed/turn gating parameters

Outputs a bounded gait reward.

Recommended structure:

1. Obtain foot contact states.
2. Compute diagonal synchronization.
3. Compute alternating support.
4. Multiply by locomotion-command gate.
5. Return a bounded value, preferably approximately `[0, 1]`.

---

### Function 2 — `foot_clearance_10cm`

Inputs:

- contact sensor
- foot body IDs
- minimum clearance
- deficit scale

Recommended:

\[
d_i=\max(0,0.10-h_i)
\]

and:

\[
R_{\text{clearance}}
=
-\text{mean}
\left[
\left(
\frac{d_i}{0.05}
\right)^2
\right]
\]

over swing feet.

Return zero when all relevant swing feet meet the 10 cm minimum.

---

### Contact penalties

Reuse the existing contact infrastructure.

Avoid duplicate contact penalties.

Separate calf/head/arm contact categories where necessary so their weights can be independently controlled.

---

# 28. File: `mdp/__init__.py`

Export the new reward functions.

For example:

```python
gait_diagonal_coordination
foot_clearance_10cm
```

Use the project's existing export style.

---

# 29. File: `quadruped_go2_locomotion_env_cfg.py`

Add new `RewTerm`s.

Conceptual form:

```python
gait_coordination = RewTerm(
    func=mdp.gait_diagonal_coordination,
    weight=0.30,
    params={
        ...
    },
)

foot_clearance = RewTerm(
    func=mdp.foot_clearance_10cm,
    weight=0.25,
    params={
        "min_height": 0.10,
        "deficit_scale": 0.05,
        ...
    },
)
```

Exact parameter names should follow the implemented function signatures.

Separate calf and arm contact weights rather than adding duplicate penalties.

---

# 30. Arm Termination Change

Current behavior includes a separate arm-contact termination condition.

For the initial Round 2 experiment:

- disable arm-contact termination;
- retain arm-contact soft penalty;
- retain base/head catastrophic termination.

This should be validated with a short run before full training.

If the robot repeatedly collapses onto the arm and remains in pathological states, restore arm termination or introduce a stronger contact-duration criterion.

---

# 31. Testing Before Training

Do not launch a long training run immediately after editing the reward code.

Run unit/sanity tests first.

Construct or inspect controlled cases:

### Case A — All feet on ground

Expected:

- gait reward low/moderate, not maximal
- no clearance reward/penalty for stance feet

### Case B — Diagonal stance/swing

```text
FL = stance
RR = stance
FR = swing
RL = swing
```

Expected:

- strong gait reward
- swing feet evaluated for clearance

### Case C — Opposite diagonal

```text
FL = swing
RR = swing
FR = stance
RL = stance
```

Expected:

- strong gait reward

### Case D — All four feet on ground

Expected:

- gait reward should NOT be maximal

This tests that the support term is working.

### Case E — Swing foot at 5 cm

Expected:

- meaningful clearance penalty

### Case F — Swing foot at 10 cm

Expected:

- approximately zero clearance deficit

### Case G — Swing foot at 15 cm

Expected:

- no additional clearance penalty

### Case H — Calf contact

Expected:

- small negative reward
- no automatic termination

### Case I — Arm contact

Expected:

- small negative reward
- no automatic termination

---

# 32. Training Strategy

Do not immediately commit to 10,000 iterations for debugging.

Use a staged process.

### Stage 1 — Reward validation

Short run with enough environments to expose numerical/contact problems.

Goal:

- no NaNs
- no tensor shape errors
- reward terms activate correctly
- no unexpected reward explosions

### Stage 2 — Short learning run

Run enough iterations to see whether:

- gait reward increases
- clearance improves
- calf contacts decrease
- arm contacts decrease
- velocity tracking remains reasonable

### Stage 3 — Full training

Only after Stage 2 behaves sensibly.

Use the same full training configuration as Policy A.

---

# 33. Evaluation Must Be Separate from Training Reward

Do not decide that Policy B is better merely because its total reward is higher.

Policy B has a different objective and therefore a different reward scale.

Evaluate independently.

Recommended metrics:

### Velocity tracking

\[
e_v =
\left\|
v_{\text{cmd}}-v_{\text{actual}}
\right\|
\]

### Yaw tracking

\[
e_{\omega}
=
|\omega_{z,\text{cmd}}-\omega_{z,\text{actual}}|
\]

### Foot clearance

Measure:

- mean swing-foot height
- minimum swing-foot height
- percentage of swing samples with \(h \ge 0.10m\)

### Gait

Measure:

- diagonal contact correlation
- stance/swing alternation
- step frequency
- duty factor
- consistency across episodes

### Foot slip

Measure stance-foot velocity.

### Contacts

Measure:

- calf/knee contacts per episode
- arm contacts per episode
- contact duration

### Stability

Measure:

- base roll RMS
- base pitch RMS
- body-height variation

### Robustness

Measure:

- fall rate
- episode duration
- recovery after disturbances

### Energy

Use a consistent actuator-energy proxy such as:

\[
E_{\text{proxy}}
=
\sum_i |\tau_i\dot q_i|
\]

over the same evaluation horizon.

---

# 34. Evaluation Protocol

Policy A and Policy B must be evaluated using:

- identical command sequences
- identical terrain seeds
- identical initial-state distributions
- identical episode lengths
- identical disturbance conditions

This makes the comparison paired rather than anecdotal.

For each condition, evaluate enough episodes to reduce variance.

Recommended conditions:

1. Standing
2. Slow forward walking
3. Fast forward walking
4. Lateral motion
5. Turning
6. Forward + turning
7. Flat terrain
8. Rough terrain

The gait reward should not be judged only on straight-line walking.

---

# 35. What Success Looks Like

Policy B should ideally demonstrate:

### Improvement

- more consistent diagonal coordination
- fewer low-clearance swing events
- at least 10 cm foot clearance during swing
- fewer calf contacts
- fewer arm contacts

### No significant degradation

- velocity tracking
- yaw tracking
- stability
- terrain robustness
- energy consumption
- command responsiveness

The strongest result is therefore not:

> "Policy B has a higher reward."

It is:

> "Policy B maintains comparable locomotion performance while producing more consistent gait coordination, ≥10 cm swing-foot clearance, and fewer undesirable contacts."

---

# 36. Ablation Plan

After the main two-policy comparison, further experiments can identify which change caused which behavior.

Recommended ablations:

### A — Baseline

Current reward.

### B — Baseline + clearance

Tests the 10 cm requirement alone.

### C — Baseline + contact penalties

Tests calf/arm avoidance alone.

### D — Baseline + gait reward

Tests gait shaping alone.

### E — Full Round 2

Gait + clearance + calf + arm.

This is scientifically stronger than only comparing A and E.

If compute is limited, at minimum compare:

```text
Baseline
vs
Full Round 2
```

and log the new reward components separately.

---

# 37. Potential Failure Modes

## Failure 1 — Robot stops moving

Possible cause:

- gait/contact penalties too strong
- clearance objective interfering with velocity tracking

Response:

- reduce new reward weights
- inspect weighted reward contributions

---

## Failure 2 — Robot keeps feet high

Possible cause:

- clearance reward accidentally rewards height above 10 cm

Response:

- use deficit-only penalty
- ensure heights above 10 cm do not receive increasing reward

---

## Failure 3 — Robot moves feet while standing

Possible cause:

- gait reward is active at zero command

Response:

- apply command-speed gating

---

## Failure 4 — Turning gets worse

Possible cause:

- trot prior is too strong during yaw commands

Response:

- reduce gait reward during strong turning
- consider yaw-dependent gating

---

## Failure 5 — Robot becomes extremely conservative

Possible cause:

- calf/arm penalties are too strong

Response:

- reduce penalties
- avoid termination on ordinary calf/arm contacts

---

## Failure 6 — Reward explodes

Possible cause:

- contact/clearance term is unbounded
- reward is calculated incorrectly for no swing feet

Response:

- bound/normalize the gait reward
- return zero when no foot is swinging
- protect denominators
- inspect raw and weighted reward statistics

---

## Failure 7 — Gait reward is high but robot does not walk

Possible cause:

- contact synchronization reward is being satisfied without meaningful stepping

Response:

- retain alternating-support constraint
- use actual locomotion-command gating
- inspect contact sequences rather than reward alone

---

# 38. Engineering Recommendation on Gait Complexity

Do not implement a full prescribed gait phase generator in this round.

The first version should be:

```text
command-conditioned
+
diagonal synchronization
+
alternating support
+
10 cm foot clearance
+
existing foot-slip penalty
```

If this produces a stable trot-like gait, stop there.

If a later research question requires a true four-beat walk, introduce explicit phase estimation or phase-conditioned gait rewards as a separate experiment.

This keeps Round 2 interpretable.

---

# 39. Engineering Recommendation on Terrain

The current environment includes flat and rough terrain generation.

The current environment initialization also disables the terrain-level curriculum:

```python
self.curriculum.terrain_levels = None
```

Do not change terrain curriculum in Round 2.

Terrain curriculum would be another independent experimental variable and could obscure whether reward shaping caused the observed improvement.

---

# 40. Engineering Recommendation on PPO

Do not increase network size or change optimizer settings simply because the new reward is more complicated.

The current actor/critic:

```text
[256, 512, 128]
```

is capable of representing the existing locomotion policy and should be the first baseline for Round 2.

If learning becomes unstable, diagnose the reward first before changing the network.

---

# 41. Recommended Git Workflow

Create a separate branch before modifying the reward system:

```bash
git checkout -b gait-aware-reward
```

Keep Policy A's code/configuration intact.

Suggested experiment structure:

```text
experiments/
    baseline/
        config/
        logs/
        checkpoints/
        evaluation/

    gait_aware_v1/
        config/
        logs/
        checkpoints/
        evaluation/
```

Record the Git commit hash for every training run.

This makes results reproducible.

---

# 42. Recommended Configuration Naming

Use explicit names.

For example:

```python
experiment_name = "go2_vanilla_baseline"
```

for Policy A.

For Policy B:

```python
experiment_name = "go2_gait_clearance_contact_v1"
```

Do not overwrite baseline checkpoints.

---

# 43. Recommended First Implementation Sequence

1. Freeze Policy A baseline code.
2. Train Policy A.
3. Save checkpoints and evaluation metrics.
4. Inspect baseline gait/contact behavior.
5. Refactor the existing contact penalties into separate calf/head/arm categories.
6. Disable arm termination for Round 2 after baseline contact analysis.
7. Replace/refactor `foot_lift_exp` into a true minimum-clearance objective.
8. Set minimum swing-foot clearance to 0.10 m.
9. Implement diagonal gait coordination.
10. Add command-speed gating.
11. Add initial reward weights:
    - gait: +0.30
    - clearance: deficit-only, starting scale approximately 0.25
    - calf: -0.25
    - arm: -0.25
12. Run reward sanity tests.
13. Run a short training test.
14. Inspect reward decomposition and gait/contact metrics.
15. Train Policy B from scratch.
16. Evaluate A vs B under identical conditions.
17. If needed, perform ablations.

---

# 44. Core Mathematical Specification

The intended Round 2 objective can be summarized as:

\[
R_B
=
R_A
+
w_gR_{\text{gait}}
+
R_{\text{clearance}}
-
w_cC_{\text{calf}}
-
w_aC_{\text{arm}}
\]

where:

\[
w_g=0.30
\]

\[
w_c=0.25
\]

\[
w_a=0.25
\]

and:

\[
R_{\text{gait}}
=
R_{\text{sync}}
R_{\text{support}}
G(v_{\text{cmd}},\omega_{\text{cmd}})
\]

with:

\[
R_{\text{sync}}
=
1-
\frac{
|c_{FL}-c_{RR}|
+
|c_{FR}-c_{RL}|
}{2}
\]

and:

\[
R_{\text{support}}
=
1-
\left|
\frac{c_{FL}+c_{RR}}{2}
+
\frac{c_{FR}+c_{RL}}{2}
-1
\right|
\]

The clearance penalty is:

\[
R_{\text{clearance}}
=
-\frac{1}{N_{\text{swing}}}
\sum_i
\left[
\frac{
\max(0,0.10-h_i)
}{0.05}
\right]^2
\]

with the sum only over swing feet.

The critical design requirement is:

\[
\boxed{
h_i\ge0.10m
\quad\Rightarrow\quad
\text{no clearance penalty}
}
\]

rather than rewarding arbitrarily high feet.

---

# 45. Final Design Philosophy

The second policy should not be thought of as:

> "A manually programmed gait."

It is better described as:

> **A PPO locomotion policy trained with additional behavioral constraints that encourage diagonal coordination, sufficient swing-foot clearance, and avoidance of non-foot ground contacts.**

The learning problem remains:

\[
a_t=\pi_\theta(o_t)
\]

The actor still receives the same observations and produces the same joint-position actions.

The difference is that PPO receives a more informative definition of desirable locomotion:

```text
Track commanded motion
        +
Remain stable
        +
Move feet with coordinated diagonal timing
        +
Clear the ground by ≥10 cm during swing
        +
Avoid calf/knee ground contact
        +
Avoid arm ground contact
```

The robot is still free to discover the detailed joint trajectories required to satisfy those objectives.

---

# 46. References

1. SMEAC, `Quadruped_Go2_Locomotion` repository:
   `https://github.com/SMEAC/Quadruped_Go2_Locomotion`

2. Project environment configuration:
   `source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/quadruped_go2_locomotion_env_cfg.py`

3. Project reward implementation:
   `source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/mdp/rewards.py`

4. Project MDP exports:
   `source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/mdp/__init__.py`

5. Project PPO configuration:
   `source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/agents/rsl_rl_ppo_cfg.py`

6. Project training script:
   `scripts/rsl_rl/train.py`

7. Literature precedent for explicit foot-clearance and locomotion reward shaping:
   Scientific Reports, "Learning-based quadrupedal locomotion..." and related quadruped RL literature.

---

## Engineering Decision Summary

**Implement now:**

- diagonal gait coordination reward
- command-conditioned gait activation
- true 10 cm swing-foot clearance penalty
- separate small calf/knee contact penalty
- separate small arm contact penalty
- remove/disable arm contact as automatic termination for this experiment
- retain catastrophic base/head termination
- keep observations/actions/PPO/terrain/commands unchanged

**Do not implement yet:**

- exact joint trajectory tracking
- explicit gait-phase observation
- full prescribed four-beat gait
- PPO architecture changes
- terrain curriculum changes
- simultaneous observation-space changes

**Initial hypothesis:**

> A modest gait prior combined with a correctly formulated 10 cm swing-foot clearance objective and small non-foot contact penalties will produce a more deliberate and visually/quantitatively coordinated walking gait while preserving the baseline policy's velocity tracking and stability.

The hypothesis must be tested empirically through reward decomposition, gait/contact metrics, and paired evaluation of the vanilla and gait-aware policies.
