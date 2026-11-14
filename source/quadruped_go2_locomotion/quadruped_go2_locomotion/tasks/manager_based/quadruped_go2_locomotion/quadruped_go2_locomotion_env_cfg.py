# --- Hierarchical Policy Utilities ---
import torch
import os



class LowLevelPolicyWrapper:
    """Wrapper for loading and running the pretrained low-level policy (Network B)."""
    def __init__(self, policy_path=None, device="cpu", freeze=True):
        if policy_path is None:
            # Default path to LL policy
            policy_path = os.path.join(os.path.dirname(__file__), "../../../assets/LL_policy/exported/policy.pt")
        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        if freeze:
            for param in self.policy.parameters():
                param.requires_grad = False

    def forward(self, obs):
        with torch.no_grad():
            return self.policy(obs)

    def __call__(self, obs):
        return self.forward(obs)
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

# my libs
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise



from . import mdp

##
# Pre-defined configs
##

from .unitree_go2witharm_cfg import UNITREE_GO2WITHARM_CFG  # isort: skip


##
# Scene definition
##


@configclass
class QuadrupedLocomotionSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            # 70 m: episode travel is bounded by sqrt(2) m/s * 40 s ~= 57 m, so even an edge tile's robot
            # cannot reach the world edge. The border is four flat boxes, so width costs nothing.
            border_width=70.0,
            num_rows=32,
            num_cols=32,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0),
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
                ),
            },
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
    )
    """        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False, 
    """

    # robot
    robot: ArticulationCfg = UNITREE_GO2WITHARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # Go2 contact sensor (monitors legs + feet)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # Open Manipulator X contact sensor. The manipulator is a sibling of base (not nested under it) named
    # "open_manipulator_x_static" in the current go2withOpenXstatic.usd (verified directly by traversing
    # the stage -- was "OpenManipulatorX" in the earlier, incorrect-mounting .usda, renamed by whatever
    # exported the corrected-mounting version), fixed to base purely via a joint rather than USD parenting.
    # It must stay a sibling, not nested under base: Isaac Lab's activate_contact_sensors never recurses
    # past a rigid body, so nesting under the rigid base body would leave it permanently unreachable.
    contact_forces_arm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/open_manipulator_x_static/.*", history_length=3, track_air_time=True
    )

    """     depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/DepthCamera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.25, 0.0, 0.7), rot=(0.66446, 0.24184, -0.24184, -0.66446), convention="opengl"),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.15,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1.0),
        ),
        width=320,
        height=240,
    ) """
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfgWithPitch(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,  # was 0.12; more pure-standing practice (robot fell at/near zero command)
        # Oversample the low-speed regime: ~35% of the non-standing envs get |lin_vel| <= 0.3 m/s so slow
        # walking is dense, not the sparse tail of the uniform (-1,1) range. See mdp/commands.py.
        rel_low_speed_envs=0.35,
        low_speed_max=0.3,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfgWithPitch.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
            ang_pos_x=(-0.3, 0.3),
            ang_pos_y=(-0.6, 0.6),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Same affine map as before (default + 0.25 * action, no clipping), but each applied target lags the policy output by
    # 0-2 control steps (random per env per episode): the robot's command path is not instantaneous. See mdp/actions.py.
    # The delay range scales with env.difficulty_scale (none at 0, the full 0-2 at 1).
    joint_pos = mdp.DelayedJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, max_delay_steps=2
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # func=mdp.logged_* wraps the underlying isaaclab.envs.mdp function to log a per-step
        # min/max/mean/finite summary for every term to this session's step_diagnostics.log (see
        # mdp/diagnostics.py), without changing the term's value. clip on the velocity terms bounds a
        # runaway inf/huge-finite physics-solver spike (doesn't catch true NaN, see diagnostics.py's
        # finite-check for that).
        # base_lin_vel is deliberately NOT an actor observation: on real hardware it is only available from
        # SportModeState, which STOPS publishing once release_motion_control() runs (required to send LowCmd),
        # so it was dead-zero on every deploy tick -- a heavily-weighted input the policy could never see,
        # which is why the robot drifted and tipped. Training the actor without it makes the sim and hardware
        # observations identical. The critic keeps it as a privileged input (CriticCfg below); the critic is
        # discarded at deploy, so this is free. Actor obs: 50 -> 47 dims.
        base_ang_vel = ObsTerm(
            func=mdp.logged_base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-50.0, 50.0),
        )
        projected_gravity = ObsTerm(
            func=mdp.logged_projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.logged_generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.logged_joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(
            func=mdp.logged_joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-50.0, 50.0),
        )
        actions = ObsTerm(func=mdp.logged_last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations (never deployed): clean proprio (no sensor noise) in the SAME order
        as the policy group, PLUS base_lin_vel which the actor cannot observe on hardware. Asymmetric
        actor-critic -- the value function learns with the true body velocity while the actor stays
        hardware-realistic. Mirrors the stairs env's CriticCfg."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-50.0, 50.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-50.0, 50.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-50.0, 50.0))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Randomised per environment: real floors and foot rubber vary, and the old fixed 0.8/0.6 let the policy
            # tune to one value. make_consistent keeps dynamic <= static.
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    # Curriculum-scaled randomisation of the trunk mass, its centre of mass and the push forces (robot_mass, robot_com,
    # push_robot_force below). Each range is multiplied by env.difficulty_scale (0 -> 1, see CurriculumCfg.difficulty
    # and mdp/curriculums.py:difficulty_scale), so training starts with none of it and gets harder only as the policy
    # keeps the robot up. The ranges here are the FULL strength, reached at difficulty 1. Full strength applied from
    # iteration 0 stalled the previous run (52% falls vs 4% without it). They are reset events (redrawn each episode)
    # because a startup event is drawn once and cannot follow a curriculum.
    #
    # The velocity push (push_robot) and the action delay (ActionsCfg.joint_pos) follow the same scale since iter4
    # phase-2. Everything else in this class (friction, PD gains, armature, joint friction) stays on at full strength
    # from the start: they are the hardware-parity randomisations the previous good policy trained with.
    robot_mass = EventTerm(
        func=mdp.randomize_base_mass_curriculum,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            # kg added to the base (battery, payload, arm differences) at full difficulty. iter4 realistic: +-5 -> +-2 kg
            # (~+-13% of the ~15.6 kg robot). The Go2 sim-to-real norm is ~+-1 kg; +-2 allows arm/payload spread while
            # staying a realistic disturbance, not an adversarial load. The inertia is scaled with the mass.
            "max_abs_kg": 2.0,
        },
    )
    # Centre-of-mass offset on the base at full difficulty: a payload/battery is rarely bolted exactly at the
    # geometric centre. It moves the balance point (fore/aft shift fights the pitch command, lateral shift fights
    # lean/roll). Kept inside the support polygon so every draw stays balanceable: the Go2 foot stance spans
    # ~+-0.19 m fore/aft and ~+-0.12 m lateral, x/y are held to ~55% and ~65% of that. z is vertical, so it can be wider.
    robot_com = EventTerm(
        func=mdp.randomize_base_com_curriculum,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "max_xyz": (0.06, 0.05, 0.08),  # meters; iter4 realistic: tightened from (0.10, 0.08, 0.15)
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.75, 1.5),  # x nominal JOINT_ARMATURE; iter4 realistic: 4x range -> 2x
            "operation": "scale",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.1),  # Nm of joint friction (the robot has some; the sim had none)
            "operation": "abs",
        },
    )
    # iter4 phase-2: the kick now follows env.difficulty_scale (0 -> none, 1 -> +-0.5 m/s), like the force push below.
    push_robot = EventTerm(
        func=mdp.push_velocity_curriculum,
        mode="interval",
        interval_range_s=(8.0, 15.0),
        params={"max_velocity": 0.5},
    )
    # External force/torque shoves on the trunk, in ADDITION to the velocity kick above: a real push is a
    # force the policy must actively resist, not an instantaneous velocity reset. Applied to the base body and
    # re-sampled every interval (persisting between), so it is a shifting disturbance field. Curriculum-scaled: at full
    # difficulty +-15 N on each axis (robot ~15.6 kg, weight ~153 N, so ~10% of weight) and +-2 Nm torque.
    # iter4 realistic: reduced from +-40 N / +-5 Nm. At 40 N (26% of weight) held for seconds the policy learned to
    # crouch and leave the default pose to avoid falling (posture eroded at difficulty >0.5); 15 N is a mild
    # slope/lean/drag, a realistic disturbance rather than an adversary. Disabled in the PLAY cfg, like push_robot.
    push_robot_force = EventTerm(
        func=mdp.push_force_curriculum,
        mode="interval",
        interval_range_s=(5.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "max_force": 15.0,
            "max_torque": 2.0,
        },
    )

    # Corrects 2.0 kg of phantom mass, 13% of the robot. `imu` and `radar` are massless sensor frames in
    # Unitree's URDF and are authored `physics:mass = 0.0` in our USD, but a rigid body cannot have zero mass
    # in PhysX, so it silently substitutes its 1.0 kg default for EACH of them. Measured via
    # root_physx_view.get_masses(): the robot loaded at 17.623 kg against a ~15.6 kg real weight, and because
    # `radar` sits at (+0.289, 0, -0.047) that put 1 kg at the nose, 29 cm forward of the base origin, shifting
    # the centre of mass ~1 cm forward and inflating pitch inertia. Our hardware has no radar fitted at all.
    #
    # Notes on the specific arguments:
    #  - "abs" sets an absolute value and the degenerate (x, x) range makes this a deterministic setter, the
    #    same trick physics_material above uses. 1 g rather than 0 because the term clamps at min_mass=1e-6.
    #  - mode="startup" runs once after sim start, by which point the actuator's cached default_mass holds
    #    PhysX's substituted 1.0 kg; "abs" then overwrites it. The term re-reads default_mass before applying,
    #    so it is idempotent. "prestartup" would not work with replicate_physics=True.
    #  - recompute_inertia=False because PhysX substituted only the MASS: the authored
    #    physics:diagonalInertia = 1e-5 is non-zero and was kept. Rescaling it to 1e-8 buys nothing, and these
    #    are not uniform-density solids. Either way both bodies are welded to `base` by fixed joints, so 2e-5
    #    against the base's ~0.1 kg*m^2 is a 0.02% effect.
    #  - body_names are literals matched with re.fullmatch, so they hit exactly these two bodies and raise at
    #    startup if either name ever disappears, instead of silently doing nothing. Neither body has any child
    #    prim, geometry or CollisionAPI, and neither appears in any reward/termination/contact body_names list,
    #    so this change is confined to dynamics.
    #  - UsdFileCfg.mass_props cannot do this: modify_mass_properties is @apply_nested and would overwrite all
    #    47 rigid bodies' masses.
    fix_massless_sensor_bodies = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["imu", "radar"]),
            "mass_distribution_params": (0.001, 0.001),
            "operation": "abs",
            "recompute_inertia": False,
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.3, 0.3),
                "z": (-0.25, 0.25),
                "roll": (-0.15, 0.15),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # Velocity tracking is a TWO-SCALE sum (total weights unchanged: 10 linear, 2 yaw):
    #  * a TIGHT speed-scaled term (std = max(floor, 0.5*|cmd|), equal to the old fixed 0.5 at >= 1 m/s) carries most of
    #    the weight, so a robot that is not following the command scores ~0 from it at any command >= 0.2 m/s -- the fix
    #    for the old dead zone (a fixed std 0.5 let a STANDING robot keep 0.70 of the reward at 0.3 m/s, 0.96 at 0.1);
    #  * a BROAD fixed-std term (the original kernel) at lower weight keeps a dense gradient everywhere. Calibration
    #    showed the tight term alone is flat for a robot that has not started moving (400-iteration run: linear
    #    tracking stuck at ~0.65 and twice the velocity error of the old run), so exploration never found walking.
    # Standing at cmd 0.3 now scores 0.4*10*0.70 + 0.6*10*0.02 = 2.9/10; tracking perfectly scores 10.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp_rel,
        weight=6.0,
        params={"command_name": "base_velocity", "std_floor": 0.08, "rel": 0.5},
    )
    track_lin_vel_xy_broad = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=4.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp_rel,
        weight=1.2,
        params={"command_name": "base_velocity", "std_floor": 0.1, "rel": 0.5},
    )
    track_ang_vel_z_broad = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.8, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # Standing-when-commanded penalty: the exponential kernels above still let a standing robot bank most of the BROAD
    # term at a small command (~2.9/10 at 0.3 m/s, ~5/10 at 0.1), so the gradient off standstill is weak -- the dead
    # zone. This penalises the speed error along the commanded direction as a FRACTION of the command (bounded [0, 1]):
    # standing costs the full weight at any command size, a tracker pays ~0, overshoot is penalised too. It replaces an
    # L1 m/s deficit that was negligible at 0.1 m/s (weight -1.5) or, when raised (-4.5), so large at high speed that the
    # policy fell and overshot 1.6-1.9x. Hard-gated off for a zero linear command. -4.0 picked over -2.0 from 400-iteration
    # calibration: best velocity error (0.91), no falls at 0.5/0.8 m/s, steps from 0.2 m/s (see
    # low_speed_standing_penalty_plan.md; single seed, so re-check low-speed stepping on later checkpoints).
    standing_when_commanded = RewTerm(
        func=mdp.standing_when_commanded, weight=-4.0, params={"command_name": "base_velocity"}
    )
    # Yaw analogue (L1 rad/s shortfall, one-sided); the dead zone is mainly linear, so this stays light.
    track_ang_vel_deficit = RewTerm(
        func=mdp.track_ang_vel_deficit, weight=-0.5, params={"command_name": "base_velocity"}
    )
    track_pitch_exp = RewTerm(
        func=mdp.track_pitch_exp,
        # Weight 3.0 (Round 3). std tightened sqrt(0.1) -> sqrt(0.04): at the old std a policy that ignores the
        # +-0.6 rad pitch command still earned 46% of the reward; now 30%, so more of the weight is real gradient.
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.04)},
    )
    track_lean_exp = RewTerm(
        func=mdp.track_lean_exp,
        # Weight 3.0 (Round 3). std tightened sqrt(0.1) -> sqrt(0.02): the old kernel was wider than the whole
        # +-0.3 rad command range, so ignoring lean earned 77% of the reward; now 42%.
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.02)},
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Smoothness / effort penalties, strengthened 2026-10-30 so a smooth, low-torque gait is LEARNED rather
    # than only clipped at deploy (the real robot showed high torque and jerky corrections). These trade off
    # against command tracking -- retune if tracking suffers; watch peak_tau in the sim2sim summary.
    # Every quadratic rate/effort penalty below is CLAMPED (max_value, in the unit of the summed-square value): the
    # stock terms are unbounded, and a physics edge case that gives a large-but-finite joint velocity/torque would
    # otherwise produce a huge negative spike (the failure class behind base_height's max_penalty). Each clamp sits
    # ~4-5x ABOVE the value a noisy-but-sane policy produces (trained noise std ~0.5), so the gradient is unchanged in
    # normal operation and only pathological outliers are cut. Worst-case clamped total per second is about -11 (trio
    # reverted to heavy), plus up to ~-2.6 from the bounded speed-deficit terms, vs +10 tracking + ~+6 pitch/lean + +2
    # alive in-distribution -- so a sane step stays net-positive and falling is never the cheaper option.
    # The smoothness trio was briefly eased (2026-11-04) to relieve the dead zone / foot-drag, then reverted: the
    # one-sided speed-deficit penalty (track_lin/ang_vel_deficit below) now carries the "move when commanded" load
    # directly, so the trio can stay heavy for hardware smoothness. Weights are back to the iter1 values.
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2_clamped, weight=-3.0e-5, params={"max_value": 6.0e3}
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2_clamped, weight=-5.0e-7, params={"max_value": 5.0e6}
    )
    # action_rate back to -0.03. It penalizes any motion, but the high-frequency buzz is caught more specifically by
    # action_jerk_l2 and joint_torque_rate_l2 below (2nd-difference / torque-rate penalties bite the 24.9 Hz Nyquist
    # limit cycle seen on hardware without flattening a clean trot), so this stays a broad smoothness term.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.03, params={"max_value": 50.0})
    dof_vel_l2 = RewTerm(  # penalize fast joint motion directly
        func=mdp.joint_vel_l2_clamped, weight=-1.0e-4, params={"max_value": 1.5e3}
    )
    # Weights calibrated on the 2026-11-04 hardware traces (per-step value summed over 12 joints, deterministic
    # policy): jerk good gait 0.0026 / buzz 10.96; torque-rate (Nm^2) good gait 0.17 / buzz 162. They are computed
    # on SAMPLED actions though: at the trained noise std ~0.5 the exploration noise alone gives jerk ~18 and
    # torque-rate ~300 per step, so the weights are sized for a noise tax of ~0.5/s and ~0.3/s (about 5% of the
    # ~10/s tracking reward each) -- the buzz then adds a comparable, consistent cost the policy can remove for free.
    action_jerk_l2 = RewTerm(  # iter2 NEW: 2nd diff of action (anti-buzz)
        func=mdp.action_jerk_l2, weight=-0.03, params={"max_value": 150.0}
    )
    joint_torque_rate_l2 = RewTerm(  # iter2 NEW: d(applied torque)/step, anti-chatter
        func=mdp.joint_torque_rate_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_value": 2.0e3},
    )
    # -- safety (P0). A terminating contact has no reward penalty of its own, so without these a net-negative step
    # (jerky/noisy early policy) makes ending the episode by falling the cheapest escape. is_alive keeps every
    # in-distribution step net-positive; the fall penalty makes termination strictly worse than any bad state the
    # clamped penalties above can produce. Reward-manager terms are weight * value * step_dt (0.02), so
    # is_alive 2.0 is +2/s and the fall -500 is a one-off -10 return (is_terminated excludes the time_out and
    # terrain_out_of_bounds truncations, and includes body/arm contact and invalid_state).
    is_alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # iter4 phase-2: -500 -> -750 (fall return -10 -> -15). Falling gets strictly costlier as the difficulty
    # curriculum ramps in phase 2; is_alive stays +2.0 so "don't fall" doesn't over-weight command tracking.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-750.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*calf", "Head_lower", "Head_upper"]),
            "threshold": 1.0,
        },
    )
    # Undesired contacts on Open Manipulator X links
    undesired_contacts_arm = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_arm", body_names=["link1", "link2", "link3", "link4", "link5", "gripper_left_link", "gripper_right_link"]),
            "threshold": 1.0,
        },
    )
    # additional penalties to encourage more natural motions
    #
    # target_height=0.337 is the base height the default joint pose produces (thigh 0.8/1.0, calf -1.5, plus
    # the foot collision radius) -- matches what joint_deviation_l1 already pulls toward, so the two rewards
    # agree instead of fighting. pitch_sensitivity lowered from -0.5 (which drove the target to an
    # unreachable 0.037m at the +-0.6 rad pitch limit) to -0.1, allowing ~6cm of crouch at full pitch instead
    # of 30cm -- the base barely moves vertically under pitch anyway, since the hips are +-0.193m fore/aft of
    # center. There is no height COMMAND any more (see mdp.commands' module docstring): no reliable height
    # measurement exists on real hardware, so this fixed target is the only height regulation left.
    height_penalty = RewTerm(
        func=mdp.base_height_l1_pitch,
        # iter3: L2 -> L1 and -4.0 -> -8.0. The quadratic was flat near small errors (a ~10 cm droop logged only
        # ~0.04/s and the policy knelt); L1 has a constant gradient so 5 cm always costs 0.4/s, 10 cm 0.8/s,
        # meaningful vs ~+14/s of tracking but below track_lin (~6-10/s) so it won't dominate. No transient blow-up.
        # iter4 phase-2: -8.0 -> -12.0 (10 cm costs 1.2/s, 2 cm 0.24/s) to hold height as randomisation ramps in.
        # iter4 realistic: -12.0 -> -15.0; with disturbances also reduced, posture holds without a crouch-freeze.
        weight=-15.0,
        params={
            "command_name": "base_velocity",
            "target_height": 0.337,
            "pitch_sensitivity": -0.1,
            "lean_sensitivity": 0.0,
        },
    )

    hip_crossing = RewTerm(
        func=mdp.hip_crossing_l2,
        weight=-0.5,  # tune starting point
        params={
            "joint_ids": [0, 1, 2, 3],  # hip_roll joint indices
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.4,
      },
    )
    # foot sliding penalty -- use same contact sensor as undesired_contacts
    foot_sliding = RewTerm(
        func=mdp.foot_sliding_exp,
        weight=-1.0,  # tune: increase magnitude for stronger penalty
        params={
            "std": math.sqrt(0.04),  # 0.2 m/s e-folding speed
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "feet_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        },
    )
    # Trot shaping (dense, partial credit for weight shifts between diagonals); 0 at a zero command.
    gait_coordination = RewTerm(
        func=mdp.GaitDiagonalCoordination,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "command_name": "base_velocity"},
    )
    # Stand on all four feet in the default pose when commanded to stand; 0 once a velocity is commanded.
    stand_still = RewTerm(
        func=mdp.stand_still_default_pose,
        # iter3: 1.0 -> 2.5. iter2's stand_still collapsed to ~0 (even zero-command envs left default pose), so the
        # default-pose attractor was too weak. Now also factors in an explicit zero-velocity term (vel_scale=0.1):
        # reward is planted * default_pose * exp(-|v|^2/vel_scale), still hard-gated to an all-zero command.
        # iter4 phase-2: 2.5 -> 4.0 (gated to zero-command envs, so no cost to tracking).
        # iter4 realistic: 4.0 -> 5.0 to hold the default stance harder under the (now gentler) disturbances.
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "feet_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "command_name": "base_velocity",
            "vel_scale": 0.1,
        },
    )
    # foot lift reward -- reward feet above min_height when not in contact
    foot_lift = RewTerm(
        func=mdp.foot_lift_exp,
        weight=0.5,  # iter2: 0.1 -> 0.5 so clearing a foot actually pays vs the sliding/smoothness penalties
        params={
            "std": math.sqrt(0.01),  # 0.1 m e-folding height
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "feet_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "min_height": 0.03,  # iter2: 0.10 -> 0.03; 10 cm was unreachable so the policy skimmed/dragged instead
        },
    )
    # -- optional penalties
    #flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_lower", "Head_upper"]),
            "threshold": 1.0,
        },
    )
    # Illegal contact on Open Manipulator X links
    body_contact_arm = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_arm", body_names=["link1", "link2", "link3", "link4", "link5", "gripper_left_link", "gripper_right_link"]),
            "threshold": 1.0,
        },
    )
    # Safety net for non-finite (NaN/Inf) root/joint state from a rare physics-solver edge case,
    # independent of body_contact's base/head-only scope. See mdp/terminations.py.
    invalid_state = DoneTerm(func=mdp.invalid_state)
    # Safety net that should never fire: the terrain border is wide enough that no episode can reach the
    # edge (robots walking off it and free-falling caused the old -4k reward dips). time_out=True makes it a
    # truncation, so the value function bootstraps instead of learning that walking far is dangerous.
    # Any non-zero Episode_Termination/terrain_out_of_bounds therefore flags something unexpected.
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # Strength of the mass / centre-of-mass / push-force randomisation in EventCfg, 0 -> 1. Rises while the policy
    # survives (>= 80% of ended episodes are time-outs, not falls), falls back if it does not (<= 40%).
    difficulty = CurrTerm(
        func=mdp.difficulty_scale,
        params={
            "up_thresh": 0.8,
            "down_thresh": 0.4,
            # iter4 realistic: step 0.05 -> 0.025 (half the increment, ~2x slower ramp, gentler jumps) so posture
            # consolidates at each level instead of being outrun by big difficulty steps.
            "step": 0.025,
            "dwell_steps": 6000,  # env steps per window, 24 per PPO iteration -> 250 iterations
            "settle_steps": 2000,  # one full episode: don't count episodes that began at the old difficulty
            "min_episodes": 500,
        },
    )


##
# Environment configuration
##

@configclass
class QuadrupedLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: QuadrupedLocomotionSceneCfg = QuadrupedLocomotionSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Viewer
    viewer = ViewerCfg(
        eye=(10.5, 10.5, 3.0), origin_type="world", env_index=0, asset_name="robot"
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 40.0
        # simulation settings
        self.sim.dt = 0.005 
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        """scene"""
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # terrain curriculum
        self.curriculum.terrain_levels = None

    



@configclass
class QuadrupedLocomotionEnvCfg_PLAY(QuadrupedLocomotionEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # nominal physics for play and for sim2sim comparisons: no pushes, no action latency, no startup randomisation
        # (armature stays at its nominal JOINT_ARMATURE); friction back to the single nominal 0.8 / 0.6 pair.
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.robot_mass = None
        self.events.robot_com = None
        self.events.actuator_gains = None
        self.events.joint_armature = None
        self.events.joint_friction = None
        self.events.push_robot = None
        self.events.push_robot_force = None
        self.actions.joint_pos.max_delay_steps = 0
        
        # terrain curriculum
        self.curriculum.terrain_levels = None
        # no difficulty scale in play: the mass / COM / push events are disabled above, so there is nothing to ramp
        self.curriculum.difficulty = None

        # NOTE: origin_type="asset_root" subscribes to a per-frame render
        # callback that continuously re-reads the robot's live pose
        # (isaaclab/envs/ui/viewport_camera_controller.py). On at least one
        # low-VRAM laptop this caused full-laptop hangs; origin_type="env"
        # (static, computed once) is a safe substitute with the same close
        # framing. Rather than hardcode that workaround here for everyone,
        # docker/isaaclab-shell.sh applies it via a Hydra CLI override
        # (env.viewer.origin_type=env) only when it detects weak GPU
        # hardware, so capable machines keep this original camera.
        # set the view to be closer
        self.viewer = ViewerCfg(
            eye=(2.0, 2.0, 1.0),
            origin_type="asset_root",
            env_index=0,
            asset_name="robot",
        )