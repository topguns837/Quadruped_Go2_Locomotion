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
            border_width=20.0,
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
        rel_standing_envs=0.02,
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

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


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
        base_lin_vel = ObsTerm(
            func=mdp.logged_base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-50.0, 50.0),
        )
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

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
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
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.2, 0.5), "yaw": (-3.14, 3.14)},
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
            "position_range": (0.5, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_pitch_exp = RewTerm(
        func=mdp.track_pitch_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.1)},
    )
    track_lean_exp = RewTerm(
        func=mdp.track_lean_exp,
        weight=0.3,
        params={"command_name": "base_velocity", "std": math.sqrt(0.1)},
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
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
        func=mdp.base_height_l2_pitch,
        weight=-1.65,
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
    # foot lift reward -- reward feet above min_height when not in contact
    foot_lift = RewTerm(
        func=mdp.foot_lift_exp,
        weight=0.1,  # tune: increase for stronger lift encouragement
        params={
            "std": math.sqrt(0.01),  # 0.1 m e-folding height
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "feet_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "min_height": 0.05,  # 5 cm
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


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


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
        
        # terrain curriculum
        self.curriculum.terrain_levels = None

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