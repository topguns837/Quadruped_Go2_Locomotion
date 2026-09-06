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
            num_rows=9,
            num_cols=21,
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
    # Open Manipulator X contact sensor. Path matches how
    # scripts/compose_go2_with_arm.py places the manipulator reference: as a
    # sibling of base (not nested under it) named "OpenManipulatorX", fixed
    # to base purely via a joint rather than USD parenting. See that script
    # for why (Isaac Lab's activate_contact_sensors never recurses past a
    # rigid body, so nesting under the rigid base body would leave it
    # permanently unreachable).
    contact_forces_arm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/OpenManipulatorX/.*", history_length=3, track_air_time=True
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
            lin_pos_z=(0.2, 0.4),
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

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
    height_penalty = RewTerm(
        func=mdp.base_height_l2_pitch,
        weight=-1.65,
        params={
            "command_name": "base_velocity",
            "pitch_sensitivity": -0.5,
            "lean_sensitivity": 0.05,
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