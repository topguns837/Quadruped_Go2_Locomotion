# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stair-climbing variant of the Go2 locomotion task.

Separate task from `Quadruped-Locomotion-Go2` (not a modification of it) -- only the terrain and the
addition of a depth camera differ; everything else (rewards, actions, commands, events) is inherited
unchanged from `env_cfg.py`.

Scope note: this environment only adds the camera SENSOR to the scene -- it is deliberately NOT wired
into `ObservationsCfg.PolicyCfg`. How depth/point-cloud data reaches the policy (downsampling, point
cloud conversion, network architecture) is a separate, later decision; `mdp.depth_array` (fixed this
session, see mdp/observations.py) is available to use whenever that's decided, but isn't used yet.

The depth camera below is configured to match a real Intel RealSense D435f (the NIR-filtered D435
variant intended for robotics -- same optical specs as the standard D435). See DEPTH_CAMERA's own
comments for the exact numbers and sources. Only the MOUNT POSE (pos/rot) is still a placeholder,
since that depends on where the real camera ends up physically mounted on the robot -- retune once
that's decided.
"""

import math

import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, TiledCameraCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from . import mdp
from .env_cfg import (
    ObservationsCfg,
    QuadrupedLocomotionEnvCfg,
    QuadrupedLocomotionSceneCfg,
)

##
# RealSense D435f depth-camera match
##

# Sourced from Intel's official D435/D435f product pages (D435f = D435 with a 750nm NIR bandpass filter
# added for robotics use; same optical specs otherwise):
#   - Depth FOV: 87 deg (H) x 58 deg (V), +/-3 deg measurement tolerance (Intel's own stated tolerance).
#   - Depth stream resolution: 424x240 is one of the camera's own natively supported depth modes (also
#     supports 1280x720/848x480/640x480/640x360/480x270) -- picked here for simulation throughput at
#     1024+ parallel envs while keeping the real 16:9 aspect ratio, not an arbitrary downscale.
#   - Range: ideal 0.3-3 m, ~0.28 m minimum depth distance at max resolution (Intel's own spec; a wider
#     0.2-10 m figure exists in some retailer listings but is lighting/scene-dependent, not the datasheet
#     number, so not used here).
# Isaac Lab's PinholeCameraCfg takes focal_length + horizontal_aperture (mm), not FOV directly, and infers
# vertical_aperture (hence vertical FOV) from the aperture/resolution ratio when left unset. horizontal_aperture
# is kept at Isaac Lab's own default (20.955mm, an arbitrary but standard "film size" scale -- any value
# works as long as focal_length is solved to match it); focal_length is solved below so the resulting
# horizontal FOV is exactly 87 deg, which independently yields a vertical FOV of ~56.5 deg given the
# 424x240 aspect ratio -- within Intel's own +/-3 deg tolerance of the spec's 58 deg.
_D435F_DEPTH_WIDTH = 424
_D435F_DEPTH_HEIGHT = 240
_D435F_HORIZONTAL_APERTURE_MM = 20.955
_D435F_HORIZONTAL_FOV_DEG = 87.0
_D435F_FOCAL_LENGTH_MM = _D435F_HORIZONTAL_APERTURE_MM / (2 * math.tan(math.radians(_D435F_HORIZONTAL_FOV_DEG / 2)))

# Resolution the camera actually RENDERS at for the policy -- a quarter of the D435f's 424x240 native depth
# mode in each axis (106x60), giving exactly the same 16:9 aspect ratio, hence the same horizontal and
# vertical FOV (FOV is set by aperture/focal_length, not pixel count, so lowering resolution doesn't change
# what's in frame). Rendering directly at this size, rather than rendering 424x240 and downsampling, is what
# makes the depth render affordable at 1000+ parallel envs; it's also close to the effective detail a small
# CNN encoder uses. Retune upward only if stair edges turn out under-resolved for the policy.
_DEPTH_RENDER_WIDTH = 106
_DEPTH_RENDER_HEIGHT = 60

# Depth stream frame rate. 30 fps is a native D435 depth mode (supported set is 5/15/30/60/90 fps; 424x240
# supports all) and the most common real deployment rate -- set to the hardware spec, not an arbitrary
# number. update_period below is 1/this. Control runs at 50 Hz, so a fresh depth frame arrives roughly
# every 1.7 control steps; between updates the policy reuses the latest frame, exactly as on the robot.
_DEPTH_CAMERA_FPS = 30.0

# Depth normalize/clip range (metres) and time-stacking for the DepthImage observation (see mdp/observations.py).
# far=2.0 (not the 3.0 render clip): stairs past ~2 m don't affect the next footstep and D435 stereo noise
# grows quickly with range, so clamping sooner keeps the policy input in its reliable band.
#
# The depth observation is (num_envs, num_frames, H, W): the num_frames most recent frames stacked along a
# CHANNEL axis (the CNN's first conv takes in_chans=num_frames), NOT flattened -- each frame keeps its full
# H x W spatial structure. 3 frames, each 5 control-steps (=0.1 s, ~3 camera frames at 30 fps) apart, span
# ~0.2 s of history: enough to read approach motion and remember the step currently under the front feet
# once it leaves the field of view.
_DEPTH_NEAR_M = 0.28
_DEPTH_FAR_M = 2.0
_DEPTH_NUM_FRAMES = 3
_DEPTH_FRAME_STRIDE_STEPS = 5

##
# Camera mount pose -- single source of truth
##

# Both depth_camera (the real sensor) and realsense_mesh (the cosmetic housing, purely so the mount is
# visible in the viewport) read pos/rot from these same constants. They used to be two independently
# hardcoded tuples that drifted apart the moment only one was edited -- confirmed directly as the cause of
# "the mesh doesn't move when I change the camera's rotation" (depth_camera's rot had been edited,
# realsense_mesh's hadn't). Edit the mount pose here, in exactly one place, and both move together.
_CAMERA_MOUNT_POS = (0.0, 0.0, 0.1)

# Tune these three numbers by eye (Stage tree move/rotate gizmo on DepthCamera or RealSenseMesh, Camera
# Inspector for the live feed, the coordinate-frame marker in play.py) instead of ever hand-deriving a
# quaternion again -- every mount-pose mistake this session traced back to hand quaternion math. Standard
# ROS/REP103 body frame this project uses elsewhere: +X forward, +Y left, +Z up. All zero = facing
# straight forward, level.
_CAMERA_MOUNT_ROLL_DEG = 0.0
_CAMERA_MOUNT_PITCH_DEG = 30.0  # positive = tilt down
_CAMERA_MOUNT_YAW_DEG = 0.0  # positive = turn left (toward +Y), standard right-hand rule about +Z

# Isaac Lab's OffsetCfg(convention="opengl") takes the camera's OWN view-axis orientation directly as
# `rot` -- NOT a body-frame-identity rotation (verified by rendering this session: a raw identity rot
# makes the camera look straight DOWN in opengl convention, not forward). This is the one quaternion that
# needed hand-verification at all: it's the "zero tilt" baseline -- camera view direction (-Z) -> body
# +X, camera up (+Y) -> body +Z -- confirmed earlier this session by rotating the camera's own basis
# vectors through it and checking they land on +X/+Z, then independently confirmed again by rendering a
# clean forward-facing depth image with it. The tunable euler angles above are composed onto this fixed
# baseline by _compute_camera_mount_rot(), so this is the only place in the file a raw quaternion appears.
_FORWARD_LEVEL_OPENGL_ROT = (0.5, 0.5, -0.5, -0.5)


def _quat_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """XYZ-Euler (degrees) -> quaternion (w, x, y, z), the same convention isaaclab.utils.math's
    quat_from_euler_xyz / euler_xyz_from_quat use elsewhere in this project (e.g. mdp/commands.py,
    deploy_real.py's quat_to_pitch_lean) -- reimplemented in plain Python (not torch) since this only
    runs once at config-definition time and a scalar/tensor roundtrip isn't worth the dependency here."""
    roll, pitch, yaw = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _quat_mul(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton product (w, x, y, z) * (w, x, y, z) -- applies q2's rotation first, then q1's."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _compute_camera_mount_rot() -> tuple[float, float, float, float]:
    """Composes the tunable body-frame euler tilt ON TOP of the fixed, already-verified forward-level
    baseline: body_euler (applied second, in body-frame terms) * baseline (applied first, camera->body)."""
    euler_quat = _quat_from_euler_deg(_CAMERA_MOUNT_ROLL_DEG, _CAMERA_MOUNT_PITCH_DEG, _CAMERA_MOUNT_YAW_DEG)
    return _quat_mul(euler_quat, _FORWARD_LEVEL_OPENGL_ROT)


_CAMERA_MOUNT_ROT = _compute_camera_mount_rot()

# Correction for the converted RealSense mesh ASSET's own modeled orientation, applied to realsense_mesh
# ONLY -- never to depth_camera. The TiledCamera sensor interprets `rot` under the OpenGL camera
# convention (forward = -Z); the mesh is a plain static prop whose "forward" (lens-pointing) axis is
# whatever the raw Collada/USD file happened to be authored with, which doesn't have to agree with that
# convention. Measured directly (visual comparison, GUI): the mesh's modeled axes are pitched 90 deg
# relative to where depth_camera actually looks. -90.0 was tried first and visually confirmed backwards
# (GUI check); +90.0 is the verified-correct sign. Independent of the tunable mount-pose angles above and
# of depth_camera, which is unaffected either way -- a property of this specific mesh file only.
_REALSENSE_MESH_ASSET_ROT_CORRECTION = _quat_from_euler_deg(-90.0, 0.0, 0.0)

##
# Scene definition
##


@configclass
class QuadrupedLocomotionStairsSceneCfg(QuadrupedLocomotionSceneCfg):
    """Same scene as the flat-ground task, but with a stairs-capable terrain and a depth camera."""

    # Multiple stairs variants (fixed step_width per entry -- Isaac Lab's MeshPyramidStairsTerrainCfg
    # only ranges step_height, not step_width, so "multiple step sizes" means multiple sub-terrain
    # entries), plus a little flat/rough so the robot also sees approach/landing ground. step_height_range
    # here is a starting point (scaled down from Isaac Lab's own ANYmal-scale (0.05, 0.23) reference for
    # Go2's smaller size) -- expect retuning once training starts.
    #
    # curriculum=True (off by default, and off in the base flat-ground task too) makes step_height within
    # each range scale with the per-row difficulty via mdp.terrain_levels_vel (see QuadrupedLocomotionStairsEnvCfg
    # below, which re-enables that curriculum term) -- confirmed terrain-type-agnostic, no changes needed there.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            # Minimum safe value, not an arbitrary shrink: mdp.terrain_out_of_bounds (the existing
            # out-of-bounds termination) ends an episode once the robot is within 3m of the TOTAL world
            # edge (grid + border, confirmed in isaaclab_tasks' terrain_out_of_bounds source), and the
            # worst-case travel distance for the unchanged command range/episode length is
            # sqrt(2) * 1.0 m/s * 40s =~ 57m, independent of terrain type (the command manager doesn't
            # know or care what's commandable on stairs vs flat). 60m keeps a small margin above that
            # bound; going much lower would be genuinely unsafe under the current command/episode
            # settings, not just conservative.
            border_width=60.0,
            # Grown from 32x32 (256m) so the fixed 60m safety border is a much smaller fraction of the
            # total world (384m inner grid + 60m/side border = 504m total -> border ~42% of world area,
            # vs ~58% at 32x32/70m) -- more columns also means more distinct stairs-type/difficulty cells
            # overall, not just a higher stairs proportion within a fixed-size grid.
            num_rows=48,
            num_cols=48,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0),
            curriculum=True,
            use_cache=False,
            # No flat/random_rough: removed entirely (previously a 5% residual) so every column is a
            # stairs variant -- maximizes stairs exposure, and removes the one terrain type where a robot
            # could sustain full commanded speed uninterrupted (the scenario the border-width safety
            # margin above is sized for). Also fixes env placement with a small --num_envs: env-to-column
            # assignment is `floor(env_index / (num_envs/num_cols))` (confirmed in
            # isaaclab.terrains.terrain_importer source), so e.g. --num_envs 1 always lands on column 0 --
            # previously always "flat", now always a stairs variant.
            sub_terrains={
                "stairs_narrow": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.25, step_height_range=(0.03, 0.15), step_width=0.25,
                    platform_width=1.5, border_width=0.5,
                ),
                "stairs_wide": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.25, step_height_range=(0.03, 0.15), step_width=0.35,
                    platform_width=1.5, border_width=0.5,
                ),
                "stairs_narrow_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.25, step_height_range=(0.03, 0.15), step_width=0.25,
                    platform_width=1.5, border_width=0.5,
                ),
                "stairs_wide_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.25, step_height_range=(0.03, 0.15), step_width=0.35,
                    platform_width=1.5, border_width=0.5,
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

    # RealSense D435f stand-in -- see the _D435F_* constants above for sourcing. data_types=["distance_to_camera"]
    # is enough to support either a depth image OR a point cloud later with no sensor-config change --
    # Isaac Lab's own isaaclab.sensors.camera.utils.create_pointcloud_from_depth unprojects this same
    # output using the camera's intrinsics, so that choice is purely downstream/observation-side.
    #
    # Mounted on the head (not base -- the earlier base mount put the Open Manipulator X arm directly in
    # frame, confirmed by actually rendering a depth image). "Head_upper" is the frontmost head link (the
    # same one RewardsCfg already treats as a collision/contact body -- see undesired_contacts above).
    # pos/rot come from the shared _CAMERA_MOUNT_POS/_CAMERA_MOUNT_ROT constants above -- edit those, not
    # here, and realsense_mesh below stays in sync automatically. This assumes Head_upper's local frame is
    # aligned with the base frame (confirmed true this session: measured relative orientation is identity,
    # for a rigid, non-articulated head link).
    depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Head_upper/DepthCamera",
        # 1/30 s = the D435's native 30 fps depth rate (see _DEPTH_CAMERA_FPS), not every physics step.
        # This is the dominant render cost at 1000+ envs; the policy still runs at 50 Hz and reuses the
        # latest depth frame between camera updates, as it would on the robot.
        update_period=1.0 / _DEPTH_CAMERA_FPS,
        offset=TiledCameraCfg.OffsetCfg(pos=_CAMERA_MOUNT_POS, rot=_CAMERA_MOUNT_ROT, convention="opengl"),
        # distance_to_image_plane = z-depth (perpendicular distance to the image plane), which is what the
        # real D435 reports. NOT distance_to_camera (Euclidean ray length); training on ray length would not
        # match the hardware depth stream. The DepthImage observation term consumes this key.
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_D435F_FOCAL_LENGTH_MM,
            focus_distance=400.0,
            horizontal_aperture=_D435F_HORIZONTAL_APERTURE_MM,
            clipping_range=(0.28, 3.0),
        ),
        width=_DEPTH_RENDER_WIDTH,
        height=_DEPTH_RENDER_HEIGHT,
    )

    # Privileged terrain sensor for the asymmetric critic ONLY -- never an actor/deploy input. A downward
    # ray grid under the base gives the exact local heightmap, which makes value learning on stairs far
    # faster and more stable than forcing the critic to infer terrain from the same depth image the actor
    # sees. Thrown away at deploy time (the critic isn't exported), so using ground-truth here is free.
    # 1.6 x 1.0 m at 0.1 m resolution = 17 x 11 = 187 points; mdp.height_scan reads this sensor.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Visual-only D435 mesh, purely so the camera's mount position is visible in the viewport/Stage tree --
    # no collision, not read by any sensor or reward. Converted this session from the Intel RealSense mesh
    # bundled with Isaac Sim's own Kaya-robot URDF sample (that sample robot's documented camera is a
    # D435), via omni.kit.asset_converter (Collada -> USD), since Isaac Sim's own asset library only ships
    # a D455 mesh, not a D435 (confirmed directly against this install's Nucleus asset root). Reads
    # pos/rot from the SAME _CAMERA_MOUNT_POS/_CAMERA_MOUNT_ROT constants depth_camera uses above -- this
    # used to be an independently hardcoded tuple that drifted out of sync whenever only depth_camera's
    # was edited (confirmed directly as the cause of "the mesh doesn't move when the camera's rotation
    # changes"); now structurally impossible for the two to disagree.
    #
    # scale=0.001, NOT the original Kaya URDF's own 0.1 -- measured directly (UsdGeom.BBoxCache) after
    # spawning at 0.1: world bbox came out ~9m on its longest axis, 100x too big for a real D435's ~90mm
    # housing. The asset_converter evidently doesn't interpret this file's internal units the same way
    # URDF's own mesh loader did, so Kaya's scale factor doesn't transfer. 0.001 brings the measured
    # longest axis down to ~90mm, matching the real D435. rot applies the mesh-asset-specific correction
    # (see _REALSENSE_MESH_ASSET_ROT_CORRECTION above) in the mesh's own local frame FIRST, then the
    # shared mount pose on top -- so the asset quirk and the tunable mount angles stay independent.
    realsense_mesh = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Head_upper/RealSenseMesh",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "resources", "realsense_d435", "realsense_d435.usd"),
            scale=(0.001, 0.001, 0.001),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_CAMERA_MOUNT_POS,
            rot=_quat_mul(_CAMERA_MOUNT_ROT, _REALSENSE_MESH_ASSET_ROT_CORRECTION),
        ),
    )


##
# Observations
##


@configclass
class StairsObservationsCfg(ObservationsCfg):
    """Inherits the proprioceptive `policy` group unchanged and adds two groups for the vision policy:

    - `depth`: the frame-stacked depth image the actor sees, kept in its own group so its 4D (N, K, H, W)
      shape survives into the policy (a group with one image term is returned as-is, not flattened).
    - `critic`: privileged, value-function-only observations -- the same proprio vector as `policy`, in the
      same term order (so a warm-started critic's first 50 input columns line up), plus a ground-truth
      height scan. Never exported/deployed.

    The vision agent config wires these up as obs_groups = {"policy": ["policy", "depth"], "critic": ["critic"]}.
    """

    @configclass
    class DepthCfg(ObsGroup):
        """Frame-stacked depth image, shape (num_envs, num_frames, H, W)."""

        image = ObsTerm(
            func=mdp.DepthImage,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "near": _DEPTH_NEAR_M,
                "far": _DEPTH_FAR_M,
                "num_frames": _DEPTH_NUM_FRAMES,
                "frame_stride_steps": _DEPTH_FRAME_STRIDE_STEPS,
                "augment": True,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            # DepthImage applies its own training-time augmentation (gated by its `augment` param); no
            # group-level additive noise on top.
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations: clean proprio (no sensor noise -- the critic is never deployed)
        in the SAME order as the policy group, then the ground-truth height scan (187 points)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-50.0, 50.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-50.0, 50.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-50.0, 50.0))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    depth: DepthCfg = DepthCfg()
    critic: CriticCfg = CriticCfg()


##
# Environment configuration
##


@configclass
class QuadrupedLocomotionStairsEnvCfg(QuadrupedLocomotionEnvCfg):
    """Stairs training config: same MDP (rewards/actions/commands/events) as the flat-ground task,
    stairs-capable terrain, depth camera in the scene, and the terrain-level curriculum re-enabled
    (the base class's __post_init__ disables it; stairs need per-row difficulty scaling to be useful)."""

    scene: QuadrupedLocomotionStairsSceneCfg = QuadrupedLocomotionStairsSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: StairsObservationsCfg = StairsObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # re-enable terrain-level curriculum (base __post_init__ sets this to None) -- mdp.terrain_levels_vel
        # is terrain-type-agnostic, so no changes to it were needed, only re-enabling it here.
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class QuadrupedLocomotionStairsEnvCfg_PLAY(QuadrupedLocomotionStairsEnvCfg):
    """Play/eval config, mirroring QuadrupedLocomotionEnvCfg_PLAY's overrides exactly (duplicated, not
    inherited, since that class is a sibling built on the flat-ground scene, not an ancestor of this one)."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # clean depth for play/eval too: no sensor augmentation (the critic group is already noise-free).
        self.observations.depth.image.params["augment"] = False
        # nominal physics for play and for sim2sim comparisons: no pushes, no action latency, no startup
        # randomisation (armature stays at its nominal JOINT_ARMATURE); friction back to the single
        # nominal 0.8 / 0.6 pair.
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.robot_mass = None
        self.events.actuator_gains = None
        self.events.joint_armature = None
        self.events.joint_friction = None
        self.events.push_robot = None
        self.actions.joint_pos.max_delay_steps = 0

        # terrain curriculum off for play too, same as the flat-ground task's play config (fixed terrain
        # rows during evaluation, no dynamic reassignment).
        self.curriculum.terrain_levels = None

        # set the view to be closer, same as the flat-ground task's play config
        self.viewer = ViewerCfg(
            eye=(2.0, 2.0, 1.0),
            origin_type="asset_root",
            env_index=0,
            asset_name="robot",
        )
