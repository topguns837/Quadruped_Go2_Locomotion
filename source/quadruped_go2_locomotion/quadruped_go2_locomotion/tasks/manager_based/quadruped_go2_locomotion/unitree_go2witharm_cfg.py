# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom Unitree Go2 with Open Manipulator X configuration for this project.

Bundles the Go2 + Open Manipulator X USD and actuator model into the package
so it can be used without depending on isaaclab_assets.robots.unitree.
"""

from __future__ import annotations

import pathlib

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Resolve path to the resources/ directory alongside this file
_RESOURCES_DIR = pathlib.Path(__file__).resolve().parent / "resources"


UNITREE_GO2WITHARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"file://{_RESOURCES_DIR / 'go2withArm' / 'go2withOpenXstatic.usd'}",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Two groups, not one, because the Go2's calf is NOT the same motor as its hip and thigh: it sits
        # behind a 1.9169:1 knee reduction. Unitree's own go2_description.urdf gives hip/thigh as
        # effort=23.7 Nm velocity=30.1 rad/s and calf as effort=45.43 Nm velocity=15.70 rad/s, and this
        # project's own USD carries the same numbers on its joint drives (calf
        # drive:angular:physics:maxForce = 45.43, physxJoint:maxJointVelocity = 899.54 deg/s = 15.70 rad/s).
        #
        # Until now all 12 joints shared one DCMotorCfg at 23.5 Nm / 30.0 rad/s, which discarded those USD
        # values and trained the calf at 0.52x its real torque with a torque-speed curve rolling off at 1.9x
        # the real speed. Isaac Lab's own stock Go2 config had the identical bug; see isaac-sim/IsaacLab
        # PR #7564 "Fix Unitree Go1 and Go2 calf actuator limits ignoring the knee reduction".
        #
        # The split is needed (rather than one group with per-joint dicts) because DCMotorCfg.saturation_effort
        # is a scalar `float` in Isaac Lab v2.3.2 -- effort_limit/velocity_limit accept dicts, saturation_effort
        # does not. PR #7564 added dict support upstream; on a newer Isaac Lab these two groups could be merged.
        #
        # Policies trained before this change (models/9_10_26_vanilla, models/9_13_26_*) depend on the old weak
        # calf: they command calf targets far past the joint limit and rely on the 23.5 Nm clamp to turn that
        # into a steady push. Give them the real torque and they destabilise, so they are NOT valid under this
        # config and must be retrained, not resumed. See sim2sim/README.md for the measurements.
        "hip_thigh": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "calves": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=45.43,
            saturation_effort=45.43,
            velocity_limit=15.70,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
