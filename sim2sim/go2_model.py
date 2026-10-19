# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Builds the MuJoCo Go2 + static OpenManipulator-X model used for sim2sim.

Starts from MuJoCo Menagerie's `unitree_go2/go2.xml` (fetched by `sim2sim/fetch_go2_model.sh`) and edits it with
`mujoco.MjSpec` instead of a copied XML file, so the upstream model stays untouched and the edits are in one place:

- Adds the arm as rigid child bodies of `base`, mounted on the head where the trained asset
  (`go2withOpenXstatic.usd`) mounts it, see ARM_MOUNT_POS. Masses, centers of mass and inertias come from
  `resources/openManipulator/configuration/open_manipulator_x_physics.usda`; link offsets from that file's joint
  `localPos0` values at the zero pose. In Isaac the arm joints are held at zero by stiff USD drives, so a rigid arm
  is a close approximation. Collision shapes are simple capsules sized to the link offsets (only used for the
  fall check and ground contact, not for self-collision).
- Matches the training actuator setup (`unitree_go2witharm_cfg.py`, DCMotorCfg with friction=0.0 and no armature
  override): joint damping and frictionloss are zeroed, since Isaac's explicit actuator does all PD in software.
  Menagerie's armature (0.01) is kept; the Isaac USD value is not known (`armature: null` in params/env.yaml).
- Adds a floor with friction 1.0 (training terrain material: static/dynamic friction 1.0).
"""

from __future__ import annotations

import os

import mujoco

MENAGERIE_GO2_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mujoco", "menagerie", "unitree_go2", "go2.xml"
)

# Arm "world" body position in the Go2 base frame, read from the asset the committed models were trained on
# (go2withOpenXstatic.usd: open_manipulator_x_static translate, and FixedJoint localPos0 = -this). This is on the
# head, NOT the (0, 0, 0.109) that scripts/compose_go2_with_arm.py produces; that script did not build the
# trained asset.
ARM_MOUNT_POS = (0.2191266564592013, 0.0, 0.10596202282924586)

# name: (position in arm frame at zero pose, mass, com, diag inertia, principal axes quat wxyz)
ARM_LINKS = {
    "world": ((0.0, 0.0, 0.0), 0.01, (0.0, 0.0, 0.0), (1e-4, 1e-4, 1e-4), (1.0, 0.0, 0.0, 0.0)),
    "link1": (
        (0.0, 0.0, 0.0), 0.079119965, (0.00030876155, 0.0, -0.00012176461),
        (0.000012500524, 0.000021898364, 0.000019272073), (0.99991304, 0.0, -0.013189722, 0.0),
    ),
    "link2": (
        (0.012, 0.0, 0.0), 0.09840684, (-0.0003018487, 0.0005404368, 0.047433466),
        (0.000034552955, 0.000032689233, 0.000018840883), (0.9999148, 0.00096008566, 0.012220063, -0.0044987164),
    ),
    "link3": (
        (0.012, 0.0, 0.0595), 0.13850917, (0.010308393, 0.00037743364, 0.10170197),
        (0.0003359318, 0.00034291513, 0.000054957833), (0.9975529, -0.0033182802, 0.06932859, -0.008419083),
    ),
    "link4": (
        (0.036, 0.0, 0.1875), 0.13274562, (0.09090959, 0.00038929816, 0.00022413279),
        (0.00003064615, 0.00024231059, 0.00025155093), (0.99999505, -0.00071134296, -0.0006102599, 0.0030147866),
    ),
    "link5": (
        (0.160, 0.0, 0.1875), 0.14327572, (0.044206753, 3.6839984e-7, 0.008914222),
        (0.00008078715, 0.00007598047, 0.000093210976), (0.99915695, 0.0, -0.04105357, 0.0),
    ),
    "gripper_left_link": ((0.2417, 0.021, 0.1875), 0.001, (0.0, 0.0, 0.0), (1e-6, 1e-6, 1e-6), (1.0, 0.0, 0.0, 0.0)),
    "gripper_right_link": ((0.2417, -0.021, 0.1875), 0.001, (0.0, 0.0, 0.0), (1e-6, 1e-6, 1e-6), (1.0, 0.0, 0.0, 0.0)),
}

# Collision capsules in the arm frame: (from, to, radius). Approximate link envelopes.
ARM_COLLIDERS = [
    ((0.0, 0.0, 0.0), (0.012, 0.0, 0.0), 0.02),  # link1 base
    ((0.012, 0.0, 0.0), (0.012, 0.0, 0.0595), 0.015),  # link2
    ((0.012, 0.0, 0.0595), (0.036, 0.0, 0.1875), 0.015),  # link3
    ((0.036, 0.0, 0.1875), (0.160, 0.0, 0.1875), 0.015),  # link4
    ((0.160, 0.0, 0.1875), (0.26, 0.0, 0.1875), 0.02),  # link5 + gripper
]

# Collision bits: robot geoms keep Menagerie's contype=conaffinity=1; arm geoms use bit 2 so they hit the floor but
# never the Go2's own geoms (matching the FilteredPairsAPI in compose_go2_with_arm.py).
ARM_CONTYPE = 2
FLOOR_CONAFFINITY = 1 | ARM_CONTYPE

FLOOR_FRICTION = 1.0
TIMESTEP = 0.005  # sim.dt in quadruped_go2_locomotion_env_cfg.py


def build_model(menagerie_xml: str = MENAGERIE_GO2_XML) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    if not os.path.isfile(menagerie_xml):
        raise FileNotFoundError(f"{menagerie_xml} not found. Run sim2sim/fetch_go2_model.sh first.")
    spec = mujoco.MjSpec.from_file(menagerie_xml)
    spec.option.timestep = TIMESTEP

    world = spec.worldbody
    world.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        friction=[FLOOR_FRICTION, 0.005, 0.0001],
        contype=1,
        conaffinity=FLOOR_CONAFFINITY,
        rgba=[0.25, 0.3, 0.35, 1],
    )

    arm = spec.body("base").add_body(name="arm_mount", pos=list(ARM_MOUNT_POS))
    for name, (pos, mass, com, inertia, axes) in ARM_LINKS.items():
        link = arm.add_body(name=f"arm_{name}", pos=list(pos))
        link.explicitinertial = True
        link.mass = mass
        link.ipos = list(com)
        link.inertia = list(inertia)
        link.iquat = list(axes)
    for i, (p0, p1, radius) in enumerate(ARM_COLLIDERS):
        arm.add_geom(
            name=f"arm_collider_{i}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*p0, *p1],
            size=[radius, 0, 0],
            contype=ARM_CONTYPE,
            conaffinity=0,
            density=0.0,
            rgba=[0.8, 0.5, 0.2, 1],
        )

    model = spec.compile()
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    return model, spec
