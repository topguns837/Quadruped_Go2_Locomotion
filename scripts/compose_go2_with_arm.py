# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# One-off script: composes the standard Unitree Go2 USD (fetched from Nucleus)
# with the bundled OpenManipulator-X static asset, mounted on top of the Go2's
# main body ("base"), and saves the result to
# resources/go2withArm/go2withOpenXStatic.usda.
#
# Run once via: isaaclab -p scripts/compose_go2_with_arm.py
"""Compose the Go2 + OpenManipulator-X USD asset."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

GO2_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
_RESOURCES_DIR = (
    f"{os.path.dirname(os.path.abspath(__file__))}/../source/quadruped_go2_locomotion/"
    "quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/resources"
)
OUTPUT_PATH = f"{_RESOURCES_DIR}/go2withArm/go2withOpenXStatic.usda"
# Reference the manipulator asset relative to the output file's own directory
# (both live under resources/), not by its absolute container mount path, so
# the composed asset stays valid if the repo is mounted somewhere else.
MANIPULATOR_USDA_RELATIVE = "../openManipulator/open_manipulator_x_static.usda"

# Go2's main body ("base") sits at identity relative to go2_description
# (queried directly from go2.usd: xformOp:translate on /go2_description/base
# is (0, 0, 0)), and its bounding box top surface is at roughly z=0.089
# (also queried directly: ComputeWorldBound on /go2_description/base). The
# arm is placed as a sibling of base (see the placement note below for why),
# so its own translate is given in that same go2_description-local frame:
# centered over the body, just above that top surface.
BASE_TRANSLATE = Gf.Vec3d(0.0, 0.0, 0.089)
MOUNT_TRANSLATE = BASE_TRANSLATE + Gf.Vec3d(0.0, 0.0, 0.02)
MOUNT_ORIENT = Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))  # identity


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    stage = Usd.Stage.CreateNew(OUTPUT_PATH)

    # Root prim: references the Go2's own default prim wholesale.
    root_prim = stage.DefinePrim("/go2_description", "Xform")
    root_prim.GetReferences().AddReference(GO2_USD_PATH)
    stage.SetDefaultPrim(root_prim)

    # Placement: a SIBLING of base, not a descendant of it. Two independent
    # problems both required this:
    #   1. PhysX rejects a RigidBodyAPI subtree nested inside another enabled
    #      rigid body's Xform hierarchy (base is itself an articulation link)
    #      unless that subtree resets its xform stack (observed directly:
    #      every arm link logged "missing xformstack reset ... in
    #      hierarchy", and the arm's own internal joints then failed with "no
    #      bodies defined at body0 and body1" even though their USD
    #      relationship targets resolved fine).
    #   2. Even after fixing (1) with a stack reset, Isaac Lab's own
    #      `activate_contact_sensors` (isaaclab/sim/schemas/schemas.py) walks
    #      the spawned prim tree but deliberately never recurses past a prim
    #      that already has RigidBodyAPI, on the assumption (per the PhysX
    #      SDK) that nested rigid bodies never occur. Since base has
    #      RigidBodyAPI, anything nested under it in USD is structurally
    #      unreachable by that walk regardless of (1), so contact_forces_arm
    #      would still find zero bodies.
    # go2_description itself (unlike base) carries no RigidBodyAPI, so a
    # sibling placement avoids both problems at once, and needs no xform
    # stack reset: MOUNT_TRANSLATE above is just this asset's ordinary local
    # transform under go2_description, same as base's own.
    arm_path = "/go2_description/OpenManipulatorX"
    arm_prim = stage.DefinePrim(arm_path, "Xform")
    arm_prim.GetReferences().AddReference(MANIPULATOR_USDA_RELATIVE)

    xformable = UsdGeom.Xformable(arm_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(MOUNT_TRANSLATE)
    # The manipulator asset's own composed xform ops already declare
    # xformOp:orient as quatd (double precision); AddOrientOp()'s default
    # (float) collides with that existing typeName, so match it explicitly.
    xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(MOUNT_ORIENT)

    # Deactivate the manipulator's own "fixed to global world frame" joint:
    # body0=[] (global frame), body1=world. Mounting it on a moving robot
    # needs a joint to the Go2's head instead, not the simulation's global frame.
    root_joint_override = stage.OverridePrim(f"{arm_path}/root_joint")
    root_joint_override.SetActive(False)

    # New fixed joint: Go2 base <-> manipulator's "world" anchor body. A
    # USD/PhysX joint only needs body0Rel/body1Rel (plus local offsets); it
    # doesn't require the two bodies to be USD Xform parent/child, so this
    # cross-body constraint works the same whether or not they're siblings.
    mount_joint = UsdPhysics.FixedJoint.Define(stage, "/go2_description/ArmMountJoint")
    mount_joint.CreateBody0Rel().SetTargets([Sdf.Path("/go2_description/base")])
    mount_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{arm_path}/world")])
    mount_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.02))
    mount_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Exclude collision between the arm and base. A FixedJoint (unlike an
    # articulation link, which gets this automatically for adjacent links)
    # does NOT disable collision between the bodies it connects, and base's
    # torso collision volume is large enough to overlap with the arm's mount
    # point sitting right on top of it. Observed directly: without this,
    # env creation didn't crash or error, it just pegged one CPU core at
    # ~150% indefinitely (5+ minutes, never progressing) inside PhysX's
    # first-frame contact/cooking step, consistent with a contact-generation
    # blowup from two overlapping rigid bodies rather than an actual crash.
    filtered = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/go2_description/base"))
    filtered.CreateFilteredPairsRel().SetTargets([
        Sdf.Path(f"{arm_path}/world"),
        Sdf.Path(f"{arm_path}/link1"),
        Sdf.Path(f"{arm_path}/link2"),
        Sdf.Path(f"{arm_path}/link3"),
        Sdf.Path(f"{arm_path}/link4"),
        Sdf.Path(f"{arm_path}/link5"),
        Sdf.Path(f"{arm_path}/gripper_left_link"),
        Sdf.Path(f"{arm_path}/gripper_right_link"),
    ])

    stage.Save()
    print(f"[INFO] Saved composed asset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()
