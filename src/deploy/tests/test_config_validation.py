# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-profile validation (deploy_real._validate_raw_config): bad profiles fail at load, naming the problem."""

import copy
import os
import sys
import unittest

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import deploy_real as dr
except Exception as e:  # pragma: no cover - environment without the deploy deps
    dr = None
    _ERR = e

HERE = os.path.dirname(__file__)
PROFILES = os.path.join(HERE, "..", "profiles")


def _good():
    with open(os.path.join(PROFILES, "go2_obs_stacking.yaml")) as f:
        return yaml.safe_load(f)


@unittest.skipIf(dr is None, "deploy_real not importable here")
class ValidateRawConfig(unittest.TestCase):
    def check(self, raw, **kw):
        dr._validate_raw_config(raw, "p.yaml", need_policy_path=kw.get("need_policy_path", True))

    def test_every_shipped_profile_is_valid(self):
        for name in os.listdir(PROFILES):
            if name.endswith(".yaml"):
                with open(os.path.join(PROFILES, name)) as f:
                    self.check(yaml.safe_load(f))

    def test_not_a_mapping(self):
        with self.assertRaisesRegex(ValueError, "must be a YAML mapping.*list"):
            self.check([1, 2])
        with self.assertRaisesRegex(ValueError, "empty file"):
            self.check(None)

    def test_missing_keys_are_listed(self):
        raw = _good()
        del raw["kp"], raw["effort_limit"]
        with self.assertRaisesRegex(ValueError, r"missing required key\(s\) \['kp', 'effort_limit'\]"):
            self.check(raw)

    def test_policy_path_only_needed_without_cli_override(self):
        raw = _good()
        del raw["policy_path"]
        with self.assertRaisesRegex(ValueError, "policy_path"):
            self.check(raw)
        self.check(raw, need_policy_path=False)

    def test_non_positive_values(self):
        for key, val in (("control_dt", 0), ("action_scale", -1), ("effort_limit", 0), ("kp", -1.0), ("kd", -0.1)):
            raw = _good()
            raw[key] = val
            with self.assertRaisesRegex(ValueError, key):
                self.check(raw)

    def test_zero_gains_are_allowed(self):
        raw = _good()
        raw["kp"] = 0.0
        raw["kd"] = 0.0
        self.check(raw)

    def test_joint_order_length_and_duplicates(self):
        raw = _good()
        raw["isaac_joint_order"] = raw["isaac_joint_order"][:-1]
        with self.assertRaisesRegex(ValueError, "isaac_joint_order must list 12 distinct joints, got 11"):
            self.check(raw)
        raw = _good()
        raw["sdk_joint_order"][0] = raw["sdk_joint_order"][1]
        with self.assertRaisesRegex(ValueError, "sdk_joint_order must list 12 distinct joints"):
            self.check(raw)

    def test_orders_must_name_the_same_joints(self):
        raw = _good()
        raw["sdk_joint_order"][0] = "FL_banana_joint"
        with self.assertRaisesRegex(ValueError, "different joints.*banana"):
            self.check(raw)

    def test_default_pose_must_cover_the_joints(self):
        raw = _good()
        raw["default_joint_pos"] = copy.deepcopy(raw["default_joint_pos"])
        raw["default_joint_pos"].pop("FL_hip_joint")
        with self.assertRaisesRegex(ValueError, "default_joint_pos and isaac_joint_order"):
            self.check(raw)


if __name__ == "__main__":
    unittest.main()
