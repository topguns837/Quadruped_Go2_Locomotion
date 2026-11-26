# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""--deploy_profile (per-policy recipe incl. observation stack depth) and --safety_profile (session caution).

Run in the container:  /workspace/isaaclab/_isaac_sim/python.sh -m unittest deploy.tests.test_deploy_profile
"""

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import deploy_real as dr

    _ERR = None
except Exception as e:  # noqa: BLE001
    dr, _ERR = None, e

SKIP = unittest.skipIf(dr is None, f"deploy_real not importable here: {_ERR}")


def _parse(*argv):
    with mock.patch.object(sys, "argv", ["deploy_real.py", *argv]), contextlib.redirect_stdout(io.StringIO()) as out:
        return dr.parse_args(), out.getvalue()


@SKIP
class DeployProfileTests(unittest.TestCase):
    def test_obs_stacking_profile_sets_history_6_and_coupled_gains(self):
        args, _ = _parse("--deploy_profile", "go2_obs_stacking")
        self.assertTrue(args.config.endswith(os.path.join("profiles", "go2_obs_stacking.yaml")))
        cfg = dr.load_config(args.config, "lo")
        self.assertEqual((cfg.history_length, cfg.kp, cfg.kd, cfg.joint_vel_filter_hz), (6, 50.0, 1.0, 10.0))

    def test_locomotion_profile_is_single_frame(self):
        args, _ = _parse("--deploy_profile", "go2_locomotion")
        cfg = dr.load_config(args.config, "lo")
        self.assertEqual((cfg.history_length, cfg.kp, cfg.joint_vel_filter_hz), (1, 40.0, None))

    def test_default_is_locomotion(self):
        args, _ = _parse()
        self.assertEqual(os.path.abspath(args.config), os.path.abspath(dr.DEFAULT_PROFILE))

    def test_unknown_profile_errors(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parse("--deploy_profile", "nope")

    def test_config_and_deploy_profile_are_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parse("--deploy_profile", "go2_obs_stacking", "--config", dr.DEFAULT_PROFILE)

    def test_explicit_history_override_wins(self):
        args, _ = _parse("--deploy_profile", "go2_obs_stacking", "--history_length", "1")
        self.assertEqual(dr.load_config(args.config, "lo", cli_history_length=args.history_length).history_length, 1)

    def test_safety_profile_applies_limits_and_is_orthogonal(self):
        args, _ = _parse("--deploy_profile", "go2_obs_stacking", "--safety_profile", "cautionary")
        self.assertEqual(args.safety_profile, "cautionary")
        self.assertEqual(args.max_target_dev_deg, 15.0)
        self.assertTrue(args.config.endswith("go2_obs_stacking.yaml"))

    def test_test_profile_alias_works_with_warning(self):
        args, out = _parse("--test_profile", "confident")
        self.assertEqual(args.safety_profile, "confident")
        self.assertIn("deprecated", out)

    def test_safety_profile_cannot_select_the_policy(self):
        valid = {"deploy_profile", "config", "safety_profile"}
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("bad:\n  deploy_profile: go2_obs_stacking\n")
        with self.assertRaises(ValueError):
            dr.load_safety_profile(f.name, "bad", valid)
        os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
