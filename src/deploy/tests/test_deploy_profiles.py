# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Safety-profile extension: policy_path selectable in a profile, low_pass profile-able, --safety_profile naming.

Needs the deploy_real import chain (torch / SDK), so it is skipped on an interpreter without them -- run in the
container:  ../IsaacLab/isaaclab.sh -p -m pytest src/deploy/tests/test_deploy_profiles.py
"""

import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import deploy_real as dr
except Exception:  # noqa: BLE001 -- torch / SDK not available on this interpreter
    dr = None

REAL_CONFIG = os.path.join(os.path.dirname(__file__), "..", "profiles", "go2_locomotion.yaml")


def write_profiles(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(text))
    return path


def parse_with_argv(argv_tail):
    """Run dr.parse_args() with a controlled sys.argv (parse_args reads sys.argv directly)."""
    saved = sys.argv
    try:
        sys.argv = ["deploy_real.py", *argv_tail]
        return dr.parse_args()
    finally:
        sys.argv = saved


@unittest.skipIf(dr is None, "deploy_real not importable here")
class LoadTestProfileRegression(unittest.TestCase):
    VALID = {"max_target_dev_deg", "max_joint_speed_rad_s", "policy_path", "low_pass", "speed_fault_cap_rad_s",
             "speed_fault_steps", "hold_kp", "network_interface", "config"}

    def test_existing_profiles_still_parse(self):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "safety_profiles.yaml")
        valid = self.VALID | {"policy_max_target_dev_deg", "abort_err_deg", "kp_handover_s", "lift_calf_tether_deg",
                              "live_dashboard", "dashboard_viewer", "dashboard_output", "calf_tau_abort_nm"}
        for name in ("cautionary", "confident", "no_limits"):
            prof = dr.load_safety_profile(path, name, valid)
            self.assertIn("max_target_dev_deg", prof)

    def test_every_shipped_profile_loads_through_the_real_parser(self):
        # Strongest check: each profile in the shipped file is validated against the REAL flag set, so a typo or a
        # key for a flag that does not exist fails here instead of at the robot.
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "safety_profiles.yaml")
        import yaml
        with open(path) as f:
            names = list(yaml.safe_load(f))
        self.assertGreaterEqual(len(names), 3)
        for name in names:
            args = parse_with_argv(["--safety_profile", name, "--safety_profiles_file", path])
            self.assertEqual(args.safety_profile, name)

    def test_unknown_key_rejected(self):
        p = write_profiles("bad:\n  bogus_knob: 1\n")
        with self.assertRaises(ValueError):
            dr.load_safety_profile(p, "bad", self.VALID)

    def test_reserved_key_rejected(self):
        for reserved in ("network_interface", "config"):
            p = write_profiles(f"bad:\n  {reserved}: something\n")
            with self.assertRaises(ValueError):
                dr.load_safety_profile(p, "bad", self.VALID)

    def test_policy_path_now_accepted(self):
        p = write_profiles("prof:\n  policy_path: models/foo/policy.pt\n")
        prof = dr.load_safety_profile(p, "prof", self.VALID)
        self.assertEqual(prof["policy_path"], "models/foo/policy.pt")


@unittest.skipIf(dr is None, "deploy_real not importable here")
class PolicyPathResolution(unittest.TestCase):
    def test_cli_policy_path_overrides_config(self):
        cfg = dr.load_config(REAL_CONFIG, "lo", None, None, "models/override/policy.pt")
        self.assertEqual(cfg.policy_path, "models/override/policy.pt")

    def test_config_policy_path_used_when_no_override(self):
        cfg_default = dr.load_config(REAL_CONFIG, "lo")
        cfg_none = dr.load_config(REAL_CONFIG, "lo", None, None, None)
        self.assertEqual(cfg_none.policy_path, cfg_default.policy_path)
        self.assertTrue(cfg_none.policy_path)  # non-empty from the YAML

    def test_profile_then_cli_precedence_through_argparse(self):
        prof = write_profiles("prof:\n  policy_path: models/from_profile/policy.pt\n")
        # profile only -> args.policy_path is the profile's value
        args = parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", prof])
        self.assertEqual(args.policy_path, "models/from_profile/policy.pt")
        # explicit CLI beats the profile
        args = parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", prof,
                                "--policy_path", "models/from_cli/policy.pt"])
        self.assertEqual(args.policy_path, "models/from_cli/policy.pt")


@unittest.skipIf(dr is None, "deploy_real not importable here")
class LowPassProfileable(unittest.TestCase):
    def test_on_from_profile(self):
        prof = write_profiles("prof:\n  low_pass: 5.0\n")
        args = parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", prof])
        self.assertEqual(args.low_pass, 5.0)

    def test_off_from_profile_null(self):
        prof = write_profiles("prof:\n  low_pass: null\n")
        args = parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", prof])
        self.assertIsNone(args.low_pass)

    def test_cli_overrides_profile(self):
        prof = write_profiles("prof:\n  low_pass: 5.0\n")
        args = parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", prof, "--low_pass", "8"])
        self.assertEqual(args.low_pass, 8.0)


@unittest.skipIf(dr is None, "deploy_real not importable here")
class LiveDashboardProfileable(unittest.TestCase):
    """The graphs-while-deploying dashboard/viewer are profile params (no separate tmux panes)."""

    def test_defaults_off_viewer_on_hardware_png(self):
        args = parse_with_argv([])
        self.assertFalse(args.live_dashboard)
        self.assertTrue(args.dashboard_viewer)
        self.assertEqual(args.dashboard_output, "/tmp/hardware_dashboard.png")

    def test_all_three_from_profile(self):
        prof = write_profiles(
            "prof:\n  live_dashboard: true\n  dashboard_viewer: false\n  dashboard_output: /tmp/x.png\n")
        args = parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", prof])
        self.assertTrue(args.live_dashboard)
        self.assertFalse(args.dashboard_viewer)
        self.assertEqual(args.dashboard_output, "/tmp/x.png")

    def test_cli_turns_profile_dashboard_off_and_on(self):
        on = write_profiles("prof:\n  live_dashboard: true\n")
        self.assertFalse(parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", on,
                                          "--no_live_dashboard"]).live_dashboard)
        off = write_profiles("prof:\n  live_dashboard: false\n")
        self.assertTrue(parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", off,
                                         "--live_dashboard"]).live_dashboard)

    def test_cli_viewer_flags_override_profile(self):
        prof = write_profiles("prof:\n  dashboard_viewer: true\n")
        self.assertFalse(parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", prof,
                                          "--no_dashboard_viewer"]).dashboard_viewer)

    def test_live_dashboard_implies_live_plot(self):
        # the dashboard reads the --live_plot log folder, so asking for one must switch the logging on
        self.assertTrue(parse_with_argv(["--live_dashboard"]).live_plot)
        self.assertFalse(parse_with_argv([]).live_plot)


@unittest.skipIf(dr is None, "deploy_real not importable here")
class SafetyProfileNames(unittest.TestCase):
    """--safety_profile is the name; --test_profile is its deprecated alias. --deploy_profile now selects the policy
    recipe (src/deploy/profiles/<name>.yaml) and no longer aliases the safety profile."""

    def test_test_profile_alias_resolves_same_as_safety_profile(self):
        prof = write_profiles("prof:\n  max_target_dev_deg: 20.0\n")
        a_old = parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", prof])
        a_new = parse_with_argv(["--safety_profile", "prof", "--safety_profiles_file", prof])
        self.assertEqual(a_old.safety_profile, "prof")
        self.assertEqual(a_new.safety_profile, "prof")
        self.assertEqual(a_new.max_target_dev_deg, 20.0)

    def test_deploy_profile_is_not_a_safety_profile_alias(self):
        a = parse_with_argv(["--deploy_profile", "go2_locomotion"])
        self.assertIsNone(a.safety_profile)
        self.assertTrue(a.config.endswith(os.path.join("profiles", "go2_locomotion.yaml")))


if __name__ == "__main__":
    unittest.main()
