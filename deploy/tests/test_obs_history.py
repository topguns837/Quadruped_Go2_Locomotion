# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation stacking + height-command support in deploy_real.

Needs the deploy_real import chain with a REAL torch (it builds tensors), so it is skipped elsewhere. Run in the
container, e.g. in a throwaway no-network one:
    /workspace/isaaclab/_isaac_sim/python.sh -m unittest deploy.tests.test_obs_history
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import torch

    import deploy_real as dr

    _ERR = None
except Exception as e:  # noqa: BLE001 -- torch / SDK not available on this interpreter
    torch, dr, _ERR = None, None, e

REAL_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "go2_locomotion.yaml")
SKIP = unittest.skipIf(dr is None or not hasattr(torch, "nn"), f"deploy_real/torch not usable here: {_ERR}")


def fake_policy(in_dim: int):
    return torch.nn.Sequential(torch.nn.Linear(in_dim, 8), torch.nn.ELU(), torch.nn.Linear(8, 12))


def frame(width: int, base: float) -> torch.Tensor:
    """(1, width) frame whose entries are base + 0.001*index, so every term/frame stays distinguishable."""
    return (base + 0.001 * torch.arange(width, dtype=torch.float32)).unsqueeze(0)


@SKIP
class Layouts(unittest.TestCase):
    def test_48_is_the_height_command_layout(self):
        self.assertEqual(dr.OBS_LAYOUT[48], {"cmd_width": 6, "has_height_obs": False, "has_lin_vel": False})

    def test_existing_layouts_unchanged(self):
        self.assertEqual(dr.OBS_LAYOUT[47], {"cmd_width": 5, "has_height_obs": False, "has_lin_vel": False})
        self.assertEqual(dr.OBS_LAYOUT[50]["has_lin_vel"], True)
        self.assertEqual(dr.OBS_LAYOUT[51]["cmd_width"], 6)
        self.assertEqual(dr.OBS_LAYOUT[52]["has_height_obs"], True)

    def test_term_widths_48(self):
        self.assertEqual(dr.frame_term_widths(48, dr.OBS_LAYOUT[48]), [3, 3, 6, 12, 12, 12])

    def test_term_widths_47_unchanged(self):
        self.assertEqual(dr.frame_term_widths(47, dr.OBS_LAYOUT[47]), [3, 3, 5, 12, 12, 12])


@SKIP
class PolicyObsLayout(unittest.TestCase):
    def test_current_single_frame_policy_is_unaffected(self):
        self.assertEqual(dr.policy_obs_layout(fake_policy(47), 1), (47, dr.OBS_LAYOUT[47]))

    def test_default_history_is_one(self):
        self.assertEqual(dr.policy_obs_layout(fake_policy(47)), (47, dr.OBS_LAYOUT[47]))

    def test_stacked_height_policy(self):
        self.assertEqual(dr.policy_obs_layout(fake_policy(288), 6), (48, dr.OBS_LAYOUT[48]))

    def test_history_mismatch_is_an_error_not_a_scramble(self):
        with self.assertRaises(ValueError):
            dr.policy_obs_layout(fake_policy(288), 5)  # 288 % 5 != 0
        with self.assertRaises(ValueError):
            dr.policy_obs_layout(fake_policy(288), 4)  # 72 per frame: divides, but not a known layout
        with self.assertRaises(ValueError):
            dr.policy_obs_layout(fake_policy(282), 1)  # a stacked policy deployed with history 1

    def test_zero_history_rejected(self):
        with self.assertRaises(ValueError):
            dr.policy_obs_layout(fake_policy(47), 0)


@SKIP
class ObsHistory(unittest.TestCase):
    def test_history_one_is_an_exact_identity(self):
        h = dr.ObsHistory(1)
        for i in range(5):
            f = frame(47, float(i))
            self.assertTrue(torch.equal(h.push_and_stack(f), f))

    def test_history_one_does_not_need_term_widths(self):
        self.assertEqual(dr.ObsHistory(1).push_and_stack(frame(47, 0.0)).shape, (1, 47))

    def test_startup_pads_with_the_first_frame(self):
        widths = [2, 3]
        h = dr.ObsHistory(3, widths)
        f0 = frame(5, 1.0)
        out = h.push_and_stack(f0)
        self.assertEqual(out.shape, (1, 15))
        self.assertTrue(torch.equal(out, torch.cat([f0[:, :2]] * 3 + [f0[:, 2:]] * 3, dim=1)))

    def test_term_major_newest_last(self):
        widths = [2, 3]
        h = dr.ObsHistory(3, widths)
        frames = [frame(5, float(i)) for i in range(4)]
        for f in frames:
            out = h.push_and_stack(f)
        newest3 = frames[1:]  # oldest -> newest
        expected = torch.cat([f[:, :2] for f in newest3] + [f[:, 2:] for f in newest3], dim=1)
        self.assertTrue(torch.equal(out, expected))
        # the newest frame's last term sample is the LAST element of the vector
        self.assertTrue(torch.equal(out[:, -3:], frames[3][:, 2:]))

    def test_window_slides_and_drops_the_oldest(self):
        h = dr.ObsHistory(2, [1])
        h.push_and_stack(frame(1, 1.0))
        h.push_and_stack(frame(1, 2.0))
        out = h.push_and_stack(frame(1, 3.0))
        self.assertTrue(torch.equal(out, torch.tensor([[2.0, 3.0]]) + torch.tensor([[0.0, 0.0]])))

    def test_reset_repads_from_the_next_frame(self):
        h = dr.ObsHistory(2, [1])
        h.push_and_stack(frame(1, 1.0))
        h.push_and_stack(frame(1, 2.0))
        h.reset()
        out = h.push_and_stack(frame(1, 9.0))
        self.assertTrue(torch.equal(out, torch.tensor([[9.0, 9.0]])))

    def test_full_48_dim_stack_shape(self):
        h = dr.ObsHistory(6, dr.frame_term_widths(48, dr.OBS_LAYOUT[48]))
        self.assertEqual(h.push_and_stack(frame(48, 0.0)).shape, (1, 288))

    def test_term_major_without_widths_is_an_error(self):
        with self.assertRaises(ValueError):
            dr.ObsHistory(3).push_and_stack(frame(5, 0.0))


@SKIP
class HeightCommand(unittest.TestCase):
    def setUp(self):
        self.L47 = dr.OBS_LAYOUT[47]
        self.L48 = dr.OBS_LAYOUT[48]

    def test_five_wide_layout_gets_five_values(self):
        self.assertEqual(dr.command_for_layout([0.1, 0, 0, 0.2, 0.3], self.L47, 0.31), [0.1, 0, 0, 0.2, 0.3])

    def test_five_wide_layout_drops_a_height_it_does_not_take(self):
        self.assertEqual(len(dr.command_for_layout([0, 0, 0, 0, 0, 0.30], self.L47)), 5)

    def test_six_wide_layout_appends_the_height(self):
        self.assertEqual(dr.command_for_layout([0.1, 0, 0, 0.2, 0.3], self.L48, 0.31), [0.1, 0, 0, 0.2, 0.3, 0.31])

    def test_six_value_command_carries_its_own_height(self):
        self.assertEqual(dr.command_for_layout([0, 0, 0, 0, 0, 0.30], self.L48, 0.35)[5], 0.30)

    def test_height_is_clamped_to_the_trained_range(self):
        lo, hi = dr.HEIGHT_COMMAND_RANGE_M
        self.assertEqual(dr.command_for_layout([0, 0, 0, 0, 0, 0.10], self.L48)[5], lo)
        self.assertEqual(dr.command_for_layout([0, 0, 0, 0, 0, 0.90], self.L48)[5], hi)

    def test_default_height_is_the_nominal_stance(self):
        self.assertAlmostEqual(dr.command_for_layout([0, 0, 0, 0, 0], self.L48)[5], dr.DEFAULT_HEIGHT_M)
        self.assertAlmostEqual(dr.DEFAULT_HEIGHT_M, 0.337)

    def test_bad_length_rejected(self):
        with self.assertRaises(ValueError):
            dr.command_for_layout([0, 0, 0], self.L48)

    def test_trained_range_matches_the_env_cfg(self):
        # the deploy clamp must equal the training range, or the clamp silently changes what the policy was taught
        cfg = os.path.join(os.path.dirname(__file__), "..", "..", "source", "quadruped_go2_locomotion",
                           "quadruped_go2_locomotion", "tasks", "manager_based", "quadruped_go2_locomotion",
                           "quadruped_go2_locomotion_env_cfg.py")
        with open(cfg) as f:
            src = f.read()
        import re

        m = re.search(r"height=\(([\d.]+), ?([\d.]+)\)", src)
        self.assertIsNotNone(m, "no height=(lo, hi) range in the env cfg")
        self.assertEqual(dr.HEIGHT_COMMAND_RANGE_M, (float(m.group(1)), float(m.group(2))))


@SKIP
class ManualBuffer(unittest.TestCase):
    def test_get_stays_five_wide(self):
        b = dr.ManualCommandBuffer()
        self.assertEqual(b.get(), [0.0] * 5)
        b.set([1, 2, 3, 4, 5])
        self.assertEqual(b.get(), [1, 2, 3, 4, 5])

    def test_height_defaults_and_updates(self):
        b = dr.ManualCommandBuffer()
        self.assertAlmostEqual(b.get_height(), dr.DEFAULT_HEIGHT_M)
        b.set([0] * 5, height=0.30)
        self.assertEqual(b.get_height(), 0.30)

    def test_set_without_height_keeps_the_last_height(self):
        b = dr.ManualCommandBuffer()
        b.set([0] * 5, height=0.30)
        b.set([1, 0, 0, 0, 0])
        self.assertEqual(b.get_height(), 0.30)

    def test_parse_command_json_with_and_without_height(self):
        base = {"lin_vel_x": 1, "lin_vel_y": 2, "ang_vel_z": 3, "pitch": 4, "lean": 5}
        self.assertEqual(dr.parse_command_json(base), ([1, 2, 3, 4, 5], None))  # old slider: no height key
        self.assertEqual(dr.parse_command_json({**base, "height": 0.3}), ([1, 2, 3, 4, 5], 0.3))
        with self.assertRaises(KeyError):
            dr.parse_command_json({"lin_vel_x": 1})


@SKIP
class DeployConfigDefaults(unittest.TestCase):
    def test_shipped_config_keeps_single_frame(self):
        cfg = dr.load_config(REAL_CONFIG, "lo")
        self.assertEqual(cfg.history_length, 1)

    def test_history_length_is_read_from_yaml(self):
        import tempfile

        import yaml

        with open(REAL_CONFIG) as f:
            raw = yaml.safe_load(f)
        raw["history_length"] = 6
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(raw, f)
        self.assertEqual(dr.load_config(f.name, "lo").history_length, 6)

    def test_cli_history_length_overrides_the_config(self):
        self.assertEqual(dr.load_config(REAL_CONFIG, "lo", None, None, None, 6).history_length, 6)
        self.assertEqual(dr.load_config(REAL_CONFIG, "lo", None, None, None, None).history_length, 1)

    def test_invalid_cli_history_length_rejected(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                dr.load_config(REAL_CONFIG, "lo", None, None, None, bad)

    def test_profile_can_set_history_length_and_cli_beats_it(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("p:\n  policy_path: models/x/policy.pt\n  history_length: 6\n")
        saved = sys.argv
        try:
            sys.argv = ["deploy_real.py", "--test_profile", "p", "--test_profiles_file", f.name]
            self.assertEqual(dr.parse_args().history_length, 6)
            sys.argv += ["--history_length", "1"]
            self.assertEqual(dr.parse_args().history_length, 1)
            sys.argv = ["deploy_real.py"]
            self.assertIsNone(dr.parse_args().history_length)  # unset -> config decides (default 1)
        finally:
            sys.argv = saved


if __name__ == "__main__":
    unittest.main()
