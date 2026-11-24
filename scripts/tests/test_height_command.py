# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Height command maths (mdp/height_command.py) -- needs torch only, no Isaac Sim / Kit.

Skipped on an interpreter without torch; run in the container (Isaac Sim's python has it):
    /workspace/isaaclab/_isaac_sim/python.sh -m unittest scripts.tests.test_height_command
"""

import importlib.util
import os
import re
import unittest

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

MDP = os.path.join(
    os.path.dirname(__file__), "..", "..", "source", "quadruped_go2_locomotion", "quadruped_go2_locomotion",
    "tasks", "manager_based", "quadruped_go2_locomotion", "mdp",
)


def _load_height_module():
    # Loaded by file path: importing the mdp package would pull in isaaclab (and pxr), which this must not need.
    spec = importlib.util.spec_from_file_location("height_command", os.path.join(MDP, "height_command.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hc = _load_height_module() if torch is not None else None


@unittest.skipIf(torch is None, "torch not available on this interpreter")
class SampleTargetHeight(unittest.TestCase):
    def test_shape_and_range(self):
        h = hc.sample_target_height(5000, (0.28, 0.38), nominal=0.337, device="cpu")
        self.assertEqual(h.shape, (5000,))
        self.assertGreaterEqual(h.min().item(), 0.28)
        self.assertLessEqual(h.max().item(), 0.38)

    def test_covers_the_range_not_a_constant(self):
        h = hc.sample_target_height(5000, (0.28, 0.38), nominal=0.337, device="cpu")
        self.assertLess(h.min().item(), 0.29)
        self.assertGreater(h.max().item(), 0.37)
        self.assertGreater(h.std().item(), 0.02)

    def test_no_range_falls_back_to_nominal(self):
        h = hc.sample_target_height(10, None, nominal=0.337, device="cpu")
        self.assertTrue(torch.allclose(h, torch.full((10,), 0.337)))

    def test_zero_envs(self):
        self.assertEqual(hc.sample_target_height(0, (0.28, 0.38), 0.337, "cpu").shape, (0,))

    def test_bad_range_rejected(self):
        with self.assertRaises(ValueError):
            hc.sample_target_height(4, (0.38, 0.28), 0.337, "cpu")  # low > high


@unittest.skipIf(torch is None, "torch not available on this interpreter")
class AdaptiveHeightTarget(unittest.TestCase):
    def cmd(self, rows):
        return torch.tensor(rows, dtype=torch.float32)

    def test_reads_height_from_column_5(self):
        c = self.cmd([[0, 0, 0, 0, 0, 0.30], [0, 0, 0, 0, 0, 0.36]])
        t = hc.adaptive_height_target(c, pitch_sensitivity=-0.1, lean_sensitivity=0.0)
        self.assertTrue(torch.allclose(t, torch.tensor([0.30, 0.36])))

    def test_pitch_offset_is_symmetric_in_sign(self):
        c = self.cmd([[0, 0, 0, 0.5, 0, 0.34], [0, 0, 0, -0.5, 0, 0.34]])
        t = hc.adaptive_height_target(c, pitch_sensitivity=-0.1, lean_sensitivity=0.0)
        self.assertAlmostEqual(t[0].item(), 0.34 - 0.05, places=6)
        self.assertAlmostEqual(t[0].item(), t[1].item(), places=6)

    def test_lean_offset(self):
        c = self.cmd([[0, 0, 0, 0, -0.2, 0.34]])
        t = hc.adaptive_height_target(c, pitch_sensitivity=0.0, lean_sensitivity=0.08)
        self.assertAlmostEqual(t.item(), 0.34 + 0.08 * 0.2, places=6)

    def test_standing_env_keeps_its_height_target(self):
        # an all-zero velocity/pitch/lean command still carries a non-zero height target (never zeroed to 0 m)
        c = self.cmd([[0, 0, 0, 0, 0, 0.31]])
        self.assertAlmostEqual(hc.adaptive_height_target(c, -0.1, 0.0).item(), 0.31, places=6)

    def test_legacy_five_wide_command_uses_fallback(self):
        c = self.cmd([[0.5, 0, 0, 0.2, 0.1]])
        t = hc.adaptive_height_target(c, -0.1, 0.0, fallback_height=0.337)
        self.assertAlmostEqual(t.item(), 0.337 - 0.02, places=6)

    def test_five_wide_without_fallback_is_an_error(self):
        with self.assertRaises(ValueError):
            hc.adaptive_height_target(self.cmd([[0, 0, 0, 0, 0]]), -0.1, 0.0)

    def test_does_not_mutate_the_command(self):
        c = self.cmd([[0.1, 0.2, 0.3, 0.4, 0.5, 0.33]])
        before = c.clone()
        hc.adaptive_height_target(c, -0.1, 0.08)
        self.assertTrue(torch.equal(c, before))


class GateStaysOnFirstFiveColumns(unittest.TestCase):
    """The hard stand-still gate must ignore column 5 (height): standing envs keep a non-zero height target, and
    if the gate read it, a standing env would never count as 'zero command'. Source-level guard (rewards.py
    imports isaaclab, so it cannot be imported here)."""

    def test_any_commanded_reads_only_columns_0_to_4(self):
        with open(os.path.join(MDP, "rewards.py")) as f:
            src = f.read()
        body = re.search(r"def _any_commanded\(.*?\n(?=\n\n|\ndef |\nclass )", src, re.S).group(0)
        self.assertRegex(body, r"cmd\[:, ?:5\]")
        self.assertNotRegex(body, r"cmd\[:, ?:6\]")
        self.assertNotRegex(body, r"cmd\[:, ?5\]")


if __name__ == "__main__":
    unittest.main()
