# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The EMA filter maths (mdp/obs_filters.py) + the deploy ObsEmaFilter, which must share one formula."""

import importlib.util
import math
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
OBS_FILTERS = os.path.join(HERE, "..", "quadruped_go2_locomotion", "mdp", "obs_filters.py")
spec = importlib.util.spec_from_file_location("obs_filters", OBS_FILTERS)
of = importlib.util.module_from_spec(spec)
spec.loader.exec_module(of)


class AlphaFromCutoff(unittest.TestCase):
    def test_matches_the_one_pole_formula(self):
        self.assertAlmostEqual(of.alpha_from_cutoff(5.0, 0.02), 1.0 - math.exp(-2 * math.pi * 5.0 * 0.02))

    def test_in_unit_interval(self):
        for fc in (0.5, 1, 5, 10, 24):
            self.assertTrue(0.0 < of.alpha_from_cutoff(fc, 0.02) <= 1.0)

    def test_higher_cutoff_smooths_less(self):
        self.assertLess(of.alpha_from_cutoff(2.0, 0.02), of.alpha_from_cutoff(10.0, 0.02))

    def test_matches_deploy_action_lowpass_formula(self):
        # deploy's action low-pass uses the identical expression (alpha = 1 - exp(-2*pi*fc*dt)); keep them equal
        self.assertAlmostEqual(of.alpha_from_cutoff(5.0, 0.02), 0.4665, places=3)

    def test_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            of.alpha_from_cutoff(0.0, 0.02)
        with self.assertRaises(ValueError):
            of.alpha_from_cutoff(5.0, 0.0)


# deploy's ObsEmaFilter needs torch; run this part in the container.
sys.path.insert(0, os.path.join(HERE, "..", "..", "deploy"))
try:
    import torch

    import deploy_real as dr

    _ERR = None
except Exception as e:  # noqa: BLE001
    torch, dr, _ERR = None, None, e


@unittest.skipIf(dr is None, f"deploy_real/torch not usable here: {_ERR}")
class DeployObsEmaFilter(unittest.TestCase):
    def test_disabled_is_passthrough(self):
        f = dr.ObsEmaFilter(cutoff_hz=None, dt=0.02)
        for v in ([1.0] * 12, [2.0] * 12):
            self.assertEqual(f.filter(v), v)

    def test_first_sample_seeds_without_transient(self):
        f = dr.ObsEmaFilter(cutoff_hz=5.0, dt=0.02)
        self.assertEqual(f.filter([0.5] * 12), [0.5] * 12)  # first call returns the input unchanged

    def test_step_matches_the_formula(self):
        f = dr.ObsEmaFilter(cutoff_hz=5.0, dt=0.02)
        a = of.alpha_from_cutoff(5.0, 0.02)
        f.filter([0.0] * 3)
        out = f.filter([1.0] * 3)
        self.assertAlmostEqual(out[0], a, places=6)  # 0 + a*(1-0)

    def test_attenuates_an_alternating_signal(self):
        # +1,-1,+1,-1 (the ~Nyquist buzz band) must come out much smaller than +-1
        f = dr.ObsEmaFilter(cutoff_hz=5.0, dt=0.02)
        f.filter([0.0])
        outs = [f.filter([1.0 if i % 2 == 0 else -1.0])[0] for i in range(40)]
        self.assertLess(max(abs(x) for x in outs[10:]), 0.6)

    def test_reset(self):
        f = dr.ObsEmaFilter(cutoff_hz=5.0, dt=0.02)
        f.filter([5.0])
        f.reset()
        self.assertEqual(f.filter([0.2]), [0.2])  # re-seeds


if __name__ == "__main__":
    unittest.main()
