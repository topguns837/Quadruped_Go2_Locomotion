# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""scripts/manual_command_slider.py: the height slider and its command file (pure functions, no display needed)."""

import importlib.util
import os
import re
import unittest

HERE = os.path.dirname(__file__)
ENV_CFG = os.path.join(
    HERE, "..", "..", "source", "quadruped_go2_locomotion", "quadruped_go2_locomotion", "tasks", "manager_based",
    "quadruped_go2_locomotion", "quadruped_go2_locomotion_env_cfg.py",
)

try:
    spec = importlib.util.spec_from_file_location("slider", os.path.join(HERE, "..", "manual_command_slider.py"))
    slider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slider)  # imports tkinter at module level
    _ERR = None
except Exception as e:  # noqa: BLE001 -- no tkinter on this interpreter
    slider, _ERR = None, e


def _cfg_range(name):
    with open(ENV_CFG) as f:
        m = re.search(rf"{name}=\((-?[\d.]+), ?(-?[\d.]+)\)", f.read())
    return float(m.group(1)), float(m.group(2))


@unittest.skipIf(slider is None, f"tkinter not importable here: {_ERR}")
class HeightSlider(unittest.TestCase):
    def by_key(self):
        return {s[0]: s for s in slider.SLIDERS}

    def test_six_sliders_height_last(self):
        self.assertEqual([s[0] for s in slider.SLIDERS], ["lin_vel_x", "lin_vel_y", "ang_vel_z", "pitch", "lean", "height"])

    def test_every_slider_range_matches_the_env_cfg(self):
        s = self.by_key()
        for key, cfg_name in (("lin_vel_x", "lin_vel_x"), ("lin_vel_y", "lin_vel_y"), ("ang_vel_z", "ang_vel_z"),
                              ("pitch", "ang_pos_y"), ("lean", "ang_pos_x"), ("height", "height")):
            self.assertEqual((s[key][2], s[key][3]), _cfg_range(cfg_name), key)

    def test_height_starts_at_the_nominal_stance(self):
        self.assertAlmostEqual(self.by_key()["height"][4], 0.337)

    def test_build_command_dict_back_compat_five_args(self):
        d = slider.build_command_dict(1, 2, 3, 4, 5)
        self.assertEqual([d[k] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "pitch", "lean")], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(d["height"], 0.337)

    def test_build_command_dict_with_height(self):
        self.assertEqual(slider.build_command_dict(0, 0, 0, 0, 0, height=0.30)["height"], 0.30)

    def test_neutral_values_zero_motion_but_nominal_height(self):
        v = slider.neutral_values()
        self.assertTrue(all(x == 0.0 for k, x in v.items() if k != "height"))
        self.assertAlmostEqual(v["height"], 0.337)

    def test_neutral_values_are_inside_every_slider_range(self):
        # Tk clamps a value outside its slider, which would desync the file from what the operator sees
        for key, _label, lo, hi, _init in [s[:5] for s in slider.SLIDERS]:
            self.assertTrue(lo <= slider.neutral_values()[key] <= hi, key)

    def test_command_file_roundtrip_is_what_deploy_parses(self):
        import json
        import tempfile

        path = os.path.join(tempfile.mkdtemp(), "cmd.json")
        slider.write_command_file(path, slider.build_command_dict(0.1, 0, 0, 0.2, 0.1, height=0.31))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["height"], 0.31)
        self.assertEqual(set(data), {"lin_vel_x", "lin_vel_y", "ang_vel_z", "pitch", "lean", "height"})


if __name__ == "__main__":
    unittest.main()
