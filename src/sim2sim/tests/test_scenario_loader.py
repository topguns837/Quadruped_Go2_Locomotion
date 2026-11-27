# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""sim2sim scenario loading: valid files parse, malformed ones fail with a message that names the problem."""

import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
s2s = pytest.importorskip("sim2sim_mujoco")

SCENARIOS = os.path.join(os.path.dirname(__file__), "..", "scenarios")


def write(tmp_path, text):
    p = tmp_path / "s.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_shipped_scenarios_load():
    for name in os.listdir(SCENARIOS):
        if name.endswith(".yaml"):
            assert s2s.load_scenario(os.path.join(SCENARIOS, name))


def test_five_and_six_value_commands(tmp_path):
    segs = s2s.load_scenario(write(tmp_path, """
        segments:
          - {name: a, duration: 1.0, command: [0, 0, 0, 0, 0]}
          - {name: b, duration: 2.5, command: [0, 0, 0, 0, 0, 0.3]}
    """))
    assert [s.name for s in segs] == ["a", "b"] and segs[1].duration == 2.5 and len(segs[1].command) == 6


@pytest.mark.parametrize("text, message", [
    ("name: nothing", "non-empty 'segments:'"),
    ("segments: []", "non-empty 'segments:'"),
    ("- 1\n- 2\n", "non-empty 'segments:'"),
    ("segments:\n  - {name: a, command: [0,0,0,0,0]}\n", r"lacks \['duration'\]"),
    ("segments:\n  - {name: a, duration: 1.0, command: [0,0,0]}\n", "5 or 6 values"),
    ("segments:\n  - {name: a, duration: -1.0, command: [0,0,0,0,0]}\n", "duration must be > 0"),
    ("segments:\n  - {name: a, duration: 0, command: [0,0,0,0,0]}\n", "duration must be > 0"),
])
def test_malformed_scenarios(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        s2s.load_scenario(write(tmp_path, text))


def test_summary_says_n_a_for_a_segment_without_steady_state_samples(capsys):
    import numpy as np

    short = s2s.SegmentStats("too_short")  # never received a steady-state sample
    ok = s2s.SegmentStats("ok")
    ok.add({n: 0.1 for n in s2s.METRIC_NAMES})
    names = [f"j{i}" for i in range(12)]
    s2s.print_summary([short, ok], "p.pt", 0.337, np.zeros(12), names)
    out = capsys.readouterr().out
    short_line = next(line for line in out.splitlines() if "too_short" in line)
    assert "n/a" in short_line and "nan" not in out


def test_sim_config_carries_history_length_and_joint_vel_filter_from_the_profile():
    profiles = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "profiles")
    stacked = s2s.load_sim_config(os.path.join(profiles, "go2_obs_stacking.yaml"), None)
    single = s2s.load_sim_config(os.path.join(profiles, "go2_locomotion.yaml"), None)
    assert stacked.history_length == 6 and stacked.joint_vel_filter_hz == 10.0
    assert single.history_length == 1 and single.joint_vel_filter_hz is None
