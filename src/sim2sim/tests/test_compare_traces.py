# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""compare_traces: every observation width a shipped policy uses must be understood, including the stacked 288."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import compare_traces as ct  # noqa: E402


@pytest.mark.parametrize("width", [47, 48, 50, 51, 52])
def test_terms_tile_the_whole_observation(width):
    terms = ct.obs_terms(width)
    assert terms[0][1] == 0 and terms[-1][2] == width
    assert all(a[2] == b[1] for a, b in zip(terms, terms[1:]))


def test_unknown_width_is_rejected_with_the_known_ones():
    with pytest.raises(ValueError, match="unknown observation width 49"):
        ct.obs_terms(49)


def test_stacked_288_reduces_to_the_newest_48_frame_term_major():
    h, base = 6, 48
    frames = np.arange(base * h, dtype=float)[None, :].repeat(3, axis=0)
    out = ct.newest_frame(frames)
    assert out.shape == (3, base)
    # term-major: base_ang_vel (w=3) occupies columns 0..17 as 6 samples of 3 values, newest last -> columns 15..17
    assert out[0, :3].tolist() == [15.0, 16.0, 17.0]
    # gravity (w=3) starts at column 18: its newest sample is columns 33..35
    assert out[0, 3:6].tolist() == [33.0, 34.0, 35.0]


def test_single_frame_passes_through():
    x = np.zeros((2, 47))
    assert ct.newest_frame(x) is x


def test_self_comparison_of_a_stacked_trace(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    np.savez(tmp_path / "t.npz", obs=rng.normal(size=(n, 288)), action=rng.normal(size=(n, 12)), dt=0.02)
    stats = ct.term_stats(ct.newest_frame(np.load(tmp_path / "t.npz")["obs"]), settle=10)
    assert "command" in stats and len(stats) == 6
