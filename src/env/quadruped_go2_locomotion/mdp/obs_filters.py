# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared, pure maths for the observation low-pass (EMA) filter -- no Isaac Sim imports, so unit-testable anywhere.

The same first-order EMA is applied to the joint-velocity observation in training (mdp.diagnostics.JointVelFiltered),
in deploy (deploy_real.ObsEmaFilter) and in sim2sim, so the policy sees the SAME filtered joint_vel in sim and on
hardware. That is what makes the policy resilient to real joint-velocity sensor noise instead of merely trained on a
quieter sim: the filter attenuates the noise band (~ Nyquist) before it reaches the net, identically everywhere.
"""

from __future__ import annotations

import math


def alpha_from_cutoff(cutoff_hz: float, dt: float) -> float:
    """First-order EMA coefficient for a -3 dB cutoff at ``cutoff_hz``, sampled every ``dt`` seconds.

    ``y += alpha * (x - y)`` with this alpha is the discrete one-pole low-pass. alpha in (0, 1]; a higher cutoff
    (or larger dt) means less smoothing (alpha -> 1 = passthrough).
    """
    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be > 0, got {cutoff_hz}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
