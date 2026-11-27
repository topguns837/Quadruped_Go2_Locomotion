# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ramp_to_default_pose's settle phase against a stub robot (needs the deploy_real import chain: run in the container)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import deploy_real as dr
except Exception:  # noqa: BLE001 -- torch / SDK not available on this interpreter
    dr = None

NAMES = [f"j{i}" for i in range(12)]


class StubMonitor:
    """Robot whose joints chase the held target at `rate` rad/s, optionally never reaching it."""

    def __init__(self, start, buf, rate, max_dev=math.radians(15.0), stuck=()):
        self.q = list(start)
        self.buf, self.rate, self.max_dev, self.stuck = buf, rate, max_dev, set(stuck)
        self.max_dev_rad = max_dev
        self.checks = 0

    def actual_isaac_order(self):
        return list(self.q)

    def actual_tau_isaac_order(self):
        return [0.0] * 12

    def limit_target(self, tgt):
        return dr.tether_limit(tgt, self.q, self.max_dev, NAMES)

    def check(self, held, kp_fraction):
        self.checks += 1
        for i, h in enumerate(held):
            if i in self.stuck:
                continue
            step = max(-self.rate * 0.02, min(self.rate * 0.02, h - self.q[i]))
            self.q[i] += step
        return 0.0, 0.0


@unittest.skipIf(dr is None, "deploy_real not importable here")
class SettlePhase(unittest.TestCase):
    def _run(self, start_off, settle_s, **kw):
        default = [0.5] * 12
        start = [d - start_off for d in default]
        buf = dr.LowCmdTarget(start, initial_kp=0.0, initial_kd=0.0)
        mon = StubMonitor(start, buf, **kw)
        dr.ramp_to_default_pose(buf, start, default, 0.2, 0.02, NAMES, start_kp=6.0, full_kp=60.0, monitor=mon,
                                settle_s=settle_s, settle_tol_rad=math.radians(6.0))
        return mon, default

    def test_slow_joints_arrive_during_settle(self):
        # 70 deg away, 60 deg/s: a 0.2 s ramp leaves them far off; settle must wait them in.
        mon, default = self._run(math.radians(70), 5.0, rate=math.radians(60))
        self.assertLess(max(abs(d - q) for d, q in zip(default, mon.q)), math.radians(6.5))

    def test_no_settle_keeps_old_failure(self):
        with self.assertRaises(dr.TrackingFault):
            self._run(math.radians(70), 0.0, rate=math.radians(60))

    def test_stuck_joint_still_faults_with_degrees(self):
        with self.assertRaises(dr.TrackingFault) as cm:
            self._run(math.radians(70), 0.6, rate=math.radians(600), stuck=[9])
        self.assertIn("j9", str(cm.exception))
        self.assertIn("deg", str(cm.exception))


@unittest.skipIf(dr is None, "deploy_real not importable here")
class HandoverFade(unittest.TestCase):
    def test_fade_is_monotonic_and_ends_at_policy_kp(self):
        vals = [dr.handover_kp(i, 25, 60.0, 25.0) for i in range(30)]
        self.assertTrue(all(a >= b for a, b in zip(vals, vals[1:])))
        self.assertLess(vals[0], 60.0)
        self.assertEqual(vals[24], 25.0)
        self.assertEqual(vals[29], 25.0)

    def test_zero_ticks_is_immediate_drop(self):
        self.assertEqual(dr.handover_kp(0, 0, 60.0, 25.0), 25.0)


class _Snap:
    wireless_remote = bytes(40)
    gyro = (0.0, 0.0, 0.0)

    def __init__(self, tau):
        self.joint_pos, self.joint_vel, self.tau_est = [0.0] * 12, [0.0] * 12, tau

    def age_s(self):
        return 0.0


class _State:
    def __init__(self, tau):
        self._snap = _Snap(tau)

    def snapshot(self):
        return self._snap


REAL = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint",
        "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]


@unittest.skipIf(dr is None, "deploy_real not importable here")
class LiftMode(unittest.TestCase):
    def _mon(self, tau):
        return dr.SafetyMonitor(_State(tau), list(range(12)), dr.TorqueResponseWatchdog(25), REAL,
                                max_dev_rad=math.radians(15), abort_err_rad=math.radians(30), abort_steps=3,
                                tau_abort_nm=21.15)

    def test_tether_per_joint_list(self):
        lim, names = dr.tether_limit([1.0, 1.0], [0.0, 0.0], [0.1, 0.5], ["a", "b"])
        self.assertEqual(lim, [0.1, 0.5])
        self.assertEqual(names, ["a", "b"])

    def test_calf_torque_aborts_by_default_but_not_in_lift_mode(self):
        tau = [0.0] * 12
        tau[10] = 30.0  # RL_calf
        with self.assertRaises(dr.TrackingFault):
            self._mon(tau).check([0.0] * 12, 1.0)
        m = self._mon(tau)
        m.set_lift_mode(math.radians(25), 40.9)
        m.check([0.0] * 12, 1.0)  # no raise
        m.end_lift_mode()
        with self.assertRaises(dr.TrackingFault):
            m.check([0.0] * 12, 1.0)

    def test_hip_and_thigh_keep_the_normal_abort_in_lift_mode(self):
        for idx in (2, 6):
            tau = [0.0] * 12
            tau[idx] = 25.0
            m = self._mon(tau)
            m.set_lift_mode(math.radians(25), 40.9)
            with self.assertRaises(dr.TrackingFault):
                m.check([0.0] * 12, 1.0)

    def test_lift_mode_only_widens_calf_leash(self):
        m = self._mon([0.0] * 12)
        m.set_lift_mode(math.radians(25), 40.9)
        lim, names = m.limit_target([1.0] * 12)  # actual = 0 everywhere
        self.assertAlmostEqual(lim[0], math.radians(15))
        self.assertAlmostEqual(lim[10], math.radians(25))
        self.assertEqual(len(names), 12)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(dr is None, "deploy_real not importable here")
class HandoverSlew(unittest.TestCase):
    def _step_rad(self, policy_tick, hold_kp, kp, handover_ticks, handover_slew, main_slew, dt=0.02):
        # mirrors the loop: conservative cap during the fade, full cap after
        max_step = main_slew * dt
        ho_step = min(handover_slew, main_slew) * dt
        in_fade = hold_kp != kp and policy_tick < handover_ticks
        return ho_step if in_fade else max_step

    def test_conservative_during_fade_full_after(self):
        dt = 0.02
        # during fade (tick 0..24): 2.618 rad/s cap even though main is 15
        self.assertAlmostEqual(self._step_rad(0, 60, 25, 25, 2.618, 15.0), 2.618 * dt)
        self.assertAlmostEqual(self._step_rad(24, 60, 25, 25, 2.618, 15.0), 2.618 * dt)
        # after fade: full 15 rad/s
        self.assertAlmostEqual(self._step_rad(25, 60, 25, 25, 2.618, 15.0), 15.0 * dt)

    def test_no_fade_when_hold_equals_kp(self):
        self.assertAlmostEqual(self._step_rad(0, 25, 25, 25, 2.618, 15.0), 15.0 * 0.02)

    def test_handover_slew_never_exceeds_main(self):
        # if someone sets handover higher than main, it is clamped to main
        self.assertAlmostEqual(self._step_rad(0, 60, 25, 25, 99.0, 10.0), 10.0 * 0.02)
