# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for src/deploy/impair.py. Run: python3 -m unittest discover -s src/deploy/tests"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import impair as imp  # noqa: E402


class Sink:
    def __init__(self):
        self.got = []
        self.times = []
        self.lock = threading.Lock()

    def __call__(self, pkt):
        with self.lock:
            self.got.append(pkt)
            self.times.append(time.monotonic())


class ScheduleTests(unittest.TestCase):
    def test_soak15_covers_15_minutes_without_gaps(self):
        s = imp.build_schedule("soak15")
        self.assertEqual(imp.schedule_duration(s), 900)
        for (_, e1, _, _), (s2, _, _, _) in zip(s, s[1:]):
            self.assertEqual(e1, s2)  # contiguous phases

    def test_scaling_compresses_time_and_strengthens_impairments(self):
        s = imp.build_schedule("soak15", time_scale=0.1, impair_scale=2.0)
        self.assertAlmostEqual(imp.schedule_duration(s), 90.0)
        loss3 = [p for _, _, name, p in s if name == "loss3"][0]
        self.assertAlmostEqual(loss3["loss_p"], 0.06)
        stall = [p for _, _, name, p in s if name == "stall50"][0]
        self.assertAlmostEqual(stall["stall_ms"], 100.0)
        self.assertAlmostEqual(stall["stall_every_s"], 1.0)  # period scales with time, length with strength

    def test_phase_at(self):
        s = imp.build_schedule("soak15")
        self.assertEqual(imp.phase_at(s, 0)[0], "clean")
        self.assertEqual(imp.phase_at(s, 241)[0], "loss1")
        self.assertEqual(imp.phase_at(s, 899.9)[0], "recovery")
        self.assertEqual(imp.phase_at(s, 5000)[0], "recovery")  # last phase persists

    def test_unknown_schedule_is_an_error(self):
        with self.assertRaises(ValueError):
            imp.build_schedule("nope")

    def test_only_loopback_is_allowed(self):
        imp.check_allowed("lo")
        for iface in ("eth0", "wlan0", "enp2s0", ""):
            with self.assertRaises(PermissionError):
                imp.check_allowed(iface)


class SanitizeTests(unittest.TestCase):
    def test_valid_profiles_pass_and_become_floats(self):
        self.assertEqual(imp.sanitize_profile({}), {})
        self.assertEqual(imp.sanitize_profile({"loss_p": 0.05, "delay_ms": 10}), {"loss_p": 0.05, "delay_ms": 10.0})
        for _, _, _, prof in imp.SCHEDULES["soak15"]:
            imp.sanitize_profile(prof)  # every shipped phase must be acceptable to the relay

    def test_bad_profiles_are_rejected(self):
        bad = [{"loss_p": 0.9}, {"delay_ms": 60000}, {"nope": 1}, {"loss_p": "0.1"}, {"loss_p": True}, {"loss_p": -0.1},
               {"stall_ms": 50}, {"stall_ms": 50, "stall_every_s": 0.1}, [], "x", None]
        for prof in bad:
            with self.assertRaises(ValueError, msg=repr(prof)):
                imp.sanitize_profile(prof)

    def test_scaled_schedules_stay_valid_for_the_supported_scales(self):
        for ts, isc in ((1.0, 1.0), (0.1, 1.0), (1.0, 0.5), (0.1, 2.0)):
            for _, _, _, prof in imp.build_schedule("soak15", ts, isc):
                imp.sanitize_profile(prof)


class ShimTests(unittest.TestCase):
    def test_clean_profile_is_a_direct_synchronous_send(self):
        sink = Sink()
        im = imp.Impairment(sink, seed=1)
        for i in range(50):
            self.assertEqual(im.send(bytes([i])), (imp.PASS, 0.0))
        self.assertEqual(len(sink.got), 50)  # delivered before send() returned, in order
        self.assertEqual(sink.got, [bytes([i]) for i in range(50)])
        self.assertIsNone(im._worker)  # no thread was ever needed
        im.close()

    def test_loss_rate_matches_the_profile(self):
        sink = Sink()
        im = imp.Impairment(sink, seed=7)
        im.set_profile("loss5", {"loss_p": 0.05})
        n = 20000
        actions = [im.send(b"x")[0] for _ in range(n)]
        dropped = actions.count(imp.DROP) / n
        self.assertAlmostEqual(dropped, 0.05, delta=0.008)
        self.assertEqual(len(sink.got), n - actions.count(imp.DROP))
        im.close()

    def test_same_seed_gives_the_same_decisions(self):
        def run(seed):
            im = imp.Impairment(Sink(), seed=seed)
            im.set_profile("p", {"loss_p": 0.1, "delay_ms": 0.0})
            out = [im.send(b"x")[0] for _ in range(500)]
            im.close()
            return out

        self.assertEqual(run(3), run(3))
        self.assertNotEqual(run(3), run(4))

    def test_delay_is_bounded_and_can_reorder(self):
        sink = Sink()
        im = imp.Impairment(sink, seed=2)
        im.set_profile("d", {"delay_ms": 30})
        n = 200
        t0 = time.monotonic()
        delays = []
        for i in range(n):
            delays.append(im.send(i.to_bytes(2, "big"))[1])
            time.sleep(0.002)
        time.sleep(0.2)  # let the worker drain
        self.assertEqual(len(sink.got), n)  # delay never loses packets
        self.assertTrue(all(0.0 <= d <= 30.0 for d in delays))
        order = [int.from_bytes(p, "big") for p in sink.got]
        self.assertNotEqual(order, sorted(order), "30 ms random delay at 500 Hz must reorder some packets")
        self.assertEqual(sorted(order), list(range(n)))
        im.close()
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_stall_holds_everything_then_releases_together(self):
        sink = Sink()
        clk = [100.0]
        im = imp.Impairment(sink, seed=0, clock=lambda: clk[0])
        im.set_profile("stall", {"stall_ms": 80, "stall_every_s": 1.0})
        # before the first period: no stall
        clk[0] = 100.5
        self.assertEqual(im.send(b"a")[0], imp.PASS)
        # inside the first stall window (period 1.0 s, 80 ms long): held until its end
        clk[0] = 101.02
        action, delay_ms = im.send(b"b")
        self.assertEqual(action, imp.STALL)
        self.assertAlmostEqual(delay_ms, 60.0, delta=1.0)  # 101.08 - 101.02
        # just after the window: passes again
        clk[0] = 101.2
        self.assertEqual(im.send(b"c")[0], imp.PASS)
        im.close()

    def test_close_releases_held_packets(self):
        sink = Sink()
        im = imp.Impairment(sink, seed=0)
        im.set_profile("d", {"delay_ms": 5000})
        for i in range(5):
            im.send(bytes([i]))
        self.assertEqual(sink.got, [])
        im.close()  # must not hang, and releases everything still held right away
        self.assertEqual(sorted(sink.got), [bytes([i]) for i in range(5)])

    def test_counts_add_up(self):
        im = imp.Impairment(Sink(), seed=5)
        im.set_profile("m", {"loss_p": 0.2, "reorder_p": 0.2})
        for _ in range(1000):
            im.send(b"x")
        self.assertEqual(sum(im.counts.values()), 1000)
        im.close()


if __name__ == "__main__":
    unittest.main()
