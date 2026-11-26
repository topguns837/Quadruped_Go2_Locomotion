# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the soak analyzer's parsing and statistics helpers (no run folder needed)."""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
try:
    import analyze_soak as A  # noqa: E402
except ImportError:  # numpy is only needed on the desktop, where the analyzer runs
    A = None


@unittest.skipIf(A is None, "numpy not installed")
class AnalyzeHelpers(unittest.TestCase):
    def test_read_csv_skips_torn_last_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            with open(p, "w") as f:
                f.write("a,b,phase\n1,2.5,clean\n3,nan,loss1\n4,5")
            c = A.read_csv(p)
            self.assertEqual(len(c["a"]), 2)
            self.assertTrue(math.isnan(c["b"][1]))
            self.assertEqual(list(c["phase"]), ["clean", "loss1"])

    def test_read_csv_missing(self):
        self.assertIsNone(A.read_csv("/nonexistent/x.csv"))

    def test_parse_icmp(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.log")
            with open(p, "w") as f:
                f.write("PING x\n[1793358486.10] 64 bytes from x: icmp_seq=1 ttl=64 time=3.2 ms\n"
                        "[1793358486.30] 64 bytes from x: icmp_seq=3 ttl=64 time=60.5 ms\n")
            r = A.parse_icmp(p)
            self.assertEqual(list(r["seq"]), [1, 3])
            self.assertAlmostEqual(r["rtt"][1], 60.5)

    def test_stats_counts_and_nan(self):
        s = A.stats([1.0, 2.0, float("nan"), 60.0, 200.0])
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["n_over_50"], 2)
        self.assertEqual(s["n_over_150"], 1)
        self.assertEqual(s["max"], 200.0)
        self.assertEqual(A.stats([])["n"], 0)

    def test_fmt(self):
        self.assertEqual(A.fmt(float("nan")), "n/a")
        self.assertEqual(A.fmt(3), "3")
        self.assertEqual(A.fmt(1.234, 2), "1.23")


if __name__ == "__main__":
    unittest.main()
