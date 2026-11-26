# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for src/deploy/runlog.py. No robot, no network, no SDK. Run: python3 -m unittest discover -s src/deploy/tests"""

import csv
import json
import math
import os
import random
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import runlog as rlg  # noqa: E402

WIRELESS = """\
Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE
 face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22
wlan0: 0000   70.  -40.  -256        0      0      0      0      0        0
"""
NETDEV = """\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo:  100  10  0  0  0  0  0  0  100  10  0  0  0  0  0  0
wlan0: 123456 789 1 2 0 0 0 0 654321 987 3 4 0 0 0 0
"""


class HistogramTests(unittest.TestCase):
    def test_percentiles_are_close_to_exact_on_random_data(self):
        rng = random.Random(1)
        xs = [rng.lognormvariate(math.log(5.0), 0.6) for _ in range(20000)]
        h = rlg.Histogram()
        for x in xs:
            h.add(x)
        xs.sort()
        for p in (50, 95, 99):
            exact = xs[int(p / 100 * len(xs)) - 1]
            est = h.percentile(p)
            self.assertLess(abs(est - exact) / exact, 0.15, f"p{p}: est {est} vs exact {exact}")
        self.assertEqual(h.percentile(100), max(xs))  # the maximum is exact
        self.assertEqual(h.count, len(xs))

    def test_threshold_counts_are_exact(self):
        h = rlg.Histogram(thresholds=(4.0, 10.0))
        for x in (1.0, 4.0, 4.1, 9.9, 10.0, 10.1, 50.0):
            h.add(x)
        self.assertEqual(h.above[4.0], 5)  # 4.1, 9.9, 10.0, 10.1, 50.0: strictly greater than 4.0
        self.assertEqual(h.above[10.0], 2)

    def test_empty_and_overflow_are_safe(self):
        h = rlg.Histogram()
        self.assertEqual(h.percentile(99), 0.0)
        self.assertEqual(h.summary()["count"], 0)
        h.add(1e9)  # beyond the top bin
        self.assertEqual(h.percentile(100), 1e9)
        h.add(0.0)  # at/below the bottom bin
        self.assertEqual(h.count, 2)

    def test_merge_adds_everything(self):
        a, b = rlg.Histogram(), rlg.Histogram()
        for x in (1, 2, 3):
            a.add(x)
        for x in (4, 100):
            b.add(x)
        a.merge(b)
        self.assertEqual(a.count, 5)
        self.assertEqual(a.max, 100)
        self.assertEqual(a.min, 1)
        self.assertEqual(a.above[20.0], 1)


class CsvTests(unittest.TestCase):
    def test_rows_roundtrip_and_float_format(self):
        with tempfile.TemporaryDirectory() as d:
            c = rlg.CsvLog(os.path.join(d, "a.csv"), ["t", "n", "s"])
            c.row(1.5, 3, "x")
            c.row(float("nan"), 4, "y")
            c.close()
            with open(os.path.join(d, "a.csv")) as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["t", "n", "s"])
            self.assertEqual(rows[1], ["1.500000", "3", "x"])
            self.assertEqual(rows[2][0], "nan")

    def test_wrong_column_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c = rlg.CsvLog(os.path.join(d, "a.csv"), ["a", "b"])
            with self.assertRaises(ValueError):
                c.row(1)
            c.close()

    def test_concurrent_writers_lose_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            c = rlg.CsvLog(os.path.join(d, "a.csv"), ["k", "i"])

            def worker(k):
                for i in range(2000):
                    c.row(k, i)
                    if i % 500 == 0:
                        c.flush()

            ts = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            c.close()
            with open(os.path.join(d, "a.csv")) as f:
                rows = list(csv.reader(f))[1:]
            self.assertEqual(len(rows), 12000)
            self.assertEqual(c.rows, 12000)
            for k in range(6):
                got = sorted(int(r[1]) for r in rows if r[0] == str(k))
                self.assertEqual(got, list(range(2000)))


class ProcParsers(unittest.TestCase):
    def test_wireless(self):
        w = rlg.parse_proc_net_wireless(WIRELESS)
        self.assertEqual(w["wlan0"], {"link": 70.0, "level": -40.0, "noise": -256.0})
        self.assertEqual(rlg.parse_proc_net_wireless(""), {})

    def test_net_dev(self):
        d = rlg.parse_proc_net_dev(NETDEV)
        self.assertEqual(d["wlan0"]["rx_packets"], 789)
        self.assertEqual(d["wlan0"]["tx_drop"], 4)
        self.assertEqual(d["wlan0"]["rx_errs"], 1)
        self.assertIn("lo", d)

    def test_proc_cpu_returns_a_sane_number(self):
        c = rlg.ProcCpu()
        x = 0
        for i in range(300000):
            x += i * i
        pct = c.percent()
        self.assertGreaterEqual(pct, 0.0)
        self.assertLess(pct, 1000.0)


class RunLogTests(unittest.TestCase):
    def test_disabled_runlog_is_a_noop(self):
        r = rlg.RunLog(None, "x")
        r.event("hello")
        r.csv("a.csv", ["a"]).row(1)
        r.write_json("m.json", {"a": 1})
        r.tee_stdio()
        r.start_flusher()
        r.close()  # nothing raised, nothing written

    def test_files_events_tee_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            r = rlg.RunLog(os.path.join(d, "run", "comp"), "comp")
            c = r.csv("x.csv", ["a", "b"])
            r.start_flusher(0.05)
            r.tee_stdio()
            print("a printed line")
            c.row(1, 2)
            r.write_json("meta.json", {"nan": float("nan"), "ok": [1, 2], "p": ("a",)})
            r.event("direct event")
            import time
            time.sleep(0.2)  # the flusher must have written without close()
            with open(os.path.join(r.dir, "x.csv")) as f:
                self.assertEqual(len(list(csv.reader(f))), 2)
            r.close()
            with open(os.path.join(r.dir, "events.log")) as f:
                ev = f.read()
            self.assertIn("a printed line", ev)
            self.assertIn("direct event", ev)
            with open(os.path.join(r.dir, "meta.json")) as f:
                m = json.load(f)
            self.assertIsNone(m["nan"])  # NaN becomes valid JSON null
            self.assertEqual(m["ok"], [1, 2])

    def test_base_meta_hashes_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.txt")
            with open(p, "w") as f:
                f.write("abc")
            m = rlg.base_meta("comp", "RUN", ["a", "b"], files=[p, os.path.join(d, "missing")])
            self.assertEqual(m["file_sha256"]["f.txt"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            self.assertEqual(m["file_sha256"]["missing"], "unreadable")
            self.assertEqual(m["run_id"], "RUN")


if __name__ == "__main__":
    unittest.main()
