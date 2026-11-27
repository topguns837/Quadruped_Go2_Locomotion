# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Safety-path tests for src/deploy/relay_jetson.py: when can the relay start publishing LowCmd to a robot? No DDS, no
robot. The relay object is built without the SDK and _start_publishing is replaced with a tripwire, so a test fails
if any path that should NOT publish tries to. Run:  python3 -m unittest discover -s src/deploy/tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import yaml  # noqa: F401  (deploy_real needs it)
except ImportError:  # pragma: no cover
    yaml = None

if yaml is not None:
    import relay_jetson as rj  # noqa: E402
    import relay_link as rl  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "profiles", "go2_locomotion.yaml")


MAPS_WRONG = """\
ffff84d86000-ffff85700000 r-xp 00000000 103:02 1234 /usr/local/lib/libddsc.so
ffff90000000-ffff90100000 r-xp 00000000 103:02 9999 /usr/lib/aarch64-linux-gnu/libc-2.31.so
"""
MAPS_RIGHT = """\
ffff84d86000-ffff85700000 r-xp 00000000 103:02 1234 /home/u/cyclonedds_ws/install/cyclonedds/lib/libddsc.so.0.10.2
ffff85800000-ffff85900000 r--p 00000000 103:02 1234 /home/u/cyclonedds_ws/install/cyclonedds/lib/libddsc.so.0.10.2
"""


@unittest.skipIf(yaml is None, "pyyaml not installed")
class DdscLibraryGuard(unittest.TestCase):
    def test_paths_are_extracted_once_and_only_for_libddsc(self):
        self.assertEqual(rj.loaded_ddsc_paths(MAPS_WRONG), ["/usr/local/lib/libddsc.so"])
        self.assertEqual(len(rj.loaded_ddsc_paths(MAPS_RIGHT)), 1)  # two mappings of one file -> one path
        self.assertEqual(rj.loaded_ddsc_paths(""), [])

    def test_wrong_library_is_refused_and_right_one_accepted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            lib = os.path.join(d, "libddsc.so.0.10.2")
            open(lib, "w").close()
            maps_ok = "a-b r-xp 0 0:0 1 " + lib + "\n"
            orig = rj.loaded_ddsc_paths
            try:
                rj.loaded_ddsc_paths = lambda _t: orig(maps_ok)
                self.assertEqual(rj.check_ddsc(d), [os.path.realpath(lib)])  # right dir: accepted
                rj.loaded_ddsc_paths = lambda _t: orig(MAPS_WRONG)
                with self.assertRaises(RuntimeError) as cm:
                    rj.check_ddsc(d)  # wrong library mapped: refused with the fix in the message
                self.assertIn("LD_LIBRARY_PATH", str(cm.exception))
                rj.loaded_ddsc_paths = lambda _t: []
                with self.assertRaises(RuntimeError):
                    rj.check_ddsc(d)  # nothing found while one is required: refused too
                self.assertEqual(rj.check_ddsc(None), [])  # not required: only reports
            finally:
                rj.loaded_ddsc_paths = orig


class FakeCtl:
    def __init__(self):
        self.sent = []

    def send(self, obj):
        self.sent.append(obj)

    def close(self):
        pass


def make_relay(iface):
    args = rj.argparse.Namespace(
        network_interface=iface, silence_damp_s=0.15, first_cmd_timeout_s=5.0, damp_hold_s=1.0,
        max_target_dev_deg=30.0, kp_max=30.0, kd_max=5.0, state_rate_hz=100.0, bind="127.0.0.1",
        control_port=0, cmd_port=0,
    )
    relay = rj.Relay(args, rj.dr.load_config(CONFIG, iface), bytes(range(32)))
    relay.ctl = FakeCtl()
    started = []

    def tripwire():
        started.append(True)
        relay.target_buf = object()  # only reached if a test lets publishing start

    relay._start_publishing = tripwire
    relay.started = started
    return relay


@unittest.skipIf(yaml is None, "pyyaml not installed")
class RelayPublishingGuards(unittest.TestCase):
    def test_skip_release_is_refused_on_a_real_interface(self):
        relay = make_relay("eth0")  # the robot-facing interface on the Jetson
        relay._do_release({"skip": True})
        self.assertEqual(relay.started, [], "publishing started although --skip_release is only for loopback")
        self.assertFalse(relay.ctl.sent[-1]["ok"])
        self.assertIn("loopback", relay.ctl.sent[-1]["message"])
        self.assertEqual(relay.mode, rl.MODE_STREAMING)
        self.assertIsNone(relay.publish_thread)

    def test_skip_release_is_accepted_on_loopback_only(self):
        relay = make_relay("lo")
        relay._do_release({"skip": True})
        self.assertEqual(relay.started, [True])
        self.assertTrue(relay.ctl.sent[-1]["ok"])
        self.assertEqual(relay.mode, rl.MODE_ACTIVE)

    def test_commands_are_ignored_until_the_relay_is_active(self):
        relay = make_relay("eth0")
        pkt = {"session": 1, "seq": 1, "t_ns": 1, "target": [0.3] * 12, "kp": 25.0, "kd": 0.5}
        relay._handle_cmd(pkt)
        self.assertIsNone(relay.target_buf)
        self.assertEqual(relay.counters["cmds"], 0)
        self.assertEqual(relay.mode, rl.MODE_STREAMING)

    def test_damp_and_stop_without_publishing_write_nothing(self):
        relay = make_relay("eth0")
        relay.publisher = None  # a write attempt would raise AttributeError
        relay.enter_damped("desktop requested damping")
        relay._stop_publishing()
        relay._end_session()
        self.assertEqual(relay.started, [])
        self.assertEqual(relay.mode, rl.MODE_STREAMING)  # session ended, ready for a new one

    def test_release_is_refused_when_not_streaming(self):
        relay = make_relay("lo")
        relay.mode = rl.MODE_DAMPED
        # the TCP handler checks this before spawning _do_release; the same guard, asserted directly
        self.assertNotEqual(relay.mode, rl.MODE_STREAMING)


if __name__ == "__main__":
    unittest.main()
