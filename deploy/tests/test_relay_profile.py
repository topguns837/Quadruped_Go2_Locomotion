# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The optional `relay:` block of a deploy profile (deploy/relay_profile.py): pure YAML, runs on any interpreter.

The wired deploy_real.py must keep accepting a profile that carries a `relay:` block (it ignores it); the
torch-dependent half of that is checked at the bottom and skipped where deploy_real cannot be imported.
"""

import os
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import relay_profile as rp  # noqa: E402

try:
    import deploy_real as dr
except Exception:  # noqa: BLE001 -- torch / SDK not available on this interpreter
    dr = None

PROFILES = os.path.join(HERE, "..", "configs", "test_profiles.yaml")


def write(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(text))
    return path


class LoadRelayBlock(unittest.TestCase):
    def test_no_relay_block_gives_empty(self):
        p = write("""
            a:
              low_pass: 5.0
            """)
        self.assertEqual(rp.load_relay_block(p, "a"), {})

    def test_maps_block_keys_to_flag_dests(self):
        p = write("""
            a:
              relay:
                host: 192.168.50.20
                key_file: /tmp/k
                control_port: 9000
                cmd_port: 9001
                cmd_rate_hz: 80
                log_nic: enp233s0
            """)
        self.assertEqual(rp.load_relay_block(p, "a"), {
            "relay_host": "192.168.50.20", "relay_key_file": "/tmp/k", "relay_control_port": 9000,
            "relay_cmd_port": 9001, "relay_cmd_rate_hz": 80.0, "log_nic": "enp233s0"})

    def test_partial_block_only_returns_what_is_set(self):
        p = write("""
            a:
              relay:
                host: 10.0.0.2
            """)
        self.assertEqual(rp.load_relay_block(p, "a"), {"relay_host": "10.0.0.2"})

    def test_unknown_key_fails_loudly(self):
        p = write("""
            a:
              relay:
                hots: 10.0.0.2
            """)
        with self.assertRaises(ValueError):
            rp.load_relay_block(p, "a")

    def test_bad_types_fail(self):
        for body in ("host: 5", "control_port: abc", "control_port: 0", "control_port: 70000", "cmd_rate_hz: -1",
                     "key_file: ''"):
            p = write(f"a:\n  relay:\n    {body}\n")
            with self.assertRaises(ValueError, msg=body):
                rp.load_relay_block(p, "a")

    def test_relay_must_be_a_mapping(self):
        p = write("""
            a:
              relay: 5
            """)
        with self.assertRaises(ValueError):
            rp.load_relay_block(p, "a")

    def test_missing_profile_fails(self):
        p = write("a:\n  low_pass: 5.0\n")
        with self.assertRaises(ValueError):
            rp.load_relay_block(p, "nope")

    def test_host_refuses_the_robot_motor_subnet(self):
        # 192.168.123.x is the robot's wired motor LAN: a relay host there means the modem collides with eth0.
        p = write("a:\n  relay:\n    host: 192.168.123.18\n")
        with self.assertRaises(ValueError):
            rp.load_relay_block(p, "a")


class PrecedenceViaArgparse(unittest.TestCase):
    """The profile block becomes argparse defaults, so an explicit CLI flag must still win."""

    def parser(self):
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--relay_host", default=None)
        ap.add_argument("--relay_control_port", type=int, default=9871)
        ap.add_argument("--relay_cmd_port", type=int, default=9870)
        return ap

    def test_cli_beats_profile_beats_default(self):
        ap = self.parser()
        ap.set_defaults(**{"relay_host": "10.0.0.2", "relay_control_port": 9000})
        a = ap.parse_args(["--relay_control_port", "1234"])
        self.assertEqual((a.relay_host, a.relay_control_port, a.relay_cmd_port), ("10.0.0.2", 1234, 9870))

    def test_every_returned_key_is_a_real_relay_flag_dest(self):
        self.assertEqual(set(rp.BLOCK_TO_DEST.values()), set(rp.RELAY_DESTS))


class ShippedProfiles(unittest.TestCase):
    def test_every_shipped_profile_has_valid_relay_block(self):
        import yaml

        with open(PROFILES) as f:
            names = list(yaml.safe_load(f))
        for n in names:
            rp.load_relay_block(PROFILES, n)  # must not raise

    def test_wireless_profile_exists_with_a_relay_block(self):
        got = rp.load_relay_block(PROFILES, "wireless")
        self.assertIn("relay_host", got)


@unittest.skipIf(dr is None, "deploy_real not importable here")
class WiredParserIgnoresRelayBlock(unittest.TestCase):
    def test_wired_loader_tolerates_relay_key(self):
        valid = {"low_pass"}
        p = write("a:\n  low_pass: 5.0\n  relay:\n    host: 10.0.0.2\n")
        self.assertEqual(dr.load_test_profile(p, "a", valid), {"low_pass": 5.0})

    def test_every_shipped_profile_still_loads_through_the_real_parser(self):
        saved = sys.argv
        try:
            for name in ("cautionary", "confident", "no_limits", "iter4_realistic", "wireless"):
                sys.argv = ["deploy_real.py", "--deploy_profile", name]
                dr.parse_args()
        finally:
            sys.argv = saved


if __name__ == "__main__":
    unittest.main()
