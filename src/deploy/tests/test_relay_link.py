# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for src/deploy/relay_link.py (wire protocol, authentication, sequence handling). No robot, no DDS,
no torch. Run:  python3 -m unittest discover -s src/deploy/tests -v
"""

import os
import socket
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import relay_link as rl  # noqa: E402

KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))


def _state_pkt(key=KEY, session=7, seq=1, **over):
    kw = dict(
        session=session, seq=seq, t_ns=123456789, echo_seq=4, echo_t_ns=99, mode=rl.MODE_ACTIVE, flags=1, cmd_age_ms=12,
        joint_pos=[0.1 * i for i in range(12)], joint_vel=[-0.01 * i for i in range(12)],
        tau_est=[0.5 * i for i in range(12)], quat_wxyz=(1.0, 0.0, 0.0, 0.0), gyro=(0.1, 0.2, 0.3),
        sport_vel=(0.4, 0.5, 0.6), wireless_remote=bytes(range(40)),
    )
    kw.update(over)
    return rl.pack_state(key, **kw), kw


class PacketTests(unittest.TestCase):
    def test_state_roundtrip(self):
        data, kw = _state_pkt()
        got = rl.unpack_state(KEY, data)
        self.assertIsNotNone(got)
        self.assertEqual(got["session"], 7)
        self.assertEqual(got["seq"], 1)
        self.assertEqual(got["mode"], rl.MODE_ACTIVE)
        self.assertEqual(got["wireless_remote"], bytes(range(40)))
        for a, b in zip(got["joint_pos"], kw["joint_pos"]):
            self.assertAlmostEqual(a, b, places=5)  # float32 on the wire
        for a, b in zip(got["tau_est"], kw["tau_est"]):
            self.assertAlmostEqual(a, b, places=5)
        self.assertEqual(got["quat_wxyz"], (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(got["echo_t_ns"], 99)

    def test_cmd_roundtrip(self):
        target = [0.01 * i - 0.05 for i in range(12)]
        got = rl.unpack_cmd(KEY, rl.pack_cmd(KEY, 9, 3, 55, target, 25.0, 0.5))
        self.assertIsNotNone(got)
        self.assertEqual((got["session"], got["seq"], got["t_ns"]), (9, 3, 55))
        for a, b in zip(got["target"], target):
            self.assertAlmostEqual(a, b, places=5)
        self.assertAlmostEqual(got["kp"], 25.0)
        self.assertAlmostEqual(got["kd"], 0.5)

    def test_wrong_key_rejected(self):
        data, _ = _state_pkt()
        self.assertIsNone(rl.unpack_state(OTHER_KEY, data))
        self.assertIsNone(rl.unpack_cmd(OTHER_KEY, rl.pack_cmd(KEY, 1, 1, 1, [0.0] * 12, 1.0, 1.0)))

    def test_any_single_bit_flip_rejected(self):
        data = rl.pack_cmd(KEY, 1, 1, 1, [0.1] * 12, 25.0, 0.5)
        for i in range(len(data)):
            tampered = bytearray(data)
            tampered[i] ^= 0x01
            self.assertIsNone(rl.unpack_cmd(KEY, bytes(tampered)), f"flip in byte {i} was accepted")

    def test_wrong_length_and_type_rejected(self):
        data = rl.pack_cmd(KEY, 1, 1, 1, [0.0] * 12, 1.0, 1.0)
        self.assertIsNone(rl.unpack_cmd(KEY, data[:-1]))
        self.assertIsNone(rl.unpack_cmd(KEY, data + b"\0"))
        self.assertIsNone(rl.unpack_state(KEY, data))  # a CMD is not a STATE
        self.assertIsNone(rl.unpack_cmd(KEY, b""))
        self.assertIsNone(rl.unpack_cmd(KEY, os.urandom(len(data))))

    def test_wrong_version_rejected(self):
        data = bytearray(rl.pack_cmd(KEY, 1, 1, 1, [0.0] * 12, 1.0, 1.0))
        data[3] = rl.PROTOCOL_VERSION + 1  # version byte; the tag no longer matches either, both must reject
        self.assertIsNone(rl.unpack_cmd(KEY, bytes(data)))

    def test_session_is_bound_by_the_tag(self):
        old = rl.pack_cmd(KEY, 1, 5, 1, [0.0] * 12, 1.0, 1.0)
        got = rl.unpack_cmd(KEY, old)
        self.assertEqual(got["session"], 1)  # the caller compares this with the live session id and drops it

    def test_short_remote_is_padded_not_crashing(self):
        data, _ = _state_pkt(wireless_remote=b"\x01\x02")
        self.assertEqual(rl.unpack_state(KEY, data)["wireless_remote"], b"\x01\x02" + bytes(38))

    def test_all_finite(self):
        self.assertTrue(rl.all_finite([0.0, 1.5, -2.0]))
        self.assertFalse(rl.all_finite([0.0, float("nan")]))
        self.assertFalse(rl.all_finite([float("inf")]))


class SequenceTests(unittest.TestCase):
    def test_first_packet_is_newer(self):
        self.assertTrue(rl.seq_is_newer(0, None))

    def test_ordering(self):
        self.assertTrue(rl.seq_is_newer(11, 10))
        self.assertFalse(rl.seq_is_newer(10, 10))  # duplicate
        self.assertFalse(rl.seq_is_newer(9, 10))  # old

    def test_wraparound(self):
        self.assertTrue(rl.seq_is_newer(2, 0xFFFFFFFE))
        self.assertFalse(rl.seq_is_newer(0xFFFFFFFE, 2))


class AuthTests(unittest.TestCase):
    def test_challenge_response_is_direction_bound(self):
        nonce = rl.new_nonce()
        a = rl.challenge_response(KEY, nonce, "client")
        b = rl.challenge_response(KEY, nonce, "relay")
        self.assertNotEqual(a, b)
        self.assertEqual(a, rl.challenge_response(KEY, nonce, "client"))
        self.assertNotEqual(a, rl.challenge_response(OTHER_KEY, nonce, "client"))

    def test_key_file_roundtrip_and_permissions(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "relay.key")
            rl.init_key(path)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertGreaterEqual(len(rl.load_key(path)), 32)
            with self.assertRaises(FileExistsError):
                rl.init_key(path)  # never overwrite an existing key

    def test_short_key_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "k")
            with open(path, "w") as f:
                f.write("abcd\n")
            with self.assertRaises(ValueError):
                rl.load_key(path)


class JsonLineTests(unittest.TestCase):
    def test_roundtrip_split_and_close(self):
        a, b = socket.socketpair()
        ja, jb = rl.JsonLineSocket(a), rl.JsonLineSocket(b)
        ja.send({"type": "x", "n": 1})
        ja.send({"type": "y"})
        self.assertEqual(jb.recv(1.0), {"type": "x", "n": 1})
        self.assertEqual(jb.recv(1.0), {"type": "y"})
        self.assertIsNone(jb.recv(0.05))  # timeout, not an error
        ja.close()
        with self.assertRaises(ConnectionError):
            jb.recv(1.0)
        jb.close()

    def test_partial_line_waits_for_the_rest(self):
        a, b = socket.socketpair()
        jb = rl.JsonLineSocket(b)
        a.sendall(b'{"type":"sp')
        self.assertIsNone(jb.recv(0.05))
        a.sendall(b'lit"}\n')
        self.assertEqual(jb.recv(1.0), {"type": "split"})
        a.close()
        jb.close()


if __name__ == "__main__":
    unittest.main()
