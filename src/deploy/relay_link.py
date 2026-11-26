# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wire protocol and desktop-side client for the Jetson relay (src/deploy/relay_jetson.py).

Why this exists: the policy runs on the desktop, but the robot's low-level DDS only works on the robot's own
192.168.123.0/24 network, which the desktop is not on when it is wireless. So a relay on the Jetson (which IS on
that network, through its untouched eth0) speaks DDS to the robot and plain UDP/TCP to the desktop. This module
holds everything both ends share, and the desktop's client. It imports neither torch nor the Unitree SDK, so
it runs anywhere and is unit-testable on its own.

Channels:
  * UDP STATE  relay -> desktop, ~100 Hz: the fields deploy_real.LatestRobotState holds, plus relay status and an
    echo of the last command's sequence/timestamp (the desktop turns that into a round-trip time).
  * UDP CMD    desktop -> relay, ~100 Hz: 12 joint targets (SDK joint order), kp, kd.
  * TCP control (JSON lines): authentication, release of the onboard controller (which needs an operator
    confirmation on the desktop), damp, shutdown.

Security: the relay drives motors and listens on a shared lab network, so nothing is accepted unauthenticated.
Both machines hold the same secret key (a file, see init_key). The TCP handshake is a mutual HMAC challenge, and
every UDP packet carries a truncated HMAC-SHA256 tag over header+body, so a stranger on the network can neither
command the robot nor feed the desktop fake state. The tag also binds a per-session id, so packets recorded
from an earlier session are rejected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import socket
import struct
import threading
import time
from collections import deque
from types import SimpleNamespace

import impair as imp
import runlog as rlg

PROTOCOL_VERSION = 1
NUM_JOINTS = 12
MAGIC = b"G2R"
TYPE_STATE = 1
TYPE_CMD = 2
TAG_LEN = 8

DEFAULT_CONTROL_PORT = 9871
DEFAULT_CMD_PORT = 9870

# Relay modes, reported in every STATE packet.
MODE_STREAMING = 0  # state only; nothing is being published to the robot
MODE_ACTIVE = 1  # LowCmd is being published (after release)
MODE_DAMPED = 2  # latched damping after a stall, a bad packet stream, a fault or a damp request

_HDR = struct.Struct("<3sBB")  # magic, protocol version, packet type
_STATE_BODY = struct.Struct("<IIQIQBBH" + "12f12f12f4f3f3f" + "40s")
_CMD_BODY = struct.Struct("<IIQ" + "12f" + "ff" + "B")

# Largest forward jump in the u32 sequence that still counts as "newer" (anything else is treated as old).
_SEQ_HALF = 0x7FFFFFFF


# --------------------------------------------------------------------------------------------------------
# Keys and authentication
# --------------------------------------------------------------------------------------------------------


def init_key(path: str) -> None:
    """Create a new random secret at `path`, readable only by the owner. Refuses to overwrite an existing key."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(32) + "\n")


def load_key(path: str) -> bytes:
    with open(path) as f:
        key = bytes.fromhex(f.read().strip())
    if len(key) < 16:
        raise ValueError(f"Key in {path} is too short ({len(key)} bytes); make a new one with --init_key.")
    return key


def _mac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def challenge_response(key: bytes, nonce_hex: str, label: str) -> str:
    """HMAC over a nonce and a direction label, so a reply to one direction can't be replayed as the other."""
    return _mac(key, label.encode() + bytes.fromhex(nonce_hex)).hex()


def new_nonce() -> str:
    return secrets.token_hex(16)


def seq_is_newer(seq: int, last: int | None) -> bool:
    """True if `seq` is ahead of `last` (u32 wraparound aware). The first packet of a session is always newer."""
    if last is None:
        return True
    return 0 < ((seq - last) & 0xFFFFFFFF) <= _SEQ_HALF


# --------------------------------------------------------------------------------------------------------
# Packets
# --------------------------------------------------------------------------------------------------------


def _seal(key: bytes, ptype: int, body: bytes) -> bytes:
    head = _HDR.pack(MAGIC, PROTOCOL_VERSION, ptype)
    return head + body + _mac(key, head + body)[:TAG_LEN]


def _open(key: bytes, data: bytes, ptype: int, body_size: int) -> bytes | None:
    """Returns the body if the packet is the right size, version and type and its tag verifies, else None."""
    if len(data) != _HDR.size + body_size + TAG_LEN:
        return None
    magic, version, got_type = _HDR.unpack_from(data, 0)
    if magic != MAGIC or version != PROTOCOL_VERSION or got_type != ptype:
        return None
    tag = data[-TAG_LEN:]
    if not hmac.compare_digest(tag, _mac(key, data[:-TAG_LEN])[:TAG_LEN]):
        return None
    return data[_HDR.size:-TAG_LEN]


def pack_state(
    key: bytes, session: int, seq: int, t_ns: int, echo_seq: int, echo_t_ns: int, mode: int, flags: int,
    cmd_age_ms: int, joint_pos, joint_vel, tau_est, quat_wxyz, gyro, sport_vel, wireless_remote: bytes,
) -> bytes:
    body = _STATE_BODY.pack(
        session & 0xFFFFFFFF, seq & 0xFFFFFFFF, t_ns & 0xFFFFFFFFFFFFFFFF, echo_seq & 0xFFFFFFFF,
        echo_t_ns & 0xFFFFFFFFFFFFFFFF, mode, flags, min(max(int(cmd_age_ms), 0), 0xFFFF),
        *joint_pos, *joint_vel, *tau_est, *quat_wxyz, *gyro, *sport_vel, bytes(wireless_remote)[:40].ljust(40, b"\0"),
    )
    return _seal(key, TYPE_STATE, body)


def unpack_state(key: bytes, data: bytes) -> dict | None:
    body = _open(key, data, TYPE_STATE, _STATE_BODY.size)
    if body is None:
        return None
    v = _STATE_BODY.unpack(body)
    return {
        "session": v[0], "seq": v[1], "t_ns": v[2], "echo_seq": v[3], "echo_t_ns": v[4], "mode": v[5],
        "flags": v[6], "cmd_age_ms": v[7], "joint_pos": list(v[8:20]), "joint_vel": list(v[20:32]),
        "tau_est": list(v[32:44]), "quat_wxyz": tuple(v[44:48]), "gyro": tuple(v[48:51]),
        "sport_vel": tuple(v[51:54]), "wireless_remote": bytes(v[54]),
    }


def pack_cmd(key: bytes, session: int, seq: int, t_ns: int, target_sdk_order, kp: float, kd: float) -> bytes:
    body = _CMD_BODY.pack(
        session & 0xFFFFFFFF, seq & 0xFFFFFFFF, t_ns & 0xFFFFFFFFFFFFFFFF, *target_sdk_order, kp, kd, 0
    )
    return _seal(key, TYPE_CMD, body)


def unpack_cmd(key: bytes, data: bytes) -> dict | None:
    body = _open(key, data, TYPE_CMD, _CMD_BODY.size)
    if body is None:
        return None
    v = _CMD_BODY.unpack(body)
    return {"session": v[0], "seq": v[1], "t_ns": v[2], "target": list(v[3:15]), "kp": v[15], "kd": v[16]}


def all_finite(values) -> bool:
    return all(math.isfinite(x) for x in values)


# --------------------------------------------------------------------------------------------------------
# TCP control channel (JSON lines)
# --------------------------------------------------------------------------------------------------------


class JsonLineSocket:
    """Newline-delimited JSON over a connected TCP socket. send() is thread-safe; recv() is for one reader."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._buf = b""
        self._send_lock = threading.Lock()
        # Latency/liveness hints only: never let an unsupported option (e.g. on a Unix socket in tests) break the link.
        for level, opt in ((socket.IPPROTO_TCP, socket.TCP_NODELAY), (socket.SOL_SOCKET, socket.SO_KEEPALIVE)):
            try:
                sock.setsockopt(level, opt, 1)
            except OSError:
                pass

    def send(self, obj: dict) -> None:
        data = json.dumps(obj, separators=(",", ":")).encode() + b"\n"
        with self._send_lock:
            self._sock.sendall(data)

    def recv(self, timeout: float | None) -> dict | None:
        """Next message, or None on timeout. Raises ConnectionError if the peer closed the connection."""
        while b"\n" not in self._buf:
            self._sock.settimeout(timeout)
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                return None
            if not chunk:
                raise ConnectionError("control connection closed by peer")
            self._buf += chunk
            if len(self._buf) > 65536:
                raise ConnectionError("control message too large")
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line.decode())

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------------------------------------
# Desktop-side client
# --------------------------------------------------------------------------------------------------------


class RelayError(RuntimeError):
    pass


class RelayClient:
    """The desktop's end of the relay link. Feeds a deploy_real.LatestRobotState (via its existing
    update_from_low_state / update_from_sportmode_state methods, plus its `ready` flag) from STATE packets, and
    streams the contents of a LowCmdTarget (get/get_kp/get_kd) to the relay as CMD packets. `isaac_to_sdk` maps the policy's joint order to the SDK order the relay expects
    (the relay never needs to know the joint maps)."""

    def __init__(self, host: str, key: bytes, state, isaac_to_sdk: list[int], control_port: int = DEFAULT_CONTROL_PORT,
                 cmd_port: int = DEFAULT_CMD_PORT, cmd_rate_hz: float = 100.0, run_log=None,
                 allow_impairment: bool = False, impairment_seed: int = 0):
        """run_log: an optional runlog.RunLog; when given, every command sent and every state packet received is
        written to tx_cmd.csv / rx_state.csv. allow_impairment: lets set_impairment() inject loss/delay/reorder/
        stalls into the COMMAND path (the relay must agree, and only does on the loopback fake-robot setup)."""
        self.host = host
        self.key = key
        self.state = state
        self.isaac_to_sdk = isaac_to_sdk
        self.control_port = control_port
        self.cmd_port = cmd_port
        self.cmd_period_s = 1.0 / cmd_rate_hz
        self.session: int | None = None
        self.relay_mode = MODE_STREAMING
        self.relay_flags = 0
        self.relay_cmd_age_ms = 0
        self.relay_damp_reason = ""
        self._ctl: JsonLineSocket | None = None
        self._udp_state: socket.socket | None = None
        self._udp_cmd: socket.socket | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_state_seq: int | None = None
        self._cmd_seq = 0
        self._cmd_lock = threading.Lock()
        self._rtts_ms: deque[float] = deque(maxlen=400000)
        self._events: deque[dict] = deque()
        self._events_cv = threading.Condition()
        self._target_buf = None
        self._closed_reason: str | None = None
        self.bad_packets = 0
        self.stale_packets = 0
        self.log = run_log if run_log is not None else rlg.RunLog(None, "client")
        self._tx_log = self.log.csv("tx_cmd.csv", ["t_mono", "t_wall", "seq", "action", "delay_ms", "kp", "kd", "phase"])
        self._rx_log = self.log.csv("rx_state.csv", [
            "t_mono", "t_wall", "seq", "status", "gap_ms", "echo_seq", "rtt_ms", "cmd_age_ms", "mode", "flags", "phase"])
        self.phase = "clean"
        self._allow_impairment = allow_impairment
        self._impairment_seed = impairment_seed
        self._impair = None
        self._impair_ack = None
        self._impair_ack_evt = threading.Event()
        self._last_state_mono: float | None = None
        thr = (30.0, 50.0, 100.0, 150.0)
        self.hist_rtt = rlg.Histogram(thresholds=thr)
        self.hist_gap = rlg.Histogram(thresholds=thr)
        self._win = {"rtt": rlg.Histogram(thresholds=thr), "gap": rlg.Histogram(thresholds=thr), "ok": 0, "stale": 0,
                     "bad": 0, "tx": 0, "tx_dropped": 0}
        self._win_lock = threading.Lock()
        self.cmds_sent = 0
        self.cmds_dropped_by_impairment = 0
        self.clock_offset_s: float | None = None
        self.clock_offset_unc_s: float | None = None

    # ---- connection -------------------------------------------------------------------------------------

    def connect(self, timeout_s: float = 5.0) -> None:
        sock = socket.create_connection((self.host, self.control_port), timeout=timeout_s)
        self._ctl = JsonLineSocket(sock)
        msg = self._ctl.recv(timeout_s)
        if not msg or msg.get("type") != "challenge":
            raise RelayError(f"Relay at {self.host}:{self.control_port} did not start with a challenge: {msg!r}")
        if msg.get("version") != PROTOCOL_VERSION:
            raise RelayError(f"Relay speaks protocol {msg.get('version')}, this client speaks {PROTOCOL_VERSION}.")

        self._udp_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_state.bind(("0.0.0.0", 0))
        self._udp_state.settimeout(0.2)
        client_nonce = new_nonce()
        t_hello = time.time()
        self._ctl.send({
            "type": "hello", "version": PROTOCOL_VERSION, "client_nonce": client_nonce,
            "response": challenge_response(self.key, msg["nonce"], "client"),
            "state_port": self._udp_state.getsockname()[1],
        })
        ack = self._ctl.recv(timeout_s)
        t_ack = time.time()
        if not ack or ack.get("type") != "hello_ack":
            raise RelayError(f"Relay refused the connection (wrong key?): {ack!r}")
        expected = challenge_response(self.key, client_nonce, "relay")
        if not hmac.compare_digest(str(ack.get("mutual", "")), expected):
            raise RelayError("Relay failed mutual authentication: it does not hold the same key.")
        self.session = int(ack["session"])
        self.cmd_port = int(ack.get("cmd_port", self.cmd_port))
        self._udp_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if ack.get("relay_wall") is not None:
            # NTP-style estimate: the relay stamped its wall clock somewhere between our send and our receive, so
            # assume the midpoint; the uncertainty is half the round trip. Positive = the relay's clock is ahead.
            self.clock_offset_s = float(ack["relay_wall"]) - (t_hello + t_ack) / 2.0
            self.clock_offset_unc_s = (t_ack - t_hello) / 2.0
            self.log.event(f"[clock] relay clock minus desktop clock = {self.clock_offset_s:+.3f} s "
                           f"(+/- {self.clock_offset_unc_s * 1000:.1f} ms)")
        if self._allow_impairment:
            self._impair = imp.Impairment(
                lambda pkt: self._udp_cmd.sendto(pkt, (self.host, self.cmd_port)), seed=self._impairment_seed)

        for name, fn in (("relay-state-rx", self._state_loop), ("relay-ctl-rx", self._control_loop)):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def _state_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._udp_state.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            if addr[0] != self.host:
                self.bad_packets += 1
                with self._win_lock:
                    self._win["bad"] += 1
                continue
            pkt = unpack_state(self.key, data)
            if pkt is None or pkt["session"] != self.session:
                self.bad_packets += 1
                with self._win_lock:
                    self._win["bad"] += 1
                continue
            now_mono = time.monotonic()
            if not seq_is_newer(pkt["seq"], self._last_state_seq):
                self.stale_packets += 1
                with self._win_lock:
                    self._win["stale"] += 1
                self._rx_log.row(self.log.clock.mono(), time.time(), pkt["seq"], "stale", math.nan, pkt["echo_seq"],
                                 math.nan, pkt["cmd_age_ms"], pkt["mode"], pkt["flags"], self.phase)
                continue
            self._last_state_seq = pkt["seq"]
            self.relay_mode = pkt["mode"]
            self.relay_flags = pkt["flags"]
            self.relay_cmd_age_ms = pkt["cmd_age_ms"]
            rtt_ms = math.nan
            if pkt["echo_t_ns"]:
                rtt_ms = (time.monotonic_ns() - pkt["echo_t_ns"]) / 1e6
                self._rtts_ms.append(rtt_ms)
            gap_ms = math.nan if self._last_state_mono is None else (now_mono - self._last_state_mono) * 1000.0
            self._last_state_mono = now_mono
            with self._win_lock:
                self._win["ok"] += 1
                if not math.isnan(rtt_ms):
                    self.hist_rtt.add(rtt_ms)
                    self._win["rtt"].add(rtt_ms)
                if not math.isnan(gap_ms):
                    self.hist_gap.add(gap_ms)
                    self._win["gap"].add(gap_ms)
            self._rx_log.row(self.log.clock.mono(), time.time(), pkt["seq"], "ok", gap_ms, pkt["echo_seq"], rtt_ms,
                             pkt["cmd_age_ms"], pkt["mode"], pkt["flags"], self.phase)
            # Feed the existing LatestRobotState through its own update methods, with objects shaped like the
            # DDS messages, so deploy_real.py needs no change (and the staleness/gap tracking is reused as is).
            low = SimpleNamespace(
                motor_state=[
                    SimpleNamespace(q=q, dq=dq, tau_est=tau)
                    for q, dq, tau in zip(pkt["joint_pos"], pkt["joint_vel"], pkt["tau_est"])
                ],
                wireless_remote=pkt["wireless_remote"],
                imu_state=SimpleNamespace(quaternion=pkt["quat_wxyz"], gyroscope=pkt["gyro"]),
            )
            self.state.update_from_low_state(low)
            self.state.update_from_sportmode_state(SimpleNamespace(velocity=pkt["sport_vel"]))

    def _control_loop(self) -> None:
        try:
            while not self._stop.is_set():
                msg = self._ctl.recv(0.5)
                if msg is None:
                    continue
                if msg.get("type") == "damped":
                    self.relay_damp_reason = str(msg.get("reason", ""))
                    self.log.event(f"[relay] damped: {self.relay_damp_reason}")
                if msg.get("type") == "impair_ack":
                    self._impair_ack = msg
                    self._impair_ack_evt.set()
                    continue
                with self._events_cv:
                    self._events.append(msg)
                    self._events_cv.notify_all()
        except (ConnectionError, OSError, json.JSONDecodeError) as e:
            if not self._stop.is_set():
                self._closed_reason = str(e)
            with self._events_cv:
                self._events.append({"type": "closed", "message": str(e)})
                self._events_cv.notify_all()

    def _next_event(self, timeout_s: float) -> dict | None:
        end = time.monotonic() + timeout_s
        with self._events_cv:
            while not self._events:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self._events_cv.wait(left)
            return self._events.popleft()

    # ---- operations -------------------------------------------------------------------------------------

    def wait_for_state(self, timeout_s: float = 10.0) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if self.state.ready:
                return True
            time.sleep(0.02)
        return False

    def request_release(self, stand_down: bool, skip: bool, on_need_confirm, timeout_s: float = 600.0) -> None:
        """Ask the relay to release the onboard controller (a DDS RPC only the relay can make). If a mode is
        active the relay asks for operator confirmation first; `on_need_confirm` (the existing checklist prompt)
        runs here on the desktop and its return is the confirmation. Raises RelayError if the release failed."""
        self._ctl.send({"type": "release", "stand_down": bool(stand_down), "skip": bool(skip)})
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            msg = self._next_event(1.0)
            if msg is None:
                continue
            kind = msg.get("type")
            if kind == "need_confirm":
                on_need_confirm()
                self._ctl.send({"type": "confirm"})
            elif kind == "release_done":
                if not msg.get("ok"):
                    raise RelayError(f"Relay could not release the onboard controller: {msg.get('message')}")
                return
            elif kind in ("error", "closed"):
                raise RelayError(f"Relay link failed during release: {msg.get('message')}")
        raise RelayError("Timed out waiting for the relay to finish the release.")

    def start_command_stream(self, target_buf) -> None:
        self._target_buf = target_buf
        t = threading.Thread(target=self._cmd_loop, name="relay-cmd-tx", daemon=True)
        t.start()
        self._threads.append(t)

    def _cmd_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            self._send_cmd_once()
            next_t += self.cmd_period_s
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def _send_cmd_once(self, force_damp: bool = False) -> None:
        if self.session is None or self._target_buf is None:
            return
        target_isaac = self._target_buf.get()
        target_sdk = [0.0] * NUM_JOINTS
        for isaac_i, sdk_i in enumerate(self.isaac_to_sdk):
            target_sdk[sdk_i] = target_isaac[isaac_i]
        kp, kd = (0.0, self._target_buf.get_kd()) if force_damp else (self._target_buf.get_kp(), self._target_buf.get_kd())
        with self._cmd_lock:
            self._cmd_seq = (self._cmd_seq + 1) & 0xFFFFFFFF
            seq = self._cmd_seq
            pkt = pack_cmd(self.key, self.session, seq, time.monotonic_ns(), target_sdk, kp, kd)
        action, delay_ms = imp.PASS, 0.0
        try:
            if self._impair is not None:
                action, delay_ms = self._impair.send(pkt)
            else:
                self._udp_cmd.sendto(pkt, (self.host, self.cmd_port))
        except OSError:
            action = "send_error"
        self.cmds_sent += 1
        with self._win_lock:
            self._win["tx"] += 1
            if action == imp.DROP:
                self._win["tx_dropped"] += 1
        if action == imp.DROP:
            self.cmds_dropped_by_impairment += 1
        self._tx_log.row(self.log.clock.mono(), time.time(), seq, action, delay_ms, kp, kd, self.phase)

    def set_impairment(self, phase: str, profile: dict, timeout_s: float = 3.0) -> None:
        """Switch the test impairment profile for BOTH directions. The relay is told first and must agree (it only
        does on the loopback fake-robot setup); only then is the local command-path shim switched, so a relay on a
        real interface can never end up with one-sided impairments. Raises RelayError on refusal."""
        if self._impair is None:
            raise RelayError("impairments were not enabled for this client (allow_impairment=False)")
        self._impair_ack_evt.clear()
        self._ctl.send({"type": "impair", "phase": phase, "profile": profile})
        if not self._impair_ack_evt.wait(timeout_s):
            raise RelayError("the relay did not acknowledge the impairment request")
        if not self._impair_ack.get("ok"):
            raise RelayError(f"the relay refused the impairment request: {self._impair_ack.get('message')}")
        self._impair.set_profile(phase, profile)
        self.phase = phase
        self.log.event(f"[impair] phase={phase} profile={profile}")

    def take_window(self) -> dict:
        """Statistics since the last call (for the 1 Hz metrics row), then reset the window."""
        with self._win_lock:
            w, self._win = self._win, {
                "rtt": rlg.Histogram(thresholds=(30.0, 50.0, 100.0, 150.0)),
                "gap": rlg.Histogram(thresholds=(30.0, 50.0, 100.0, 150.0)),
                "ok": 0, "stale": 0, "bad": 0, "tx": 0, "tx_dropped": 0}
        return w

    def raise_if_unhealthy(self) -> None:
        """Called from the safety monitor each tick. The run is over if the relay latched damping, or if the
        control connection died (the relay damps on its own when that happens, so stopping here is consistent)."""
        if self._closed_reason is not None:
            raise RelayError(f"Relay control connection lost ({self._closed_reason}).")
        if self.relay_mode == MODE_DAMPED:
            raise RelayError(f"Relay is damping the robot ({self.relay_damp_reason or 'reason not reported yet'}).")

    def rtt_stats(self) -> dict:
        s = sorted(self._rtts_ms)
        if not s:
            return {"count": 0}
        return {"count": len(s), "median_ms": s[len(s) // 2], "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))], "max_ms": s[-1]}

    def damp_and_close(self) -> None:
        """Exit path: tell the relay to damp, keep sending damping commands briefly, then close. Safe to call twice."""
        if self._ctl is not None:
            try:
                self._ctl.send({"type": "damp"})
            except OSError:
                pass
        if self._target_buf is not None:
            for _ in range(30):
                self._send_cmd_once(force_damp=True)
                time.sleep(0.01)
        self.close()

    def close(self) -> None:
        self._stop.set()
        if self._impair is not None:
            self._impair.close()
        if self._ctl is not None:
            try:
                self._ctl.send({"type": "shutdown"})
            except OSError:
                pass
            self._ctl.close()
        for s in (self._udp_state, self._udp_cmd):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
