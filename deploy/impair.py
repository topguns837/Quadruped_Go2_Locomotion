# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Network impairment shim for the relay's UDP send paths (test use only).

Wraps a `send(bytes)` function and, according to a profile, drops, delays, reorders or stalls packets, so the
receivers see what real WiFi loss, jitter and burst-release would look like. Deterministic (seeded), and every
decision is returned to the caller for per-packet logging.

Why in our own code and not `tc netem`: netem needs root and kernel modules, would touch the NICs that carry our
own SSH sessions, and cannot report per-packet decisions.

SAFETY: impairments exist only for fake-robot test runs. The relay refuses an impair request unless its DDS interface
is the loopback one, and deploy_real_wireless.py only enables them together with --skip_release. This module does not
enforce that itself (it cannot know), so callers must call check_allowed().

Profile keys (all optional; missing = off):
    loss_p            probability of dropping a packet
    delay_ms          extra delay, uniform in [0, delay_ms]; random delays also reorder packets
    reorder_p         probability of holding a packet back by reorder_hold_ms so it arrives after later ones
    reorder_hold_ms   hold time for reordered packets (default 25 ms, i.e. 2-3 packet periods at 100 Hz)
    stall_ms          every stall_every_s seconds, hold ALL packets for stall_ms, then release them together
    stall_every_s     period of the stalls (first stall one period after the phase starts)
"""

from __future__ import annotations

import heapq
import random
import threading
import time

PASS, DROP, DELAY, REORDER, STALL = "pass", "drop", "delay", "reorder", "stall"

# (start_s, end_s, phase_name, profile). Times are seconds from the start of the session.
SCHEDULES = {
    "soak15": [
        (0, 240, "clean", {}),
        (240, 300, "loss1", {"loss_p": 0.01}),
        (300, 360, "loss3", {"loss_p": 0.03}),
        (360, 420, "loss5", {"loss_p": 0.05}),
        (420, 480, "delay10", {"delay_ms": 10}),
        (480, 540, "delay30", {"delay_ms": 30}),
        (540, 600, "reorder3", {"reorder_p": 0.03, "delay_ms": 5}),
        (600, 720, "stall50", {"stall_ms": 50, "stall_every_s": 10}),
        (720, 900, "recovery", {}),
    ],
}


def build_schedule(name: str, time_scale: float = 1.0, impair_scale: float = 1.0):
    """The named schedule with durations multiplied by `time_scale` (to compress a test) and impairment strength
    multiplied by `impair_scale` (loss/reorder probabilities, delays and stall lengths; probabilities capped at 1)."""
    if name not in SCHEDULES:
        raise ValueError(f"unknown impairment schedule '{name}'; known: {sorted(SCHEDULES)}")
    out = []
    for start, end, phase, prof in SCHEDULES[name]:
        p = dict(prof)
        for k in ("loss_p", "reorder_p"):
            if k in p:
                p[k] = min(1.0, p[k] * impair_scale)
        for k in ("delay_ms", "stall_ms"):
            if k in p:
                p[k] = p[k] * impair_scale
        if "stall_every_s" in p:
            p["stall_every_s"] = p["stall_every_s"] * time_scale
        out.append((start * time_scale, end * time_scale, phase, p))
    return out


def schedule_duration(schedule) -> float:
    return max(end for _, end, _, _ in schedule)


def phase_at(schedule, t: float):
    """(phase_name, profile) active at session time t; the last phase persists past the end."""
    for start, end, phase, prof in schedule:
        if start <= t < end:
            return phase, prof
    last = schedule[-1]
    return last[2], last[3]


# key -> (min, max). Bounds keep a buggy or hostile (but authenticated) peer from asking for absurd impairments.
PROFILE_LIMITS = {
    "loss_p": (0.0, 0.5), "delay_ms": (0.0, 500.0), "reorder_p": (0.0, 0.5), "reorder_hold_ms": (0.0, 500.0),
    "stall_ms": (0.0, 500.0), "stall_every_s": (0.0, 3600.0),
}


def sanitize_profile(profile) -> dict:
    """Validate a profile received over the network. Returns a clean dict of floats; raises ValueError on an unknown
    key, a non-numeric value or a value outside PROFILE_LIMITS."""
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object")
    out = {}
    for k, v in profile.items():
        if k not in PROFILE_LIMITS:
            raise ValueError(f"unknown profile key '{k}'")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"profile value for '{k}' must be a number")
        lo, hi = PROFILE_LIMITS[k]
        if not (lo <= float(v) <= hi):
            raise ValueError(f"profile value {k}={v} outside [{lo}, {hi}]")
        out[k] = float(v)
    if out.get("stall_ms", 0.0) > 0.0 and out.get("stall_every_s", 0.0) < 0.5:
        raise ValueError("stall_ms needs stall_every_s >= 0.5")
    return out


def check_allowed(network_interface: str) -> None:
    """Raise unless impairments are safe: only on the loopback fake-robot setup."""
    if network_interface != "lo":
        raise PermissionError(
            f"network impairments are only allowed on the loopback fake-robot setup (interface 'lo'), "
            f"not '{network_interface}'"
        )


class Impairment:
    """Callable shim: impairment.send(packet) -> (action, extra_delay_ms). Clean profile = direct synchronous send."""

    def __init__(self, send_fn, seed: int = 0, clock=time.monotonic):
        self._send = send_fn
        self._rng = random.Random(seed)
        self._clock = clock
        self.phase = "clean"
        self.profile = {}
        self._phase_start = clock()
        self._heap = []
        self._seq = 0
        self._cv = threading.Condition()
        self._worker = None
        self._closed = False
        self.counts = {PASS: 0, DROP: 0, DELAY: 0, REORDER: 0, STALL: 0}

    def set_profile(self, phase: str, profile: dict) -> None:
        with self._cv:
            self.phase = phase
            self.profile = dict(profile or {})
            self._phase_start = self._clock()

    # ---- decision ---------------------------------------------------------------------------------------

    def _decide(self, now: float):
        """Returns (action, delay_s) for one packet under the current profile."""
        p = self.profile
        if not p:
            return PASS, 0.0
        if p.get("loss_p", 0.0) > 0.0 and self._rng.random() < p["loss_p"]:
            return DROP, 0.0
        action, delay = PASS, 0.0
        if p.get("delay_ms", 0.0) > 0.0:
            delay = self._rng.uniform(0.0, p["delay_ms"]) / 1000.0
            action = DELAY
        if p.get("reorder_p", 0.0) > 0.0 and self._rng.random() < p["reorder_p"]:
            delay = max(delay, p.get("reorder_hold_ms", 25.0) / 1000.0)
            action = REORDER
        every, stall = p.get("stall_every_s", 0.0), p.get("stall_ms", 0.0) / 1000.0
        if every > 0.0 and stall > 0.0:
            t_rel = now - self._phase_start
            k = int(t_rel // every)
            if k >= 1 and (t_rel - k * every) < stall:
                stall_end = self._phase_start + k * every + stall
                delay = max(delay, stall_end - now)
                action = STALL
        return action, delay

    # ---- send -------------------------------------------------------------------------------------------

    def send(self, packet: bytes):
        with self._cv:
            now = self._clock()
            action, delay = self._decide(now)
            self.counts[action] += 1
            if action == DROP:
                return DROP, 0.0
            if delay <= 0.0 and not self._heap:
                direct = True
            else:
                direct = False
                self._seq += 1
                heapq.heappush(self._heap, (now + delay, self._seq, packet))
                self._ensure_worker()
                self._cv.notify()
        if direct:
            self._send(packet)
        return action, delay * 1000.0

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="impair-release", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._heap and not self._closed:
                    self._cv.wait()
                if self._closed and not self._heap:
                    return
                due, _, pkt = self._heap[0]
                wait = due - self._clock()
                if wait > 0 and not self._closed:
                    self._cv.wait(wait)
                    continue
                heapq.heappop(self._heap)
            try:
                self._send(pkt)
            except OSError:
                pass

    def close(self) -> None:
        """Release anything still held (immediately) and stop the worker."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
