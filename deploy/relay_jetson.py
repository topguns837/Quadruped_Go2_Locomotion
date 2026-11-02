# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-side relay: the Jetson's half of the wireless deployment (the desktop half is deploy_real_wireless.py).

The Jetson is on the robot's 192.168.123.0/24 network through its untouched eth0, so it can speak the robot's
low-level DDS; the desktop cannot, because it is on WiFi. This process bridges the two at the application level:
  * robot -> desktop: subscribes rt/lowstate and rt/sportmodestate, sends the values as signed UDP packets;
  * desktop -> robot: receives signed UDP joint targets and republishes them as LowCmd at 500 Hz;
  * releases the robot's onboard controller on request (a DDS RPC only this side can make), after the operator
    confirms on the desktop.
It reuses deploy_real.py's own tested helpers (release_motion_control, init_low_cmd, LowCmdTarget,
publish_low_cmd, tether_limit) rather than copying them. It needs no torch, so on the Jetson (which has none) a
stub module satisfies deploy_real's module-level `import torch`; nothing the relay calls ever uses it.

Safety lives on THIS side, independent of the desktop, because the WiFi link is the thing most likely to fail:
  * silence watchdog: no valid command for --silence_damp_s -> latched damping (kp=0, kd=3), then publishing
    stops, mirroring deploy_real.py's exit sequence;
  * a lost control connection, a stalled robot LowState, a stop request, SIGINT/SIGTERM all do the same;
  * every command must be authenticated (shared key), belong to the current session, come from the desktop that
    opened the control connection, be newer than the last, be finite, and have kp/kd in range -- an
    authenticated-but-invalid command is a desktop bug and damps immediately;
  * every target is limited to --max_target_dev_deg of the ACTUAL joint position (deploy_real.tether_limit), so
    PD torque stays bounded no matter what arrives, looser than the desktop's own tether so it only acts on
    anomalies.
There is deliberately no "resume": once damped, the run is over; start a new session to try again.

Usage on the Jetson (venv active; the login profile puts ROS libraries on LD_LIBRARY_PATH, so set it explicitly):
    export LD_LIBRARY_PATH=$HOME/cyclonedds_ws/install/cyclonedds/lib
    python deploy/relay_jetson.py --network_interface eth0 --key_file ~/.go2_relay_key \\
        --require_ddsc_dir ~/cyclonedds_ws/install/cyclonedds/lib
    python deploy/relay_jetson.py --init_key ~/.go2_relay_key      # once, then copy the file to the desktop
Before a real run, stop the ROS node that can send sport commands (see deploy.md): go2_highroscontrol.
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import signal
import socket
import sys
import threading
import time
import types

try:
    import torch  # noqa: F401  (present on the desktop; absent on the Jetson)
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deploy_real as dr  # noqa: E402
import impair as imp  # noqa: E402
import relay_link as rl  # noqa: E402
import runlog as rlg  # noqa: E402

DAMP_KD = dr.HOLD_DAMPING_KD
IDENTITY_MAP = list(range(rl.NUM_JOINTS))
JOINT_LABELS = [str(i) for i in range(rl.NUM_JOINTS)]

METRIC_COLUMNS = [
    "t_mono", "t_wall", "phase", "mode", "cmds_ok", "cmds_stale", "cmds_invalid", "cmds_auth_rejected", "cmds_ignored",
    "cmd_gap_n", "cmd_gap_p50_ms", "cmd_gap_p99_ms", "cmd_gap_max_ms", "cmd_gap_n30", "cmd_gap_n50", "cmd_gap_n100",
    "cmd_gap_n150", "states_sent", "pub_n", "pub_p50_ms", "pub_p99_ms", "pub_max_ms", "pub_n4", "pub_n10", "pub_n20",
    "pubcall_p99_ms", "pubcall_max_ms", "lowstate_age_ms", "cpu_pct", "load1", "temp_c", "wifi_link", "wifi_level_dbm",
    "wifi_noise_dbm", "wlan_rx_pkts", "wlan_tx_pkts", "wlan_rx_errs", "wlan_tx_errs", "wlan_rx_drop", "wlan_tx_drop",
    "metrics_cost_ms",
]
CMD_GAP_THRESHOLDS = (30.0, 50.0, 100.0, 150.0)


def loaded_ddsc_paths(maps_text: str) -> list[str]:
    """Real paths of every libddsc mapped into this process, from /proc/self/maps text. Order-preserving."""
    paths: list[str] = []
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and "libddsc.so" in os.path.basename(parts[5]):
            path = os.path.realpath(parts[5].split(" (deleted)")[0])
            if path not in paths:
                paths.append(path)
    return paths


def check_ddsc(required_dir: str | None) -> list[str]:
    """Print which CycloneDDS C library is loaded and, if `required_dir` is given, refuse to continue unless every
    loaded libddsc lives there. Why: on the Jetson the Python bindings are built against the Unitree workspace's
    0.10.2, but a login-profile LD_LIBRARY_PATH can make the loader pick a different /usr/local/lib/libddsc instead.
    Reading DDS works with the wrong one; the first WRITE segfaults (observed). A relay that dies while driving the
    motors cannot send its damping burst, so it must never start on a mismatched library."""
    try:
        with open("/proc/self/maps") as f:
            paths = loaded_ddsc_paths(f.read())
    except OSError:
        paths = []
    print(f"[RELAY] libddsc loaded from: {paths or 'unknown (no /proc/self/maps)'}", flush=True)
    if required_dir:
        want = os.path.realpath(os.path.expanduser(required_dir))
        if not paths or any(os.path.dirname(p) != want for p in paths):
            raise RuntimeError(
                f"Wrong CycloneDDS library: loaded {paths or 'none found'}, required directory {want}. Start the "
                f"relay with  LD_LIBRARY_PATH={want}  (and nothing else on it) so the bindings use the library "
                "they were built against."
            )
    return paths


class Relay:
    def __init__(self, args, cfg, key: bytes):
        self.args = args
        self.cfg = cfg
        self.key = key
        self.state = dr.LatestRobotState(num_joints=rl.NUM_JOINTS)
        self.stop = threading.Event()
        self.lock = threading.RLock()

        self.mode = rl.MODE_STREAMING
        self.damp_reason = ""
        self.damp_since = 0.0
        self.session: int | None = None
        self.peer_ip: str | None = None
        self.client_state_addr: tuple[str, int] | None = None
        self.ctl: rl.JsonLineSocket | None = None
        self.confirm_event = threading.Event()
        self.releasing = False

        self.target_buf = None
        self.low_cmd = None
        self.publish_thread = None
        self.publisher = None
        self.crc = None

        self.last_cmd_time: float | None = None
        self.active_since = 0.0
        self.last_cmd_seq: int | None = None
        self.echo_seq = 0
        self.echo_t_ns = 0
        self.counters = {"cmds": 0, "rejected_auth": 0, "rejected_stale": 0, "rejected_invalid": 0, "tethered": 0, "states_sent": 0}

        self.cmd_sock: socket.socket | None = None
        self.state_sock: socket.socket | None = None

        # --- run logging: every part is a harmless no-op when --log_dir is not given ---
        self.log = rlg.RunLog(getattr(args, "log_dir", None), "relay")
        self.per_packet = getattr(args, "log_level", "full") == "full"
        self.phase = "clean"
        null = rlg._NullCsv
        self.rx_log = self.log.csv("rx_cmd.csv", [
            "t_mono", "t_wall", "seq", "status", "reason", "inter_ms", "kp", "kd", "tethered", "phase"]) if self.per_packet else null()
        self.tx_log = self.log.csv("tx_state.csv", [
            "t_mono", "t_wall", "seq", "lowstate_age_ms", "action", "delay_ms", "mode", "phase"]) if self.per_packet else null()
        self.gap_log = self.log.csv("publish_gaps.csv", ["t_mono", "t_wall", "gap_ms", "phase"])
        self.metrics_log = self.log.csv("metrics_1hz.csv", METRIC_COLUMNS)
        self.hist_cmd_gap = rlg.Histogram(thresholds=CMD_GAP_THRESHOLDS)
        self.win_cmd_gap = rlg.Histogram(thresholds=CMD_GAP_THRESHOLDS)
        self.hist_pub = rlg.Histogram(thresholds=(4.0, 10.0, 20.0))
        self.win_pub = rlg.Histogram(thresholds=(4.0, 10.0, 20.0))
        self.hist_pubcall = rlg.Histogram(thresholds=(1.0, 2.0, 5.0))
        self.win_pubcall = rlg.Histogram(thresholds=(1.0, 2.0, 5.0))
        self.win_counts = {"ok": 0, "stale": 0, "invalid": 0, "auth": 0, "ignored": 0}
        self.win_lock = threading.Lock()
        self.pub_lock = threading.Lock()
        self._pub_last_ns = 0
        self._gap_events: list = []
        self.damp_events: list = []
        self.sessions_started = 0
        self._wifi_iface = getattr(args, "wifi_iface", "wlan0")
        self._metrics_cost_ms = 0.0
        self.log_row_cost_us = math.nan
        # Impairment shim for STATE packets: exists only on the loopback fake-robot setup, never on a real interface.
        self._state_imp = None

    def _win_count(self, key: str) -> None:
        with self.win_lock:
            self.win_counts[key] += 1

    # ---- run logging ------------------------------------------------------------------------------------

    def start_logging(self, argv) -> None:
        """Tee stdout into events.log, write meta.json, benchmark the logger, start the 1 Hz flusher."""
        if not self.log.enabled:
            return
        self.log.tee_stdio()
        here = os.path.dirname(os.path.abspath(__file__))
        files = [os.path.join(here, f) for f in (
            "relay_jetson.py", "relay_link.py", "deploy_real.py", "impair.py", "runlog.py", "fake_robot.py",
            os.path.join("configs", "go2_locomotion.yaml"))]
        meta = rlg.base_meta("relay", getattr(self.args, "run_id", ""), argv, files)
        meta["args"] = {k: v for k, v in vars(self.args).items()}
        meta["config"] = {"kp": self.cfg.kp, "kd": self.cfg.kd, "link_lost_after_s": self.cfg.link_lost_after_s,
                          "control_dt": self.cfg.control_dt}
        meta["wifi_iface"] = self._wifi_iface
        meta["wireless_at_start"] = rlg.wireless_now(self._wifi_iface)
        try:
            import subprocess
            out = subprocess.run(["iwconfig", self._wifi_iface], capture_output=True, text=True, timeout=3)
            meta["iwconfig"] = (out.stdout + out.stderr).strip()
        except Exception as e:  # noqa: BLE001 -- iwconfig may not exist; that is fine
            meta["iwconfig"] = f"unavailable: {e!r}"
        meta["ddsc_paths"] = loaded_ddsc_paths(open("/proc/self/maps").read()) if os.path.exists("/proc/self/maps") else []
        bench = rlg.CsvLog(os.path.join(self.log.dir, "bench.tmp"), list("abcdefgh"))
        t0 = time.perf_counter()
        for i in range(20000):
            bench.row(1.5, i, "ok", 3, 4.25, 5, 6, "p")
        self.log_row_cost_us = (time.perf_counter() - t0) / 20000 * 1e6
        bench.close()
        os.remove(os.path.join(self.log.dir, "bench.tmp"))
        meta["log_row_cost_us"] = self.log_row_cost_us
        self.log.write_json("meta.json", meta)
        self.log.start_flusher(1.0)

    def _emit_metrics(self, cpu, net) -> None:
        with self.win_lock:
            gap, self.win_cmd_gap = self.win_cmd_gap, rlg.Histogram(thresholds=CMD_GAP_THRESHOLDS)
            counts, self.win_counts = self.win_counts, {"ok": 0, "stale": 0, "invalid": 0, "auth": 0, "ignored": 0}
        with self.pub_lock:
            pub, self.win_pub = self.win_pub, rlg.Histogram(thresholds=(4.0, 10.0, 20.0))
            call, self.win_pubcall = self.win_pubcall, rlg.Histogram(thresholds=(1.0, 2.0, 5.0))
            events, self._gap_events = self._gap_events, []
        for t_mono, t_wall, gap_ms in events:
            self.gap_log.row(t_mono, t_wall, gap_ms, self.phase)
        w = rlg.wireless_now(self._wifi_iface)
        d = net.delta()
        nan = math.nan
        self.metrics_log.row(
            self.log.clock.mono(), time.time(), self.phase, self.mode, counts["ok"], counts["stale"], counts["invalid"],
            counts["auth"], counts["ignored"], gap.count, gap.percentile(50), gap.percentile(99), gap.max,
            gap.above[30.0], gap.above[50.0], gap.above[100.0], gap.above[150.0], self.counters["states_sent"],
            pub.count, pub.percentile(50), pub.percentile(99), pub.max, pub.above[4.0], pub.above[10.0], pub.above[20.0],
            call.percentile(99), call.max, self.state.age_s() * 1000.0, cpu.percent(), rlg.read_loadavg()[0],
            rlg.read_thermal_max_c(), w.get("link", nan), w.get("level", nan), w.get("noise", nan),
            d.get("rx_packets", nan), d.get("tx_packets", nan), d.get("rx_errs", nan), d.get("tx_errs", nan),
            d.get("rx_drop", nan), d.get("tx_drop", nan), self._metrics_cost_ms,
        )

    def _metrics_loop(self) -> None:
        cpu, net = rlg.ProcCpu(), rlg.NetCounters(self._wifi_iface)
        next_t = time.monotonic() + 1.0
        while not self.stop.wait(max(0.0, next_t - time.monotonic())):
            next_t += 1.0
            t0 = time.perf_counter()
            try:
                self._emit_metrics(cpu, net)
            except Exception as e:  # noqa: BLE001 -- logging must never hurt control
                self.log.event(f"[runlog] metrics error: {e!r}")
            self._metrics_cost_ms = (time.perf_counter() - t0) * 1000.0

    def write_summary(self) -> None:
        if not self.log.enabled:
            return
        with self.pub_lock:
            pub, call = self.hist_pub.to_dict(), self.hist_pubcall.to_dict()
        self.log.write_json("summary.json", {
            "run_id": getattr(self.args, "run_id", ""), "sessions_started": self.sessions_started,
            "counters": dict(self.counters), "damp_events": self.damp_events,
            "cmd_gap_ms": self.hist_cmd_gap.to_dict(), "publish_interval_ms": pub, "publish_call_ms": call,
            "impairment_counts": dict(self._state_imp.counts) if self._state_imp is not None else None,
            "log_row_cost_us": self.log_row_cost_us, "csv_rows": {c.path.split("/")[-1]: c.rows for c in self.log._csvs},
            "lowstate_age_ms_at_end": self.state.age_s() * 1000.0, "finished_wall": time.time(),
        })

    def close_logs(self) -> None:
        self.write_summary()
        self.log.close()

    # ---- DDS -------------------------------------------------------------------------------------------

    def start_dds(self) -> None:
        if not dr._UNITREE_SDK_AVAILABLE:
            raise RuntimeError("unitree_sdk2py is not importable here (activate ~/relay_env).")
        check_ddsc(self.args.require_ddsc_dir)
        dr.ChannelFactoryInitialize(0, self.args.network_interface)
        self.crc = dr.CRC()
        low_sub = dr.ChannelSubscriber("rt/lowstate", dr.LowState_)
        low_sub.Init(self.state.update_from_low_state, 10)
        sport_sub = dr.ChannelSubscriber("rt/sportmodestate", dr.SportModeState_)
        sport_sub.Init(self.state.update_from_sportmode_state, 10)
        self.publisher = dr.ChannelPublisher("rt/lowcmd", dr.LowCmd_)
        self.publisher.Init()
        print(f"[RELAY] Waiting for LowState on '{self.args.network_interface}'...", flush=True)
        while not self.state.ready and not self.stop.is_set():
            time.sleep(0.05)
        print("[RELAY] Robot state stream connected.", flush=True)

    def _start_publishing(self) -> None:
        snap = self.state.snapshot()
        self.low_cmd = dr.init_low_cmd(self.cfg)
        self.target_buf = dr.LowCmdTarget(list(snap.joint_pos), initial_kp=0.0, initial_kd=0.0)
        self._pub_last_ns = 0
        self.publish_thread = dr.RecurrentThread(
            interval=0.002, target=self._timed_publish, name="lowcmd_publish",
            args=(self.low_cmd, self.target_buf, IDENTITY_MAP, self.publisher, self.crc),
        )
        self.publish_thread.Start()
        print("[RELAY] LowCmd publishing started at 500 Hz with ZERO gains (kp=kd=0 -> no torque).", flush=True)

    def _timed_publish(self, *args) -> None:
        """dr.publish_low_cmd, plus timing: the gap since the previous call (the real 500 Hz period) and how long the
        call itself took. Recording is two histogram adds under a lock, with the disk work left to the 1 Hz thread."""
        t0 = time.perf_counter_ns()
        last, self._pub_last_ns = self._pub_last_ns, t0
        try:
            dr.publish_low_cmd(*args)
        finally:
            if last:
                gap_ms, call_ms = (t0 - last) / 1e6, (time.perf_counter_ns() - t0) / 1e6
                with self.pub_lock:
                    self.hist_pub.add(gap_ms)
                    self.win_pub.add(gap_ms)
                    self.hist_pubcall.add(call_ms)
                    self.win_pubcall.add(call_ms)
                    if gap_ms > 4.0:
                        self._gap_events.append((self.log.clock.mono(), time.time(), gap_ms))

    def _stop_publishing(self) -> None:
        """Same sequence as deploy_real's exit: stop the 500 Hz thread first so it cannot overwrite the damping
        burst, then ~0.5 s of damping frames (a single dropped frame must not leave the robot without one)."""
        with self.lock:
            thread, self.publish_thread = self.publish_thread, None
        if thread is None:
            return
        thread.Wait(1.0)
        print("[RELAY] Sending damping burst (kp=0, kd=3) before going quiet.", flush=True)
        damp_cmd = dr.init_low_cmd(self.cfg)
        for i in range(rl.NUM_JOINTS):
            damp_cmd.motor_cmd[i].q = 0.0
            damp_cmd.motor_cmd[i].kp = 0.0
            damp_cmd.motor_cmd[i].kd = DAMP_KD
        damp_cmd.crc = self.crc.Crc(damp_cmd)
        for _ in range(100):
            self.publisher.Write(damp_cmd)
            time.sleep(0.005)

    # ---- safety ----------------------------------------------------------------------------------------

    def enter_damped(self, reason: str) -> None:
        """Latch damping. Idempotent. Keeps the 500 Hz heartbeat alive with kp=0, kd=3 for --damp_hold_s."""
        with self.lock:
            if self.mode == rl.MODE_DAMPED:
                return
            prev = self.mode
            self.mode = rl.MODE_DAMPED
            self.damp_reason = reason
            self.damp_since = time.monotonic()
            self.damp_events.append({"t_mono": self.log.clock.mono(), "t_wall": time.time(), "reason": reason,
                                     "was": "active" if prev == rl.MODE_ACTIVE else "streaming", "phase": self.phase})
            if self.target_buf is not None:
                self.target_buf.set_kp(0.0)
                self.target_buf.set_kd(DAMP_KD)
            ctl = self.ctl
        print(f"[RELAY] DAMPING (was {'active' if prev == rl.MODE_ACTIVE else 'streaming'}): {reason}", flush=True)
        if ctl is not None:
            try:
                ctl.send({"type": "damped", "reason": reason})
            except OSError:
                pass

    def _watchdog_loop(self) -> None:
        while not self.stop.is_set():
            time.sleep(0.01)
            now = time.monotonic()
            with self.lock:
                mode = self.mode
                last = self.last_cmd_time
                since_active = now - self.active_since
                damp_since = self.damp_since
                publishing = self.publish_thread is not None
            if mode == rl.MODE_ACTIVE:
                if last is None:
                    if since_active > self.args.first_cmd_timeout_s:
                        self.enter_damped(f"no command arrived within {self.args.first_cmd_timeout_s:.0f} s of release")
                elif now - last > self.args.silence_damp_s:
                    self.enter_damped(f"no valid command for {(now - last) * 1000:.0f} ms (limit {self.args.silence_damp_s * 1000:.0f} ms)")
                if self.state.age_s() > self.cfg.link_lost_after_s:
                    self.enter_damped(f"no LowState from the robot for {self.state.age_s():.1f} s")
            elif mode == rl.MODE_DAMPED and publishing and now - damp_since > self.args.damp_hold_s:
                self._stop_publishing()  # the mode stays latched DAMPED for the rest of this session

    # ---- command path ----------------------------------------------------------------------------------

    def _cmd_loop(self) -> None:
        while not self.stop.is_set():
            try:
                data, addr = self.cmd_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            with self.lock:
                session, peer = self.session, self.peer_ip
            pkt = rl.unpack_cmd(self.key, data)
            if pkt is None or session is None or pkt["session"] != session or addr[0] != peer:
                self.counters["rejected_auth"] += 1
                self._win_count("auth")
                self.rx_log.row(self.log.clock.mono(), time.time(), -1, "reject_auth", f"from {addr[0]}", math.nan,
                                math.nan, math.nan, 0, self.phase)
                continue
            self._handle_cmd(pkt)

    def _handle_cmd(self, pkt: dict) -> None:
        invalid_reason = None
        mono, wall = self.log.clock.mono(), time.time()
        with self.lock:
            if not rl.seq_is_newer(pkt["seq"], self.last_cmd_seq):
                self.counters["rejected_stale"] += 1
                self._win_count("stale")
                self.rx_log.row(mono, wall, pkt["seq"], "stale", "", math.nan, pkt["kp"], pkt["kd"], 0, self.phase)
                return
            self.last_cmd_seq = pkt["seq"]
            if self.mode != rl.MODE_ACTIVE or self.target_buf is None:
                self._win_count("ignored")
                self.rx_log.row(mono, wall, pkt["seq"], "ignored", "relay not active", math.nan, pkt["kp"], pkt["kd"], 0,
                                self.phase)
                return
            kp, kd, target = pkt["kp"], pkt["kd"], pkt["target"]
            if not (rl.all_finite(target) and rl.all_finite((kp, kd)) and 0.0 <= kp <= self.args.kp_max
                    and 0.0 <= kd <= self.args.kd_max):
                self.counters["rejected_invalid"] += 1
                bad = f"kp={kp}, kd={kd}" if rl.all_finite(target) else "non-finite target"
                invalid_reason = f"authenticated command failed validation ({bad})"
                self._win_count("invalid")
                self.rx_log.row(mono, wall, pkt["seq"], "invalid", bad, math.nan, kp, kd, 0, self.phase)
        if invalid_reason is not None:
            # Authenticated but invalid means a desktop bug: damp (outside the lock, since it sends on the network).
            self.enter_damped(invalid_reason)
            return
        with self.lock:
            if self.mode != rl.MODE_ACTIVE or self.target_buf is None:
                return
            snap = self.state.snapshot()
            limited, clipped = dr.tether_limit(
                target, list(snap.joint_pos), self.args.max_target_dev_deg * 3.141592653589793 / 180.0, JOINT_LABELS
            )
            if clipped:
                self.counters["tethered"] += 1
            self.target_buf.set(limited)
            self.target_buf.set_kp(kp)
            self.target_buf.set_kd(kd)
            now_m = time.monotonic()
            inter_ms = math.nan if self.last_cmd_time is None else (now_m - self.last_cmd_time) * 1000.0
            self.last_cmd_time = now_m
            self.echo_seq, self.echo_t_ns = pkt["seq"], pkt["t_ns"]
            self.counters["cmds"] += 1
            with self.win_lock:
                self.win_counts["ok"] += 1
                if not math.isnan(inter_ms):
                    self.win_cmd_gap.add(inter_ms)
                    self.hist_cmd_gap.add(inter_ms)
            self.rx_log.row(mono, wall, pkt["seq"], "ok", "", inter_ms, kp, kd, 1 if clipped else 0, self.phase)

    # ---- state path ------------------------------------------------------------------------------------

    def _state_loop(self) -> None:
        period = 1.0 / self.args.state_rate_hz
        seq = 0
        last_sent_update = None
        next_t = time.monotonic()
        while not self.stop.is_set():
            next_t += period
            with self.lock:
                addr, session, mode = self.client_state_addr, self.session, self.mode
                cmd_age_ms = 0 if self.last_cmd_time is None else (time.monotonic() - self.last_cmd_time) * 1000.0
                echo_seq, echo_t = self.echo_seq, self.echo_t_ns
            snap = self.state.snapshot()
            # Only fresh robot data is forwarded. If the robot's LowState stalls, the desktop must see silence
            # (and trip its own link-lost guard), not a stream of repeated stale values.
            if addr is not None and session is not None and snap.last_update is not None \
                    and snap.last_update != last_sent_update:
                last_sent_update = snap.last_update
                seq = (seq + 1) & 0xFFFFFFFF
                pkt = rl.pack_state(
                    self.key, session, seq, time.monotonic_ns(), echo_seq, echo_t, mode,
                    1 if mode == rl.MODE_DAMPED else 0, cmd_age_ms, snap.joint_pos, snap.joint_vel, snap.tau_est,
                    snap.quat_wxyz, snap.gyro, snap.sportmode_velocity, snap.wireless_remote,
                )
                action, delay_ms = imp.PASS, 0.0
                try:
                    if self._state_imp is not None:
                        action, delay_ms = self._state_imp.send(pkt)  # clean profile = direct, synchronous send
                    else:
                        self.state_sock.sendto(pkt, addr)
                    if action != imp.DROP:
                        self.counters["states_sent"] += 1
                except OSError:
                    action = "send_error"
                self.tx_log.row(self.log.clock.mono(), time.time(), seq, snap.age_s() * 1000.0, action, delay_ms, mode,
                                self.phase)
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    # ---- control connection ----------------------------------------------------------------------------

    def serve(self) -> None:
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind((self.args.bind, self.args.cmd_port))
        self.cmd_sock.settimeout(0.2)
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.args.network_interface == "lo":
            self._state_imp = imp.Impairment(self._send_state_packet, seed=getattr(self.args, "impair_seed", 0))
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.args.bind, self.args.control_port))
        listener.listen(1)
        listener.settimeout(0.5)
        for name, fn in (("cmd-rx", self._cmd_loop), ("state-tx", self._state_loop), ("watchdog", self._watchdog_loop)):
            threading.Thread(target=fn, name=name, daemon=True).start()
        threading.Thread(target=self._status_loop, name="status", daemon=True).start()
        if self.log.enabled:
            threading.Thread(target=self._metrics_loop, name="metrics", daemon=True).start()
        print(f"[RELAY] Listening: control tcp/{self.args.control_port}, commands udp/{self.args.cmd_port}. "
              "Waiting for the desktop...", flush=True)
        while not self.stop.is_set():
            try:
                conn, addr = listener.accept()
            except socket.timeout:
                continue
            try:
                self._serve_client(conn, addr)
            except Exception as e:  # noqa: BLE001 -- a bad client must never take the relay down
                print(f"[RELAY] Client session ended with an error: {e!r}", flush=True)
            finally:
                self._end_session()
            if getattr(self.args, "exit_after_session", False) and self.sessions_started > 0:
                print("[RELAY] --exit_after_session: the session is over, shutting down.", flush=True)
                self.stop.set()
        listener.close()

    def _send_state_packet(self, pkt: bytes) -> None:
        addr = self.client_state_addr  # read at release time: a delayed packet must not outlive its session
        if addr is not None:
            self.state_sock.sendto(pkt, addr)

    def _serve_client(self, conn: socket.socket, addr) -> None:
        ctl = rl.JsonLineSocket(conn)
        try:
            self._serve_client_inner(ctl, addr)
        finally:
            ctl.close()

    def _serve_client_inner(self, ctl: "rl.JsonLineSocket", addr) -> None:
        with self.lock:
            busy = self.session is not None
        if busy:
            ctl.send({"type": "error", "message": "relay already has a session; restart it for a new run"})
            return
        nonce = rl.new_nonce()
        ctl.send({"type": "challenge", "nonce": nonce, "version": rl.PROTOCOL_VERSION})
        hello = ctl.recv(5.0)
        ok = False
        try:
            ok = (hello is not None and hello.get("type") == "hello" and hello.get("version") == rl.PROTOCOL_VERSION
                  and isinstance(hello.get("response"), str) and isinstance(hello.get("state_port"), int)
                  and isinstance(hello.get("client_nonce"), str) and bytes.fromhex(hello["client_nonce"]) is not None
                  and hmac.compare_digest(hello["response"], rl.challenge_response(self.key, nonce, "client")))
        except ValueError:
            ok = False
        if not ok:
            print(f"[RELAY] Rejected a connection from {addr[0]} (bad or missing authentication).", flush=True)
            ctl.send({"type": "error", "message": "authentication failed"})
            return
        with self.lock:
            self.session = int.from_bytes(os.urandom(4), "little")
            self.peer_ip = addr[0]
            self.client_state_addr = (addr[0], int(hello["state_port"]))
            self.ctl = ctl
            self.last_cmd_seq = None
            self.last_cmd_time = None
            self.mode = rl.MODE_STREAMING
            self.sessions_started += 1
        ctl.send({"type": "hello_ack", "session": self.session, "cmd_port": self.args.cmd_port,
                  "relay_wall": time.time(),
                  "mutual": rl.challenge_response(self.key, hello["client_nonce"], "relay")})
        print(f"[RELAY] Desktop {addr[0]} authenticated (session {self.session:08x}).", flush=True)
        while not self.stop.is_set():
            try:
                msg = ctl.recv(0.5)
            except ConnectionError as e:
                self.enter_damped(f"control connection lost ({e})")
                return
            if msg is None:
                continue
            kind = msg.get("type")
            if kind == "release":
                if self.releasing or self.mode != rl.MODE_STREAMING:
                    ctl.send({"type": "release_done", "ok": False, "message": "not in a state that can release"})
                    continue
                threading.Thread(target=self._do_release, args=(msg,), name="release", daemon=True).start()
            elif kind == "confirm":
                self.confirm_event.set()
            elif kind == "impair":
                self._handle_impair(ctl, msg)
            elif kind == "damp":
                self.enter_damped("desktop requested damping")
            elif kind == "shutdown":
                self.enter_damped("desktop closed the session")
                return

    def _handle_impair(self, ctl, msg: dict) -> None:
        """Test-only: switch the impairment profile for STATE packets. Refused on anything but the loopback fake robot."""
        try:
            if self.args.network_interface != "lo" or self._state_imp is None:
                raise PermissionError("impairments are only allowed on the loopback fake-robot setup")
            profile = imp.sanitize_profile(msg.get("profile") or {})
            phase = str(msg.get("phase", "?"))[:40]
            self._state_imp.set_profile(phase, profile)
            self.phase = phase
            self.log.event(f"[impair] phase={phase} profile={profile}")
            ctl.send({"type": "impair_ack", "ok": True})
        except (PermissionError, ValueError) as e:
            print(f"[RELAY] Refused an impairment request: {e}", flush=True)
            ctl.send({"type": "impair_ack", "ok": False, "message": str(e)})

    def _do_release(self, msg: dict) -> None:
        ctl = self.ctl
        self.releasing = True
        try:
            skip = bool(msg.get("skip"))
            if skip and self.args.network_interface != "lo":
                raise RuntimeError("skip is only accepted on the loopback fake robot (--network_interface lo)")
            if skip:
                print("[RELAY] Skipping release (loopback fake robot only).", flush=True)
            else:
                def _ask() -> None:
                    self.confirm_event.clear()
                    ctl.send({"type": "need_confirm"})
                    if not self.confirm_event.wait(600.0):
                        raise RuntimeError("operator did not confirm the pre-release checklist in time")

                dr.release_motion_control(stand_down=bool(msg.get("stand_down")), before_release=_ask)
            with self.lock:
                self._start_publishing()
                self.mode = rl.MODE_ACTIVE
                self.active_since = time.monotonic()
            ctl.send({"type": "release_done", "ok": True, "message": "released"})
        except Exception as e:  # noqa: BLE001 -- report to the desktop, never crash the relay
            print(f"[RELAY] Release failed: {e}", flush=True)
            try:
                ctl.send({"type": "release_done", "ok": False, "message": str(e)})
            except OSError:
                pass
        finally:
            self.releasing = False

    def _end_session(self) -> None:
        """Client gone (clean or not). Damp if publishing, stop publishing, and forget the session."""
        with self.lock:
            was_publishing = self.publish_thread is not None
        if was_publishing:
            self.enter_damped("control connection ended")
            self._stop_publishing()
        with self.lock:
            if self.ctl is not None:
                self.ctl.close()
            self.ctl = None
            self.session = None
            self.peer_ip = None
            self.client_state_addr = None
            self.mode = rl.MODE_STREAMING
            self.damp_reason = ""
            self.target_buf = None
            if self._state_imp is not None:
                self._state_imp.set_profile("clean", {})
            self.phase = "clean"
        print("[RELAY] Session closed. Ready for a new desktop connection.", flush=True)
        self.write_summary()

    def _status_loop(self) -> None:
        while not self.stop.wait(5.0):
            with self.lock:
                m = {rl.MODE_STREAMING: "streaming", rl.MODE_ACTIVE: "ACTIVE", rl.MODE_DAMPED: "DAMPED"}[self.mode]
                c = dict(self.counters)
            print(f"[RELAY] status: {m}  cmds={c['cmds']} states_sent={c['states_sent']} tethered={c['tethered']} "
                  f"rejected(auth/stale/invalid)={c['rejected_auth']}/{c['rejected_stale']}/{c['rejected_invalid']} "
                  f"lowstate_age={self.state.age_s() * 1000:.0f}ms", flush=True)

    def shutdown(self) -> None:
        self.stop.set()
        with self.lock:
            publishing = self.publish_thread is not None
        if publishing:
            self.enter_damped("relay stopping")
            self._stop_publishing()
        if self._state_imp is not None:
            self._state_imp.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jetson relay for wireless Go2 deployment.")
    p.add_argument("--network_interface", type=str, help="Robot-facing interface for DDS (eth0 on the Jetson).")
    p.add_argument("--key_file", type=str, default=os.path.expanduser("~/.go2_relay_key"))
    p.add_argument("--init_key", type=str, default=None, metavar="PATH",
                   help="Create a new random key at PATH (mode 600) and exit; copy it to the desktop.")
    p.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "go2_locomotion.yaml"))
    p.add_argument("--bind", type=str, default="0.0.0.0")
    p.add_argument("--control_port", type=int, default=rl.DEFAULT_CONTROL_PORT)
    p.add_argument("--cmd_port", type=int, default=rl.DEFAULT_CMD_PORT)
    p.add_argument("--state_rate_hz", type=float, default=100.0)
    p.add_argument("--silence_damp_s", type=float, default=0.15,
                   help="Damp if no valid command arrives for this long. Too small trips on WiFi jitter; too "
                   "large leaves the robot driven by a stale target. Tune from the measured link (default 0.15).")
    p.add_argument("--first_cmd_timeout_s", type=float, default=5.0)
    p.add_argument("--damp_hold_s", type=float, default=1.0, help="Keep the damping heartbeat this long, then go quiet.")
    p.add_argument("--max_target_dev_deg", type=float, default=30.0,
                   help="Backstop: limit every target to this far from the actual joint angle (the desktop applies a "
                   "tighter tether of its own; this one only acts on anomalies).")
    p.add_argument("--require_ddsc_dir", type=str, default=None, metavar="DIR",
                   help="Refuse to start unless every loaded libddsc is in DIR (on the Jetson: "
                   "~/cyclonedds_ws/install/cyclonedds/lib). Prevents the wrong-library segfault on first write.")
    p.add_argument("--log_dir", type=str, default=None,
                   help="Write this run's logs here (events.log, per-packet and 1 Hz CSVs, meta.json, summary.json). "
                   "Default: no logging.")
    p.add_argument("--run_id", type=str, default="", help="Label stored in meta.json/summary.json.")
    p.add_argument("--log_level", choices=("full", "light"), default="full",
                   help="'light' skips the per-packet CSVs (rx_cmd.csv, tx_state.csv); metrics and summaries stay.")
    p.add_argument("--exit_after_session", action="store_true", default=False,
                   help="Shut down (writing summary.json) once the first authenticated session ends.")
    p.add_argument("--wifi_iface", type=str, default="wlan0", help="Interface whose signal/packet counters are logged.")
    p.add_argument("--impair_seed", type=int, default=0, help="Seed for the (loopback-only) test impairment shim.")
    p.add_argument("--kp_max", type=float, default=30.0)
    p.add_argument("--kd_max", type=float, default=5.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.init_key:
        rl.init_key(args.init_key)
        print(f"[RELAY] Wrote a new key to {args.init_key} (mode 600). Copy the SAME file to the desktop's key path.")
        return 0
    if not args.network_interface:
        print("--network_interface is required (eth0 on the Jetson).", file=sys.stderr)
        return 2
    if not os.path.exists(args.key_file):
        print(f"No key at {args.key_file}. Create one with:  python {sys.argv[0]} --init_key {args.key_file}", file=sys.stderr)
        return 2
    key = rl.load_key(args.key_file)
    cfg = dr.load_config(args.config, args.network_interface)
    relay = Relay(args, cfg, key)

    def _on_signal(signum, _frame) -> None:
        print(f"\n[RELAY] Signal {signum}: shutting down safely.", flush=True)
        relay.shutdown()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    relay.start_logging(sys.argv)
    try:
        relay.start_dds()
        relay.serve()
    finally:
        relay.shutdown()
        relay.close_logs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
