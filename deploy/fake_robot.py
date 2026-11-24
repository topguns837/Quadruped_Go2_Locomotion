# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic Go2 DDS publisher for testing deploy_real.py without a physical robot.

Publishes plausible `LowState_`/`SportModeState_` messages on the loopback interface (or any interface you
point it at) so deploy_real.py's subscribers actually receive real DDS traffic -- not zeroed stand-in state
like --dry_run uses -- exercising the real message-parsing/observation-assembly/inference/publish path
end-to-end. Also subscribes to `rt/lowcmd` and logs what it receives, so you can see deploy_real.py's
actual computed targets (CRC-valid or not, sane joint positions or not) without a robot to send them to.

Does NOT implement the SportClient/MotionSwitcherClient RPC service side (CheckMode/ReleaseMode) -- that's
a request/response service protocol, not a pub/sub topic, and convincingly faking it is a separate, deeper
piece of work. Run deploy_real.py's `release_motion_control()` against this fake robot and it will correctly
fail loud after retrying (RPC calls time out, `code != 0`, never silently treated as "released" -- see that
function's own comments) -- this is a genuine, useful verification of that fail-safe behavior, not a gap in
this tool.

Usage (see deploy.md's hardware usage guide for the full picture):
    /workspace/isaaclab/_isaac_sim/python.sh deploy/fake_robot.py --network_interface lo
"""

from __future__ import annotations

import argparse
import array
import math
import os
import signal
import sys
import threading
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog as rlg  # noqa: E402

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_, unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


def parse_args():
    parser = argparse.ArgumentParser(description="Publish synthetic Go2 DDS state for testing deploy_real.py.")
    parser.add_argument("--network_interface", type=str, default="lo")
    parser.add_argument(
        "--config",
        type=str,
        default="deploy/profiles/go2_locomotion.yaml",
        help="Read default_joint_pos/sdk_joint_order from here so the synthetic LowState looks like a "
        "robot standing at its actual default pose, not all-zeros.",
    )
    parser.add_argument("--rate_hz", type=float, default=50.0, help="Publish rate for LowState/SportModeState.")
    parser.add_argument(
        "--start_offset_deg", type=float, default=0.0,
        help="Added to every joint's published start position, so deploy_real.py's ramp has real distance to "
        "cover (default 0 = the robot already sits at default_joint_pos).",
    )
    parser.add_argument(
        "--responsive", action="store_true", default=False,
        help="Joints track the commanded q (when the command's kp > 0) and tau_est reports kp*error. Without "
        "this the fake robot is 'dead': it never moves and reports tau_est=0 -- the signature the "
        "torque-response watchdog in deploy_real.py exists to catch.",
    )
    parser.add_argument(
        "--press_estop_after", type=float, default=None,
        help="After this many seconds, publish the handheld-controller bits for L2+B (the e-stop combo) in "
        "LowState.wireless_remote.",
    )
    parser.add_argument(
        "--fake_high_dq_after", type=float, default=None,
        help="After this many seconds, report an artificially large dq (100 rad/s) on every joint in "
        "LowState -- for testing deploy_real.py's JointSpeedWatchdog/JointSpeedFault without a real robot.",
    )
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Write this run's logs here (events.log, metrics_1hz.csv, lowcmd_gaps.csv, raw "
                        "lowcmd_intervals.f32, meta.json, summary.json). Default: no logging.")
    parser.add_argument("--run_id", type=str, default="", help="Label stored in meta.json/summary.json.")
    parser.add_argument("--duration_s", type=float, default=None, help="Stop by itself after this many seconds.")
    parser.add_argument("--print_every", type=int, default=50, help="Print every Nth received LowCmd (default 50).")
    parser.add_argument("--crc_check_every", type=int, default=10,
                        help="Recompute and compare the CRC of every Nth received LowCmd (0 = never).")
    return parser.parse_args()


GAP_THRESHOLDS = (4.0, 10.0, 20.0)
METRIC_COLUMNS = [
    "t_mono", "t_wall", "lowcmd_n", "int_p50_ms", "int_p99_ms", "int_max_ms", "int_n4", "int_n10", "int_n20",
    "kp_min", "kp_max", "crc_checked", "crc_bad", "header_bad", "level_bad", "mode_bad", "lowstate_pub_n",
    "lowstate_pub_p99_ms", "lowstate_pub_max_ms", "cpu_pct", "load1",
]


class Monitor:
    """What the fake robot measures about the LowCmd stream it receives, in memory; written out once a second."""

    def __init__(self, args, log):
        self.log = log
        self.crc = CRC()
        self.crc_every = args.crc_check_every
        self.lock = threading.Lock()
        self.hist = rlg.Histogram(thresholds=GAP_THRESHOLDS)
        self.win = rlg.Histogram(thresholds=GAP_THRESHOLDS)
        self.pub_hist = rlg.Histogram(thresholds=GAP_THRESHOLDS)
        self.pub_win = rlg.Histogram(thresholds=GAP_THRESHOLDS)
        self.raw = array.array("f")
        self.count = 0
        self.first_t = None
        self.last_t = None
        self._last_ns = 0
        self._pub_last_ns = 0
        self.last_nonzero_kp_idx = 0
        self.max_kp = 0.0
        self.win_kp = [math.inf, -math.inf]
        self.win_checks = {"crc_checked": 0, "crc_bad": 0, "header_bad": 0, "level_bad": 0, "mode_bad": 0}
        self.tot_checks = dict(self.win_checks)
        self.gap_events = []
        self.gaps = log.csv("lowcmd_gaps.csv", ["t_mono", "t_wall", "gap_ms"])
        self.metrics = log.csv("metrics_1hz.csv", METRIC_COLUMNS)
        self.cpu = rlg.ProcCpu()

    def on_cmd(self, msg) -> None:
        now = time.perf_counter_ns()
        gap_ms = (now - self._last_ns) / 1e6 if self._last_ns else None
        self._last_ns = now
        kp0 = msg.motor_cmd[0].kp
        head_bad = 0 if (msg.head[0] == 254 and msg.head[1] == 239) else 1
        level_bad = 0 if msg.level_flag == 255 else 1
        mode_bad = 0 if all(msg.motor_cmd[i].mode == 1 for i in range(20)) else 1
        crc_checked = crc_bad = 0
        if self.crc_every and self.count % self.crc_every == 0:
            crc_checked = 1
            crc_bad = 0 if self.crc.Crc(msg) == msg.crc else 1
        with self.lock:
            self.count += 1
            if self.first_t is None:
                self.first_t = time.time()
            self.last_t = time.time()
            if gap_ms is not None:
                self.hist.add(gap_ms)
                self.win.add(gap_ms)
                self.raw.append(gap_ms)
                if gap_ms > 4.0:
                    self.gap_events.append((self.log.clock.mono(), time.time(), gap_ms))
            if kp0 > 0.0:
                self.last_nonzero_kp_idx = self.count
                self.max_kp = max(self.max_kp, kp0)
            self.win_kp[0] = min(self.win_kp[0], kp0)
            self.win_kp[1] = max(self.win_kp[1], kp0)
            for k, v in (("crc_checked", crc_checked), ("crc_bad", crc_bad), ("header_bad", head_bad),
                         ("level_bad", level_bad), ("mode_bad", mode_bad)):
                self.win_checks[k] += v
                self.tot_checks[k] += v

    def on_publish(self) -> None:
        now = time.perf_counter_ns()
        if self._pub_last_ns:
            gap_ms = (now - self._pub_last_ns) / 1e6
            with self.lock:
                self.pub_hist.add(gap_ms)
                self.pub_win.add(gap_ms)
        self._pub_last_ns = now

    def emit(self) -> None:
        with self.lock:
            win, self.win = self.win, rlg.Histogram(thresholds=GAP_THRESHOLDS)
            pw, self.pub_win = self.pub_win, rlg.Histogram(thresholds=GAP_THRESHOLDS)
            checks, self.win_checks = self.win_checks, {k: 0 for k in self.win_checks}
            kp, self.win_kp = self.win_kp, [math.inf, -math.inf]
            events, self.gap_events = self.gap_events, []
            n = self.count
        for t_mono, t_wall, gap_ms in events:
            self.gaps.row(t_mono, t_wall, gap_ms)
        self.metrics.row(
            self.log.clock.mono(), time.time(), n, win.percentile(50), win.percentile(99), win.max, win.above[4.0],
            win.above[10.0], win.above[20.0], kp[0] if kp[0] != math.inf else math.nan,
            kp[1] if kp[1] != -math.inf else math.nan, checks["crc_checked"], checks["crc_bad"], checks["header_bad"],
            checks["level_bad"], checks["mode_bad"], pw.count, pw.percentile(99), pw.max, self.cpu.percent(),
            rlg.read_loadavg()[0],
        )

    def dump_raw(self) -> None:
        if self.log.enabled:
            with self.lock:
                data = array.array("f", self.raw)
            with open(os.path.join(self.log.dir, "lowcmd_intervals.f32.tmp"), "wb") as f:
                data.tofile(f)
            os.replace(os.path.join(self.log.dir, "lowcmd_intervals.f32.tmp"),
                       os.path.join(self.log.dir, "lowcmd_intervals.f32"))

    def summary(self) -> dict:
        with self.lock:
            span = (self.last_t - self.first_t) if (self.first_t and self.last_t) else 0.0
            return {
                "lowcmd_total": self.count, "span_s": span, "rate_hz": (self.count / span) if span > 0 else 0.0,
                "interval_ms": self.hist.to_dict(), "lowstate_publish_interval_ms": self.pub_hist.to_dict(),
                "max_kp_seen": self.max_kp,
                # LowCmd frames received after the last frame with kp > 0: the damping burst and hold (kp=0, kd=3).
                "kp_zero_tail_frames": self.count - self.last_nonzero_kp_idx if self.max_kp > 0 else 0,
                "checks_total": dict(self.tot_checks), "crc_check_every": self.crc_every,
            }


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sdk_joint_order = cfg["sdk_joint_order"]
    default_joint_pos = cfg["default_joint_pos"]
    offset_rad = args.start_offset_deg * 3.141592653589793 / 180.0
    default_q_sdk_order = [default_joint_pos[name] + offset_rad for name in sdk_joint_order]
    log = rlg.RunLog(args.log_dir, "fake_robot")
    log.tee_stdio()
    monitor = Monitor(args, log)
    log.start_flusher(1.0)
    log.write_json("meta.json", dict(
        rlg.base_meta("fake_robot", args.run_id, sys.argv, [os.path.abspath(__file__)]), args=vars(args)))
    q_state = list(default_q_sdk_order)  # current published joint positions (only move when --responsive)
    tau_state = [0.0] * 12
    last_cmd = {"q": [0.0] * 12, "kp": [0.0] * 12}
    start_time = time.monotonic()

    ChannelFactoryInitialize(0, args.network_interface)

    low_state_pub = ChannelPublisher("rt/lowstate", LowState_)
    low_state_pub.Init()
    sportmode_pub = ChannelPublisher("rt/sportmodestate", SportModeState_)
    sportmode_pub.Init()

    received_lowcmd_count = 0

    def _on_low_cmd(msg: "LowCmd_"):
        nonlocal received_lowcmd_count
        received_lowcmd_count += 1
        try:
            monitor.on_cmd(msg)
        except Exception as e:  # noqa: BLE001 -- measuring must never break the fake robot
            log.event(f"[fake_robot] monitor error: {e!r}")
        last_cmd["q"] = [m.q for m in msg.motor_cmd[:12]]
        last_cmd["kp"] = [m.kp for m in msg.motor_cmd[:12]]
        if received_lowcmd_count % args.print_every == 1:
            qs = [f"{m.q:.3f}" for m in msg.motor_cmd[:12]]
            kp0 = msg.motor_cmd[0].kp  # kp is uniform across all 12 joints in this project -- one is enough
            print(f"[FAKE_ROBOT] received LowCmd #{received_lowcmd_count}: head={list(msg.head)} "
                  f"level_flag={msg.level_flag} mode0={msg.motor_cmd[0].mode} kp0={kp0:.2f} crc={msg.crc} q={qs}")

    low_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    low_cmd_sub.Init(_on_low_cmd, 10)

    def _publish_state():
        monitor.on_publish()
        low_state = unitree_go_msg_dds__LowState_()
        for i in range(12):
            if args.responsive and last_cmd["kp"][i] > 0.0:
                err = last_cmd["q"][i] - q_state[i]
                tau_state[i] = last_cmd["kp"][i] * err
                q_state[i] += 0.5 * err
            else:
                tau_state[i] = 0.0
            fake_dq = (
                100.0 if args.fake_high_dq_after is not None and time.monotonic() - start_time >= args.fake_high_dq_after
                else 0.0
            )
            low_state.motor_state[i].q = q_state[i]
            low_state.motor_state[i].dq = fake_dq
            low_state.motor_state[i].tau_est = tau_state[i]
        if args.press_estop_after is not None and time.monotonic() - start_time >= args.press_estop_after:
            remote = [0] * 40
            remote[2] |= 1 << 5  # L2
            remote[3] |= 1 << 1  # B
            low_state.wireless_remote = remote
        low_state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]  # identity: level, facing +x
        low_state.imu_state.gyroscope = [0.0, 0.0, 0.0]
        low_state_pub.Write(low_state)

        sportmode_state = unitree_go_msg_dds__SportModeState_()
        sportmode_state.velocity = [0.0, 0.0, 0.0]
        sportmode_state.position = [0.0, 0.0, 0.3]
        sportmode_pub.Write(sportmode_state)

    interval = 1.0 / args.rate_hz
    thread = RecurrentThread(interval=interval, target=_publish_state, name="fake_robot_publish")
    thread.Start()

    print(f"[FAKE_ROBOT] Publishing synthetic LowState/SportModeState on '{args.network_interface}' "
          f"at {args.rate_hz} Hz. Ctrl+C to stop.")
    print("[FAKE_ROBOT] NOTE: does not answer CheckMode/ReleaseMode RPC calls -- deploy_real.py's "
          "release_motion_control() will correctly time out and fail loud against this tool (see module "
          "docstring). That's expected, not a bug in this fake robot.")
    stop = threading.Event()

    def _request_stop(signum, _frame):
        print(f"\n[FAKE_ROBOT] Signal {signum}: stopping.")
        stop.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    t_end = time.monotonic() + args.duration_s if args.duration_s else None
    next_raw_dump = time.monotonic() + 60.0
    while not stop.wait(1.0):
        try:
            monitor.emit()
            if time.monotonic() >= next_raw_dump:
                monitor.dump_raw()
                next_raw_dump += 60.0
        except Exception as e:  # noqa: BLE001
            log.event(f"[fake_robot] logging error: {e!r}")
        if t_end is not None and time.monotonic() >= t_end:
            print("[FAKE_ROBOT] --duration_s reached: stopping.")
            break
    print(f"[FAKE_ROBOT] Stopping. Received {received_lowcmd_count} LowCmd messages total.")
    thread.Wait(1.0)
    try:
        monitor.emit()
        monitor.dump_raw()
        log.write_json("summary.json", dict(monitor.summary(), run_id=args.run_id, finished_wall=time.time(),
                                            csv_rows={c.path.split("/")[-1]: c.rows for c in log._csvs}))
    finally:
        log.close()


if __name__ == "__main__":
    main()
