# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Desktop policy loop for the WIRELESS deployment. deploy_real.py (wired DDS) is left untouched; this is its
sibling. The policy, observation building, slew limit, tether, ramp, watchdogs and prompts are deploy_real's own
code, imported unchanged. What differs is only the robot I/O: instead of talking DDS to the robot (impossible
from a desktop that is not on the robot's 192.168.123.0/24 network), this talks to deploy/relay_jetson.py running
on the Jetson over authenticated UDP/TCP (see relay_link.py).

Differences from the wired run that matter for safety:
  * the Jetson, not this process, owns the 500 Hz LowCmd heartbeat and damps by itself if commands stop (it
    does not wait for this process to notice a dead link);
  * robot state reaches the policy delayed by the WiFi one-way latency, so it is staler than on the cable. The
    policy was trained with 0-2 steps of ACTION delay, but not with stale observations: quantify with the
    round-trip stats printed at exit before trusting this on the ground;
  * the relay's tether (default 30 deg) is a backstop looser than this script's own (--max_target_dev_deg).

Usage (the relay must already be running on the Jetson; see relay_jetson.py and deploy.md):
    python deploy/deploy_real_wireless.py --relay_host 192.168.152.215 --manual_commands --live_plot
All of deploy_real.py's flags (--max_target_dev_deg, --startup_ramp_s, --steps_per_confirm, ...) work here too.
--dry_run is not supported (use deploy_real.py --dry_run for that); --network_interface is not used.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deploy_real as dr  # noqa: E402
import impair as imp  # noqa: E402
import relay_link as rl  # noqa: E402
import relay_profile  # noqa: E402
import runlog as rlg  # noqa: E402


class RelaySafetyMonitor(dr.SafetyMonitor):
    """deploy_real's SafetyMonitor plus the relay's health: if the Jetson latched damping or the control
    connection died, the run is over (reported through the same LinkLost path as a dead link)."""

    def __init__(self, *args, relay_client: rl.RelayClient, **kwargs):
        super().__init__(*args, **kwargs)
        self._relay = relay_client

    def _raise_if_unsafe(self, snap) -> None:
        super()._raise_if_unsafe(snap)
        try:
            self._relay.raise_if_unhealthy()
        except rl.RelayError as e:
            raise dr.LinkLost(str(e)) from e


class _DurationReached(Exception):
    """--duration_s elapsed: a normal, clean end of a timed run."""


LOOP_THRESHOLDS = (20.0, 25.0, 40.0)  # ms: the 50 Hz period is 20 ms, so anything over it is an overrun
GATE_THRESHOLDS = (30.0, 50.0, 100.0, 150.0)
METRIC_COLUMNS = [
    "t_mono", "t_wall", "phase", "states_ok", "states_stale", "states_bad", "state_gap_p50_ms", "state_gap_p99_ms",
    "state_gap_max_ms", "state_gap_n30", "state_gap_n50", "state_gap_n100", "state_gap_n150", "rtt_p50_ms",
    "rtt_p95_ms", "rtt_p99_ms", "rtt_max_ms", "rtt_n30", "rtt_n50", "rtt_n100", "rtt_n150", "cmds_tx",
    "cmds_dropped_by_impairment", "relay_mode", "relay_cmd_age_ms", "loop_n", "loop_p99_ms", "loop_max_ms",
    "infer_p99_ms", "overruns", "cpu_pct", "load1", "nic_rx_pkts", "nic_tx_pkts", "nic_rx_errs", "nic_rx_drop",
]


def _git_info() -> dict:
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, timeout=5)
        return {"commit": head.stdout.strip(), "dirty_files": len([l for l in dirty.stdout.splitlines() if l.strip()])}
    except Exception as e:  # noqa: BLE001 -- inside a container without git, or no repo
        return {"unavailable": repr(e)}


class Conductor(threading.Thread):
    """Walks the impairment schedule: at each phase boundary tells the relay and the local shim to switch. Runs in a
    thread so a slow acknowledgement can never delay the 50 Hz control loop. On any failure it records the error and
    the control loop stops the run (an unacknowledged impairment would make the logs lie about what was applied)."""

    def __init__(self, client, schedule, t0_mono: float, log):
        super().__init__(name="impair-conductor", daemon=True)
        self.client, self.schedule, self.t0, self.log = client, schedule, t0_mono, log
        self.stop = threading.Event()
        self.error = None
        self.phase_started = {}

    def run(self) -> None:
        for start, _end, phase, profile in self.schedule:
            wait = self.t0 + start - time.monotonic()
            if wait > 0 and self.stop.wait(wait):
                return
            try:
                self.client.set_impairment(phase, profile)
                self.phase_started[phase] = self.log.clock.mono()
            except Exception as e:  # noqa: BLE001
                self.error = e
                return


def parse_all_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    relay_parser = argparse.ArgumentParser(add_help=False, description="Wireless deployment (relay) options.")
    relay_parser.add_argument("--relay_host", type=str, help="Jetson address reachable from this machine (required).")
    relay_parser.add_argument("--relay_key_file", type=str, default=os.path.expanduser("~/.go2_relay_key"),
                              help="Shared secret; the SAME file as on the Jetson (see relay_jetson.py --init_key).")
    relay_parser.add_argument("--relay_control_port", type=int, default=rl.DEFAULT_CONTROL_PORT)
    relay_parser.add_argument("--relay_cmd_port", type=int, default=rl.DEFAULT_CMD_PORT)
    relay_parser.add_argument("--relay_cmd_rate_hz", type=float, default=100.0)
    relay_parser.add_argument("--log_dir", type=str, default=None,
                              help="Write this run's logs here (events.log, per-packet and 1 Hz CSVs, loop.csv, "
                              "meta.json, summary.json). Default: no logging.")
    relay_parser.add_argument("--run_id", type=str, default="", help="Label stored in meta.json/summary.json.")
    relay_parser.add_argument("--log_level", choices=("full", "light"), default="full",
                              help="'light' skips the per-packet CSVs (tx_cmd.csv, rx_state.csv, loop.csv).")
    relay_parser.add_argument("--log_nic", type=str, default="enp1s0f0",
                              help="Network interface whose packet/error counters are logged each second.")
    relay_parser.add_argument("--duration_s", type=float, default=None,
                              help="End the run cleanly this long after the policy loop starts (a schedule sets a "
                              "default of its own length).")
    relay_parser.add_argument("--impair_schedule", type=str, default=None,
                              help=f"TEST ONLY: inject network impairments per this schedule ({sorted(imp.SCHEDULES)}); "
                              "needs --skip_release, and the relay only agrees on a loopback fake robot.")
    relay_parser.add_argument("--impair_time_scale", type=float, default=1.0, help="Multiply phase durations.")
    relay_parser.add_argument("--impair_scale", type=float, default=1.0, help="Multiply impairment strength.")
    relay_parser.add_argument("--impair_seed", type=int, default=1, help="Seed for the impairment shim.")
    # A profile's optional `relay:` block (deploy/relay_profile.py) becomes these flags' defaults, so the precedence
    # stays  explicit CLI flag > profile > built-in default. dr.parse_args() reads the rest of the same profile.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--test_profile", "--deploy_profile", dest="test_profile", default=None)
    pre.add_argument("--test_profiles_file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "configs", "test_profiles.yaml"))
    pre_args, _ = pre.parse_known_args()
    if pre_args.test_profile is not None:
        relay_parser.set_defaults(**relay_profile.load_relay_block(pre_args.test_profiles_file, pre_args.test_profile))
    relay_args, rest = relay_parser.parse_known_args()
    relay_args.relay_key_file = os.path.expanduser(relay_args.relay_key_file)
    if any(a in ("-h", "--help") for a in rest):
        print(relay_parser.format_help())
        print("Also accepts every deploy_real.py flag. deploy_real.py --help follows:\n")
    sys.argv = [sys.argv[0]] + rest
    args = dr.parse_args()
    return relay_args, args


def _raise_keyboard_interrupt(signum, _frame) -> None:
    raise KeyboardInterrupt


def main() -> int:
    # Make Ctrl+C AND a plain `kill` (SIGTERM) run the same damp-and-exit path below. Python ignores SIGINT if it
    # was started as a background job, and by default SIGTERM would end the process with no exit damping at all.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    relay_args, args = parse_all_args()
    if args.dry_run:
        print("--dry_run is not supported here (there is no robot I/O to skip). Use deploy_real.py --dry_run.")
        return 2
    if not relay_args.relay_host:
        print("--relay_host is required (the Jetson's address as seen from this machine).")
        return 2
    if not os.path.exists(relay_args.relay_key_file):
        print(f"No key at {relay_args.relay_key_file}. Copy the Jetson's key file here (same path, mode 600).")
        return 2

    schedule = None
    if relay_args.impair_schedule:
        if not args.skip_release:
            print("--impair_schedule is test-only and needs --skip_release (fake robot on the loopback relay).")
            return 2
        try:
            schedule = imp.build_schedule(relay_args.impair_schedule, relay_args.impair_time_scale,
                                          relay_args.impair_scale)
            for _s, _e, _n, _p in schedule:
                imp.sanitize_profile(_p)
        except ValueError as e:
            print(f"Bad impairment schedule: {e}")
            return 2

    log = rlg.RunLog(relay_args.log_dir, "desktop")
    log.tee_stdio()
    per_packet = relay_args.log_level == "full"
    loop_log = log.csv("loop.csv", [
        "t_mono", "t_wall", "step", "phase", "loop_ms", "infer_ms", "sleep_ms", "overrun", "max_err_deg", "max_tau",
        "clipped_n", "tethered_n", "kp", "state_age_ms"]) if per_packet else rlg._NullCsv()
    metrics_log = log.csv("metrics_1hz.csv", METRIC_COLUMNS)
    log.start_flusher(1.0)

    cfg = dr.load_config(args.config, args.network_interface or "udp-relay", args.link_lost_after,
                         args.hold_kp, args.policy_path, args.history_length)
    key = rl.load_key(relay_args.relay_key_file)
    sdk_to_isaac, isaac_to_sdk = dr.build_joint_index_maps(cfg)
    default_joint_pos_isaac_order = [cfg.default_joint_pos[name] for name in cfg.isaac_joint_order]

    print(f"[INFO] Loading policy from: {cfg.policy_path}")
    policy = torch.jit.load(cfg.policy_path)
    policy.eval()
    base_dim, obs_layout = dr.policy_obs_layout(policy, cfg.history_length)
    include_base_lin_vel = obs_layout["has_lin_vel"]
    obs_history = dr.ObsHistory(cfg.history_length, dr.frame_term_widths(base_dim, obs_layout))
    jv_filter = dr.ObsEmaFilter(cfg.joint_vel_filter_hz, cfg.control_dt)

    state = dr.LatestRobotState(num_joints=12)

    manual_cmd = dr.ManualCommandBuffer()
    if args.manual_commands:
        threading.Thread(
            target=dr._manual_command_file_poll_loop, args=(manual_cmd, args.command_file), daemon=True
        ).start()
        dr.launch_slider(args.command_file)
        print(f"[INFO] Manual command mode: slider window launched (writing to {args.command_file}).")

    live_plot_writer = None
    if args.live_plot:
        from torch.utils.tensorboard import SummaryWriter

        live_plot_dir = os.path.join("logs", "hardware_dashboard", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        live_plot_writer = SummaryWriter(log_dir=live_plot_dir)
        print(f"[INFO] Live plot logging to: {live_plot_dir}")
        if args.live_dashboard:
            dr.dashboard_launch.start(live_plot_dir, args.dashboard_output, viewer=args.dashboard_viewer)

    # --- relay connection (replaces deploy_real's DDS init) ---
    client_log = log if per_packet else rlg.RunLog(None, "desktop")
    client = rl.RelayClient(
        relay_args.relay_host, key, state, isaac_to_sdk, control_port=relay_args.relay_control_port,
        cmd_port=relay_args.relay_cmd_port, cmd_rate_hz=relay_args.relay_cmd_rate_hz, run_log=client_log,
        allow_impairment=schedule is not None, impairment_seed=relay_args.impair_seed,
    )
    meta = rlg.base_meta("desktop", relay_args.run_id, sys.argv, [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in (
            "deploy_real_wireless.py", "relay_link.py", "deploy_real.py", "impair.py", "runlog.py", "relay_jetson.py",
            os.path.join("configs", "go2_locomotion.yaml"))] + [cfg.policy_path])
    meta.update({
        "relay_args": vars(relay_args), "deploy_args": {k: v for k, v in vars(args).items()},
        "config": {"kp": cfg.kp, "kd": cfg.kd, "link_lost_after_s": cfg.link_lost_after_s, "control_dt": cfg.control_dt,
                   "policy_path": cfg.policy_path},
        "schedule": schedule, "torch": getattr(torch, "__version__", "unknown"), "git": _git_info(),
    })
    log.write_json("meta.json", meta)
    target_buf = None
    try:
        print(f"[INFO] Connecting to the relay at {relay_args.relay_host}:{relay_args.relay_control_port}...")
        client.connect()
        print("[INFO] Relay authenticated (mutual).")
        if not client.wait_for_state(10.0):
            raise rl.RelayError("The relay is up but no robot state arrived within 10 s.")
        print("[INFO] Robot state stream connected (via the relay).")
        if schedule is not None:
            client.set_impairment("setup", {})  # labels every row from here until the policy loop starts
            print(f"[INFO] Impairment schedule '{relay_args.impair_schedule}' enabled (TEST ONLY): "
                  f"{len(schedule)} phases over {imp.schedule_duration(schedule):.0f} s.")

        # MUST happen before LowCmd means anything. The relay performs the release (a DDS RPC) and asks for the
        # operator's confirmation, which is collected here with the same checklist prompt as the wired script.
        if args.skip_release:
            print("[WARN] --skip_release: asking the relay to skip the release (it only agrees on a loopback fake robot).")
        client.request_release(
            stand_down=args.stand_down, skip=args.skip_release,
            on_need_confirm=lambda: dr.wait_for_space(dr.PRE_RELEASE_CHECKLIST, flush=True),
        )

        startup_kp = cfg.kp * args.startup_kp_fraction
        initial_snap = state.snapshot()
        hold_isaac_order = [initial_snap.joint_pos[sdk_i] for sdk_i in sdk_to_isaac]
        target_buf = dr.LowCmdTarget(hold_isaac_order, initial_kp=0.0, initial_kd=0.0)
        client.start_command_stream(target_buf)
        print("[INFO] Command stream started (relay is publishing LowCmd at 500 Hz with ZERO gains).")
    except (rl.RelayError, OSError) as e:
        print(f"[FATAL] Could not set up the wireless link: {e}")
        client.damp_and_close()
        log.write_json("summary.json", {"run_id": relay_args.run_id, "outcome": "setup_failed", "error": str(e)})
        log.close()
        return 1

    last_action = [0.0] * 12
    step = 0
    control_dt = cfg.control_dt

    gate = dr.StepConfirmGate(args.steps_per_confirm) if args.steps_per_confirm is not None else None
    if gate is not None:
        print(f"[INFO] Manual step-confirm mode: every target printed in degrees, {gate.steps_per_confirm} "
              f"command(s) sent per SPACE press. Not real-time while this is on.")

    max_joint_step_rad = args.max_joint_speed_rad_s * control_dt
    print(f"[INFO] Per-tick slew-rate limit: max {args.max_joint_speed_rad_s:.3f} rad/s "
          f"({math.degrees(max_joint_step_rad):.1f} deg/joint/tick).")

    speed_watchdog = dr.JointSpeedWatchdog(
        consecutive_steps=args.speed_fault_steps,
        caps_by_type=dict(zip(("hip", "thigh", "calf"), args.speed_fault_cap_rad_s)),
        joint_names=cfg.isaac_joint_order,
    )
    monitor = RelaySafetyMonitor(
        state, sdk_to_isaac, dr.TorqueResponseWatchdog(consecutive_steps=5 if gate is not None else 25),
        joint_names=cfg.isaac_joint_order,
        max_dev_rad=math.radians(args.max_target_dev_deg),
        abort_err_rad=math.radians(args.abort_err_deg),
        abort_steps=3,
        tau_abort_nm=0.9 * cfg.effort_limit,
        speed_watchdog=speed_watchdog,
        writer=live_plot_writer,
        link_lost_after_s=cfg.link_lost_after_s,
        calf_tau_abort_nm=args.calf_tau_abort_nm,
        relay_client=client,
    )
    loop_win = {"loop": rlg.Histogram(thresholds=LOOP_THRESHOLDS), "infer": rlg.Histogram(thresholds=LOOP_THRESHOLDS),
                "overruns": 0}
    loop_total = {"loop": rlg.Histogram(thresholds=LOOP_THRESHOLDS), "infer": rlg.Histogram(thresholds=LOOP_THRESHOLDS),
                  "overruns": 0, "ticks": 0}
    loop_lock = threading.Lock()
    metrics_stop = threading.Event()

    def _metrics_loop() -> None:
        cpu, nic = rlg.ProcCpu(), rlg.NetCounters(relay_args.log_nic)
        next_t = time.monotonic() + 1.0
        while not metrics_stop.wait(max(0.0, next_t - time.monotonic())):
            next_t += 1.0
            try:
                w = client.take_window()
                with loop_lock:
                    lw = dict(loop_win)
                    loop_win["loop"] = rlg.Histogram(thresholds=LOOP_THRESHOLDS)
                    loop_win["infer"] = rlg.Histogram(thresholds=LOOP_THRESHOLDS)
                    loop_win["overruns"] = 0
                g, r, nan = w["gap"], w["rtt"], math.nan
                d = nic.delta()
                metrics_log.row(
                    log.clock.mono(), time.time(), client.phase, w["ok"], w["stale"], w["bad"], g.percentile(50),
                    g.percentile(99), g.max, g.above[30.0], g.above[50.0], g.above[100.0], g.above[150.0],
                    r.percentile(50), r.percentile(95), r.percentile(99), r.max, r.above[30.0], r.above[50.0],
                    r.above[100.0], r.above[150.0], w["tx"], w["tx_dropped"], client.relay_mode,
                    client.relay_cmd_age_ms, lw["loop"].count, lw["loop"].percentile(99), lw["loop"].max,
                    lw["infer"].percentile(99), lw["overruns"], cpu.percent(), rlg.read_loadavg()[0],
                    d.get("rx_packets", nan), d.get("tx_packets", nan), d.get("rx_errs", nan), d.get("rx_drop", nan),
                )
            except Exception as e:  # noqa: BLE001 -- logging must never hurt control
                log.event(f"[runlog] metrics error: {e!r}")

    if log.enabled:
        threading.Thread(target=_metrics_loop, name="desktop-metrics", daemon=True).start()
    print(f"[INFO] Tether: targets limited to {args.max_target_dev_deg:.0f} deg of actual "
          f"(max PD torque ~ kp*dev = {cfg.kp * math.radians(args.max_target_dev_deg):.1f} Nm); abort if error "
          f"> {args.abort_err_deg:.0f} deg for 3 steps or torque >= {0.9 * cfg.effort_limit:.1f} Nm "
          f"(calves {args.calf_tau_abort_nm:.1f} Nm).")
    print(f"[INFO] Joint-speed fault: abort if actual |velocity| exceeds "
          f"{args.speed_fault_cap_rad_s[0]:.1f}/{args.speed_fault_cap_rad_s[1]:.1f}/{args.speed_fault_cap_rad_s[2]:.1f} "
          f"rad/s (hip/thigh/calf) for {args.speed_fault_steps} consecutive steps.")

    print(f"[INFO] Entering control loop at {1.0 / control_dt:.0f} Hz (wireless relay). Ctrl+C to stop.")
    outcome = "unknown"
    conductor = None
    loop_t0 = loop_wall0 = None
    with dr.cbreak_terminal() if gate is not None else contextlib.nullcontext():
        try:
            # 1) Zero-gain heartbeat (no torque), with the controller e-stop already live.
            heartbeat_end = time.monotonic() + dr.ZERO_GAIN_HEARTBEAT_S
            while time.monotonic() < heartbeat_end:
                monitor.poll()
                time.sleep(0.02)

            # 2) Hold damped and pause so the robot can be hoisted while released.
            if not args.no_pause_for_hoist:
                target_buf.set_kd(dr.HOLD_DAMPING_KD)
                seen_buttons: set[str] = set()

                def _pause_poll() -> None:
                    nonlocal seen_buttons
                    pressed = {
                        n for n, v in dr.decode_controller_buttons(state.snapshot().wireless_remote).items() if v
                    }
                    if pressed != seen_buttons:
                        if pressed:
                            print(f"[CONTROLLER] pressed: {sorted(pressed)}", flush=True)
                        seen_buttons = pressed
                    monitor.poll()

                dr.wait_for_space(dr.HOIST_PAUSE_PROMPT, poll=_pause_poll, flush=True)

            # 3) Start the ramp from where the robot ACTUALLY is now, with target == actual.
            start_snap = state.snapshot()
            start_isaac_order = [start_snap.joint_pos[sdk_i] for sdk_i in sdk_to_isaac]
            target_buf.set(start_isaac_order)
            target_buf.set_kd(cfg.kd)
            target_buf.set_kp(startup_kp)
            print(f"[INFO] Holding actual pose at gentle kp={startup_kp:.1f}, kd={cfg.kd:.1f} (full kp={cfg.kp:.1f}).")

            dr.ramp_to_default_pose(
                target_buf, start_isaac_order, default_joint_pos_isaac_order, args.startup_ramp_s,
                cfg.control_dt, cfg.isaac_joint_order, start_kp=startup_kp, full_kp=cfg.kp, gate=gate,
                monitor=monitor,
            )

            loop_t0 = time.monotonic()
            loop_wall0 = time.time()
            if schedule is not None:
                conductor = Conductor(client, schedule, loop_t0, log)
                conductor.start()
            run_duration = relay_args.duration_s if relay_args.duration_s is not None else (
                imp.schedule_duration(schedule) if schedule is not None else None)
            t_end = loop_t0 + run_duration if run_duration else None
            if run_duration:
                print(f"[INFO] Timed run: the policy loop ends cleanly after {run_duration:.0f} s.")
            log.event(f"[run] policy loop started (t_mono={log.clock.mono():.3f})")

            while True:
                tick_start = time.perf_counter()
                if conductor is not None and conductor.error is not None:
                    raise rl.RelayError(f"impairment schedule failed: {conductor.error}")
                if t_end is not None and time.monotonic() >= t_end:
                    raise _DurationReached()
                snap = state.snapshot()
                cmd = manual_cmd.get()

                obs_cmd = dr.command_for_layout(cmd, obs_layout, manual_cmd.get_height())  # see deploy_real
                frame = dr.build_observation(cfg, snap, sdk_to_isaac, default_joint_pos_isaac_order, obs_cmd,
                                             last_action, include_base_lin_vel=include_base_lin_vel,
                                             joint_vel_filter=jv_filter)
                obs = obs_history.push_and_stack(frame)
                t_inf = time.perf_counter()
                with torch.inference_mode():
                    action = policy(obs)[0].tolist()
                infer_ms = (time.perf_counter() - t_inf) * 1000.0

                target_isaac_order = [
                    default_joint_pos_isaac_order[i] + cfg.action_scale * action[i] for i in range(12)
                ]

                held_target = target_buf.get()
                clamped_target, clipped_names = dr.clamp_step(
                    held_target, target_isaac_order, max_joint_step_rad, cfg.isaac_joint_order
                )
                if clipped_names:
                    policy_rates = {
                        name: (target_isaac_order[i] - held_target[i]) / control_dt
                        for i, name in enumerate(cfg.isaac_joint_order)
                    }
                    rate_str = ", ".join(f"{name}={policy_rates[name]:+.2f}rad/s" for name in clipped_names)
                    print(f"[WARN] step {step}: slew-rate limit clipped {len(clipped_names)} joint(s) "
                          f"to {args.max_joint_speed_rad_s:.3f} rad/s (policy wanted: {rate_str})")
                max_err, max_tau = monitor.check(target_buf.get(), kp_fraction=1.0)
                clamped_target, tethered = monitor.limit_target(clamped_target)
                if tethered:
                    print(f"[WARN] step {step}: tether limited {len(tethered)} joint(s) to within "
                          f"{args.max_target_dev_deg:.0f} deg of actual: {', '.join(tethered)}")
                if gate is not None:
                    print(f"[STEP {step}] {dr.format_degrees(clamped_target, cfg.isaac_joint_order)}  "
                          f"| max|err|={math.degrees(max_err):.1f}deg max|tau_est|={max_tau:.2f}Nm\n")
                    gate.wait(poll=monitor.poll)
                target_buf.set(clamped_target)
                # last_action MUST reflect what was actually sent (clamped_target), not the policy's raw wish --
                # see deploy_real.py's identical comment (feedback runaway confirmed on real hardware).
                last_action = [
                    (clamped_target[i] - default_joint_pos_isaac_order[i]) / cfg.action_scale for i in range(12)
                ]

                if live_plot_writer is not None:
                    lean, pitch = dr.quat_to_pitch_lean(snap.quat_wxyz)
                    dr.log_live_plot(
                        live_plot_writer, step, cmd, snap.sportmode_velocity, snap.gyro[2], pitch, lean,
                    )

                step += 1
                elapsed = time.perf_counter() - tick_start
                sleep_time = control_dt - elapsed
                if gate is None:
                    with loop_lock:
                        for h in (loop_win["loop"], loop_total["loop"]):
                            h.add(elapsed * 1000.0)
                        for h in (loop_win["infer"], loop_total["infer"]):
                            h.add(infer_ms)
                        loop_total["ticks"] += 1
                        if sleep_time < 0:
                            loop_win["overruns"] += 1
                            loop_total["overruns"] += 1
                    loop_log.row(log.clock.mono(), time.time(), step, client.phase, elapsed * 1000.0, infer_ms,
                                 max(sleep_time, 0.0) * 1000.0, 1 if sleep_time < 0 else 0, math.degrees(max_err),
                                 max_tau, len(clipped_names), len(tethered), target_buf.get_kp(), snap.age_s() * 1000.0)
                if gate is None:
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif step % 50 == 0:
                        print(f"[WARN] control loop overran by {-sleep_time * 1000:.1f}ms at step {step}")
        except _DurationReached:
            outcome = "completed"
            print("\n[INFO] --duration_s reached: ending the run cleanly.")
        except KeyboardInterrupt:
            outcome = "interrupted"
            print("\n[INFO] Ctrl+C received -- stopping control loop.")
        except dr.EmergencyStop as e:
            outcome = f"estop: {e}"
            print(f"\n[ESTOP] {e} -- stopping control loop.")
        except (dr.TorqueResponseFault, dr.TrackingFault) as e:
            outcome = f"fault: {e}"
            print(f"\n[FAULT] {e}")
        except dr.LinkLost as e:
            outcome = f"link_lost: {e}"
            print(f"\n[LINK] {e}")
        except rl.RelayError as e:
            outcome = f"relay_error: {e}"
            print(f"\n[FAULT] {e}")
        except Exception as e:  # noqa: BLE001 -- keep the exit path (damping, summary) running, then report
            outcome = f"exception: {e!r}"
            traceback.print_exc()
        finally:
            # The relay damps on its own when commands stop; this just makes the exit prompt and tidy.
            print("[INFO] Asking the relay to damp (kp=0, kd=3) and closing the session.")
            if conductor is not None:
                conductor.stop.set()
            client.damp_and_close()
            metrics_stop.set()
            gs = state.gap_stats()
            if gs["count"] > 0:
                print(
                    f"[LINK] Robot-state gaps over {gs['count']} packets: max {gs['max_s'] * 1000:.0f} ms, "
                    f"p95 {gs['p95_s'] * 1000:.0f} ms (link-loss threshold {cfg.link_lost_after_s * 1000:.0f} ms)"
                    + ("  [sample buffer capped]" if gs["capped"] else "")
                )
            rtt = client.rtt_stats()
            if rtt.get("count"):
                print(
                    f"[LINK] Command->relay->state round trip over {rtt['count']} samples: median "
                    f"{rtt['median_ms']:.1f} ms, p95 {rtt['p95_ms']:.1f} ms, max {rtt['max_ms']:.1f} ms "
                    f"(dropped/stale/bad packets: {client.stale_packets}/{client.bad_packets})"
                )
            try:
                loop_end_mono = time.monotonic()
                with loop_lock:
                    loops = {"loop_ms": loop_total["loop"].to_dict(), "infer_ms": loop_total["infer"].to_dict(),
                             "overruns": loop_total["overruns"], "ticks": loop_total["ticks"]}
                log.write_json("summary.json", {
                    "run_id": relay_args.run_id, "outcome": outcome,
                    "policy_loop_seconds": None if loop_t0 is None else loop_end_mono - loop_t0,
                    "policy_loop_start_wall": loop_wall0, "schedule": schedule,
                    "phase_started_at_mono": None if conductor is None else conductor.phase_started,
                    "rtt_ms": client.hist_rtt.to_dict(), "state_gap_ms": client.hist_gap.to_dict(),
                    "rtt_exact": client.rtt_stats(), "state_gap_exact": state.gap_stats(),
                    "packets": {"cmds_sent": client.cmds_sent, "cmds_dropped_by_impairment":
                                client.cmds_dropped_by_impairment, "stale": client.stale_packets, "bad": client.bad_packets},
                    "impairment_counts": None if client._impair is None else dict(client._impair.counts),
                    "clock_offset_s": client.clock_offset_s, "clock_offset_uncertainty_s": client.clock_offset_unc_s,
                    "relay_damp_reason": client.relay_damp_reason, "loop": loops,
                    "csv_rows": {c.path.split("/")[-1]: c.rows for c in log._csvs}, "finished_wall": time.time(),
                })
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] could not write summary.json: {e!r}")
            log.close()
    return 0 if outcome in ("completed", "interrupted") else 3


if __name__ == "__main__":
    raise SystemExit(main())
