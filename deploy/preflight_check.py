# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tier-1 (zero-movement) hardware pre-flight checks -- read-only, structurally incapable of commanding the
robot. Deliberately does NOT import ChannelPublisher, LowCmd_, or anything else that could send data to the
robot: this file only subscribes to rt/lowstate and rt/sportmodestate and prints what it receives, so it's
safe to run against a standing, rig-mounted robot before ever touching release_motion_control() (which
crouches the robot via its own StandDown()) or LowCmd publishing (see deploy.md's Tier 1 / Tier 2 split).

Reuses deploy_real.py's own helpers directly (config loading, joint-index mapping, state buffer, quaternion
math, observation assembly) rather than reimplementing them -- see that file for what each does.

Default output is a compact ~5-line status block per interval (connectivity / gravity / joints / height),
not a raw data dump -- pass --verbose for the full per-joint and full-observation-vector breakdown. On exit
(Ctrl+C, or --duration expiring) it prints a run-level SUMMARY with an explicit verdict: min/max ranges
tracked across the whole run (not just the latest instant), so a joint reading that's consistently offset in
a stable way reads differently from one that's noisy/drifting -- the summary is what answers "is there
anything here that would block testing/deploying," not any single snapshot.

Usage (see deploy.md's Tier-1 section for the full walkthrough and what each check means):
    /workspace/isaaclab/_isaac_sim/python.sh deploy/preflight_check.py --network_interface enp3s0
    /workspace/isaaclab/_isaac_sim/python.sh deploy/preflight_check.py --network_interface enp3s0 --duration 15
    /workspace/isaaclab/_isaac_sim/python.sh deploy/preflight_check.py --network_interface enp3s0 --verbose

Testable right now, no rig needed, against deploy/fake_robot.py:
    # terminal 1:
    /workspace/isaaclab/_isaac_sim/python.sh deploy/fake_robot.py --network_interface lo
    # terminal 2:
    /workspace/isaaclab/_isaac_sim/python.sh deploy/preflight_check.py --network_interface lo
"""

from __future__ import annotations

import argparse
import time

import deploy_real as dr
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

GRAVITY_TOLERANCE = 0.2  # |projected_gravity - (0,0,-1)| beyond this -> WARN
STALE_AFTER_S = 1.0  # no new message on a topic for this long -> flagged stale
JOINT_WARN_THRESHOLD = 0.5  # |delta from default_joint_pos| beyond this -> WARN
STABLE_RANGE_THRESHOLD = 0.05  # per-joint (max-min) delta narrower than this over the run -> "stable"


class TopicHealth:
    """Tracks message count/recency/rate for one DDS topic, both live and across the whole run."""

    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.last_received = None
        self.start_time = time.monotonic()
        self.ever_stale = False

    def mark(self) -> None:
        self.count += 1
        self.last_received = time.monotonic()

    def is_stale(self) -> bool:
        return self.last_received is None or (time.monotonic() - self.last_received) > STALE_AFTER_S

    def poll_stale(self) -> None:
        """Call once per interval tick (not just on message receipt) so a topic that goes silent mid-run is
        captured in ever_stale even if messages resume before the final summary is printed."""
        if self.is_stale():
            self.ever_stale = True

    def rate_hz(self) -> float:
        elapsed = time.monotonic() - self.start_time
        return self.count / elapsed if elapsed > 0 else 0.0

    def compact_status(self) -> str:
        if self.last_received is None:
            return f"{self.name} NO DATA"
        flag = "STALE" if self.is_stale() else "OK"
        return f"{self.name} {flag} ({self.count} msgs, ~{self.rate_hz():.0f} Hz)"


class JointTracker:
    """Accumulates min/max delta-from-default per joint across the whole run."""

    def __init__(self, joint_names: list[str]):
        self.names = list(joint_names)
        self.min_delta = {n: None for n in self.names}
        self.max_delta = {n: None for n in self.names}

    def update(self, deltas: dict[str, float]) -> None:
        for name, d in deltas.items():
            if self.min_delta[name] is None or d < self.min_delta[name]:
                self.min_delta[name] = d
            if self.max_delta[name] is None or d > self.max_delta[name]:
                self.max_delta[name] = d


class GravityTracker:
    """Accumulates projected_gravity error across the whole run."""

    def __init__(self):
        self.errors: list[float] = []

    def update(self, err: float) -> None:
        self.errors.append(err)

    def stats(self) -> tuple[float, float, float] | None:
        if not self.errors:
            return None
        return min(self.errors), max(self.errors), sum(self.errors) / len(self.errors)


def gravity_error(projected_gravity: tuple[float, float, float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(projected_gravity, (0.0, 0.0, -1.0))) ** 0.5


def compute_joint_deltas(cfg: "dr.DeployConfig", joint_pos_sdk_order: list[float]) -> dict[str, float]:
    default_sdk_order = [cfg.default_joint_pos[name] for name in cfg.sdk_joint_order]
    return {name: q - q0 for name, q, q0 in zip(cfg.sdk_joint_order, joint_pos_sdk_order, default_sdk_order)}


def compact_joint_summary(deltas: dict[str, float]) -> str:
    warns = [(n, d) for n, d in deltas.items() if abs(d) > JOINT_WARN_THRESHOLD]
    ok_count = len(deltas) - len(warns)
    if not warns:
        return f"{ok_count}/{len(deltas)} ok"
    warn_str = " ".join(f"{n}(Δ{d:+.2f})" for n, d in warns)
    return f"{ok_count}/{len(deltas)} ok, {len(warns)} WARN -> {warn_str}"


def verbose_joint_lines(cfg: "dr.DeployConfig", joint_pos_sdk_order: list[float], deltas: dict[str, float]) -> list[str]:
    lines = []
    default_sdk_order = [cfg.default_joint_pos[name] for name in cfg.sdk_joint_order]
    for name, q, q0 in zip(cfg.sdk_joint_order, joint_pos_sdk_order, default_sdk_order):
        d = deltas[name]
        flag = "WARN large deviation" if abs(d) > JOINT_WARN_THRESHOLD else "ok"
        lines.append(f"    {name:16s} q={q:+.3f}  default={q0:+.3f}  delta={d:+.3f}  [{flag}]")
    return lines


def print_summary(
    low_state_health: TopicHealth,
    sportmode_health: TopicHealth,
    joint_tracker: JointTracker,
    gravity_tracker: GravityTracker,
) -> None:
    print("\n" + "=" * 74)
    print("PRE-FLIGHT SUMMARY")
    print("=" * 74)
    verdict: list[str] = []

    print("\nConnectivity:")
    for health in (low_state_health, sportmode_health):
        if health.count == 0:
            flag = "NEVER CONNECTED"
        elif health.ever_stale:
            flag = "WENT STALE DURING RUN"
        else:
            flag = "OK"
        print(f"  {health.name:20s} {health.count} messages, ~{health.rate_hz():.0f} Hz avg  [{flag}]")
    if low_state_health.count > 0 and not low_state_health.ever_stale:
        verdict.append("[OK]   Connectivity confirmed working (rt/lowstate steady for the whole run).")
    else:
        verdict.append("[FAIL] rt/lowstate connectivity issue -- check network_interface, robot power, cabling.")

    print("\nIMU / gravity:")
    gstats = gravity_tracker.stats()
    if gstats is None:
        print("  no data")
        verdict.append("[FAIL] No IMU data received -- cannot verify orientation convention.")
    else:
        gmin, gmax, gmean = gstats
        print(f"  error min={gmin:.3f} max={gmax:.3f} mean={gmean:.3f} (tolerance {GRAVITY_TOLERANCE})")
        if gmax < GRAVITY_TOLERANCE:
            verdict.append("[OK]   IMU/gravity convention confirmed correct throughout the run.")
        else:
            verdict.append(f"[WARN] projected_gravity exceeded tolerance at some point (max error {gmax:.3f}) "
                            "-- investigate before trusting orientation-derived observations.")

    print("\nJoint mapping (deviation from default_joint_pos, range over the run):")
    stable_warns, noisy_warns = [], []
    for name in joint_tracker.names:
        lo, hi = joint_tracker.min_delta[name], joint_tracker.max_delta[name]
        if lo is None:
            print(f"  {name:16s} no data")
            continue
        spread = hi - lo
        is_warn = max(abs(lo), abs(hi)) > JOINT_WARN_THRESHOLD
        stability = "stable" if spread < STABLE_RANGE_THRESHOLD else f"varying by {spread:.3f}"
        flag = "WARN" if is_warn else "ok"
        print(f"  {name:16s} [{lo:+.3f}, {hi:+.3f}]  ({stability})  [{flag}]")
        if is_warn:
            (stable_warns if spread < STABLE_RANGE_THRESHOLD else noisy_warns).append(name)

    if not stable_warns and not noisy_warns:
        verdict.append("[OK]   All joint readings within expected range of default_joint_pos.")
    if stable_warns:
        verdict.append(
            f"[INFO] {len(stable_warns)} joint(s) show a STABLE large deviation from default_joint_pos: "
            f"{', '.join(stable_warns)}. This alone doesn't confirm a mapping bug -- likely the robot's own "
            "onboard-controller pose differs from this policy's trained default. Recommended: manually move "
            "ONE joint by hand while this script runs (--verbose) and confirm only that joint's reading changes."
        )
    if noisy_warns:
        verdict.append(
            f"[WARN] {len(noisy_warns)} joint(s) show a large AND unstable/varying deviation: "
            f"{', '.join(noisy_warns)}. Unlike a stable offset, this could indicate a loose connection or "
            "noisy sensor -- investigate before proceeding."
        )

    verdict.append(f"[TODO] Physically measure the robot's standing height and compare against "
                    f"DEFAULT_HEIGHT_M={dr.DEFAULT_HEIGHT_M} -- not checked by this script.")

    print("\nVERDICT:")
    for line in verdict:
        print(f"  {line}")
    print("=" * 74)


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only Tier-1 pre-flight checks (see deploy.md).")
    parser.add_argument("--config", type=str, default="deploy/configs/go2_locomotion.yaml")
    parser.add_argument("--network_interface", type=str, default=None, help="Overrides the config file's value.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between printed snapshots.")
    parser.add_argument(
        "--duration", type=float, default=None,
        help="If set, run for this many seconds then auto-stop and print the summary, instead of requiring Ctrl+C.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Also print the full per-joint breakdown and full observation vector every interval, not just "
        "the compact status line.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = dr.load_config(args.config, args.network_interface)
    sdk_to_isaac, _ = dr.build_joint_index_maps(cfg)
    default_joint_pos_isaac_order = [cfg.default_joint_pos[name] for name in cfg.isaac_joint_order]

    state = dr.LatestRobotState(num_joints=12)
    low_state_health = TopicHealth("rt/lowstate")
    sportmode_health = TopicHealth("rt/sportmodestate")
    joint_tracker = JointTracker(cfg.sdk_joint_order)
    gravity_tracker = GravityTracker()

    ChannelFactoryInitialize(0, cfg.network_interface)

    def _on_low_state(msg: LowState_):
        state.update_from_low_state(msg)
        low_state_health.mark()

    def _on_sportmode_state(msg: SportModeState_):
        state.update_from_sportmode_state(msg)
        sportmode_health.mark()

    low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_sub.Init(_on_low_state, 10)
    sportmode_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sportmode_sub.Init(_on_sportmode_state, 10)

    print(f"[INFO] Subscribed on interface '{cfg.network_interface}'. This script NEVER publishes anything "
          f"-- it cannot move the robot. Ctrl+C to stop (or pass --duration).")
    print(f"[INFO] Compact status by default -- pass --verbose for the full per-joint/observation dump. A "
          f"run summary with an explicit verdict prints when this stops.")

    start_time = time.monotonic()
    try:
        while True:
            time.sleep(args.interval)
            low_state_health.poll_stale()
            sportmode_health.poll_stale()
            elapsed = time.monotonic() - start_time

            print(f"\n=== snapshot @ {time.strftime('%H:%M:%S')} (T+{elapsed:.1f}s) ===")
            if not state.ready:
                print(f"  Connectivity : {low_state_health.compact_status()}   "
                      f"{sportmode_health.compact_status()}")
                print("  (no LowState received yet -- nothing else to report)")
            else:
                snap = state.snapshot()
                deltas = compute_joint_deltas(cfg, snap.joint_pos)
                joint_tracker.update(deltas)
                gravity = dr.quat_rotate_inverse_wxyz(snap.quat_wxyz, (0.0, 0.0, -1.0))
                err = gravity_error(gravity)
                gravity_tracker.update(err)

                print(f"  Connectivity : {low_state_health.compact_status()}   "
                      f"{sportmode_health.compact_status()}")
                print(f"  Gravity      : {'PASS' if err < GRAVITY_TOLERANCE else 'WARN'} "
                      f"(err={err:.3f}, tolerance={GRAVITY_TOLERANCE})")
                print(f"  Joints       : {compact_joint_summary(deltas)}")
                print(f"  Height       : not yet manually verified (DEFAULT_HEIGHT_M={dr.DEFAULT_HEIGHT_M})")

                if args.verbose:
                    print("  --- verbose ---")
                    print(f"  quat_wxyz={tuple('%.3f' % v for v in snap.quat_wxyz)}")
                    print(f"  sportmode_velocity={tuple('%.3f' % v for v in snap.sportmode_velocity)}")
                    print("  joint positions (SDK order):")
                    for line in verbose_joint_lines(cfg, snap.joint_pos, deltas):
                        print(line)
                    obs = dr.build_observation(
                        cfg, snap, sdk_to_isaac, default_joint_pos_isaac_order,
                        manual_cmd=[0.0] * 6, last_action=[0.0] * 12,
                    )
                    values = obs[0].tolist()
                    labels = ["base_lin_vel"] * 3 + ["base_ang_vel"] * 3 + ["projected_gravity"] * 3
                    if cfg.has_height_obs:
                        labels += ["base_height"]
                    labels += (["velocity_commands"] * 6 + ["joint_pos_rel"] * 12 + ["joint_vel"] * 12
                               + ["last_action"] * 12)
                    print(f"  observation vector ({len(values)}-dim, has_height_obs={cfg.has_height_obs}):")
                    for label, value in zip(labels, values):
                        print(f"    {label:18s} {value:+.4f}")

            if args.duration is not None and elapsed >= args.duration:
                break
    except KeyboardInterrupt:
        print()

    print_summary(low_state_health, sportmode_health, joint_tracker, gravity_tracker)


if __name__ == "__main__":
    main()
