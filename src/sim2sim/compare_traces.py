# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare per-step traces (.npz) from different simulators (or the robot) running the same policy.

Read-only: no simulator, no policy, only numpy. Each trace is written by `--trace_out` in
`src/sim2sim/sim2sim_mujoco.py` or `src/play/play.py` (same field names; fields a source lacks are skipped).

Prints (1) per observation term mean/std/min/max for every trace and the gap between the first two, and
(2) the "zero-command quality" metrics: base velocity RMS/peak, joint velocity RMS, action jitter, dominant
action frequency, how far the policy's targets sit from the default pose, peak torque, foot contact.

Usage:
    python3 src/sim2sim/compare_traces.py isaac_zero.npz mujoco_zero.npz [--settle_s 2.0]
    python3 src/sim2sim/compare_traces.py a.npz a.npz        # self-comparison sanity check: all gaps must be 0
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# Observation layouts by policy input size, mirroring deploy_real.OBS_LAYOUT (cmd_width, has_height_obs, has_lin_vel):
#   47: single frame, 5-wide command   48: 6-wide command incl. height (the stacked policy's frame, 48 x 6 = 288)
#   50: with base_lin_vel   51/52: legacy 6-wide command (+ base_height).
# The term order is that of deploy_real.build_observation / ObservationsCfg. The slice of every term depends on the
# size, so it is derived from the trace's own observation width and never assumed to be 50.
OBS_LAYOUT = {
    47: (5, False, False),
    48: (6, False, False),
    50: (5, False, True),
    51: (6, False, True),
    52: (6, True, True),
}
ACTION_SCALE = 0.25


def obs_terms(obs_dim: int) -> list[tuple[str, int, int]]:
    """[(term name, start, stop)] for an observation of width `obs_dim`."""
    if obs_dim not in OBS_LAYOUT:
        raise ValueError(f"unknown observation width {obs_dim}; known widths are {sorted(OBS_LAYOUT)}")
    cmd_width, has_height, has_lin_vel = OBS_LAYOUT[obs_dim]
    widths = []
    if has_lin_vel:
        widths.append(("base_lin_vel", 3))
    widths += [("base_ang_vel", 3), ("projected_gravity", 3)]
    if has_height:
        widths.append(("base_height", 1))
    widths += [("command", cmd_width), ("joint_pos_rel", 12), ("joint_vel", 12), ("last_action", 12)]
    terms, i = [], 0
    for name, w in widths:
        terms.append((name, i, i + w))
        i += w
    assert i == obs_dim, (i, obs_dim)
    return terms


def term_slice(obs_dim: int, name: str) -> slice | None:
    for n, a, b in obs_terms(obs_dim):
        if n == name:
            return slice(a, b)
    return None


def newest_frame(obs: np.ndarray) -> np.ndarray:
    """If the trace stacks a history of frames (stacked width = base * H, e.g. 282 = 47 * 6), return just the
    NEWEST single frame so the per-term stats stay meaningful. Single-frame traces pass through unchanged.

    Assumes the deploy/Isaac TERM-major layout (deploy_real.OBS_HISTORY_TERM_MAJOR): the stacked vector is
    [term0's H samples, term1's H samples, ...], each term's samples newest-LAST. The newest frame is the last
    `w` columns of each term's H*w block, reassembled in term order."""
    width = obs.shape[1]
    if width in OBS_LAYOUT:
        return obs
    base = h = None
    for hh in range(2, 13):
        if width % hh == 0 and (width // hh) in OBS_LAYOUT:
            base, h = width // hh, hh
            break
    if base is None:
        return obs  # fall through: obs_terms() will raise with a clear message
    out = np.empty((obs.shape[0], base), dtype=obs.dtype)
    for _, a, b in obs_terms(base):
        w = b - a
        block_start = a * h  # term blocks are consecutive H*w_i; sum of preceding widths is `a`
        out[:, a:b] = obs[:, block_start + (h - 1) * w: block_start + h * w]  # newest sample of this term
    return out


def load(path: str) -> dict[str, np.ndarray]:
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            try:
                out[k] = z[k]
            except ValueError:  # an object array (e.g. an older trace's joint-name list): not needed here
                pass
    return out


def term_stats(obs: np.ndarray, settle: int) -> dict[str, tuple[float, float, float, float]]:
    out = {}
    for name, a, b in obs_terms(obs.shape[1]):
        x = obs[settle:, a:b]
        out[name] = (float(x.mean()), float(x.std()), float(x.min()), float(x.max()))
    return out


def dominant_freq(x: np.ndarray, dt: float) -> float:
    """Frequency (Hz) with the most power in the mean action signal, ignoring the DC bin."""
    x = x - x.mean(axis=0, keepdims=True)
    spec = (np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(axis=1)
    freqs = np.fft.rfftfreq(x.shape[0], dt)
    return float(freqs[1 + int(np.argmax(spec[1:]))]) if len(spec) > 1 else 0.0


def metrics(t: dict[str, np.ndarray], settle_s: float) -> dict[str, float]:
    dt = float(t["dt"]) if "dt" in t else 0.02
    s = int(round(settle_s / dt))
    m: dict[str, float] = {"steps_used": float(len(t["obs"]) - s), "duration_s": (len(t["obs"]) - s) * dt}
    obs_dim = t["obs"].shape[1]
    # base linear velocity: the trace's own field when present; else the observation's base_lin_vel term, which only
    # a policy that observes it (not the 47-dim one) has
    lv_slice = term_slice(obs_dim, "base_lin_vel")
    lv = t["lin_vel_com_b"] if "lin_vel_com_b" in t else (t["obs"][:, lv_slice] if lv_slice is not None else None)
    if lv is not None:
        lv = lv[s:]
        m["lin_vel_xy_rms"] = float(np.sqrt((lv[:, :2] ** 2).sum(axis=1).mean()))
        m["lin_vel_xy_peak"] = float(np.abs(lv[:, :2]).max())
        m["lin_vel_x_mean"] = float(lv[:, 0].mean())
        m["lin_vel_y_mean"] = float(lv[:, 1].mean())
    av = t["obs"][s:, term_slice(obs_dim, "base_ang_vel")]
    m["ang_vel_rms"] = float(np.sqrt((av**2).sum(axis=1).mean()))
    jv = t["obs"][s:, term_slice(obs_dim, "joint_vel")]
    m["joint_vel_rms"] = float(np.sqrt((jv**2).mean()))
    act = t["action"][s:]
    m["action_step_rms"] = float(np.sqrt((np.diff(act, axis=0) ** 2).mean()))
    m["action_dominant_hz"] = dominant_freq(act, dt)
    m["target_dev_mean_deg"] = float(np.degrees(ACTION_SCALE * np.abs(act).mean()))
    m["target_dev_max_deg"] = float(np.degrees(ACTION_SCALE * np.abs(act.mean(axis=0)).max()))
    if "torque" in t:
        m["torque_peak_nm"] = float(np.abs(t["torque"][s:]).max())
        m["torque_rms_nm"] = float(np.sqrt((t["torque"][s:] ** 2).mean()))
    if "foot_force" in t:
        m["feet_down_fraction"] = float((t["foot_force"][s:] > 1.0).mean())
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", help="two or more .npz traces; the first is the reference")
    parser.add_argument("--settle_s", type=float, default=2.0, help="initial transient excluded from statistics")
    args = parser.parse_args()
    names = [os.path.basename(p) for p in args.traces]
    traces = [load(p) for p in args.traces]
    # Reduce stacked-history obs to the newest frame so per-term stats and slices stay single-frame. Keeps the
    # dedicated trace fields (lin_vel_origin_b, torque, ...) untouched; only the raw obs matrix is narrowed.
    for t in traces:
        if "obs" in t:
            t["obs"] = newest_frame(t["obs"])
    dt = float(traces[0]["dt"]) if "dt" in traces[0] else 0.02
    settle = int(round(args.settle_s / dt))

    print(f"=== Observation terms (mean / std over steps after {args.settle_s:.1f} s; shown per scalar element) ===")
    stats = [term_stats(t["obs"], settle) for t in traces]
    widths = {t["obs"].shape[1] for t in traces}
    if len(widths) != 1:
        raise SystemExit(f"traces have different observation widths {sorted(widths)}: not the same policy")
    for term, _, _ in obs_terms(widths.pop()):
        print(f"\n{term}")
        for n, st in zip(names, stats):
            mean, std, lo, hi = st[term]
            print(f"  {n:28s} mean={mean:+9.4f} std={std:8.4f} min={lo:+9.3f} max={hi:+9.3f}")
        if len(stats) >= 2:
            d = stats[1][term][0] - stats[0][term][0]
            r = stats[1][term][1] / max(stats[0][term][1], 1e-9)
            print(f"  {'-> gap (2nd - 1st)':28s} d(mean)={d:+9.4f}  std ratio={r:6.2f}x")

    print(f"\n=== Zero-command quality metrics (after {args.settle_s:.1f} s) ===")
    ms = [metrics(t, args.settle_s) for t in traces]
    keys = sorted(set().union(*[set(m) for m in ms]))
    print(f"{'metric':24s}" + "".join(f"{n[:22]:>24s}" for n in names))
    for k in keys:
        print(f"{k:24s}" + "".join(f"{m[k]:24.4f}" if k in m else f"{'-':>24s}" for m in ms))

    if len(traces) >= 2 and "lin_vel_origin_b" in traces[-1] and "lin_vel_com_b" in traces[-1]:
        t = traces[-1]
        diff = (t["lin_vel_com_b"] - t["lin_vel_origin_b"])[settle:]
        print(f"\n=== {names[-1]}: base_lin_vel reference point (COM vs frame origin) ===")
        print(f"  |COM - origin| velocity: mean {np.linalg.norm(diff, axis=1).mean():.4f} m/s, "
              f"peak {np.linalg.norm(diff, axis=1).max():.4f} m/s (this is omega x r the observation would be off by)")


if __name__ == "__main__":
    main()
