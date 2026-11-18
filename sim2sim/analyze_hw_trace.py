# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyse a hardware (or sim) policy trace for the ~25 Hz 'shaking' limit cycle.

Pure numpy -- no Isaac Sim, no torch, no robot. Reads a `policy_trace.npz` written by `deploy/deploy_real.py`
(`logs/hardware_dashboard/<ts>/policy_trace.npz`; fields: step, t, obs[:,47], raw_action[:,12], sent_target,
kp, n_slew_clipped, n_tethered, tau_est, joint_vel, foot_force, quat_wxyz, gyro, joint_order) OR a sim trace
from sim2sim_mujoco.py / play.py (obs, action, ...). It reports, per time window:

  - |Δaction|            mean absolute per-step change of the raw policy action (the thing that buzzes)
  - lag1                 lag-1 autocorrelation of Δaction; ~-1 means the target flips every tick = Nyquist buzz
  - nearNyq%             fraction of action power within 10% of Nyquist (fs/2); high => every-tick oscillation
  - gyro_rms             body angular-rate RMS (rad/s); the body actually shaking
  - tau_rms / tau_pk     joint torque RMS / peak (Nm)
  - jvel_rms             joint velocity RMS (rad/s)

and finds the ONSET time (first sustained window whose lag1 crosses --onset_lag1), the worst joints, and the
command / foot-contact state at onset. The command is decoded from the 47-dim observation layout
(ang_vel 0:3, gravity 3:6, command 6:11, joint_pos 11:23, joint_vel 23:35, last_action 35:47).

Usage:
    python3 sim2sim/analyze_hw_trace.py logs/hardware_dashboard/2026-11-18_14-19-26/policy_trace.npz
    python3 sim2sim/analyze_hw_trace.py <trace.npz> --window_s 2 --onset_lag1 -0.5 --out report.md
    python3 sim2sim/analyze_hw_trace.py hw.npz isaac.npz mujoco.npz   # compare several (pooled buzz metrics)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

CMD_SLICE = slice(6, 11)  # command terms within the 47-dim obs (see module docstring)
ACTION_KEYS = ("raw_action", "action")  # hardware vs sim field name for the policy action


def load(path: str) -> dict[str, np.ndarray]:
    out = {}
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            try:
                out[k] = z[k]
            except ValueError:
                pass
    return out


def action_of(t: dict[str, np.ndarray]) -> np.ndarray:
    for k in ACTION_KEYS:
        if k in t:
            return np.asarray(t[k], dtype=float)
    raise KeyError(f"no action field; looked for {ACTION_KEYS}, have {sorted(t)}")


def dt_of(t: dict[str, np.ndarray]) -> float:
    if "t" in t and len(t["t"]) > 1:
        return float(np.median(np.diff(t["t"])))
    if "dt" in t:
        return float(t["dt"])
    return 0.02  # 50 Hz fallback


def lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D signal (0 if flat)."""
    x = x - x.mean()
    denom = float((x * x).sum())
    return float((x[1:] * x[:-1]).sum() / denom) if denom > 1e-12 else 0.0


def buzz_metrics(act: np.ndarray, dt: float, tau=None, jv=None, gyro=None) -> dict:
    """Buzz metrics over a slab of consecutive steps (act: (T,12))."""
    out: dict[str, float] = {"steps": len(act)}
    if len(act) < 3:
        return out
    d = np.diff(act, axis=0)
    out["dact_mean"] = float(np.abs(d).mean())
    out["lag1"] = float(np.median([lag1(d[:, j]) for j in range(d.shape[1])]))
    x = act - act.mean(axis=0, keepdims=True)
    spec = (np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(axis=1)
    fr = np.fft.rfftfreq(len(x), dt)
    fs = 1.0 / dt
    denom = max(float(spec[1:].sum()), 1e-12)
    out["near_nyq_pct"] = float(spec[fr >= 0.9 * fs / 2].sum() / denom * 100.0)
    if tau is not None:
        out["tau_rms"] = float(np.sqrt((tau**2).mean()))
        out["tau_pk"] = float(np.abs(tau).max())
    if jv is not None:
        out["jvel_rms"] = float(np.sqrt((jv**2).mean()))
    if gyro is not None:
        out["gyro_rms"] = float(np.sqrt((gyro**2).sum(axis=1).mean()))
    return out


def windows(n: int, step: int):
    for a in range(0, n, step):
        yield a, min(a + step, n)


def analyse_one(path: str, window_s: float, onset_lag1: float) -> list[str]:
    t = load(path)
    act = action_of(t)
    dt = dt_of(t)
    fs = 1.0 / dt
    ts = t["t"] - t["t"][0] if "t" in t else np.arange(len(act)) * dt
    tau = t.get("tau_est")
    jv = t.get("joint_vel")
    gyro = t.get("gyro")
    obs = t.get("obs")
    names = [str(x) for x in t["joint_order"]] if "joint_order" in t else [f"j{j}" for j in range(act.shape[1])]
    w = max(2, int(round(window_s / dt)))

    lines = [f"## {os.path.basename(os.path.dirname(path)) or os.path.basename(path)}",
             f"steps={len(act)}  dur={ts[-1]:.1f}s  dt={dt * 1000:.1f}ms  fs={fs:.1f}Hz  Nyquist={fs / 2:.1f}Hz"]
    if "kp" in t:
        kp = t["kp"]
        lines.append(f"kp {kp.min():.0f}..{kp.max():.0f}  "
                     f"slew_clipped={int(t.get('n_slew_clipped', np.zeros(1)).sum())}  "
                     f"tethered={int(t.get('n_tethered', np.zeros(1)).sum())}")
    lines.append("")
    lines.append("| t(s) | |Δact| | lag1 | nearNyq% | gyro_rms | tau_pk | " + ("cmd[x y yaw pitch lean] |" if obs is not None else "|"))
    lines.append("|---|---|---|---|---|---|" + ("---|" if obs is not None else ""))
    onset_t = None
    for a, b in windows(len(act), w):
        m = buzz_metrics(act[a:b], dt,
                         tau[a:b] if tau is not None else None,
                         jv[a:b] if jv is not None else None,
                         gyro[a:b] if gyro is not None else None)
        if m.get("steps", 0) < 3:
            continue
        cmd_s = ""
        if obs is not None:
            c = obs[a:b, CMD_SLICE].mean(0)
            cmd_s = f" [{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f} {c[3]:+.2f} {c[4]:+.2f}] |"
        lines.append(f"| {ts[a]:.0f} | {m['dact_mean']:.4f} | {m['lag1']:+.2f} | {m['near_nyq_pct']:.0f} | "
                     f"{m.get('gyro_rms', float('nan')):.3f} | {m.get('tau_pk', float('nan')):.1f} |{cmd_s}")
        if onset_t is None and m["lag1"] <= onset_lag1 and m["dact_mean"] > 0.02:
            onset_t = ts[a]

    # worst joints over the WORST window (max mean |Δaction|), not the final (often quiet) one
    best_a, best_v = 0, -1.0
    for a, b in windows(len(act), w):
        if b - a >= 3:
            v = float(np.abs(np.diff(act[a:b], axis=0)).mean())
            if v > best_v:
                best_a, best_v = a, v
    wslice = slice(best_a, min(best_a + w, len(act)))
    d = np.abs(np.diff(act[wslice], axis=0)).mean(0)
    worst = np.argsort(d)[::-1][:4]
    lines.append("")
    lines.append(f"**Onset** (first window with lag1 <= {onset_lag1} and |Δact| > 0.02): "
                 + (f"**t = {onset_t:.1f}s**" if onset_t is not None else "none detected"))
    lines.append(f"**Worst joints (worst window, t≈{ts[best_a]:.0f}s)**: "
                 + ", ".join(f"{names[j]} (|Δ|={d[j]:.3f})" for j in worst))
    if obs is not None and onset_t is not None:
        oi = int(np.searchsorted(ts, onset_t))
        c = obs[oi, CMD_SLICE]
        foot = t.get("foot_force")
        foot_s = f", foot_force_sum={float(foot[oi].sum()):.0f}" if foot is not None else ""
        lines.append(f"**At onset**: cmd=[{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f} {c[3]:+.2f} {c[4]:+.2f}]{foot_s}")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="one or more policy_trace.npz / sim trace .npz")
    ap.add_argument("--window_s", type=float, default=2.0, help="window size for the per-window table (s)")
    ap.add_argument("--onset_lag1", type=float, default=-0.5, help="lag-1 autocorr threshold that marks buzz onset")
    ap.add_argument("--out", default=None, help="also write the markdown report here")
    args = ap.parse_args()
    lines = ["# Policy-trace buzz analysis", ""]
    for p in args.traces:
        lines += analyse_one(p, args.window_s, args.onset_lag1)
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
