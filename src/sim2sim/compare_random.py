# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report on a policy driven with the SAME random command schedule in Isaac and in MuJoCo.

Read-only: no simulator, no policy, only numpy + yaml. For every seed it reads three files written by
`src/sim2sim/run_sim2sim_eval.sh` into one directory:

    seed<N>.yaml          the command schedule (src/sim2sim/random_commands.py)
    seed<N>_isaac.npz     `play.py --scenario seed<N>.yaml --trace_out ...`
    seed<N>_mujoco.npz    `sim2sim_mujoco.py --scenario seed<N>.yaml --trace_out ...`

and computes the same metrics from both traces, with one definition (the trace fields have the same names in both):

* tracking error per command component: RMS of (measured - commanded) for body-frame lin_x, lin_y, ang_z (the base
  origin velocity), pitch and lean (from the base quaternion, same convention as `deploy_real.quat_to_pitch_lean`);
* height error: RMS of (base z - 0.337 m), the fixed target `height_penalty` holds in training;
* falls: steps where the base or arm touched the ground (training's `body_contact` termination; the MuJoCo runner
  uses the same 1 N threshold). The robot is reset to the default pose after a fall in both simulators;
* peak / RMS joint torque.

The first `--settle_s` seconds of every segment, and of every recovery after a fall, are excluded from the tracking
statistics (the transient after a command change or a reset), the same for both simulators.

    python3 src/sim2sim/compare_random.py --dir logs/sim2sim_eval_24999/random [--out report.md]
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import yaml

NOMINAL_HEIGHT_M = 0.337
TERMS = ("lin_x", "lin_y", "ang_z", "pitch", "lean", "height")
SIMS = ("isaac", "mujoco")


def load_trace(path: str) -> dict[str, np.ndarray]:
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            try:
                out[k] = z[k]
            except ValueError:  # an object array, not needed here
                pass
    return out


def pitch_lean(quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pitch, lean/roll) per step; the same XYZ-Euler convention as deploy_real.quat_to_pitch_lean."""
    w, x, y, z = (quat_wxyz[:, i] for i in range(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return pitch, roll


def segment_index(segments: list[dict], dt: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Per step: the segment id and the commanded 5-vector."""
    seg_id, cmds = [], []
    for i, s in enumerate(segments):
        n = round(float(s["duration"]) / dt)
        seg_id += [i] * n
        cmds += [[float(v) for v in s["command"]]] * n
    seg_id, cmds = np.array(seg_id[:n_steps]), np.array(cmds[:n_steps])
    return seg_id, cmds


def analyse(trace: dict[str, np.ndarray], segments: list[dict], settle_s: float) -> dict:
    dt = float(trace["dt"])
    n = min(len(trace["root_quat"]), sum(round(float(s["duration"]) / dt) for s in segments))
    seg_id, cmd = segment_index(segments, dt, n)
    fell = trace["fell"][:n] > 0.5 if "fell" in trace else np.zeros(n, dtype=bool)

    pitch, lean = pitch_lean(trace["root_quat"][:n])
    lv, av = trace["lin_vel_origin_b"][:n], trace["ang_vel_b"][:n]
    err = {
        "lin_x": lv[:, 0] - cmd[:, 0],
        "lin_y": lv[:, 1] - cmd[:, 1],
        "ang_z": av[:, 2] - cmd[:, 2],
        "pitch": pitch - cmd[:, 3],
        "lean": lean - cmd[:, 4],
        "height": trace["root_pos"][:n, 2] - NOMINAL_HEIGHT_M,
    }

    # steps since the segment started or since the last fall reset: both are excluded while < settle
    settle = int(round(settle_s / dt))
    since = np.zeros(n, dtype=int)
    start = 0
    for k in range(n):
        if k == 0 or seg_id[k] != seg_id[k - 1]:
            start = k
        if k > 0 and fell[k - 1]:
            start = k
        since[k] = k - start
    valid = (since >= settle) & ~fell

    tau = np.abs(trace["torque"][:n]) if "torque" in trace else np.zeros((n, 1))
    feet = (trace["foot_force"][:n] > 1.0).mean(axis=1) if "foot_force" in trace else np.full(n, np.nan)
    per_seg = []
    for i, s in enumerate(segments):
        m = seg_id == i
        mv = m & valid
        row = {
            "name": s["name"],
            "command": [float(v) for v in s["command"]],
            "samples": int(mv.sum()),
            "falls": int(fell[m].sum()),
            "peak_tau": float(tau[m].max()) if m.any() else float("nan"),
        }
        for t in TERMS:
            row[t] = float(np.sqrt((err[t][mv] ** 2).mean())) if mv.any() else float("nan")
        row["lin_xy"] = float(np.sqrt((err["lin_x"][mv] ** 2 + err["lin_y"][mv] ** 2).mean())) if mv.any() else float("nan")
        per_seg.append(row)

    total = {t: float(np.sqrt((err[t][valid] ** 2).mean())) if valid.any() else float("nan") for t in TERMS}
    total.update(
        falls=int(fell.sum()),
        segments_with_fall=len({int(seg_id[k]) for k in np.nonzero(fell)[0]}),
        peak_tau=float(tau.max()),
        rms_tau=float(np.sqrt((tau**2).mean())),
        feet_down=float(np.nanmean(feet)),
        steps=int(n),
        valid_steps=int(valid.sum()),
        seconds=n * dt,
    )
    return {"total": total, "segments": per_seg, "err": err, "valid": valid, "seg_id": seg_id, "cmd": cmd}


def fmt_cmd(c: list[float]) -> str:
    return "[" + " ".join(f"{v:+.2f}" for v in c) + "]"


def correlation(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() < 1e-9 or b[ok].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def discover_seeds(directory: str) -> list[int]:
    seeds = []
    for p in glob.glob(os.path.join(directory, "seed*.yaml")):
        m = re.fullmatch(r"seed(\d+)\.yaml", os.path.basename(p))
        if m and all(os.path.exists(os.path.join(directory, f"seed{m.group(1)}_{s}.npz")) for s in SIMS):
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def build_report(directory: str, settle_s: float, title: str) -> str:
    seeds = discover_seeds(directory)
    if not seeds:
        raise SystemExit(f"no complete seed<N>.yaml + seed<N>_isaac.npz + seed<N>_mujoco.npz sets in {directory}")
    out: list[str] = [f"# {title}", ""]
    results: dict[int, dict] = {}
    for sd in seeds:
        with open(os.path.join(directory, f"seed{sd}.yaml")) as f:
            segs = yaml.safe_load(f)["segments"]
        results[sd] = {
            s: analyse(load_trace(os.path.join(directory, f"seed{sd}_{s}.npz")), segs, settle_s) for s in SIMS
        }
        results[sd]["segs"] = segs

    n_seg = sum(len(r["segs"]) for r in results.values())
    secs = sum(r["isaac"]["total"]["seconds"] for r in results.values())
    out += [
        f"Seeds {seeds}: {n_seg} random command segments, {secs:.0f} s per simulator. Every command component is drawn "
        "uniformly over the training range; ~12% of segments are all-zero. Both simulators replay the identical "
        f"schedule with nominal physics (no disturbances). The first {settle_s:.1f} s after each command change or "
        "fall-reset is excluded from the tracking errors.",
        "",
    ]

    # ---- aggregate over all seeds (pool the valid samples) ----
    def pooled(sim: str) -> dict:
        errs = {t: np.concatenate([r[sim]["err"][t][r[sim]["valid"]] for r in results.values()]) for t in TERMS}
        tot = {t: float(np.sqrt((errs[t] ** 2).mean())) for t in TERMS}
        tot["falls"] = sum(r[sim]["total"]["falls"] for r in results.values())
        tot["seg_falls"] = sum(r[sim]["total"]["segments_with_fall"] for r in results.values())
        tot["peak_tau"] = max(r[sim]["total"]["peak_tau"] for r in results.values())
        tot["rms_tau"] = float(np.sqrt(np.mean([r[sim]["total"]["rms_tau"] ** 2 for r in results.values()])))
        return tot

    agg = {s: pooled(s) for s in SIMS}
    out += ["## Overall (all seeds pooled)", "", "| metric | Isaac | MuJoCo | MuJoCo / Isaac |", "|---|---|---|---|"]
    units = {"lin_x": "m/s", "lin_y": "m/s", "ang_z": "rad/s", "pitch": "rad", "lean": "rad", "height": "m"}
    for t in TERMS:
        a, m = agg["isaac"][t], agg["mujoco"][t]
        out.append(f"| {t} RMS error ({units[t]}) | {a:.3f} | {m:.3f} | {m / a if a > 1e-9 else float('nan'):.2f}x |")
    out.append(f"| falls (total) | {agg['isaac']['falls']} | {agg['mujoco']['falls']} | |")
    out.append(
        f"| segments with a fall (of {n_seg}) | {agg['isaac']['seg_falls']} | {agg['mujoco']['seg_falls']} | |"
    )
    out.append(f"| peak joint torque (Nm) | {agg['isaac']['peak_tau']:.1f} | {agg['mujoco']['peak_tau']:.1f} | |")
    out.append(f"| RMS joint torque (Nm) | {agg['isaac']['rms_tau']:.2f} | {agg['mujoco']['rms_tau']:.2f} | |")
    out.append("")

    # ---- standing vs moving vs high-demand ----
    def bucket(sim: str, pick) -> dict:
        rows = [(r[sim]["segments"][i], r["segs"][i]) for r in results.values() for i in range(len(r["segs"]))]
        sel = [row for row, _ in rows if pick(np.array(row["command"]))]
        out_d = {"n": len(sel)}
        for t in ("lin_xy", "ang_z", "pitch", "lean", "height"):
            vals = [row[t] for row in sel if np.isfinite(row[t])]
            out_d[t] = float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
        out_d["falls"] = sum(row["falls"] for row in sel)
        return out_d

    buckets = {
        "standing (all-zero command)": lambda c: bool(np.all(c == 0)),
        "slow (max |v| or |w| <= 0.5)": lambda c: bool(np.any(c != 0)) and bool(np.max(np.abs(c[:3])) <= 0.5),
        "fast (max |v| or |w| > 0.5)": lambda c: bool(np.max(np.abs(c[:3])) > 0.5),
        "large pitch/lean (|pitch|>0.3 or |lean|>0.15)": lambda c: bool(abs(c[3]) > 0.3 or abs(c[4]) > 0.15),
    }
    out += [
        "## By command type (RMS of per-segment errors; lin_xy = combined x/y error)", "",
        "| command type | segs | | lin_xy | ang_z | pitch | lean | height | falls |", "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, pick in buckets.items():
        for sim in SIMS:
            b = bucket(sim, pick)
            out.append(
                f"| {name if sim == 'isaac' else ''} | {b['n'] if sim == 'isaac' else ''} | {sim} | {b['lin_xy']:.3f} | "
                f"{b['ang_z']:.3f} | {b['pitch']:.3f} | {b['lean']:.3f} | {b['height']:.3f} | {b['falls']} |"
            )
    out.append("")

    # ---- do the two simulators agree on which commands are hard? ----
    out += ["## Do Isaac and MuJoCo agree on which segments are hard?", "",
            "Pearson correlation of the per-segment error across the two simulators (1.0 = they find the same "
            "commands hard; near 0 = the error is not driven by the command in the same way).", ""]
    out += ["| error | correlation | segments |", "|---|---|---|"]
    for t in ("lin_xy", "ang_z", "pitch", "lean", "height"):
        a = [row[t] for r in results.values() for row in r["isaac"]["segments"]]
        m = [row[t] for r in results.values() for row in r["mujoco"]["segments"]]
        out.append(f"| {t} | {correlation(a, m):+.2f} | {len(a)} |")
    out.append("")

    # ---- worst segments ----
    flat = []
    for sd, r in results.items():
        for i, seg in enumerate(r["segs"]):
            ri, rm = r["isaac"]["segments"][i], r["mujoco"]["segments"][i]
            flat.append((sd, i, seg["name"], ri, rm))
    worst = sorted(flat, key=lambda x: -(x[4]["falls"] * 100 + x[3]["falls"] * 100 + np.nan_to_num(x[4]["lin_xy"]) + np.nan_to_num(x[3]["lin_xy"])))[:10]
    out += ["## Ten hardest segments (falls first, then velocity error)", "",
            "| seed | segment | command [x y yaw pitch lean] | Isaac lin_xy / yaw / falls | MuJoCo lin_xy / yaw / falls |",
            "|---|---|---|---|---|"]
    for sd, i, name, ri, rm in worst:
        out.append(
            f"| {sd} | {name} | {fmt_cmd(ri['command'])} | {ri['lin_xy']:.2f} / {ri['ang_z']:.2f} / {ri['falls']} | "
            f"{rm['lin_xy']:.2f} / {rm['ang_z']:.2f} / {rm['falls']} |"
        )
    out.append("")

    # ---- per seed ----
    out += ["## Per seed", "", "| seed | sim | falls | lin_x | lin_y | ang_z | pitch | lean | height | peak tau |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for sd, r in results.items():
        for sim in SIMS:
            t = r[sim]["total"]
            out.append(
                f"| {sd} | {sim} | {t['falls']} | {t['lin_x']:.3f} | {t['lin_y']:.3f} | {t['ang_z']:.3f} | "
                f"{t['pitch']:.3f} | {t['lean']:.3f} | {t['height']:.3f} | {t['peak_tau']:.1f} |"
            )
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="directory with seed<N>.yaml and seed<N>_{isaac,mujoco}.npz")
    parser.add_argument("--settle_s", type=float, default=1.0, help="seconds excluded after each command change/fall")
    parser.add_argument("--title", default="Sim2sim random-command evaluation")
    parser.add_argument("--out", default=None, help="also write the report (markdown) to this file")
    args = parser.parse_args()
    report = build_report(args.dir, args.settle_s, args.title)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()
