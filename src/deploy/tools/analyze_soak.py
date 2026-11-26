# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyse one instrumented soak run: per-phase tables, checks against pass rules, report and plots.

Layout it reads (all optional except the desktop folder; whatever is missing is reported as n/a):
    <run>/desktop/            meta.json summary.json metrics_1hz.csv rx_state.csv tx_cmd.csv loop.csv events.log
    <run>/jetson/relay/       meta.json summary.json metrics_1hz.csv rx_cmd.csv tx_state.csv publish_gaps.csv events.log
    <run>/jetson/fake_robot/  meta.json summary.json metrics_1hz.csv lowcmd_gaps.csv lowcmd_intervals.f32 events.log
    <run>/icmp_ping.log       `ping -D` output, one line per reply

Writes <run>/report.md, <run>/report.json and (if matplotlib is installed) <run>/plots/*.png.

Usage:  python src/deploy/tools/analyze_soak.py logs/relay_runs/<RUN_ID>

Needs numpy. Phase membership is taken from the `phase` column each component stamps on its own rows, so no
cross-machine clock alignment is needed for the tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re

import numpy as np

# ---- pass rules (edit here) ---------------------------------------------------------------------------------------
RULES = {
    "clean_state_loss_pct_max": 0.0,
    "clean_rtt_p95_ms_max": 40.0,
    "clean_rtt_max_ms_max": 100.0,
    "clean_cmd_gap_max_ms_max": 100.0,
    "loss_honesty_points": 1.0,           # measured loss within this many percentage points (or 3 sigma) of injected
    "recovery_factor_max": 1.5,           # recovery RTT p95 <= factor * max(clean RTT p95, 5 ms)
    "publish_p99_ms_max": 6.0,
    "publish_max_ms_max": 20.0,
    "publish_rate_tolerance_pct": 1.0,
    "relay_cpu_pct_max": 60.0,
    "desktop_overrun_pct_max": 0.1,
    "row_count_tolerance_pct": 1.0,
    "damp_tail_frames_min": 100,
}
CLEAN_PHASES = ("clean",)
SKIP_PHASES = ("setup",)

# ---- loading ------------------------------------------------------------------------------------------------------


def read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_csv(path: str):
    """{column: numpy array}; columns that are entirely numeric become float arrays, the rest object arrays."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return None
        cols = [[] for _ in header]
        for row in r:
            if len(row) != len(header):
                continue  # a torn last line from a killed process
            for i, v in enumerate(row):
                cols[i].append(v)
    out = {}
    for name, vals in zip(header, cols):
        try:
            out[name] = np.array([float(v) if v not in ("", "nan") else math.nan for v in vals], dtype=float)
        except ValueError:
            out[name] = np.array(vals, dtype=object)
    return out


def read_f32(path: str):
    return np.fromfile(path, dtype=np.float32).astype(float) if os.path.exists(path) else None


def parse_icmp(path: str):
    """(t_wall array, rtt_ms array, seq array) from `ping -D` lines, in file order."""
    if not os.path.exists(path):
        return None
    t, rtt, seq = [], [], []
    pat = re.compile(r"\[(\d+\.\d+)\].*icmp_seq=(\d+).*time=([\d.]+) ms")
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                t.append(float(m.group(1)))
                seq.append(int(m.group(2)))
                rtt.append(float(m.group(3)))
    return {"t": np.array(t), "rtt": np.array(rtt), "seq": np.array(seq)} if t else None


class Run:
    def __init__(self, root: str):
        self.root = root
        d, rj, fr = (os.path.join(root, "desktop"), os.path.join(root, "jetson", "relay"),
                     os.path.join(root, "jetson", "fake_robot"))
        self.desktop_dir, self.relay_dir, self.fake_dir = d, rj, fr
        self.d_meta, self.d_sum = read_json(os.path.join(d, "meta.json")), read_json(os.path.join(d, "summary.json"))
        self.r_meta, self.r_sum = read_json(os.path.join(rj, "meta.json")), read_json(os.path.join(rj, "summary.json"))
        self.f_meta, self.f_sum = read_json(os.path.join(fr, "meta.json")), read_json(os.path.join(fr, "summary.json"))
        self.d_rx = read_csv(os.path.join(d, "rx_state.csv"))
        self.d_tx = read_csv(os.path.join(d, "tx_cmd.csv"))
        self.d_loop = read_csv(os.path.join(d, "loop.csv"))
        self.d_met = read_csv(os.path.join(d, "metrics_1hz.csv"))
        self.r_rx = read_csv(os.path.join(rj, "rx_cmd.csv"))
        self.r_tx = read_csv(os.path.join(rj, "tx_state.csv"))
        self.r_met = read_csv(os.path.join(rj, "metrics_1hz.csv"))
        self.r_pub = read_csv(os.path.join(rj, "publish_gaps.csv"))
        self.f_met = read_csv(os.path.join(fr, "metrics_1hz.csv"))
        self.f_gaps = read_csv(os.path.join(fr, "lowcmd_gaps.csv"))
        self.f_raw = read_f32(os.path.join(fr, "lowcmd_intervals.f32"))
        self.icmp = parse_icmp(os.path.join(root, "icmp_ping.log"))


# ---- statistics helpers -------------------------------------------------------------------------------------------


def pct(a, p):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, p)) if a.size else math.nan


def stats(a, thresholds=(30, 50, 100, 150)):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"n": 0}
    out = {"n": int(a.size), "mean": float(a.mean()), "p50": pct(a, 50), "p95": pct(a, 95), "p99": pct(a, 99),
           "max": float(a.max())}
    for t in thresholds:
        out[f"n_over_{t}"] = int((a > t).sum())
    return out


def phase_order(*cols):
    seen = []
    for c in cols:
        if c is None:
            continue
        for p in c:
            if p not in seen:
                seen.append(p)
    return seen


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


# ---- per-phase analysis ---------------------------------------------------------------------------------------------


def per_phase(run: Run, silence_ms: float):
    rx, tx, rr, rt, loop = run.d_rx, run.d_tx, run.r_rx, run.r_tx, run.d_loop
    # Rows after the policy loop stopped sending are shutdown artefacts (echo age grows); rows before the loop started
    # are the model-loading period, during which nobody reads the socket. Both are excluded from the link statistics.
    ds = run.d_sum or {}
    if rx is not None and ds.get("policy_loop_start_wall"):
        t_lo = ds["policy_loop_start_wall"] - 0.5
        t_hi = ds["policy_loop_start_wall"] + ds.get("policy_loop_seconds", 1e9) + 0.02
        rx = {k: v[(rx["t_wall"] >= t_lo) & (rx["t_wall"] <= t_hi)] for k, v in rx.items()}
    phases = [p for p in phase_order(rx["phase"] if rx else None, rr["phase"] if rr else None,
                                     tx["phase"] if tx else None) if p not in SKIP_PHASES]
    table = {}
    for ph in phases:
        row = {"phase": ph}
        if rx is not None:
            m = rx["phase"] == ph
            ok = m & (rx["status"] == "ok")
            seqs = rx["seq"][ok]
            row["duration_s"] = float(rx["t_mono"][m].max() - rx["t_mono"][m].min()) if m.sum() > 1 else 0.0
            row["state_ok"] = int(ok.sum())
            row["state_stale"] = int((m & (rx["status"] == "stale")).sum())
            span = (seqs.max() - seqs.min() + 1) if seqs.size else 0
            received = ok.sum() + (m & (rx["status"] == "stale")).sum()  # late/out-of-order packets did arrive
            lost = max(0.0, span - received) if span else 0.0
            row["state_lost"] = int(lost)
            row["state_lost_pct"] = 100.0 * lost / span if span else math.nan
            row["rtt"] = stats(rx["rtt_ms"][ok])
            row["state_gap"] = stats(rx["gap_ms"][ok])
        if rt is not None:
            m = rt["phase"] == ph
            n = int(m.sum())
            dropped = int((m & (rt["action"] == "drop")).sum())
            row["state_tx_rows"] = n
            row["state_injected_drop_pct"] = 100.0 * dropped / n if n else math.nan
            row["state_actions"] = {a: int((m & (rt["action"] == a)).sum()) for a in ("pass", "drop", "delay", "reorder", "stall")}
        if tx is not None:
            m = tx["phase"] == ph
            n = int(m.sum())
            dropped = int((m & (tx["action"] == "drop")).sum())
            row["cmd_tx_rows"] = n
            row["cmd_injected_drop_pct"] = 100.0 * dropped / n if n else math.nan
        if rr is not None:
            m = rr["phase"] == ph
            ok = m & (rr["status"] == "ok")
            row["cmd_ok"] = int(ok.sum())
            row["cmd_stale"] = int((m & (rr["status"] == "stale")).sum())
            row["cmd_rejected_other"] = int((m & ~np.isin(rr["status"], ["ok", "stale"])).sum())
            row["cmd_gap"] = stats(rr["inter_ms"][ok])
            row["cmd_gap_margin_ms"] = silence_ms - row["cmd_gap"].get("max", math.nan) if row["cmd_gap"]["n"] else math.nan
        if loop is not None:
            m = loop["phase"] == ph
            row["loop_ticks"] = int(m.sum())
            row["loop_ms"] = stats(loop["loop_ms"][m], (20, 25, 40))
            row["infer_ms"] = stats(loop["infer_ms"][m], (5, 10, 20))
            row["overruns"] = int(loop["overrun"][m].sum())
        if run.r_met is not None:
            m = run.r_met["phase"] == ph
            if m.any():
                row["relay_cpu_mean"] = float(np.nanmean(run.r_met["cpu_pct"][m]))
                row["relay_cpu_max"] = float(np.nanmax(run.r_met["cpu_pct"][m]))
                row["temp_max_c"] = float(np.nanmax(run.r_met["temp_c"][m])) if not np.all(np.isnan(run.r_met["temp_c"][m])) else math.nan
                row["wifi_level_mean_dbm"] = float(np.nanmean(run.r_met["wifi_level_dbm"][m])) if not np.all(np.isnan(run.r_met["wifi_level_dbm"][m])) else math.nan
        table[ph] = row
    return table


def attribute_fake_robot_phases(run: Run):
    """Phase of each fake-robot metrics second, by wall clock against the relay's per-second phase (same machine)."""
    if run.f_met is None or run.r_met is None:
        return None
    rt, rp = run.r_met["t_wall"], run.r_met["phase"]
    out = []
    for t in run.f_met["t_wall"]:
        i = int(np.argmin(np.abs(rt - t)))
        out.append(rp[i] if abs(rt[i] - t) < 2.0 else "?")
    return np.array(out, dtype=object)


def publisher_by_phase(run: Run):
    ph = attribute_fake_robot_phases(run)
    if ph is None:
        return {}
    out = {}
    for p in phase_order(ph):
        m = ph == p
        out[p] = {"seconds": int(m.sum()), "int_p99_max_ms": float(np.nanmax(run.f_met["int_p99_ms"][m])),
                  "int_max_ms": float(np.nanmax(run.f_met["int_max_ms"][m])), "int_n4": int(np.nansum(run.f_met["int_n4"][m])),
                  "int_n10": int(np.nansum(run.f_met["int_n10"][m])), "int_n20": int(np.nansum(run.f_met["int_n20"][m]))}
    return out


def icmp_by_phase(run: Run):
    if run.icmp is None or run.d_rx is None:
        return None
    t, rtt, seq = run.icmp["t"], run.icmp["rtt"], run.icmp["seq"]
    out = {}
    rx = run.d_rx
    for p in phase_order(rx["phase"]):
        if p in SKIP_PHASES:
            continue
        m = rx["phase"] == p
        t0, t1 = float(rx["t_wall"][m].min()), float(rx["t_wall"][m].max())
        sel = (t >= t0) & (t <= t1)
        if not sel.any():
            continue
        s = seq[sel]
        span = int(s.max() - s.min() + 1)
        out[p] = {"n": int(sel.sum()), "lost_pct": 100.0 * (span - sel.sum()) / span if span else math.nan,
                  **{k: v for k, v in stats(rtt[sel]).items() if k != "n"}}
    return out


# ---- checks ---------------------------------------------------------------------------------------------------------


def check(name, ok, detail):
    return {"check": name, "result": "PASS" if ok is True else ("FAIL" if ok is False else "INFO"), "detail": detail}


def evaluate(run: Run, table: dict, pub: dict, silence_ms: float, schedule):
    R, out = RULES, []
    inj = {}
    if schedule:
        for _s, _e, name, prof in schedule:
            inj[name] = prof
    damp = (run.r_sum or {}).get("damp_events") or []
    outcome = (run.d_sum or {}).get("outcome", "unknown")
    out.append(check("Run completed", outcome == "completed", f"desktop outcome: {outcome}"))
    unplanned = [d for d in damp if d.get("reason") != "desktop requested damping"]
    out.append(check("No unplanned damping during the run", not unplanned,
                     "; ".join(f"[{d.get('phase')}] {d.get('reason')}" for d in unplanned) or "none (only the desktop's own end-of-run damp)"))

    clean = [table[p] for p in CLEAN_PHASES if p in table]
    for row in clean:
        if "state_lost_pct" in row:
            out.append(check(f"[{row['phase']}] state loss", row["state_lost_pct"] <= R["clean_state_loss_pct_max"],
                             f"{fmt(row['state_lost_pct'], 2)} % of {row['state_ok'] + row['state_lost']} packets"))
        if row.get("rtt", {}).get("n"):
            out.append(check(f"[{row['phase']}] RTT p95 / max",
                             row["rtt"]["p95"] <= R["clean_rtt_p95_ms_max"] and row["rtt"]["max"] <= R["clean_rtt_max_ms_max"],
                             f"p95 {fmt(row['rtt']['p95'])} ms (limit {R['clean_rtt_p95_ms_max']}), max {fmt(row['rtt']['max'])} ms "
                             f"(limit {R['clean_rtt_max_ms_max']})"))
        if row.get("cmd_gap", {}).get("n"):
            out.append(check(f"[{row['phase']}] command gap max", row["cmd_gap"]["max"] <= R["clean_cmd_gap_max_ms_max"],
                             f"{fmt(row['cmd_gap']['max'])} ms (limit {R['clean_cmd_gap_max_ms_max']}, relay damps at {silence_ms:.0f})"))

    for name, prof in inj.items():
        row = table.get(name)
        if not row:
            continue
        for key, label, col in (("loss_p", "state", "state_injected_drop_pct"), ("loss_p", "command", "cmd_injected_drop_pct")):
            if key in prof and col in row and not math.isnan(row[col]):
                n = row.get("state_tx_rows" if label == "state" else "cmd_tx_rows", 0)
                p = prof[key]
                tol = max(R["loss_honesty_points"], 300.0 * math.sqrt(p * (1 - p) / n) if n else 100.0)
                out.append(check(f"[{name}] {label}-path injected loss ~ {100 * p:g}%", abs(row[col] - 100 * p) <= tol,
                                 f"measured {fmt(row[col], 2)} % over {n} packets (tolerance {fmt(tol, 2)} points)"))
        if row.get("cmd_gap", {}).get("n"):
            out.append(check(f"[{name}] command gap below the {silence_ms:.0f} ms damp limit", row["cmd_gap"]["max"] < silence_ms,
                             f"max {fmt(row['cmd_gap']['max'])} ms, margin {fmt(row['cmd_gap_margin_ms'])} ms"))

    if "recovery" in table and clean and table["recovery"].get("rtt", {}).get("n") and clean[0].get("rtt", {}).get("n"):
        lim = R["recovery_factor_max"] * max(clean[0]["rtt"]["p95"], 5.0)
        out.append(check("Recovery returns to the clean RTT", table["recovery"]["rtt"]["p95"] <= lim,
                         f"recovery p95 {fmt(table['recovery']['rtt']['p95'])} ms vs limit {fmt(lim)} ms"))

    fs = run.f_sum
    if fs:
        rate = fs.get("rate_hz", 0.0)
        out.append(check("500 Hz LowCmd rate (robot's view)", abs(rate - 500.0) <= 5.0 * R["publish_rate_tolerance_pct"],
                         f"{fmt(rate, 2)} Hz over {fmt(fs.get('span_s'), 0)} s, {fs.get('lowcmd_total')} messages"))
        iv = fs["interval_ms"]["summary"]
        out.append(check("500 Hz interval p99 / max", iv["p99"] <= R["publish_p99_ms_max"] and iv["max"] <= R["publish_max_ms_max"],
                         f"p50 {fmt(iv['p50'], 2)} ms, p99 {fmt(iv['p99'], 2)} ms (limit {R['publish_p99_ms_max']}), "
                         f"max {fmt(iv['max'], 2)} ms (limit {R['publish_max_ms_max']}); >4 ms: {iv.get('n_over_4')}, "
                         f">10 ms: {iv.get('n_over_10')}, >20 ms: {iv.get('n_over_20')}"))
        ck = fs.get("checks_total", {})
        bad = {k: ck.get(k, 0) for k in ("crc_bad", "header_bad", "level_bad", "mode_bad")}
        out.append(check("CRC, header, level, mode on received LowCmd", ck.get("crc_checked", 0) > 0 and not any(bad.values()),
                         f"CRC checked {ck.get('crc_checked')} frames; bad: {bad}"))
        out.append(check("Exit damping burst reached the robot (kp=0 frames)", fs.get("kp_zero_tail_frames", 0) >= R["damp_tail_frames_min"],
                         f"{fs.get('kp_zero_tail_frames')} frames with kp=0 after the last stiff frame (max kp seen {fmt(fs.get('max_kp_seen'), 1)})"))

    if run.r_met is not None:
        cpu = float(np.nanmean(run.r_met["cpu_pct"]))
        out.append(check("Relay CPU", cpu <= R["relay_cpu_pct_max"], f"mean {fmt(cpu)} % of one core, max {fmt(float(np.nanmax(run.r_met['cpu_pct'])))} %"))
        out.append(check("Relay logging cost", None, f"mean {fmt((run.r_sum or {}).get('log_row_cost_us'), 2)} us per logged row; "
                         f"publish call p99 {fmt(((run.r_sum or {}).get('publish_call_ms') or {}).get('summary', {}).get('p99'), 2)} ms"))
    if run.d_sum and run.d_sum.get("loop", {}).get("ticks"):
        lp = run.d_sum["loop"]
        pc = 100.0 * lp["overruns"] / lp["ticks"]
        out.append(check("Desktop control-loop overruns", pc <= R["desktop_overrun_pct_max"],
                         f"{lp['overruns']} of {lp['ticks']} ticks ({fmt(pc, 3)} %); loop p99 {fmt(lp['loop_ms']['summary']['p99'], 2)} ms, "
                         f"inference p99 {fmt(lp['infer_ms']['summary']['p99'], 2)} ms"))

    # logging health: the per-packet files should hold what the summaries counted
    if run.d_sum and run.d_tx is not None:
        sent = run.d_sum["packets"]["cmds_sent"]
        rows = len(run.d_tx["seq"])
        out.append(check("Logging: tx_cmd.csv rows vs commands sent", sent > 0 and abs(rows - sent) <= sent * R["row_count_tolerance_pct"] / 100.0,
                         f"{rows} rows vs {sent} sent"))
    if run.r_sum and run.r_rx is not None:
        cmds = run.r_sum["counters"]["cmds"]
        ok_rows = int((run.r_rx["status"] == "ok").sum())
        out.append(check("Logging: rx_cmd.csv ok rows vs commands accepted", cmds > 0 and abs(ok_rows - cmds) <= max(1, cmds * R["row_count_tolerance_pct"] / 100.0),
                         f"{ok_rows} rows vs {cmds} accepted"))
    return out


# ---- output ---------------------------------------------------------------------------------------------------------


def write_report(run: Run, table, pub, icmp, checks, silence_ms, path):
    L = []
    L.append(f"# Soak report: {os.path.basename(os.path.abspath(run.root))}\n")
    ds = run.d_sum or {}
    L.append(f"- outcome: **{ds.get('outcome', 'unknown')}**; policy loop {fmt(ds.get('policy_loop_seconds'), 0)} s")
    if run.d_sum and run.d_sum.get("clock_offset_s") is not None:
        L.append(f"- relay clock minus desktop clock: {run.d_sum['clock_offset_s']:+.3f} s "
                 f"(+/- {1000 * (run.d_sum.get('clock_offset_uncertainty_s') or 0):.1f} ms)")
    if run.r_meta:
        L.append(f"- relay: python {run.r_meta.get('python')}, `{(run.r_meta.get('iwconfig') or '').splitlines()[0] if run.r_meta.get('iwconfig') else 'no iwconfig'}`")
        pm = re.search(r"Power Management:(\w+)", run.r_meta.get("iwconfig") or "")
        L.append(f"- WiFi power management on the Jetson dongle: {pm.group(1) if pm else 'unknown'}")
    L.append(f"- relay damp limit: {silence_ms:.0f} ms of command silence\n")

    L.append("## Verdicts\n")
    L.append("| result | check | detail |\n|---|---|---|")
    for c in checks:
        L.append(f"| {c['result']} | {c['check']} | {c['detail']} |")
    fails = sum(1 for c in checks if c["result"] == "FAIL")
    L.append(f"\n**{sum(1 for c in checks if c['result'] == 'PASS')} passed, {fails} failed, "
             f"{sum(1 for c in checks if c['result'] == 'INFO')} informational.**\n")

    L.append("## Link by phase (desktop view, every packet)\n")
    L.append("| phase | s | state ok | lost % | stale | RTT p50 | p95 | p99 | max | >50ms | >100ms | state gap p99 | max |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ph, r in table.items():
        rt, sg = r.get("rtt", {}), r.get("state_gap", {})
        L.append(f"| {ph} | {fmt(r.get('duration_s'), 0)} | {r.get('state_ok', 'n/a')} | {fmt(r.get('state_lost_pct'), 2)} | {r.get('state_stale', 'n/a')} | "
                 f"{fmt(rt.get('p50'))} | {fmt(rt.get('p95'))} | {fmt(rt.get('p99'))} | {fmt(rt.get('max'))} | {rt.get('n_over_50', 'n/a')} | "
                 f"{rt.get('n_over_100', 'n/a')} | {fmt(sg.get('p99'))} | {fmt(sg.get('max'))} |")

    L.append("\n## Commands at the relay by phase (the damping-risk numbers)\n")
    L.append("| phase | ok | stale | inj. cmd drop % | inj. state drop % | gap p50 | p99 | max | margin to damp | gaps >50ms | >100ms |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for ph, r in table.items():
        g = r.get("cmd_gap", {})
        L.append(f"| {ph} | {r.get('cmd_ok', 'n/a')} | {r.get('cmd_stale', 'n/a')} | {fmt(r.get('cmd_injected_drop_pct'), 2)} | "
                 f"{fmt(r.get('state_injected_drop_pct'), 2)} | {fmt(g.get('p50'))} | {fmt(g.get('p99'))} | {fmt(g.get('max'))} | "
                 f"{fmt(r.get('cmd_gap_margin_ms'))} | {g.get('n_over_50', 'n/a')} | {g.get('n_over_100', 'n/a')} |")

    if pub:
        L.append("\n## 500 Hz LowCmd as the robot sees it (fake robot, by phase)\n")
        L.append("| phase | s | worst per-second p99 (ms) | max (ms) | >4 ms | >10 ms | >20 ms |\n|---|---|---|---|---|---|---|")
        for ph, r in pub.items():
            L.append(f"| {ph} | {r['seconds']} | {fmt(r['int_p99_max_ms'], 2)} | {fmt(r['int_max_ms'], 2)} | {r['int_n4']} | {r['int_n10']} | {r['int_n20']} |")

    if icmp:
        L.append("\n## Independent ICMP ping by phase (real link, not touched by the injector)\n")
        L.append("| phase | pings | lost % | p50 | p95 | p99 | max |\n|---|---|---|---|---|---|---|")
        for ph, r in icmp.items():
            L.append(f"| {ph} | {r['n']} | {fmt(r['lost_pct'], 2)} | {fmt(r.get('p50'))} | {fmt(r.get('p95'))} | {fmt(r.get('p99'))} | {fmt(r.get('max'))} |")

    L.append("\n## Load by phase\n")
    L.append("| phase | relay CPU mean % | max % | Jetson temp max C | WiFi level mean dBm | desktop loop p99 ms | overruns |\n|---|---|---|---|---|---|---|")
    for ph, r in table.items():
        L.append(f"| {ph} | {fmt(r.get('relay_cpu_mean'))} | {fmt(r.get('relay_cpu_max'))} | {fmt(r.get('temp_max_c'))} | "
                 f"{fmt(r.get('wifi_level_mean_dbm'))} | {fmt(r.get('loop_ms', {}).get('p99'), 2)} | {r.get('overruns', 'n/a')} |")
    L.append("\nUnits are milliseconds unless stated. Percentiles are exact (computed from every packet), except the "
             "fake-robot columns, which come from per-second summaries.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def make_plots(run: Run, outdir: str, silence_ms: float, phases_wall):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    os.makedirs(outdir, exist_ok=True)
    made = []

    def shade(ax, t0, table_t):
        for name, (a, b) in table_t.items():
            ax.axvspan(a - t0, b - t0, alpha=0.08 if name in CLEAN_PHASES else 0.18, color="tab:red" if name not in CLEAN_PHASES and name != "recovery" else "tab:green")
            ax.text((a + b) / 2 - t0, ax.get_ylim()[1] * 0.97, name, ha="center", va="top", fontsize=7, rotation=90)

    rx = run.d_rx
    if rx is not None:
        ok = rx["status"] == "ok"
        t_clean = rx["t_wall"][rx["phase"] == "clean"]
        t0 = float(t_clean.min()) if t_clean.size else float(rx["t_wall"].min())
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(rx["t_wall"][ok] - t0, rx["rtt_ms"][ok], ",", color="tab:blue", label="protocol RTT (every packet)")
        if run.icmp is not None:
            ax.plot(run.icmp["t"] - t0, run.icmp["rtt"], ".", ms=2, color="tab:orange", label="ICMP ping")
        ax.set_yscale("log")
        ax.axhline(silence_ms, color="red", lw=0.8, ls="--")
        ax.set_xlabel("seconds since the schedule started")
        ax.set_ylabel("ms")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title("Round trip over time")
        shade(ax, t0, phases_wall)
        fig.tight_layout()
        p = os.path.join(outdir, "rtt_timeline.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
        fig, ax = plt.subplots(figsize=(6, 4))
        for ph in phase_order(rx["phase"]):
            v = np.sort(rx["rtt_ms"][(rx["phase"] == ph) & ok])
            v = v[~np.isnan(v)]
            if v.size > 20 and ph not in SKIP_PHASES:
                ax.plot(v, 1.0 - np.arange(v.size) / v.size, label=ph)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("RTT ms")
        ax.set_ylabel("fraction slower")
        ax.legend(fontsize=6)
        ax.set_title("RTT CCDF by phase")
        fig.tight_layout()
        p = os.path.join(outdir, "rtt_ccdf.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
    if run.r_met is not None and run.r_rx is not None:
        m = run.r_rx["status"] == "ok"
        t0 = float(run.r_rx["t_wall"][run.r_rx["phase"] == "clean"].min()) if (run.r_rx["phase"] == "clean").any() else float(run.r_rx["t_wall"].min())
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(run.r_rx["t_wall"][m] - t0, run.r_rx["inter_ms"][m], ",", color="tab:green")
        ax.axhline(silence_ms, color="red", lw=0.8, ls="--", label=f"relay damps at {silence_ms:.0f} ms")
        ax.set_yscale("log")
        ax.set_xlabel("seconds since the schedule started")
        ax.set_ylabel("ms between accepted commands")
        ax.legend(fontsize=7)
        ax.set_title("Command gaps at the relay")
        shade(ax, t0, phases_wall)
        fig.tight_layout()
        p = os.path.join(outdir, "cmd_gap_timeline.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
    if run.f_met is not None:
        fig, ax = plt.subplots(figsize=(11, 4))
        t = run.f_met["t_mono"]
        ax.plot(t, run.f_met["int_p99_ms"], label="p99 per second")
        ax.plot(t, run.f_met["int_max_ms"], label="max per second")
        ax.axhline(4.0, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("fake-robot seconds")
        ax.set_ylabel("ms between LowCmd frames (nominal 2)")
        ax.legend(fontsize=7)
        ax.set_title("500 Hz stream as the robot sees it")
        fig.tight_layout()
        p = os.path.join(outdir, "lowcmd_interval.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
    return made


def phase_wall_ranges(run: Run):
    rx = run.d_rx
    out = {}
    if rx is None:
        return out
    for ph in phase_order(rx["phase"]):
        if ph in SKIP_PHASES:
            continue
        m = rx["phase"] == ph
        out[ph] = (float(rx["t_wall"][m].min()), float(rx["t_wall"][m].max()))
    return out


def analyse(root: str, silence_ms: float | None = None, plots: bool = True) -> dict:
    run = Run(root)
    if run.d_meta is None and run.d_rx is None:
        raise SystemExit(f"{root}/desktop has no logs (meta.json / rx_state.csv missing)")
    if silence_ms is None:
        silence_ms = 1000.0 * float(((run.r_meta or {}).get("args") or {}).get("silence_damp_s", 0.15))
    schedule = (run.d_meta or {}).get("schedule")
    table = per_phase(run, silence_ms)
    pub = publisher_by_phase(run)
    icmp = icmp_by_phase(run)
    checks = evaluate(run, table, pub, silence_ms, schedule)
    result = {"run": os.path.basename(os.path.abspath(root)), "silence_ms": silence_ms, "phases": table,
              "publisher_by_phase": pub, "icmp_by_phase": icmp, "checks": checks}
    with open(os.path.join(root, "report.json"), "w") as f:
        json.dump(json.loads(json.dumps(result, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))), f, indent=1)
    write_report(run, table, pub, icmp, checks, silence_ms, os.path.join(root, "report.md"))
    result["plots"] = make_plots(run, os.path.join(root, "plots"), silence_ms, phase_wall_ranges(run)) if plots else []
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse an instrumented soak run folder.")
    ap.add_argument("run_dir")
    ap.add_argument("--silence_ms", type=float, default=None, help="Relay damp limit (default: read from the relay's meta.json).")
    ap.add_argument("--no_plots", action="store_true")
    a = ap.parse_args()
    res = analyse(a.run_dir, a.silence_ms, plots=not a.no_plots)
    print(open(os.path.join(a.run_dir, "report.md")).read())
    print(f"wrote {a.run_dir}/report.md, report.json" + (f", plots/ ({len(res['plots'])} files)" if res["plots"] else ""))
    return 1 if any(c["result"] == "FAIL" for c in res["checks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
