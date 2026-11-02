# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run logging for the wireless relay stack: events, per-packet CSVs, histograms, /proc readers, metadata.

Used by relay_jetson.py, deploy_real_wireless.py and fake_robot.py so that a long test leaves files that can be
fetched and analysed afterwards (deploy/tools/analyze_soak.py). Pure standard library, Python 3.8 compatible,
torch-free, SDK-free.

Design rules (they protect the control loops, which matter more than the logs):
  * Recording is an in-memory append under a lock. Nothing touches the disk on the hot path; a 1 Hz flusher
    thread writes. One lost second of logs on a crash is acceptable, a blocked 500 Hz publisher is not.
  * Logging never raises into the caller. Every error is counted and written to the events log instead.
  * Every row carries t_mono (seconds since the component started, monotonic) and t_wall (epoch seconds).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import socket
import sys
import threading
import time

# ----------------------------------------------------------------------------------------------------------
# Time
# ----------------------------------------------------------------------------------------------------------


class Clock:
    """Seconds since construction (monotonic) plus wall time, so every row can carry both."""

    def __init__(self):
        self._m0 = time.monotonic()
        self.wall0 = time.time()

    def mono(self) -> float:
        return time.monotonic() - self._m0

    def wall(self) -> float:
        return time.time()


# ----------------------------------------------------------------------------------------------------------
# Events log and stdout tee
# ----------------------------------------------------------------------------------------------------------


class EventLog:
    """Timestamped text log. Thread-safe; each line is flushed immediately (events are rare and valuable)."""

    def __init__(self, path: str, clock: Clock):
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()
        self._clock = clock
        self.errors = 0

    def write(self, text: str) -> None:
        try:
            line = f"{self._clock.mono():10.3f} {self._clock.wall():.3f} {text.rstrip()}\n"
            with self._lock:
                self._f.write(line)
        except Exception:  # noqa: BLE001 -- logging must never take the caller down
            self.errors += 1

    def close(self) -> None:
        try:
            with self._lock:
                self._f.close()
        except Exception:  # noqa: BLE001
            pass


class Tee(io.TextIOBase):
    """Replaces sys.stdout/sys.stderr: everything still reaches the terminal and every complete line is also
    written to the events log. Lets existing print() calls become logged events without editing them."""

    def __init__(self, original, events: EventLog, prefix: str = ""):
        self._orig = original
        self._events = events
        self._prefix = prefix
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        try:
            self._orig.write(s)
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    self._events.write(self._prefix + line)
        return len(s)

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        try:
            return self._orig.isatty()
        except Exception:  # noqa: BLE001
            return False

    def fileno(self) -> int:
        return self._orig.fileno()


# ----------------------------------------------------------------------------------------------------------
# Buffered CSV
# ----------------------------------------------------------------------------------------------------------


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return format(v, ".6f")
    return str(v)


class CsvLog:
    """CSV file with an in-memory buffer. row() is cheap and thread-safe; flush() writes to disk."""

    def __init__(self, path: str, header):
        self.path = path
        self.header = list(header)
        self._f = open(path, "w", buffering=1 << 16)
        self._f.write(",".join(self.header) + "\n")
        self._buf = []
        self._lock = threading.Lock()
        self.rows = 0

    def row(self, *values) -> None:
        if len(values) != len(self.header):
            raise ValueError(f"{self.path}: {len(values)} values for {len(self.header)} columns")
        line = ",".join(_fmt(v) for v in values)
        with self._lock:
            self._buf.append(line)
            self.rows += 1

    def flush(self) -> None:
        with self._lock:
            lines, self._buf = self._buf, []
        if lines:
            self._f.write("\n".join(lines) + "\n")
            self._f.flush()

    def close(self) -> None:
        try:
            self.flush()
            self._f.close()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------------------------------
# Histogram
# ----------------------------------------------------------------------------------------------------------


class Histogram:
    """Fixed log-spaced histogram of non-negative values (default unit: milliseconds) with exact count, sum, min,
    max and exact counts above chosen thresholds. O(1) per sample, constant memory, so it can sit in a 500 Hz
    path. Percentiles are bin-upper-edge estimates (about 12% resolution at 20 bins per decade); the maximum and the
    threshold counts are exact."""

    def __init__(self, lo: float = 0.05, hi: float = 5000.0, bins_per_decade: int = 20, thresholds=(4.0, 10.0, 20.0)):
        self.lo, self.hi, self.bpd = lo, hi, bins_per_decade
        self._n = int(math.ceil(math.log10(hi / lo) * bins_per_decade)) + 1
        self.counts = [0] * (self._n + 1)  # last bin = overflow
        self.thresholds = tuple(thresholds)
        self.above = {t: 0 for t in self.thresholds}
        self.count = 0
        self.total = 0.0
        self.min = math.inf
        self.max = 0.0

    def add(self, x: float) -> None:
        self.count += 1
        self.total += x
        if x > self.max:
            self.max = x
        if x < self.min:
            self.min = x
        for t in self.thresholds:
            if x > t:
                self.above[t] += 1
        if x <= self.lo:
            i = 0
        else:
            i = int(math.log10(x / self.lo) * self.bpd) + 1
            if i > self._n:
                i = self._n
        self.counts[i] += 1

    def merge(self, other: "Histogram") -> None:
        self.count += other.count
        self.total += other.total
        self.max = max(self.max, other.max)
        self.min = min(self.min, other.min)
        for i, c in enumerate(other.counts):
            self.counts[i] += c
        for t in self.thresholds:
            self.above[t] += other.above.get(t, 0)

    def _edge(self, i: int) -> float:
        return self.lo * 10 ** (i / self.bpd) if i > 0 else self.lo

    def percentile(self, p: float) -> float:
        """Estimate of the p-th percentile (0-100): the upper edge of the bin that contains it. p=100 is the exact max."""
        if self.count == 0:
            return 0.0
        if p >= 100:
            return self.max
        target = p / 100.0 * self.count
        run = 0
        for i, c in enumerate(self.counts):
            run += c
            if run >= target and c:
                return min(self._edge(i), self.max)
        return self.max

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def summary(self) -> dict:
        d = {"count": self.count, "mean": self.mean, "min": 0.0 if self.count == 0 else self.min,
             "p50": self.percentile(50), "p95": self.percentile(95), "p99": self.percentile(99), "max": self.max}
        for t in self.thresholds:
            d[f"n_over_{t:g}"] = self.above[t]
        return d

    def to_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "bins_per_decade": self.bpd, "counts": self.counts,
                "summary": self.summary()}


# ----------------------------------------------------------------------------------------------------------
# /proc readers
# ----------------------------------------------------------------------------------------------------------


def parse_proc_net_wireless(text: str) -> dict:
    """{iface: {"link": q, "level": dBm, "noise": dBm}} from /proc/net/wireless text (wext drivers only)."""
    out = {}
    for line in text.splitlines():
        if ":" not in line or "|" in line:
            continue
        name, rest = line.split(":", 1)
        parts = rest.split()
        if len(parts) < 4:
            continue
        try:
            out[name.strip()] = {"link": float(parts[1].rstrip(".")), "level": float(parts[2].rstrip(".")),
                                 "noise": float(parts[3].rstrip("."))}
        except ValueError:
            continue
    return out


def parse_proc_net_dev(text: str) -> dict:
    """{iface: {rx_bytes, rx_packets, rx_errs, rx_drop, tx_bytes, tx_packets, tx_errs, tx_drop}}."""
    out = {}
    for line in text.splitlines():
        if ":" not in line or "|" in line:
            continue
        name, rest = line.split(":", 1)
        p = rest.split()
        if len(p) < 16:
            continue
        try:
            out[name.strip()] = {"rx_bytes": int(p[0]), "rx_packets": int(p[1]), "rx_errs": int(p[2]),
                                 "rx_drop": int(p[3]), "tx_bytes": int(p[8]), "tx_packets": int(p[9]),
                                 "tx_errs": int(p[10]), "tx_drop": int(p[11])}
        except ValueError:
            continue
    return out


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def read_loadavg() -> tuple:
    p = _read("/proc/loadavg").split()
    try:
        return float(p[0]), float(p[1]), float(p[2])
    except (IndexError, ValueError):
        return (math.nan, math.nan, math.nan)


def read_thermal_max_c() -> float:
    """Hottest thermal zone in degrees C (nan if none readable)."""
    best = math.nan
    base = "/sys/class/thermal"
    try:
        zones = [z for z in os.listdir(base) if z.startswith("thermal_zone")]
    except OSError:
        return best
    for z in zones:
        t = _read(os.path.join(base, z, "temp")).strip()
        try:
            v = int(t) / 1000.0
        except ValueError:
            continue
        if math.isnan(best) or v > best:
            best = v
    return best


class ProcCpu:
    """CPU use of this process, in percent of ONE core, between successive calls."""

    def __init__(self):
        self._tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self._last = (time.monotonic(), self._cpu_s())

    def _cpu_s(self) -> float:
        p = _read("/proc/self/stat")
        try:
            rest = p[p.rindex(")") + 2:].split()
            return (int(rest[11]) + int(rest[12])) / self._tck  # utime + stime
        except (ValueError, IndexError):
            return time.process_time()

    def percent(self) -> float:
        now, cpu = time.monotonic(), self._cpu_s()
        t0, c0 = self._last
        self._last = (now, cpu)
        return 100.0 * (cpu - c0) / (now - t0) if now > t0 else 0.0


class NetCounters:
    """Per-interval deltas of /proc/net/dev counters for one interface."""

    def __init__(self, iface: str):
        self.iface = iface
        self._last = parse_proc_net_dev(_read("/proc/net/dev")).get(iface)

    def delta(self) -> dict:
        cur = parse_proc_net_dev(_read("/proc/net/dev")).get(self.iface)
        if cur is None or self._last is None:
            self._last = cur
            return {}
        d = {k: cur[k] - self._last[k] for k in cur}
        self._last = cur
        return d


def wireless_now(iface: str) -> dict:
    return parse_proc_net_wireless(_read("/proc/net/wireless")).get(iface, {})


# ----------------------------------------------------------------------------------------------------------
# Metadata and the run directory
# ----------------------------------------------------------------------------------------------------------


def file_sha256(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "unreadable"


def base_meta(component: str, run_id: str, argv, files=()) -> dict:
    return {
        "component": component, "run_id": run_id, "argv": list(argv), "hostname": socket.gethostname(),
        "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
        "nproc": os.cpu_count(), "started_wall": time.time(),
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "file_sha256": {os.path.basename(p): file_sha256(p) for p in files},
    }


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class RunLog:
    """One component's log directory: events.log, any number of CSVs, meta.json, summary.json.
    `log_dir=None` makes every method a harmless no-op so callers need no `if` around logging."""

    def __init__(self, log_dir, component: str):
        self.enabled = log_dir is not None
        self.component = component
        self.clock = Clock()
        self.dir = log_dir
        self._csvs = []
        self._flush_stop = threading.Event()
        self._flusher = None
        self._orig_streams = None
        self.events = None
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.events = EventLog(os.path.join(log_dir, "events.log"), self.clock)

    def csv(self, name: str, header):
        if not self.enabled:
            return _NullCsv()
        c = CsvLog(os.path.join(self.dir, name), header)
        self._csvs.append(c)
        return c

    def event(self, text: str) -> None:
        if self.enabled:
            self.events.write(text)

    def tee_stdio(self) -> None:
        """Copy everything printed from now on into events.log (stdout lines as-is, stderr lines prefixed)."""
        if self.enabled and self._orig_streams is None:
            self._orig_streams = (sys.stdout, sys.stderr)
            sys.stdout = Tee(sys.stdout, self.events)
            sys.stderr = Tee(sys.stderr, self.events, prefix="[stderr] ")

    def write_json(self, name: str, obj) -> None:
        if not self.enabled:
            return
        try:
            tmp = os.path.join(self.dir, name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(jsonable(obj), f, indent=1, sort_keys=True)
            os.replace(tmp, os.path.join(self.dir, name))
        except Exception as e:  # noqa: BLE001
            self.event(f"[runlog] could not write {name}: {e!r}")

    def start_flusher(self, period: float = 1.0) -> None:
        if not self.enabled or self._flusher is not None:
            return

        def loop():
            while not self._flush_stop.wait(period):
                self.flush()

        self._flusher = threading.Thread(target=loop, name=f"runlog-flush-{self.component}", daemon=True)
        self._flusher.start()

    def flush(self) -> None:
        for c in self._csvs:
            try:
                c.flush()
            except Exception as e:  # noqa: BLE001
                self.event(f"[runlog] flush failed for {c.path}: {e!r}")

    def close(self) -> None:
        if not self.enabled:
            return
        self._flush_stop.set()
        if self._orig_streams is not None:
            sys.stdout, sys.stderr = self._orig_streams
            self._orig_streams = None
        self.flush()
        for c in self._csvs:
            c.close()
        if self.events is not None:
            self.events.close()


class _NullCsv:
    path = ""
    rows = 0

    def row(self, *values) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
