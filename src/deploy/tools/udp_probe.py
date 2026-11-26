# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bare UDP echo probe: separates the NETWORK from the relay stack when a WiFi link stalls.

Stdlib only and Python 3.8 compatible, so the same file runs on the Jetson and the desktop. No DDS, no relay, no
robot: it cannot move anything.

    Jetson:   python3 udp_probe.py server --port 9890
    Desktop:  python3 udp_probe.py client --host 192.168.50.20 --port 9890 --rate 100 --duration 300

The client sends numbered packets at --rate Hz; the server echoes each one at once. The server reports the gaps
between arrivals (the downlink as the Jetson sees it) and the client reports round-trip times and echo gaps. A stall
shows up as a long arrival gap followed by a burst of near-zero gaps. Compare the stall count and length to the
relay runs (about 0.45-0.49 s once every 5-40 s): if they appear here too, the network is the cause; if not, the
relay stack on the Jetson is.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
import time

_FMT = "!Id"  # seq, sender monotonic time
_PAD = b"\x00" * 120  # similar in size to a relay command packet


def _wall():
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + ".%03d" % int((t % 1) * 1000)


def _pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def _report_gaps(name, times, limits=(50, 100, 150)):
    gaps = [(times[i], 1000.0 * (times[i] - times[i - 1])) for i in range(1, len(times))]
    g = [x[1] for x in gaps]
    print("%s: n=%d  gap ms: p50 %.1f  p99 %.1f  max %.1f  | counts > %s: %s" % (
        name, len(times), _pct(g, 50), _pct(g, 99), max(g) if g else float("nan"),
        "/".join(str(x) for x in limits), "/".join(str(sum(1 for v in g if v > x)) for x in limits)))
    t0 = times[0] if times else 0.0
    big = [(round(t - t0, 2), round(v)) for t, v in gaps if v > 100]
    if big:
        print("   stalls >100 ms (t since start s, gap ms):", big[:20], "..." if len(big) > 20 else "")
    return g


def run_server(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((args.bind, args.port))
    sock.settimeout(1.0)
    print("[server] echoing on %s:%d (Ctrl+C to stop)" % (args.bind, args.port), flush=True)
    arrivals, last_print, last_n = [], time.monotonic(), 0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                if arrivals and time.monotonic() - arrivals[-1] > 5.0:
                    _report_gaps("[server] session", arrivals)
                    arrivals = []
                continue
            now = time.monotonic()
            sock.sendto(data, addr)
            if arrivals and now - arrivals[-1] > 0.1:
                print("[server] STALL %.0f ms ended at %s" % (1000.0 * (now - arrivals[-1]), _wall()), flush=True)
            arrivals.append(now)
            if now - last_print > 10.0 and len(arrivals) > last_n + 10:
                recent = arrivals[last_n:]
                g = [1000.0 * (recent[i] - recent[i - 1]) for i in range(1, len(recent))]
                print("[server] last 10 s: %d pkts, max gap %.0f ms" % (len(recent), max(g)), flush=True)
                last_print, last_n = now, len(arrivals)
    except KeyboardInterrupt:
        pass
    if arrivals:
        _report_gaps("[server] arrivals (downlink, desktop->Jetson)", arrivals)


def run_client(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.2)
    dest = (args.host, args.port)
    refused = args.refused_is_reply
    if refused:
        sock.connect(dest)  # a connected UDP socket reports the ICMP "port unreachable" as ConnectionRefusedError
    rtts, echo_t = [], []
    stop = threading.Event()

    def rx():
        while not stop.is_set():
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except ConnectionRefusedError:
                echo_t.append(time.monotonic())  # the host's ICMP reply: counts as an echo (no RTT)
                continue
            now = time.monotonic()
            seq, t_sent = struct.unpack(_FMT, data[:12])
            echo_t.append(now)
            rtts.append((now - t_sent) * 1000.0)

    th = threading.Thread(target=rx, daemon=True)
    th.start()
    period = 1.0 / args.rate
    t0 = time.monotonic()
    send_t, seq = [], 0
    print("[client] %d Hz for %.0f s to %s:%d" % (args.rate, args.duration, args.host, args.port), flush=True)
    try:
        while time.monotonic() - t0 < args.duration:
            target = t0 + seq * period
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)
            now = time.monotonic()
            try:
                if refused:
                    sock.send(struct.pack(_FMT, seq, now) + _PAD)
                else:
                    sock.sendto(struct.pack(_FMT, seq, now) + _PAD, dest)
            except ConnectionRefusedError:
                echo_t.append(time.monotonic())
            send_t.append(now)
            seq += 1
    except KeyboardInterrupt:
        pass
    time.sleep(1.0)
    stop.set()
    th.join(timeout=1.0)
    print("[client] sent %d, echoes %d, lost %d" % (seq, len(rtts) if rtts else len(echo_t), seq - (len(rtts) or len(echo_t))))
    if not rtts and echo_t:
        _report_gaps("[client] send gaps (this machine)", send_t)
        _report_gaps("[client] reply arrival gaps", echo_t)
    if rtts:
        print("[client] RTT ms: p50 %.1f  p95 %.1f  p99 %.1f  max %.1f  | > 100/150/300 ms: %d/%d/%d" % (
            _pct(rtts, 50), _pct(rtts, 95), _pct(rtts, 99), max(rtts), sum(1 for v in rtts if v > 100),
            sum(1 for v in rtts if v > 150), sum(1 for v in rtts if v > 300)))
        _report_gaps("[client] send gaps (this machine)", send_t)
        _report_gaps("[client] echo arrival gaps", echo_t)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("server")
    s.add_argument("--bind", default="0.0.0.0")
    s.add_argument("--port", type=int, default=9890)
    c = sub.add_parser("client")
    c.add_argument("--host", required=True)
    c.add_argument("--port", type=int, default=9890)
    c.add_argument("--rate", type=float, default=100.0)
    c.add_argument("--duration", type=float, default=300.0)
    c.add_argument("--refused-is-reply", dest="refused_is_reply", action="store_true",
                   help="No server needed: send to a host that answers closed UDP ports with ICMP port-unreachable "
                   "(e.g. the AP itself, port 9) and time those replies. Tests ONE path at 100 Hz.")
    args = p.parse_args(argv)
    if args.mode == "server":
        run_server(args)
        return 0
    if args.rate <= 0 or args.duration <= 0:
        print("--rate and --duration must be > 0", file=sys.stderr)
        return 2
    return run_client(args)


if __name__ == "__main__":
    sys.exit(main())
