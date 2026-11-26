#!/usr/bin/env bash
# One-command WiFi link check, RUN ON THE DESKTOP. Pings the robot through the WiFi interface only and
# reports what matters for the 50 Hz control loop: loss, latency, the longest silence between replies, and
# "bursts" (replies that arrive together after being held for seconds -- the stall signature seen on the
# dongle pair). Optionally also reads the Jetson access point's per-station transmit-failure counters.
#
# Usage:
#   src/deploy/wireless/link_check.sh <wifi_iface> [count] [target_ip]
#   JETSON=unitree@192.168.123.18 src/deploy/wireless/link_check.sh wlx3c52a12ae732 60
#
# JETSON (optional): ssh target used to read `iw dev wlan0 station dump` before and after. It goes over the
# normal route (the cable, if plugged), so the AP stats are readable even while WiFi is stalling; ssh may
# prompt for the password twice. Without it, only the desktop-side numbers are printed.
set -u

IFACE="${1:?usage: link_check.sh <wifi_iface> [count] [target_ip]}"
COUNT="${2:-60}"
TARGET="${3:-192.168.123.18}"
JETSON="${JETSON:-}"
AP_IFACE="${AP_IFACE:-wlan0}"

# Accept the target in the count position (link_check.sh wlx... 10.0.0.2 [count]).
if [[ "$COUNT" == *.* ]]; then
    swap="$COUNT"; COUNT="${3:-60}"; TARGET="$swap"
fi
[[ "$COUNT" =~ ^[0-9]+$ ]] || { echo "count must be a whole number, got: $COUNT" >&2; exit 1; }

ip link show "$IFACE" >/dev/null 2>&1 || { echo "No such interface: $IFACE" >&2; exit 1; }
if [[ "$TARGET" == 192.168.123.* ]]; then
    ip -4 addr show dev "$IFACE" | grep -q 'inet 192\.168\.123\.' || {
        echo "WARN: $IFACE has no 192.168.123.x address -- the ping will not reach the robot." >&2
    }
fi

ap_stats() {
    [ -n "$JETSON" ] || return 0
    ssh -o ConnectTimeout=5 "$JETSON" "iw dev $AP_IFACE station dump" 2>/dev/null |
        awk '/Station/ {mac=$2} /tx packets:/ {txp=$3} /tx failed:/ {txf=$3} /tx retries:/ {txr=$3} /rx packets:/ {rxp=$3}
             END {print txp+0, txf+0, txr+0, rxp+0}'
}

echo "[link_check] $COUNT pings, 1/s, via $IFACE -> $TARGET (about $((COUNT + 2)) s)"
before="$(ap_stats)"
ping -D -O -I "$IFACE" -c "$COUNT" -i 1 -W 3 "$TARGET" 2>&1 | python3 -c '
import re, sys
rows, sent, recv = [], 0, 0
for line in sys.stdin:
    m = re.match(r"\[(\d+\.\d+)\].*icmp_seq=(\d+).*time=([\d.]+) ms", line)
    if m:
        rows.append((float(m.group(1)), int(m.group(2)), float(m.group(3))))
    m = re.search(r"(\d+) packets transmitted, (\d+) received", line)
    if m:
        sent, recv = int(m.group(1)), int(m.group(2))
if not rows:
    print(f"RESULT: no replies at all ({sent} sent). The link is not carrying traffic.")
    sys.exit(0)
rtts = sorted(r[2] for r in rows)
gaps = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
big = sum(1 for r in rtts if r > 1000)
bursts = sum(1 for g, (_, _, rtt) in zip(gaps, rows[1:]) if g < 0.2 and rtt > 1000)
p95 = rtts[min(len(rtts) - 1, int(0.95 * len(rtts)))]
print(f"RESULT: sent {sent}, received {recv}, loss {100.0 * (sent - recv) / max(sent, 1):.0f}%")
print(f"        rtt ms: min {rtts[0]:.1f}  median {rtts[len(rtts) // 2]:.1f}  p95 {p95:.1f}  max {rtts[-1]:.1f}")
print(f"        longest silence between replies: {max(gaps) if gaps else 0:.1f} s")
print(f"        replies delayed over 1 s: {big}   burst-released replies: {bursts}")
ok = sent == recv and rtts[-1] < 100 and bursts == 0
print("        VERDICT:", "link looks healthy" if ok else "link is NOT fit for the control loop (loss, high latency or stalls)")
'
after="$(ap_stats)"

if [ -n "$JETSON" ] && [ -n "$before" ] && [ -n "$after" ]; then
    read -r btx bfail bret brx <<<"$before"
    read -r atx afail aret arx <<<"$after"
    echo "AP ($AP_IFACE) deltas over this run: tx packets +$((atx - btx)), tx FAILED +$((afail - bfail)), tx retries +$((aret - bret)), rx packets +$((arx - brx))"
    echo "        (tx FAILED rising while the desktop saw no replies = the AP radio is not getting its frames through)"
fi
