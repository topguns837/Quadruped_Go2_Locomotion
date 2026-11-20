#!/usr/bin/env bash
# Bench aid: run the UDP echo SERVER on the desktop's own WiFi dongle over the REAL radio, with no second machine.
#
# Problem it solves: if this host has both a wired NIC on the AP and a WiFi dongle on the same AP, the kernel hands
# packets between them internally and they never cross the radio. Here the dongle is moved into its own network
# namespace, so traffic from the wired NIC to the dongle's address must go wired NIC -> AP -> radio -> dongle,
# exactly like desktop -> AP -> Jetson dongle. Use it to test a dongle/driver against the AP at 100 Hz.
#
#   sudo bash deploy/wireless/bench_netns.sh <wifi_iface> [ssid] [ip/prefix] [port]
#   e.g.  sudo bash deploy/wireless/bench_netns.sh wlx3c52a12ae732
# then, in another terminal on the SAME desktop (NOT as root):
#   python3 deploy/tools/udp_probe.py client --host 192.168.50.31 --port 9890 --rate 100 --duration 300
# Ctrl+C here to stop; the dongle returns to the normal namespace.
#
# It touches ONLY the named WiFi interface. It never touches any wired interface, a running Docker container or any
# other namespace. For an OPEN network only (it uses `iw connect`); WPA needs wpa_supplicant in the namespace.
set -u

IFACE="${1:?usage: sudo bash bench_netns.sh <wifi_iface> [ssid] [ip/prefix] [port]}"
SSID="${2:-unitree_go2_testing_2.4GHz}"
ADDR="${3:-192.168.50.31/24}"
PORT="${4:-9890}"
NS=go2bench
HERE="$(cd "$(dirname "$0")" && pwd)"
PROBE="$HERE/../tools/udp_probe.py"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 2; }
case "$IFACE" in wl*|wlan*) ;; *) echo "refusing: '$IFACE' does not look like a WiFi interface" >&2; exit 2;; esac
[ -d "/sys/class/net/$IFACE/wireless" ] || { echo "refusing: $IFACE is not a wireless interface (is it plugged in and not already moved?)" >&2; exit 2; }
[ -f "$PROBE" ] || { echo "missing $PROBE" >&2; exit 2; }
ip netns list 2>/dev/null | grep -qw "$NS" && { echo "namespace $NS already exists; remove it first: sudo ip netns del $NS" >&2; exit 2; }
PHY="phy$(iw dev "$IFACE" info | awk '/wiphy/{print $2}')"
[ "$PHY" != "phy" ] || { echo "could not find the phy of $IFACE" >&2; exit 2; }

cleanup() {
    trap - INT TERM EXIT
    echo
    echo "[bench] stopping; returning $PHY to the normal namespace"
    # Deleting the namespace hands its physical devices back to the initial namespace.
    ip netns del "$NS" 2>/dev/null
    sleep 1
    ip link show "$IFACE" >/dev/null 2>&1 && echo "[bench] $IFACE is back" || echo "[bench] $IFACE not visible yet; replug the dongle if it does not return"
}
trap cleanup INT TERM EXIT

# The network manager would otherwise fight over the interface while we set it up.
nmcli dev set "$IFACE" managed no 2>/dev/null || true
ip netns add "$NS" || exit 1
iw phy "$PHY" set netns name "$NS" || { echo "could not move $PHY" >&2; exit 1; }
ip netns exec "$NS" ip link set lo up
ip netns exec "$NS" ip link set "$IFACE" up || exit 1
echo "[bench] connecting $IFACE to '$SSID' (open network) inside namespace $NS"
ip netns exec "$NS" iw dev "$IFACE" connect "$SSID" || { echo "connect request failed" >&2; exit 1; }
for i in $(seq 1 20); do
    ip netns exec "$NS" iw dev "$IFACE" link 2>/dev/null | grep -q "Connected to" && break
    sleep 1
done
ip netns exec "$NS" iw dev "$IFACE" link | grep -E "Connected|SSID|freq|signal|bitrate"
ip netns exec "$NS" ip addr add "$ADDR" dev "$IFACE" || exit 1
echo "[bench] $IFACE is $ADDR inside $NS. Starting the echo server on port $PORT (Ctrl+C to stop)."
echo "[bench] now run the client from another terminal (not as root):"
echo "        python3 deploy/tools/udp_probe.py client --host ${ADDR%/*} --port $PORT --rate 100 --duration 300"
ip netns exec "$NS" python3 "$PROBE" server --bind 0.0.0.0 --port "$PORT"
