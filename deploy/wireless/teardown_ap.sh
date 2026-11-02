#!/usr/bin/env bash
# Restore the robot's wired 192.168.123.x networking after setup_ap.sh. RUN ON THE ROBOT, as root.
#
# Stops hostapd, removes the 123.x NIC from the bridge, puts PC1's address back on the NIC directly, and
# deletes the bridge. Re-enable the Jetson's built-in WiFi separately if you disabled it.
#
# Env vars mirror setup_ap.sh:
#   ETH_IFACE (required), BRIDGE (default br0), PC1_IP (default 192.168.123.18/24), GATEWAY (default 192.168.123.1)
set -euo pipefail

ETH_IFACE="${ETH_IFACE:?set ETH_IFACE to the robot 192.168.123.x NIC}"
BRIDGE="${BRIDGE:-br0}"
PC1_IP="${PC1_IP:-192.168.123.18/24}"
GATEWAY="${GATEWAY:-192.168.123.1}"

[ "$(id -u)" -eq 0 ] || { echo "Must run as root." >&2; exit 1; }

echo "[teardown] stopping hostapd"
pkill -x hostapd || true
sleep 1

if ip link show "$BRIDGE" >/dev/null 2>&1; then
    echo "[teardown] detaching $ETH_IFACE from $BRIDGE and restoring $PC1_IP on it"
    ip link set "$ETH_IFACE" nomaster || true
    ip addr flush dev "$BRIDGE" || true
    ip link set "$BRIDGE" down || true
    ip link delete "$BRIDGE" type bridge || true
    ip addr replace "$PC1_IP" dev "$ETH_IFACE"
    ip link set "$ETH_IFACE" up
    ip route replace default via "$GATEWAY" dev "$ETH_IFACE"
fi

if command -v nmcli >/dev/null 2>&1; then
    nmcli dev set "$ETH_IFACE" managed yes 2>/dev/null || true
fi

echo "[teardown] done. Verify: ping 192.168.123.161"
