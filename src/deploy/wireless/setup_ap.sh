#!/usr/bin/env bash
# DISABLED 2026-10-29 -- DO NOT RUN. This guard exits before anything below can execute.
echo "setup_ap.sh is DISABLED. It bridges the Jetson's eth0 (moves 192.168.123.18 onto br0) and depends on a" >&2
echo "USB WiFi dongle that wedged the Jetson: with the dongle plugged in the Jetson stayed unreachable even" >&2
echo "after power cycles. See src/deploy/wireless/README.md for the supported approach." >&2
exit 1

# ---- original script below, kept for reference only (unreachable) ----
# Bring up the bridged WiFi AP on the robot's Jetson (PC1) so a desktop can reach the robot's
# 192.168.123.0/24 DDS network over WiFi. RUN ON THE ROBOT, as root, over SSH.
#
# WHAT IT DOES (idempotent):
#   1. turns off WiFi power save on the dongle (biggest latency win),
#   2. ensures a bridge ($BRIDGE) exists carrying 192.168.123.x, enslaving $ETH_IFACE (the robot's internal
#      123.x NIC) and moving its address onto the bridge,
#   3. starts hostapd (which adds the dongle AP into the same bridge, per hostapd.conf's `bridge=`),
#   so an associated desktop becomes an L2 host on 192.168.123.x exactly as the cable made it.
#
# !! DANGER !! If you are SSHed in OVER the wired 192.168.123.x link, step 2 briefly moves that interface's
# IP onto the bridge and CAN drop your SSH session mid-migration. Run under tmux/screen or nohup so a
# dropped session doesn't leave the network half-configured, and keep a direct console/keyboard fallback.
# This touches the robot's live motor-control LAN (Jetson<->motion controller); verify `ping 192.168.123.161` still works
# afterward (the script does this and warns if it fails).
#
# Configure via env vars (all have example defaults; override to match your robot):
#   ETH_IFACE     the robot's internal 192.168.123.x NIC to bridge (check: ip -br addr | grep 192.168.123)
#   DONGLE_IFACE  the USB WiFi dongle (check: iw dev)
#   BRIDGE        bridge name (default br0)
#   PC1_IP        the Jetson address to put on the bridge (default 192.168.123.18/24)
#   HOSTAPD_CONF  path to the filled-in hostapd.conf (default alongside this script)
#
# Example:
#   sudo ETH_IFACE=eth0 DONGLE_IFACE=wlan0 HOSTAPD_CONF=$PWD/hostapd.conf \
#     setsid nohup ./setup_ap.sh > ~/ap.log 2>&1 < /dev/null &
#
# Safety: the script ignores SIGHUP and auto-rolls back (restores PC1_IP + default route on ETH_IFACE, removes
# the bridge) on any failure or whenever hostapd stops, so the wired cable stays usable for emergencies.
set -euo pipefail

ETH_IFACE="${ETH_IFACE:?set ETH_IFACE to the robot 192.168.123.x NIC}"
DONGLE_IFACE="${DONGLE_IFACE:?set DONGLE_IFACE to the USB WiFi dongle}"
BRIDGE="${BRIDGE:-br0}"
PC1_IP="${PC1_IP:-192.168.123.18/24}"                  # the Go2 EDU Jetson (the machine you SSH into)
GATEWAY="${GATEWAY:-192.168.123.1}"                     # default route the Jetson uses on the 123.x LAN
PC1_PING_TARGET="${PC1_PING_TARGET:-192.168.123.161}"  # motion controller -- must stay reachable
HOSTAPD_CONF="${HOSTAPD_CONF:-$(dirname "$0")/hostapd.conf}"

# Survive the SSH session dying: a dropped session sends SIGHUP, which killed an earlier run halfway through
# the cutover and left the robot with no IP on the wired link.
trap '' HUP

# Emergency-cable guarantee: ETH_IFACE stays up as a bridge port, so the wired cable keeps reaching the
# Jetson at PC1_IP. If anything goes wrong after the cutover (error, bad hostapd start, hostapd exiting,
# Ctrl+C, or the motion controller becoming unreachable), rollback() puts PC1_IP and the default route back
# on ETH_IFACE directly and removes the bridge, i.e. the exact pre-script wired setup.
CUTOVER_DONE=0
rollback() {
    [ "$CUTOVER_DONE" = 1 ] || return 0
    CUTOVER_DONE=0
    echo "[ap] ROLLBACK: restoring $PC1_IP and the default route on $ETH_IFACE" >&2
    pkill -x hostapd 2>/dev/null || true
    if ip link show "$BRIDGE" >/dev/null 2>&1; then
        ip -batch - <<ROLLBACK_EOF || true
addr flush dev $BRIDGE
link set $ETH_IFACE nomaster
addr replace $PC1_IP dev $ETH_IFACE
route replace default via $GATEWAY dev $ETH_IFACE
ROLLBACK_EOF
        ip link set "$BRIDGE" down 2>/dev/null || true
        ip link delete "$BRIDGE" type bridge 2>/dev/null || true
    fi
    if command -v nmcli >/dev/null 2>&1; then
        nmcli dev set "$ETH_IFACE" managed yes 2>/dev/null || true
    fi
}
trap rollback EXIT

[ "$(id -u)" -eq 0 ] || { echo "Must run as root." >&2; exit 1; }
for b in ip iw hostapd rfkill; do command -v "$b" >/dev/null || { echo "Missing required tool: $b" >&2; exit 1; }; done
[ -f "$HOSTAPD_CONF" ] || { echo "hostapd.conf not found at $HOSTAPD_CONF" >&2; exit 1; }
sed 's/#.*//' "$HOSTAPD_CONF" | grep -q '<<' && { echo "hostapd.conf still has <<...>> placeholders -- fill them in first." >&2; exit 1; }

# NetworkManager would re-assert IPs on ETH_IFACE and fight hostapd for the dongle. Runtime-only (reverts
# on reboot or via teardown_ap.sh); harmless if NM is not installed or not running.
if command -v nmcli >/dev/null 2>&1; then
    nmcli dev set "$DONGLE_IFACE" managed no 2>/dev/null || true
    nmcli dev set "$ETH_IFACE" managed no 2>/dev/null || true
fi

echo "[ap] unblocking radios + disabling power save on $DONGLE_IFACE"
rfkill unblock wifi || true
ip link set "$DONGLE_IFACE" up || true
iw dev "$DONGLE_IFACE" set power_save off || echo "[ap] WARN: could not disable power save (check driver)"

if ! ip link show "$BRIDGE" >/dev/null 2>&1; then
    echo "[ap] creating bridge $BRIDGE"
    ip link add name "$BRIDGE" type bridge
    ip link set "$BRIDGE" up
fi

# Enslave the 123.x NIC and move its address to the bridge, only if not already done (idempotent).
eth_master_idx="$(cat /sys/class/net/$ETH_IFACE/master/ifindex 2>/dev/null || echo none)"
bridge_idx="$(cat /sys/class/net/$BRIDGE/ifindex)"
if [ "$eth_master_idx" != "$bridge_idx" ]; then
    # Atomic cutover: flush ETH_IFACE's L3, enslave it, and put the SAME IP on the bridge in ONE `ip -batch`
    # invocation. Because TCP is bound to the IP (not the interface) and the IP is only gone for the few ms
    # between these lines, an SSH session riding this link usually survives rather than resetting. Still run
    # under nohup so a worst-case drop can't kill the script mid-way.
    echo "[ap] moving $ETH_IFACE into $BRIDGE and assigning $PC1_IP to $BRIDGE (atomic cutover)"
    ip -batch - <<EOF
addr flush dev $ETH_IFACE
link set $ETH_IFACE master $BRIDGE
link set $ETH_IFACE up
addr replace $PC1_IP dev $BRIDGE
route replace default via $GATEWAY dev $BRIDGE
EOF
    CUTOVER_DONE=1
else
    CUTOVER_DONE=1
    echo "[ap] $ETH_IFACE already in $BRIDGE; ensuring $PC1_IP and default route present"
    ip addr replace "$PC1_IP" dev "$BRIDGE"
    ip route replace default via "$GATEWAY" dev "$BRIDGE"
fi

echo "[ap] verifying internal control link to the motion controller ($PC1_PING_TARGET)"
if ! ping -c 3 -W 2 "$PC1_PING_TARGET" >/dev/null 2>&1; then
    echo "[ap] ERROR: cannot reach $PC1_PING_TARGET after bridging -- rolling back to the wired setup." >&2
    exit 1
fi

# Foreground (not exec) so the EXIT trap fires and restores the wired setup whenever hostapd stops, whether
# by Ctrl+C, a bad config / unsupported channel, or being killed.
echo "[ap] starting hostapd ($HOSTAPD_CONF). Stopping it (Ctrl+C or kill) restores the wired setup automatically."
hostapd "$HOSTAPD_CONF"
