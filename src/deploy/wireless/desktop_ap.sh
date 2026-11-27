#!/usr/bin/env bash
# Run a USB WiFi dongle on the DESKTOP as an access point (hostapd), to replace a scanning enterprise AP (e.g. the
# Cisco WAP150, whose periodic off-channel scan stalls the relay downlink). The Jetson's dongle then joins THIS as a
# client and the relay runs desktop<->Jetson over it.
#
#   sudo bash src/deploy/wireless/desktop_ap.sh <ap_iface> [ssid] [channel] [ip/prefix]
#   e.g.  sudo bash src/deploy/wireless/desktop_ap.sh wlxf8d1110f7bdb go2desk 6
#
# Open network (no passphrase) for a quick test -- add WPA later. Static addressing, no DHCP: the AP is
# <ip/prefix> (default 192.168.60.1/24); point the Jetson at a static 192.168.60.20/24 on the same SSID. Ctrl+C
# to stop; the dongle is handed back to NetworkManager. Desktop-side ONLY: it never touches the Jetson or eth0.
set -u

IFACE="${1:?usage: sudo bash desktop_ap.sh <ap_iface> [ssid] [channel] [ip/prefix]}"
SSID="${2:-go2desk}"
CH="${3:-6}"
ADDR="${4:-192.168.60.1/24}"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 2; }
command -v hostapd >/dev/null || { echo "hostapd not installed: sudo apt install hostapd" >&2; exit 2; }
case "$IFACE" in wl*|wlan*) ;; *) echo "refusing: '$IFACE' does not look like a WiFi interface" >&2; exit 2;; esac
[ -d "/sys/class/net/$IFACE/wireless" ] || { echo "refusing: $IFACE is not a wireless interface" >&2; exit 2; }
iw phy "phy$(iw dev "$IFACE" info | awk '/wiphy/{print $2}')" info | grep -qE "^\s+\* AP$" \
    || { echo "refusing: $IFACE does not advertise AP mode" >&2; exit 2; }

CONF="$(mktemp /tmp/hostapd_desktop.XXXXXX.conf)"
cleanup() {
    trap - INT TERM EXIT
    echo; echo "[ap] stopping hostapd, handing $IFACE back to NetworkManager"
    ip addr flush dev "$IFACE" 2>/dev/null
    nmcli dev set "$IFACE" managed yes 2>/dev/null || true
    rm -f "$CONF"
}
trap cleanup INT TERM EXIT

rfkill unblock wifi 2>/dev/null || true
nmcli dev set "$IFACE" managed no 2>/dev/null || true   # keep NM from fighting hostapd for the radio
pkill -f "wpa_supplicant.*$IFACE" 2>/dev/null || true
ip addr flush dev "$IFACE"
ip link set "$IFACE" up
ip addr add "$ADDR" dev "$IFACE"

cat > "$CONF" <<EOF
interface=$IFACE
driver=nl80211
ssid=$SSID
country_code=US
ieee80211d=1
hw_mode=g
channel=$CH
wmm_enabled=1
auth_algs=1
ignore_broadcast_ssid=0
EOF

echo "[ap] starting AP '$SSID' on $IFACE, channel $CH, ${ADDR%/*} (open network)"
echo "[ap] On the Jetson, join it with a STATIC address (new profile, never touches eth0):"
echo "       sudo nmcli con add type wifi ifname wlan0 con-name $SSID ssid $SSID \\"
echo "         ipv4.method manual ipv4.addresses 192.168.60.20/24 ipv4.never-default yes \\"
echo "         ipv6.method ignore 802-11-wireless.powersave 2 connection.autoconnect-priority 90"
echo "       sudo nmcli con up $SSID ifname wlan0"
echo "[ap] then test from the desktop:"
echo "       python3 src/deploy/tools/udp_probe.py client --host 192.168.60.20 --port 9890 --rate 100 --duration 180"
echo "[ap] (start the server on the Jetson: python3 ~/udp_probe.py server --port 9890). Ctrl+C here to stop."
exec hostapd "$CONF"
