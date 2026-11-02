# Wireless link for Go2 deployment

**Status (2026-10-29): the Ethernet cable is the only supported link.** A Jetson-hosted WiFi access point was
tried and retired. This file records why, and what a WiFi link needs if you set one up later.

## Hard rules

- **Never plug the USB WiFi dongle into the Jetson.** With it in, the Jetson stayed unreachable even after
  several power cycles, and came back the moment the dongle was removed. Nothing of ours runs at boot, so the
  dongle or its USB port wedges the Jetson on its own. Its driver (`rt2800usb`) had also logged USB `-110`
  timeouts and a kernel oops.
- **Never change the Jetson's `eth0`:** no bridging it, no moving its IP, no touching its routes or
  NetworkManager profile. The wired link carries the control loop and is your emergency access.

## Why the Jetson-hosted AP was retired

- `setup_ap.sh` bridged `eth0` into `br0` and moved `192.168.123.18` onto the bridge. That is the only way a
  bridge on the Jetson can reach the `123.x` network, so it cannot be done without touching `eth0`.
- The AP's radio was unusable anyway: unicast frames stalled and were released in bursts, about a third of the
  AP's transmissions failed, and the desktop saw beacon loss. A phone hotspot worked fine with the desktop's
  dongle, so the stall was on the Jetson's side.
- A route or NAT from a separate WiFi subnet cannot carry the robot's DDS. Discovery is multicast, DDS puts
  IP addresses inside its payloads (NAT cannot rewrite them), the motion controller has no route back to a
  foreign subnet, and `unitree_sdk2py` ignores `CYCLONEDDS_URI` (it passes its own config to `Domain()`, see
  `unitree_sdk2py/core/channel.py`). So the desktop has to be on the robot's `192.168.123.0/24` at layer 2.

## What a WiFi link needs

One small device that bridges WiFi to Ethernet, plugged into the robot's **spare Ethernet port**. It must not
be the Jetson, and nothing is installed on the robot.

1. A router or access point in **bridge / AP mode**, with **DHCP turned off** (the `123.x` network is
   static) and a dedicated SSID. It must not also bridge your wider lab network onto `123.x`. A spare router
   you own and can configure works; so would a small travel router mounted on the robot (needed only if the
   robot has to roam, since a cable ties it to the router). Power a robot-mounted one from a USB power bank,
   not from the Jetson's USB.
2. On the desktop, join that SSID and set the WiFi connection to a static address with no gateway:
   ```
   nmcli con modify <wifi-con> ipv4.method manual ipv4.addresses 192.168.123.99/24 \
     ipv4.gateway "" ipv4.dns "" ipv4.never-default yes ipv6.method disabled 802-11-wireless.powersave 2
   nmcli con up <wifi-con>
   ```
3. For a wireless run, unplug the desktop's cable so only WiFi is on `123.x`. With both connected, replies
   can take either path and `ping -I <wifi>` tests become unreliable. Replug it for emergency access.
4. Point deploy at the WiFi interface: `network_interface` in `deploy/configs/go2_locomotion.yaml`, or
   `--network_interface <wifi>`. No code change is needed.

Bring it up in order, and stop at the first rung that fails (`deploy/Checklist.md` still governs real
hardware):

- Link: `deploy/wireless/link_check.sh <wifi> 60`. You want 0% loss, no bursts, and a max latency well under
  the 500 ms link-loss threshold. Then `sudo ping -i 0.02 -c 500 -I <wifi> 192.168.123.18`, and a 15 minute
  soak at 1 packet/s.
- `deploy/preflight_check.py --network_interface <wifi>` (real LowState, read-only).
- `deploy/fake_robot.py` on the robot with `deploy_real.py` on the desktop, watching the end-of-run
  `[LINK] LowState gaps` summary.
- Hoisted real-robot runs per `deploy/Checklist.md`. Tune `--link_lost_after` only from the measured gaps.

## Files in this directory

- `link_check.sh`: desktop-side link tester (loss, latency, longest silence, burst detection). Still in use.
- `teardown_ap.sh`: recovery only. It acts only if a leftover `br0` exists on the Jetson, to put `eth0` back.
- `setup_ap.sh`: **disabled**, exits immediately.
- `hostapd.conf`, `hostapd_open.conf`: **obsolete**, only used by `setup_ap.sh`. `hostapd.conf` contains the
  test WiFi passphrase, so keep both files out of commits.
