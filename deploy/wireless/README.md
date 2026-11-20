# Wireless link for Go2 deployment

**Status (2026-11-20): the supported wireless link is the application-level RELAY over a modem's WiFi**
(`deploy/relay_jetson.py` on the Jetson + `deploy/deploy_real_wireless.py` on the desktop, see "Jetson USB client +
relay" below). The Ethernet cable stays the default link for `deploy_real.py` and the emergency path. A Jetson-hosted
WiFi access point bridged onto `eth0` was tried and retired; the sections after "Hard rules" record why.

## Jetson USB client + relay (the current setup)

```
 desktop --(Ethernet)--> modem/router --(2.4 GHz WiFi)--> TP-Link USB dongle (Jetson, CLIENT) 
                                                              |  relay_jetson.py: DDS on eth0 (untouched)
                                                              +--> robot motion controller 192.168.123.161
```

- The dongle only gives the *Jetson* a modem-subnet address. It does NOT put the robot's `192.168.123.0/24` motor LAN
  on WiFi, so DDS never crosses the air: the relay translates DDS <-> signed UDP/TCP at the Jetson (ports 9870/9871).
- **The modem subnet must not be `192.168.123.0/24`** (collides with the Jetson's `eth0`). Use e.g. `192.168.50.0/24`.
  `relay_profile.py` refuses a `relay.host` inside `192.168.123.0/24`.
- 2.4 GHz SSID (the TL-WN722N v2 is 2.4 GHz only). Give the dongle a **stable address** (DHCP reservation on its MAC, or
  a static IP in its own NetworkManager profile): the desktop's `relay.host` must not change between runs.
- On the Jetson, the WiFi NetworkManager profile must be bound to the wlan device only
  (`connection.interface-name <wlan>`) and never edit `eth0`'s profile, IP or routes. After bring-up check that `eth0`
  is unchanged and `python deploy/lowstate_probe.py --network_interface eth0` still shows ~500 Hz.
- Settings live in the `wireless` profile of `deploy/configs/test_profiles.yaml` (a `relay:` block: host, key_file,
  ports, rate, log_nic); a run is `deploy/deploy_real_wireless.py --deploy_profile wireless --manual_commands`. Any
  explicit flag overrides the profile.
- Bench first, on the desktop alone with no robot: `deploy/tests/run_t2_loopback.sh` (stack on loopback), then
  `link_check.sh <dongle iface> 60 <modem ip>` to measure the real radio. Do NOT bind the relay to the dongle IP and
  the client to the wired NIC on one host and call it a WiFi test: both addresses are local, so the kernel hands the
  packets over internally and they never reach the radio. Use a network namespace for the dongle, or measure against
  the modem. Full ladder: `deploy.md` Stage 0b.

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
