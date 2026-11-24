# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional ``relay:`` block of a deploy profile (deploy/configs/safety_profiles.yaml) for deploy_real_wireless.py.

Pure YAML + validation, no torch / SDK, so it is unit-testable anywhere. The block's keys map onto the wireless
script's relay flags; the result is fed to argparse ``set_defaults`` so precedence stays CLI > profile > default.
deploy_real.py (wired) ignores the block.
"""

from __future__ import annotations

import ipaddress

import yaml

# block key -> argparse dest in deploy_real_wireless.py's relay parser
BLOCK_TO_DEST = {
    "host": "relay_host",
    "key_file": "relay_key_file",
    "control_port": "relay_control_port",
    "cmd_port": "relay_cmd_port",
    "cmd_rate_hz": "relay_cmd_rate_hz",
    "log_nic": "log_nic",
}
RELAY_DESTS = tuple(BLOCK_TO_DEST.values())

# The robot's wired motor LAN. A relay host (or a modem subnet) here collides with the Jetson's eth0.
ROBOT_LAN = ipaddress.ip_network("192.168.123.0/24")


def _need_str(profile: str, key: str, v) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"profile '{profile}' relay.{key} must be a non-empty string, got {v!r}")
    return v


def _need_port(profile: str, key: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 65535:
        raise ValueError(f"profile '{profile}' relay.{key} must be a port 1-65535, got {v!r}")
    return v


def _need_rate(profile: str, key: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise ValueError(f"profile '{profile}' relay.{key} must be a positive number, got {v!r}")
    return float(v)


def _host(profile: str, key: str, v) -> str:
    v = _need_str(profile, key, v)
    try:
        ip = ipaddress.ip_address(v)
    except ValueError:
        return v  # a hostname: nothing to check
    if ip in ROBOT_LAN:
        raise ValueError(
            f"profile '{profile}' relay.host {v} is on the robot's motor LAN {ROBOT_LAN}; the relay host must be "
            "the Jetson's address on the modem/WiFi subnet (never 192.168.123.x)")
    return v


_CHECK = {"host": _host, "key_file": _need_str, "control_port": _need_port, "cmd_port": _need_port,
          "cmd_rate_hz": _need_rate, "log_nic": _need_str}


def load_relay_block(path: str, name: str) -> dict:
    """Return ``{relay_flag_dest: value}`` from profile ``name``'s ``relay:`` block ({} when it has none).

    Raises ValueError on an unknown profile, a non-mapping block, an unknown key or a bad value, so a typo fails at
    startup rather than silently running against the wrong host.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if name not in data:
        raise ValueError(f"profile '{name}' not found in {path}. Available: {sorted(data)}")
    profile = data[name]
    if not isinstance(profile, dict) or "relay" not in profile:
        return {}
    block = profile["relay"]
    if not isinstance(block, dict):
        raise ValueError(f"profile '{name}' relay: must be a mapping, got {type(block).__name__}")
    bad = [k for k in block if k not in BLOCK_TO_DEST]
    if bad:
        raise ValueError(f"profile '{name}' relay: has unknown key(s) {bad}; allowed: {sorted(BLOCK_TO_DEST)}")
    return {BLOCK_TO_DEST[k]: _CHECK[k](name, k, v) for k, v in block.items()}
