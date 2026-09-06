#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Shared helpers sourced by the other docker/*.sh scripts. Not meant to be
# run directly.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME=quadruped-go2
CONTAINER_NAME=quadruped_go2_locomotion

# GPU VRAM (MiB) below which we treat this as a constrained machine and apply
# laptop-style workarounds (small num_envs, lighter rendering, a smaller
# terrain grid, a hang-safe play camera). Comfortably above this laptop's
# 6144 MiB; adjust if a new constrained machine needs a different cutoff.
WEAK_GPU_VRAM_THRESHOLD_MIB=8192

# True (exit 0) if the first GPU's total VRAM is below the threshold above,
# or if nvidia-smi isn't available at all, in which case we don't know this
# machine's capability and default to the safe/constrained assumption rather
# than assuming it can handle the full settings.
is_weak_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    local vram_mib
    vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [ -z "${vram_mib}" ]; then
        return 0
    fi
    [ "${vram_mib}" -lt "${WEAK_GPU_VRAM_THRESHOLD_MIB}" ]
}

# Ensures exactly one fresh container is running for this project: tears down
# whatever is currently there (if anything) and starts a brand new one. This
# means an in-progress process inside a previous container is lost. That's
# intentional, see docker/README.md.
recreate_container() {
    cd "${REPO_ROOT}"
    # Grant the container's X11 client (runs as root) access to the host X
    # server. Only lasts for the current host X session (resets on
    # logout/reboot), so it's cheap to redo on every container start. Skipped
    # quietly if xhost isn't installed (e.g. a headless host).
    if command -v xhost >/dev/null 2>&1; then
        xhost +local:docker >/dev/null 2>&1 || true
    fi
    docker compose down --remove-orphans
    docker compose up -d

    # docker compose up -d returns as soon as the container starts, not when
    # entrypoint.sh's pip install -e finishes, so exec-ing into it too early
    # (e.g. to open a tmux session right after this call) can race that
    # install and hit a spurious "No module named 'quadruped_go2_locomotion'".
    # Wait for entrypoint.sh's readiness marker before returning control.
    local waited=0
    until docker exec "${CONTAINER_NAME}" test -f /tmp/.entrypoint_ready 2>/dev/null; do
        sleep 0.5
        waited=$((waited + 1))
        if [ "${waited}" -ge 60 ]; then
            echo "WARNING: timed out waiting for the container to finish starting up (30s). Continuing anyway." >&2
            break
        fi
    done
}
