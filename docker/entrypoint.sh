#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Installs the mounted quadruped_go2_locomotion extension into the Isaac Lab
# python env on every container start. This runs on every start (not just at
# build time) because the project source is bind-mounted, not baked in.
set -e

EXTENSION_PATH="${PROJECT_PATH}/source/quadruped_go2_locomotion"
READY_MARKER=/tmp/.entrypoint_ready

if [ -f "${EXTENSION_PATH}/pyproject.toml" ]; then
    "${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install -q -e "${EXTENSION_PATH}"
else
    echo "WARNING: ${EXTENSION_PATH} not found. Is the repo mounted at ${PROJECT_PATH}?" >&2
fi

# `docker compose up -d` returns as soon as this script *starts*, not when it
# finishes, so a `docker exec` launched right after (as recreate_container()
# does before opening a tmux session) can race the pip install above and run
# against a container where the extension isn't installed yet. This marker
# lets recreate_container() wait for the install to actually be done first.
touch "${READY_MARKER}"

exec "$@"
