#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Opens an empty tmux shell in a freshly (re)created project container, for
# free experimentation/debugging. Same underlying setup as
# isaaclab-shell.sh (project mounted, extension pip-installed). This just
# drops you straight at a bare prompt instead of pretyping any commands.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

recreate_container
docker exec -it "${CONTAINER_NAME}" tmux new-session -s shell
