#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Builds the project's Docker image. Only needs to be re-run when the Isaac
# Sim/Isaac Lab version pins in docker-compose.yml change, not for project
# code changes, which are bind-mounted at runtime.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

cd "${REPO_ROOT}"
docker compose build
