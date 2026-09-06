#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Menu entry point for the Docker dev workflow. See docker/README.md.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}/docker"

while true; do
    echo ""
    echo "Start Menu"
    echo "  1) Open Isaac Lab"
    echo "  2) Build docker image"
    echo "  3) Open empty container"
    echo "  4) Exit"
    read -r -p "Choose an option [1-4]: " choice

    case "${choice}" in
        1) "${DOCKER_DIR}/isaaclab-shell.sh" ;;
        2) "${DOCKER_DIR}/build.sh" ;;
        3) "${DOCKER_DIR}/shell.sh" ;;
        4) exit 0 ;;
        *) echo "Invalid option: ${choice}" ;;
    esac
done
