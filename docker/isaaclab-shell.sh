#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Opens a freshly (re)created project container with a tmux session ready
# for Isaac Lab work: one window per common task (list envs / train / play),
# each with its command already typed into the pane but NOT executed. Press
# Enter on whichever one you want, or edit it first.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

recreate_container

docker exec -it "${CONTAINER_NAME}" bash -lc '
  tmux new-session -d -s isaaclab -n list-envs
  tmux send-keys -t isaaclab:list-envs "isaaclab -p scripts/list_envs.py"
  tmux new-window -t isaaclab -n train
  tmux send-keys -t isaaclab:train "isaaclab -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --rendering_mode performance --num_envs 1 --livestream 2"
  tmux new-window -t isaaclab -n play
  tmux send-keys -t isaaclab:play "isaaclab -p scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1 --rendering_mode performance --livestream 2"
  tmux select-window -t isaaclab:list-envs
  tmux attach -t isaaclab
'
