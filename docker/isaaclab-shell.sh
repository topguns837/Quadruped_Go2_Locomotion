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

# On a constrained GPU, use a small env count, a lighter rendering preset,
# a smaller terrain grid, and the hang-safe static play camera (see the
# NOTE next to ViewerCfg in quadruped_go2_locomotion_env_cfg.py for why that
# camera override exists). A capable machine gets none of this: the task's
# own defaults (1024 envs for train, 50 for play) and the original terrain
# grid/camera apply untouched. The terrain/viewer overrides use Hydra's
# `env.<dotted.path>=<value>` CLI syntax (verified against
# isaaclab_tasks/utils/hydra.py: the Hydra config root is a plain
# {"env": ..., "agent": ...} dict, so overriding existing fields this way
# needs no `+` prefix; a wrong path fails fast with "Key '...' is not in
# struct" rather than silently doing nothing).
if is_weak_gpu; then
    TRAIN_ARGS="--num_envs 1 --rendering_mode performance env.scene.terrain.terrain_generator.num_rows=3 env.scene.terrain.terrain_generator.num_cols=5 env.scene.terrain.terrain_generator.use_cache=True"
    PLAY_ARGS="--num_envs 1 --rendering_mode performance env.scene.terrain.terrain_generator.num_rows=3 env.scene.terrain.terrain_generator.num_cols=5 env.scene.terrain.terrain_generator.use_cache=True env.viewer.origin_type=env"
else
    TRAIN_ARGS=""
    PLAY_ARGS=""
fi

TRAIN_CMD="isaaclab -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 ${TRAIN_ARGS}"
PLAY_CMD="isaaclab -p scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2-Play ${PLAY_ARGS}"

# live_dashboard.py's default matplotlib backend crashes in this container (no working Qt/GTK binding --
# see docker/Dockerfile's note next to the feh apt install), so it runs in --output mode: periodically
# saves a PNG instead of opening a window, and feh --reload displays that PNG, auto-refreshing on the
# same interval. --logdir points at the experiment root, not one specific run, so it always picks up
# whichever run is most recently modified.
DASHBOARD_CMD="isaaclab -p scripts/live_dashboard.py --logdir logs/rsl_rl/go2_with_pitch_lean_and_height_control_ppo --output /tmp/dashboard.png"
VIEW_CMD="feh --reload 3 /tmp/dashboard.png"

docker exec -it "${CONTAINER_NAME}" bash -lc '
  tmux new-session -d -s isaaclab -n list-envs
  tmux send-keys -t isaaclab:list-envs "isaaclab -p scripts/list_envs.py"
  tmux new-window -t isaaclab -n train
  tmux send-keys -t isaaclab:train "'"${TRAIN_CMD}"'"
  DASH_PANE=$(tmux split-window -h -t isaaclab:train -P -F "#{pane_id}")
  tmux send-keys -t "$DASH_PANE" "'"${DASHBOARD_CMD}"'"
  VIEW_PANE=$(tmux split-window -v -t "$DASH_PANE" -P -F "#{pane_id}")
  tmux send-keys -t "$VIEW_PANE" "'"${VIEW_CMD}"'"
  tmux select-pane -t isaaclab:train.0
  tmux new-window -t isaaclab -n play
  tmux send-keys -t isaaclab:play "'"${PLAY_CMD}"'"
  tmux select-window -t isaaclab:list-envs
  tmux attach -t isaaclab
'
