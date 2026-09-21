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
PLAY_CMD="isaaclab -p scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2-Play ${PLAY_ARGS} --manual_commands --live_plot"

# live_dashboard.py's default matplotlib backend crashes in this container (no working Qt/GTK binding --
# see docker/Dockerfile's note next to the feh apt install), so it runs in --output mode: periodically
# saves a PNG instead of opening a window, and feh --reload displays that PNG, auto-refreshing on the
# same interval. --logdir points at the experiment root, not one specific run, so it always picks up
# whichever run is most recently modified.
DASHBOARD_CMD="isaaclab -p scripts/live_dashboard.py --logdir logs/rsl_rl/go2_with_pitch_lean_and_height_control_ppo --output /tmp/dashboard.png"
VIEW_CMD="feh --reload 3 /tmp/dashboard.png"

# Same live-dashboard setup as training's, but reading play.py's --live_plot output instead (a separate
# TensorBoard log under logs/play_dashboard/, written only when --live_plot is on -- see play.py). Same
# script (--exclude-panels/--y-range-multiplier default to "show everything, tight fit", so training's
# DASHBOARD_CMD above is completely unaffected by these -- they're only passed here):
#   --max-points 500: play logs per physics step, not per training iteration, so the default 100-point
#     window is only ~2s of history; 500 gives a much more readable rolling window.
#   --exclude-panels: reward/episode-length/tracking-error are training concepts play.py never logs, so
#     they'd just render as blank panels -- explicitly dropped instead.
#   --y-range-multiplier 2.0: doubles each panel's y-axis range around its tight auto-fit, for more visual
#     headroom comparing the (usually flat) commanded line against the noisier actual line.
# Single-quotes (not escaped double-quotes) around the panel list below are deliberate: this string passes
# through host-level variable expansion, then a tmux send-keys splice, before finally being typed into the
# pane and parsed by ITS shell -- single-quoting survives that whole chain without premature quote-closing,
# where escaped double-quotes would have broken partway through (verified directly).
PLAY_DASHBOARD_CMD="isaaclab -p scripts/live_dashboard.py --logdir logs/play_dashboard --output /tmp/play_dashboard.png --max-points 500 --exclude-panels 'Mean reward,Mean episode length,Velocity tracking error' --y-range-multiplier 2.0"
PLAY_VIEW_CMD="feh --reload 3 /tmp/play_dashboard.png"

# Real-hardware deployment (see deploy.md, deploy/deploy_real.py). Unlike TRAIN_CMD/PLAY_CMD this does NOT
# use the `isaaclab -p` wrapper -- deploy_real.py never touches pxr/Omniverse/SimulationApp, so launching
# the full Kit app would be pure waste (slow startup, unneeded GPU/rendering init). It's run directly under
# Isaac Sim's bundled Python instead, which already has torch/tensorboard without launching Kit (same
# interpreter this session's TensorBoard log-reading checks used) -- only unitree_sdk2py needs installing
# into it separately (see deploy_real.py's own module docstring).
#
# --dry_run is deliberately the pre-typed default -- this command is one Enter-press away from powering
# real motors once --network_interface and --dry_run are edited out, so it must never be pre-filled ready
# to send real commands. REPLACE_WITH_NIC_NAME is a literal placeholder (e.g. "enp3s0", found via
# `ifconfig` after connecting the Ethernet cable to the robot per deploy.md Stage 0) -- replace it before
# removing --dry_run. Deliberately NOT wrapped in <angle brackets>: those are shell redirection operators
# when unquoted, and this string is typed directly into the pane's own shell -- confirmed directly that
# `<FILL_IN_NIC>` silently drops --manual_commands from the command line (swallowed as a redirect target)
# and fails on a missing "FILL_IN_NIC" input file, instead of erroring obviously. A bare placeholder word
# has no such failure mode. network_mode: host (docker-compose.yml) is what makes binding a specific host
# NIC from inside this container possible at all.
HARDWARE_CMD="/workspace/isaaclab/_isaac_sim/python.sh deploy/deploy_real.py --network_interface REPLACE_WITH_NIC_NAME --manual_commands --live_plot --dry_run"

# Same live-dashboard setup as play's, reading deploy_real.py's --live_plot output instead (logs/hardware_dashboard/,
# same tag names as play.py's --live_plot -- see deploy_real.py's log_live_plot). Same --exclude-panels/
# --y-range-multiplier reasoning as PLAY_DASHBOARD_CMD above; training's DASHBOARD_CMD is unaffected.
HARDWARE_DASHBOARD_CMD="isaaclab -p scripts/live_dashboard.py --logdir logs/hardware_dashboard --output /tmp/hardware_dashboard.png --max-points 500 --exclude-panels 'Mean reward,Mean episode length,Velocity tracking error' --y-range-multiplier 2.0"
HARDWARE_VIEW_CMD="feh --reload 3 /tmp/hardware_dashboard.png"

# Tier-1 read-only pre-flight check (deploy/preflight_check.py) -- structurally incapable of publishing
# anything (no ChannelPublisher/LowCmd_ import in that file at all), so it's safe to run alongside
# HARDWARE_CMD without any risk of the two fighting over control. Same interpreter choice as HARDWARE_CMD
# (Isaac Sim's bundled Python directly, not `isaaclab -p`) for the same reason -- no Kit/SimulationApp
# involved. Shares REPLACE_WITH_NIC_NAME with HARDWARE_CMD; same placeholder-safety reasoning applies (see
# HARDWARE_CMD's comment on why it's not wrapped in <angle brackets>).
PREFLIGHT_CMD="/workspace/isaaclab/_isaac_sim/python.sh deploy/preflight_check.py --network_interface REPLACE_WITH_NIC_NAME"

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
  PLAY_DASH_PANE=$(tmux split-window -h -t isaaclab:play -P -F "#{pane_id}")
  tmux send-keys -t "$PLAY_DASH_PANE" "'"${PLAY_DASHBOARD_CMD}"'"
  PLAY_VIEW_PANE=$(tmux split-window -v -t "$PLAY_DASH_PANE" -P -F "#{pane_id}")
  tmux send-keys -t "$PLAY_VIEW_PANE" "'"${PLAY_VIEW_CMD}"'"
  tmux select-pane -t isaaclab:play.0
  tmux new-window -t isaaclab -n hardware
  HW_CMD_PANE=$(tmux display-message -p -t isaaclab:hardware -F "#{pane_id}")
  tmux send-keys -t "$HW_CMD_PANE" "'"${HARDWARE_CMD}"'"
  HW_DASH_PANE=$(tmux split-window -h -t "$HW_CMD_PANE" -P -F "#{pane_id}")
  tmux send-keys -t "$HW_DASH_PANE" "'"${HARDWARE_DASHBOARD_CMD}"'"
  HW_PREFLIGHT_PANE=$(tmux split-window -v -t "$HW_CMD_PANE" -P -F "#{pane_id}")
  tmux send-keys -t "$HW_PREFLIGHT_PANE" "'"${PREFLIGHT_CMD}"'"
  HW_VIEW_PANE=$(tmux split-window -v -t "$HW_DASH_PANE" -P -F "#{pane_id}")
  tmux send-keys -t "$HW_VIEW_PANE" "'"${HARDWARE_VIEW_CMD}"'"
  tmux select-pane -t "$HW_CMD_PANE"
  tmux select-window -t isaaclab:list-envs
  tmux attach -t isaaclab
'
