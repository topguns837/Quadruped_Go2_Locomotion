#!/usr/bin/env bash
# Train in fixed-size batches, restarting the training process between batches so all of its host RAM and its
# CUDA/VRAM context are released (a process exit is the only reliable way to free either). Each batch resumes from
# the latest checkpoint and writes to ONE fixed run folder (train.py --run_dir), so the dashboard/TensorBoard curve
# and the checkpoints stay continuous across batches. The live-graphs dashboard (src/common/live_dashboard.py + feh)
# is (re)started at the start of every batch.
#
# Run INSIDE the training container, from the repo root (/workspace/quadruped_go2_locomotion):
#   bash src/training/train_batched.sh                                  # 2000-iteration batches up to 18000, headless
#   bash src/training/train_batched.sh --num_envs 1024 --batch_iterations 1000 --total_iterations 6000
#   bash src/training/train_batched.sh --livestream 2 --rendering_mode performance
#   bash src/training/train_batched.sh --dry_run --livestream 2         # print the resolved commands, start nothing
# Run with --help for every option. Anything not recognised is passed to train.py unchanged (e.g. Hydra overrides
# such as `agent.save_interval=10 env.commands.base_velocity.debug_vis=False`).
#
# Curriculum state (difficulty, terrain levels) is saved/restored by train.py as curriculum_<N>.pt next to each
# checkpoint, so batches continue the curriculum instead of restarting it at difficulty 0.
#
# Each batch ends at the next multiple of the batch size (so a crash at iteration 1500 resumes to 2000, not 3500).
# Because RSL-RL names a batch's last checkpoint model_{end-1}, batches end at model_1999, model_3999, ...
# Ctrl-C in the terminal stops the loop (train.py exits gracefully, no further batch starts). A SIGINT sent only to
# train.py (e.g. by src/training/train_mem_guard.sh) ends just that batch; the loop then resumes from the last checkpoint.
set -uo pipefail

cd "$(dirname "$0")/../.."

usage() {
  cat <<'EOF'
Usage: src/training/train_batched.sh [options] [train.py args / Hydra overrides ...]

Batching
  --batch_iterations N     iterations per training process (default 2000; env BATCH)
  --total_iterations N     stop once N iterations are done (default 18000; env TARGET). The final checkpoint is
                           model_{N-1}.
  --run_dir NAME           run folder under logs/rsl_rl/<experiment>/ shared by all batches (default run_iter3; env RUN)
  --experiment_name NAME   experiment folder (default: read from agents/rsl_rl_ppo_cfg.py; env EXP)
  --max_fails N            give up after N consecutive batches with no new checkpoint (default 3)

Training (forwarded to train.py)
  --task ID                gym task (default Quadruped-Locomotion-Go2; env TASK)
  --num_envs N             parallel environments (default 2048; env NUM_ENVS)
  --headless               no window, no stream (the default when neither --livestream nor --gui is given)
  --livestream 1|2         stream the viewport over WebRTC (implies headless rendering); 2 = private network
  --gui                    native Isaac window (neither --headless nor --livestream)
  --allow_large_render     permit --gui/--livestream with --num_envs above the render cap (default cap 256; env
                           RENDER_ENV_CAP). Rendering every env costs huge host RAM -- 2048 OOM-killed the host
                           (~120 GB); headless is unaffected. Only raise this if you know the box has the RAM.
  --rendering_mode MODE    performance | balanced | quality (use performance with --livestream/--gui)
  --seed N                 environment/agent seed

Live graphs
  --no_dashboard           do not (re)start src/common/live_dashboard.py at each batch
  --no_viewer              do not open feh on the dashboard PNG
  --dashboard_output PATH  PNG the dashboard writes and feh shows (default /tmp/dashboard.png; env DASH_OUT)
                           Any dashboard already writing that PNG (e.g. the one in the tmux pane) is replaced.

Release check after each batch
  --vram_tol_mib N         VRAM must return within N MiB of its pre-batch level (default 500)
  --release_timeout S      give up waiting for the release after S seconds (default 180)

Other
  --dry_run                print the resolved train.py / dashboard commands for the next batch and exit
  -h, --help               this text

--max_iterations, --resume, --load_run and --checkpoint are managed by this script and are rejected.
EOF
}

die() { echo "train_batched.sh: $*" >&2; exit 2; }

# ---- defaults (environment variables stay supported as defaults) -------------------------------------------------
TASK="${TASK:-Quadruped-Locomotion-Go2}"
BATCH="${BATCH:-2000}"
TARGET="${TARGET:-18000}"
RUN="${RUN:-run_iter3}"
EXP="${EXP:-}"
EXP_ARG=""
NUM_ENVS="${NUM_ENVS:-2048}"
HEADLESS=""            # "" = decide below, 1 = --headless, 0 = native window
LIVESTREAM=""
RENDERING_MODE=""
SEED=""
EXTRA="${EXTRA:-}"     # legacy: extra train.py arguments, word-split
MAX_FAILS="${MAX_FAILS:-3}"
VRAM_TOL_MIB="${VRAM_TOL_MIB:-500}"
RELEASE_TIMEOUT="${RELEASE_TIMEOUT:-180}"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"
DASHBOARD=1
VIEWER=1
DASH_OUT="${DASH_OUT:-/tmp/dashboard.png}"
DRY=0
ALLOW_LARGE_RENDER=0
RENDER_ENV_CAP="${RENDER_ENV_CAP:-256}"
PASS=()

# ---- argument parsing ----------------------------------------------------------------------------------------------
take() {  # usage: take "$@"; sets VAL and SHIFT (handles both `--opt value` and `--opt=value`)
  if [ "$has_val" -eq 1 ]; then VAL="$val"; SHIFT=1
  else [ $# -ge 2 ] || die "$opt needs a value"; VAL="$2"; SHIFT=2; fi
}
while [ $# -gt 0 ]; do
  opt="$1"; val=""; has_val=0
  case "$opt" in --*=*) val="${opt#*=}"; opt="${opt%%=*}"; has_val=1 ;; esac
  case "$opt" in
    -h|--help) usage; exit 0 ;;
    --task) take "$@"; TASK="$VAL"; shift "$SHIFT" ;;
    --num_envs) take "$@"; NUM_ENVS="$VAL"; shift "$SHIFT" ;;
    --batch_iterations) take "$@"; BATCH="$VAL"; shift "$SHIFT" ;;
    --total_iterations) take "$@"; TARGET="$VAL"; shift "$SHIFT" ;;
    --run_dir) take "$@"; RUN="$VAL"; shift "$SHIFT" ;;
    --experiment_name) take "$@"; EXP="$VAL"; EXP_ARG="$VAL"; shift "$SHIFT" ;;
    --max_fails) take "$@"; MAX_FAILS="$VAL"; shift "$SHIFT" ;;
    --headless) HEADLESS=1; shift ;;
    --gui) HEADLESS=0; shift ;;
    --livestream) take "$@"; LIVESTREAM="$VAL"; shift "$SHIFT" ;;
    --rendering_mode) take "$@"; RENDERING_MODE="$VAL"; shift "$SHIFT" ;;
    --seed) take "$@"; SEED="$VAL"; shift "$SHIFT" ;;
    --no_dashboard) DASHBOARD=0; shift ;;
    --no_viewer) VIEWER=0; shift ;;
    --dashboard_output) take "$@"; DASH_OUT="$VAL"; shift "$SHIFT" ;;
    --vram_tol_mib) take "$@"; VRAM_TOL_MIB="$VAL"; shift "$SHIFT" ;;
    --release_timeout) take "$@"; RELEASE_TIMEOUT="$VAL"; shift "$SHIFT" ;;
    --allow_large_render) ALLOW_LARGE_RENDER=1; shift ;;
    --dry_run) DRY=1; shift ;;
    --max_iterations|--resume|--load_run|--checkpoint|--load_checkpoint)
      die "$opt is managed by this script (use --batch_iterations / --total_iterations)" ;;
    --) shift; PASS+=("$@"); break ;;
    *) PASS+=("$1"); shift ;;
  esac
done

is_pos_int() { [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]; }
is_pos_int "$BATCH" || die "--batch_iterations must be a positive integer (got '$BATCH')"
is_pos_int "$TARGET" || die "--total_iterations must be a positive integer (got '$TARGET')"
is_pos_int "$NUM_ENVS" || die "--num_envs must be a positive integer (got '$NUM_ENVS')"
is_pos_int "$MAX_FAILS" || die "--max_fails must be a positive integer (got '$MAX_FAILS')"
is_pos_int "$RELEASE_TIMEOUT" || die "--release_timeout must be a positive integer (got '$RELEASE_TIMEOUT')"
[[ "$VRAM_TOL_MIB" =~ ^[0-9]+$ ]] || die "--vram_tol_mib must be a non-negative integer (got '$VRAM_TOL_MIB')"
[ -z "$LIVESTREAM" ] || [[ "$LIVESTREAM" =~ ^[12]$ ]] || die "--livestream must be 1 or 2 (got '$LIVESTREAM')"
[ -z "$RENDERING_MODE" ] || [[ "$RENDERING_MODE" =~ ^(performance|balanced|quality)$ ]] \
  || die "--rendering_mode must be performance, balanced or quality (got '$RENDERING_MODE')"
[ -z "$SEED" ] || [[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer (got '$SEED')"

CFG="src/env/quadruped_go2_locomotion/agents/rsl_rl_ppo_cfg.py"
[ -n "$EXP" ] || EXP="$(grep -oP 'experiment_name = "\K[^"]+' "$CFG" | head -1)"
[ -n "$EXP" ] || die "cannot determine the experiment name (pass --experiment_name)"

# Rendering (native --gui or --livestream) builds every environment in the render graph, which costs enormous host
# RAM: at 2048 envs it reached ~120 GB and the kernel OOM-killed the whole host (headless uses ~8 GB). Refuse a
# render run whose env count is above the cap so it can't silently take the desktop down; --allow_large_render
# overrides. The headless default run is never affected.
rendering=0
{ [ -n "$LIVESTREAM" ] || [ "$HEADLESS" = "0" ]; } && rendering=1
if [ "$rendering" -eq 1 ] && [ "$NUM_ENVS" -gt "$RENDER_ENV_CAP" ] && [ "$ALLOW_LARGE_RENDER" -eq 0 ]; then
  die "rendering (--gui/--livestream) with --num_envs $NUM_ENVS exceeds the render cap $RENDER_ENV_CAP; rendering
every env OOM-killed the host at 2048. Use a smaller --num_envs (e.g. 64-256) to watch, keep --headless for the
full $NUM_ENVS run, or pass --allow_large_render / set RENDER_ENV_CAP to override."
fi

# Arguments forwarded to train.py on every batch.
train_args=(--task="$TASK" --num_envs "$NUM_ENVS")
if [ "$HEADLESS" = "1" ] || { [ -z "$HEADLESS" ] && [ -z "$LIVESTREAM" ]; }; then train_args+=(--headless); fi
[ -z "$LIVESTREAM" ] || train_args+=(--livestream "$LIVESTREAM")
[ -z "$RENDERING_MODE" ] || train_args+=(--rendering_mode "$RENDERING_MODE")
[ -z "$SEED" ] || train_args+=(--seed "$SEED")
# Pass the experiment name as a Hydra override, NOT the --experiment_name flag: that flag is defined in cli_args.py
# but never applied to agent_cfg (train.py writes to the config's default experiment, so the script would then look
# for checkpoints in the wrong folder and give up), and it also perturbs parsing of the following Hydra overrides.
# agent.experiment_name=... sets it cleanly and routes train.py to logs/rsl_rl/<EXP>/<RUN>.
[ -z "$EXP_ARG" ] || train_args+=("agent.experiment_name=$EXP_ARG")
# shellcheck disable=SC2206  # EXTRA is intentionally word-split
[ -z "$EXTRA" ] || train_args+=($EXTRA)
[ "${#PASS[@]}" -eq 0 ] || train_args+=("${PASS[@]}")

RUN_PATH="logs/rsl_rl/$EXP/$RUN"
STATUS="logs/batch_status_${EXP}.log"
mkdir -p logs

say() { if [ "$DRY" -eq 1 ]; then echo "$*"; else echo "$(date -Is)  $*" | tee -a "$STATUS"; fi; }

latest_iter() {  # highest model_<N>.pt in the run folder, or -1 if none
  local n
  n=$(ls "$RUN_PATH"/model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
  echo "${n:--1}"
}

gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 || echo "n/a"; }
vram_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
host_avail_mib() { free -m | awk '/Mem:/ {print $7}'; }

# After a batch's train.py returns, make "all resources released" observable instead of assumed: wait until no
# train.py of THIS run is still alive in the container AND VRAM is back within VRAM_TOL_MIB of its level just before
# the batch started (a baseline-relative check, so it also works when another process shares the GPU). Inside a
# container nvidia-smi cannot map PIDs, hence these two signals. On timeout it warns and carries on (it must never
# wedge the run); a leftover process or non-reclaimed VRAM is then visible in the status log.
wait_released() {  # $1 = VRAM MiB before the batch, $2 = host MiB available before the batch
  local base="$1" host_before="$2" waited=0 alive used
  while :; do
    alive=$(pgrep -fc "src/training/train.py.*--run_dir $RUN" || true)
    used=$(vram_mib)
    if [ "${alive:-0}" -eq 0 ] && { [ -z "$base" ] || [ -z "$used" ] || [ "$used" -le $(( base + VRAM_TOL_MIB )) ]; }; then
      say "released after ${waited}s: no train.py left; VRAM ${used:-n/a} MiB (before batch ${base:-n/a}); host available $(host_avail_mib) MiB (before batch ${host_before})"
      return 0
    fi
    if [ "$waited" -ge "$RELEASE_TIMEOUT" ]; then
      say "WARNING: not fully released after ${RELEASE_TIMEOUT}s: train.py processes=${alive:-0}, VRAM ${used:-n/a} MiB (before batch ${base:-n/a}); continuing anyway"
      return 1
    fi
    sleep 5; waited=$(( waited + 5 ))
  done
}

# ---- live graphs ---------------------------------------------------------------------------------------------------
# live_dashboard.py writes DASH_OUT periodically (its --output mode) and feh --reload shows it; this is the same
# pair of commands as the tmux training pane (docker/isaaclab-shell.sh), pointed at this run's fixed folder. The
# dashboard is stopped and started again at the start of EVERY batch so it always follows the current run folder.
# It waits for the first events file, so it can be started before train.py has created the folder.
DASH_PID=""
VIEWER_PID=""
stop_dashboard() {
  if [ -n "$DASH_PID" ]; then kill -TERM -- "-$DASH_PID" 2>/dev/null; fi
  # also any dashboard already writing the same PNG (a hand-started one / the tmux pane's), so two writers never race
  pkill -f "live_dashboard.py.*--output $DASH_OUT" 2>/dev/null
  DASH_PID=""
}
start_viewer() {
  [ "$VIEWER" -eq 1 ] || return 0
  if [ -n "$VIEWER_PID" ] && kill -0 "$VIEWER_PID" 2>/dev/null; then return 0; fi
  command -v feh >/dev/null 2>&1 || { say "viewer: feh is not installed; open $DASH_OUT yourself"; return 0; }
  [ -n "${DISPLAY:-}" ] || { say "viewer: no DISPLAY; open $DASH_OUT yourself"; return 0; }
  if pgrep -f "feh .*$DASH_OUT" >/dev/null 2>&1; then return 0; fi   # already open (e.g. the tmux viewer pane)
  setsid bash -c 'for _ in $(seq 1 90); do [ -s "$1" ] && break; sleep 2; done; exec feh --scale-down --reload 0.5 "$1"' \
    _ "$DASH_OUT" >/dev/null 2>&1 &
  VIEWER_PID=$!
  say "viewer: feh opened on $DASH_OUT (refreshes itself)"
}
start_dashboard() {
  [ "$DASHBOARD" -eq 1 ] || return 0
  stop_dashboard
  setsid bash -c '
    for _ in $(seq 1 180); do ls "$1"/events.out.tfevents.* >/dev/null 2>&1 && break; sleep 5; done
    exec "$2" -p src/common/live_dashboard.py --logdir "$1" --output "$3"
  ' _ "$RUN_PATH" "$ISAACLAB_SH" "$DASH_OUT" >/dev/null 2>&1 &
  DASH_PID=$!
  say "dashboard (re)started: pid $DASH_PID, $RUN_PATH -> $DASH_OUT"
  start_viewer
}

STOP=0
# train.py runs in the FOREGROUND on purpose: a non-interactive shell starts `&` jobs with SIGINT ignored, so a
# backgrounded train.py would never see Ctrl-C. In the foreground the terminal's Ctrl-C reaches the whole process
# group (train.py exits gracefully) and bash runs this trap once it returns, which stops the loop.
on_stop() { STOP=1; say "stop requested; no further batches will start"; }
trap on_stop INT TERM
trap stop_dashboard EXIT   # the last PNG stays on screen in feh; only the refreshing process is stopped

say "batched training start: exp=$EXP run=$RUN batch=$BATCH total=$TARGET"
say "train.py args: ${train_args[*]}"
fails=0
while :; do
  n=$(latest_iter)
  # RSL-RL numbering: a run of N iterations starting at iteration `start` runs it=start..start+N-1 and its final
  # checkpoint is model_{start+N-1}; a resume starts again AT the loaded checkpoint's number. So with latest
  # checkpoint n, iterations completed ~ n+1, and a batch ending at boundary B (a multiple of BATCH) has final
  # checkpoint model_{B-1}.
  if [ $(( n + 1 )) -ge "$TARGET" ]; then say "reached target (latest checkpoint $n, target $TARGET); done"; break; fi
  [ "$STOP" -eq 1 ] && break
  start=$(( n < 0 ? 0 : n ))
  boundary=$(( ( (n + 1) / BATCH + 1 ) * BATCH ))
  [ "$boundary" -gt "$TARGET" ] && boundary=$TARGET
  run_iters=$(( boundary - start ))

  args=("${train_args[@]}" --run_dir "$RUN" --max_iterations "$run_iters")
  if [ "$n" -ge 0 ]; then args+=(--resume --load_run "$RUN"); fi

  if [ "$DRY" -eq 1 ]; then
    say "next batch: iteration $start -> $boundary ($run_iters iterations), latest checkpoint $n"
    say "train:     $ISAACLAB_SH -p src/training/train.py ${args[*]}"
    if [ "$DASHBOARD" -eq 1 ]; then
      say "dashboard: $ISAACLAB_SH -p src/common/live_dashboard.py --logdir $RUN_PATH --output $DASH_OUT"
      [ "$VIEWER" -eq 1 ] && say "viewer:    feh --scale-down --reload 0.5 $DASH_OUT"
    fi
    exit 0
  fi

  start_dashboard
  vram_before=$(vram_mib); host_before=$(host_avail_mib)
  say "batch: iteration $start -> $boundary ($run_iters iterations) (VRAM in use before: $(gpu_mem))"
  "$ISAACLAB_SH" -p src/training/train.py "${args[@]}"
  rc=$?
  after=$(latest_iter)
  say "batch exited rc=$rc, latest checkpoint $n -> $after"
  wait_released "$vram_before" "$host_before"
  [ "$STOP" -eq 1 ] && break

  if [ "$after" -le "$n" ]; then
    fails=$(( fails + 1 ))
    say "no new checkpoint (failure $fails/$MAX_FAILS)"
    [ "$fails" -ge "$MAX_FAILS" ] && { say "giving up"; exit 1; }
    sleep $(( 30 * fails ))
  else
    fails=0
  fi

  sleep 5  # short settle before the next launch; release was already confirmed by wait_released
done
say "batched training finished (latest checkpoint $(latest_iter))"
