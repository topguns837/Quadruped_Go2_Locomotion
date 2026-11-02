#!/usr/bin/env bash
# Host-side RAM guard for the training container. Samples memory via `docker stats`, logs every sample, and when
# usage crosses THRESHOLD% of the container's memory limit sends SIGINT to the train.py process so training stops
# gracefully (resume later from the last checkpoint) instead of being hard OOM-killed. Keeps the desktop/host
# responsive. Read-only except for that one stop action, which targets only the train.py process.
#
# Run on the HOST (not inside the container) before/at the training launch, e.g.:
#   bash scripts/train_mem_guard.sh
#   THRESHOLD=85 INTERVAL=15 bash scripts/train_mem_guard.sh
# The memtrace_*.log it writes is also the memory-vs-time trace for diagnosing the leak (a climb while training
# runs points at the sim/rollout path; flat-while-training points at a side process).
set -uo pipefail

CONTAINER="${CONTAINER:-quadruped_go2_locomotion}"
THRESHOLD="${THRESHOLD:-90}"                     # percent of the container memory limit (48 GiB)
INTERVAL="${INTERVAL:-30}"                        # seconds between samples
PATTERN="${PATTERN:-scripts/rsl_rl/train.py}"    # how to find the training process inside the container
LOG="${LOG:-logs/memtrace_$(date +%F_%H-%M-%S).log}"

mkdir -p "$(dirname "$LOG")"
echo "# mem_guard start $(date -Is) container=$CONTAINER threshold=${THRESHOLD}% interval=${INTERVAL}s" | tee -a "$LOG"
stopped=0
while :; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
    echo "$(date -Is)  container not running; exiting" | tee -a "$LOG"; exit 0
  fi
  line=$(docker stats --no-stream --format '{{.MemPerc}};{{.MemUsage}}' "$CONTAINER" 2>/dev/null)
  perc=${line%%;*}; perc=${perc%\%}; usage=${line#*;}
  echo "$(date -Is)  mem=${usage:-?}  perc=${perc:-?}%" | tee -a "$LOG"
  pct_int=${perc%.*}
  if [ "$stopped" -eq 0 ] && [ -n "${pct_int:-}" ] && [ "$pct_int" -ge "$THRESHOLD" ] 2>/dev/null; then
    echo "$(date -Is)  THRESHOLD ${THRESHOLD}% crossed (${perc}%) -- SIGINT '$PATTERN' in $CONTAINER" | tee -a "$LOG"
    docker exec "$CONTAINER" pkill -INT -f "$PATTERN" || echo "$(date -Is)  no match for '$PATTERN'" | tee -a "$LOG"
    stopped=1
  fi
  if [ "$stopped" -eq 1 ] && ! docker exec "$CONTAINER" pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "$(date -Is)  training stopped; resume from the last logs/rsl_rl/<exp>/<run>/model_*.pt with --resume. exiting" | tee -a "$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
