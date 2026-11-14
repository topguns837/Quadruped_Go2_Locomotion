#!/usr/bin/env bash
# Random-command sim2sim evaluation: drive ONE checkpoint with the same seeded random command schedules in Isaac Sim and
# in MuJoCo, then print a side-by-side report (sim2sim/compare_random.py).
#
# Run INSIDE the Isaac Lab container, from the repo root (/workspace/quadruped_go2_locomotion). No GPU training run may
# be using the card at the same time as the Isaac step.
#
#   bash sim2sim/run_random_eval.sh --checkpoint logs/rsl_rl/go2_kp40_iter4_realistic/run_iter4_realistic/model_24999.pt
#   bash sim2sim/run_random_eval.sh --checkpoint <model_N.pt> --seeds "0 1 2 3 4" --segments 30 --segment_s 4
#   bash sim2sim/run_random_eval.sh --checkpoint <model_N.pt> --scale 0.5          # half the training command range
#   bash sim2sim/run_random_eval.sh --checkpoint <model_N.pt> --skip_isaac         # reuse existing Isaac traces
#
# For each seed S it writes into --out_dir:   seedS.yaml  seedS_isaac.npz  seedS_mujoco.npz   and finally report.md.
# Isaac: `play.py --scenario` (nominal physics, no disturbances, only falls reset). MuJoCo: `sim2sim_mujoco.py
# --scenario` with the policy that play.py exported from the same checkpoint. Run with --help for every option.
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=""; SEEDS="0 1 2"; SEGMENTS=20; SEGMENT_S=4; SCALE=1.0; OUT=""; POLICY=""; SETTLE=1.0
GUI=0; SKIP_ISAAC=0; SKIP_MUJOCO=0
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}"
PYTHON_SH="${PYTHON_SH:-/workspace/isaaclab/_isaac_sim/python.sh}"

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "run_random_eval.sh: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CKPT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --segments) SEGMENTS="$2"; shift 2 ;;
    --segment_s) SEGMENT_S="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --out_dir) OUT="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;
    --settle_s) SETTLE="$2"; shift 2 ;;
    --gui) GUI=1; shift ;;              # show the Isaac window instead of running headless
    --skip_isaac) SKIP_ISAAC=1; shift ;;
    --skip_mujoco) SKIP_MUJOCO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

[ -n "$CKPT" ] || die "--checkpoint is required"
[ -f "$CKPT" ] || die "checkpoint not found: $CKPT"
[ -s "$CKPT" ] || die "checkpoint is empty (a truncated save?): $CKPT"
[ -n "$OUT" ] || OUT="logs/sim2sim_eval_$(basename "$CKPT" .pt | sed 's/^model_//')/random"
[ -n "$POLICY" ] || POLICY="$(dirname "$CKPT")/exported/policy.pt"
mkdir -p "$OUT"

echo "[eval] checkpoint=$CKPT  seeds=[$SEEDS]  segments=$SEGMENTS x ${SEGMENT_S}s  scale=$SCALE  out=$OUT"
for S in $SEEDS; do
  SCEN="$OUT/seed$S.yaml"
  "$PYTHON_SH" sim2sim/random_commands.py --seed "$S" --segments "$SEGMENTS" --segment_s "$SEGMENT_S" \
    --scale "$SCALE" --out "$SCEN" || die "could not generate the schedule for seed $S"

  if [ "$SKIP_ISAAC" -eq 0 ]; then
    echo "[eval] seed $S: Isaac Sim"
    HL="--headless"; [ "$GUI" -eq 1 ] && HL=""
    "$ISAACLAB_SH" -p scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1 $HL \
      --checkpoint "$CKPT" --scenario "$SCEN" --trace_out "$OUT/seed${S}_isaac.npz" --trace_deterministic_reset \
      > "$OUT/seed${S}_isaac.log" 2>&1 || die "Isaac run failed for seed $S (see $OUT/seed${S}_isaac.log)"
  fi

  if [ "$SKIP_MUJOCO" -eq 0 ]; then
    [ -f "$POLICY" ] || die "exported policy not found: $POLICY (run the Isaac step first, or pass --policy)"
    echo "[eval] seed $S: MuJoCo"
    "$PYTHON_SH" sim2sim/sim2sim_mujoco.py --headless --policy "$POLICY" --scenario "$SCEN" \
      --trace_out "$OUT/seed${S}_mujoco.npz" > "$OUT/seed${S}_mujoco.log" 2>&1 \
      || die "MuJoCo run failed for seed $S (see $OUT/seed${S}_mujoco.log)"
  fi
done

"$PYTHON_SH" sim2sim/compare_random.py --dir "$OUT" --settle_s "$SETTLE" --out "$OUT/report.md" \
  --title "Sim2sim random-command evaluation: $(basename "$CKPT")"
