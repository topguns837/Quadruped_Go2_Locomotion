#!/usr/bin/env bash
# Desktop side of the instrumented WiFi soak. Run it INSIDE the project container, from the repo root, after the
# Jetson script says READY:
#     bash src/deploy/tests/run_soak_desktop.sh <RUN_ID> [relay_host]
#
# It runs the desktop policy loop against the Jetson relay with the 15-minute impairment schedule 'soak15'
# (test only; the Jetson relay is on loopback with the fake robot, so no motor can move) and ends by itself.
# Logs land in logs/relay_runs/<RUN_ID>/desktop/. In a SECOND terminal on the HOST, start the independent ICMP log
# right before this (it ends by itself after about 15 minutes):
#     mkdir -p logs/relay_runs/<RUN_ID>
#     ping -D -i 0.2 -w 930 -I enp2s0 192.168.50.20 > logs/relay_runs/<RUN_ID>/icmp_ping.log
set -u

RUN="${1:?usage: run_soak_desktop.sh <RUN_ID> [relay_host]}"
HOST="${2:-192.168.50.20}"   # the Jetson dongle on the modem/AP subnet
case "$RUN" in *[!A-Za-z0-9._-]*|"") echo "RUN_ID may only contain letters, digits, . _ -" >&2; exit 2;; esac

cd "$(dirname "$0")/../../.."
PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
OUT="logs/relay_runs/$RUN"
KEY="${KEY:-deploy/relay.key}"
[ -f "$KEY" ] || { echo "no key at $KEY" >&2; exit 2; }
[ -d "$OUT/desktop" ] && { echo "$OUT/desktop already exists; use a new RUN_ID" >&2; exit 2; }
mkdir -p "$OUT/desktop"

echo "[soak] run $RUN -> relay $HOST, logs in $OUT/desktop (about 15 minutes, ends by itself)"
# Optional env: DURATION=180 (seconds, default = full schedule), SCHEDULE=none (clean link only), LOG_LEVEL=light.
EXTRA=""
[ "${SCHEDULE:-soak15}" = none ] && EXTRA="--duration_s ${DURATION:-180}" || EXTRA="--impair_schedule ${SCHEDULE:-soak15} --impair_seed 1 ${DURATION:+--duration_s $DURATION}"
$PY src/deploy/deploy_real_wireless.py \
    --relay_host "$HOST" --relay_key_file "$KEY" \
    --skip_release --no_pause_for_hoist --startup_ramp_s 3 $EXTRA \
    --log_level "${LOG_LEVEL:-full}" --log_dir "$OUT/desktop" --run_id "$RUN" --log_nic "${LOG_NIC:-enp2s0}"
rc=$?
echo "[soak] desktop run finished with exit code $rc (0 = completed or Ctrl+C, 3 = it stopped early: see events.log)"
echo "[soak] when the Jetson script has finished too, fetch its logs and analyse:"
echo "  scp -r unitree@192.168.50.20:relay_logs/$RUN $OUT/jetson"
echo "  $PY src/deploy/tools/analyze_soak.py $OUT"
exit $rc
