#!/usr/bin/env bash
# The whole soak stack on ONE machine's loopback (fake robot + relay + desktop policy loop), with a compressed
# impairment schedule. Used to verify the logging, the impairments and the analyzer before a real run; also gives a
# no-WiFi software baseline. Run inside the project container (it has the SDK, torch and numpy), isolated:
#   docker run --rm --network none --entrypoint bash -v "$PWD:/repo" quadruped_go2_locomotion:latest \
#       -lc 'bash /repo/deploy/tests/run_soak_local.sh <RUN_ID> [time_scale]'
# time_scale 0.1 turns the 900 s schedule into 90 s. Output: <repo>/logs/relay_runs/<RUN_ID>/{desktop,jetson/...}
set -u
RUN="${1:?usage: run_soak_local.sh <RUN_ID> [time_scale]}"
TS="${2:-0.1}"
cd "$(dirname "$0")/../../.."
PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
export PYTHONUNBUFFERED=1
OUT="logs/relay_runs/$RUN"
KEY="/tmp/soak_local.key"
rm -f "$KEY"; $PY src/deploy/relay_jetson.py --init_key "$KEY" > /dev/null 2>&1 || { echo "no key"; exit 2; }
mkdir -p "$OUT/desktop" "$OUT/jetson/relay" "$OUT/jetson/fake_robot"

$PY src/deploy/fake_robot.py --network_interface lo --responsive --rate_hz 200 --print_every 500 \
    --log_dir "$OUT/jetson/fake_robot" --run_id "$RUN" --duration_s 3000 > "$OUT/jetson/fake_robot.stdout" 2>&1 &
FAKE=$!
sleep 7
$PY src/deploy/relay_jetson.py --network_interface lo --key_file "$KEY" --exit_after_session \
    --log_dir "$OUT/jetson/relay" --run_id "$RUN" --wifi_iface lo > "$OUT/jetson/relay.stdout" 2>&1 &
RELAY=$!
sleep 7
$PY src/deploy/deploy_real_wireless.py --relay_host 127.0.0.1 --relay_key_file "$KEY" --skip_release --no_pause_for_hoist \
    --startup_ramp_s 3 --impair_schedule soak15 --impair_time_scale "$TS" --impair_seed 1 \
    --log_dir "$OUT/desktop" --run_id "$RUN" --log_nic lo > "$OUT/desktop.stdout" 2>&1
echo "desktop exit code: $?"
for i in $(seq 1 15); do kill -0 "$RELAY" 2>/dev/null || break; sleep 1; done
# $! is the python.sh wrapper, which does not forward signals: signal the python processes by name.
pkill -INT -f "src/deploy/relay_jetson.py" 2>/dev/null; sleep 2
pkill -INT -f "src/deploy/fake_robot.py" 2>/dev/null
for i in $(seq 1 10); do pgrep -f "src/deploy/fake_robot.py" >/dev/null || break; sleep 1; done
pkill -9 -f "src/deploy/fake_robot.py" 2>/dev/null; pkill -9 -f "src/deploy/relay_jetson.py" 2>/dev/null
echo "logs in $OUT"
