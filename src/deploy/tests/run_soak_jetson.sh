#!/usr/bin/env bash
# Jetson side of the instrumented WiFi soak. Starts the FAKE ROBOT and the RELAY, both on the loopback interface
# ONLY, so nothing here can reach the real robot or eth0; waits for the desktop run to finish; stops everything
# cleanly. Logs land in ~/relay_logs/<RUN_ID>/{relay,fake_robot}/ plus jetson_env.txt.
#
# Usage (a normal Jetson shell; the venv and library path are set up here):
#     bash ~/src/deploy/tests/run_soak_jetson.sh <RUN_ID> [max_minutes]
#
# LOG_LEVEL=light bash ... drops the per-packet files (to test whether logging itself perturbs timing).
# Then start the desktop run (see src/deploy/tests/run_soak_desktop.sh). Press Ctrl+C here to abort safely.
set -u

RUN="${1:?usage: run_soak_jetson.sh <RUN_ID> [max_minutes]}"
MAXMIN="${2:-25}"
case "$RUN" in *[!A-Za-z0-9._-]*|"") echo "RUN_ID may only contain letters, digits, . _ -" >&2; exit 2;; esac
# Hard guard: this script has no interface option and nothing on its command line may name a real interface.
for a in "$@"; do case "$a" in *eth0*|*wlan0*|*enp*) echo "refusing: this test only ever runs on loopback" >&2; exit 2;; esac; done

cd "$(dirname "$0")/../../.."          # the repo root
DDSC="$HOME/cyclonedds_ws/install/cyclonedds/lib"
export LD_LIBRARY_PATH="$DDSC"      # only the library the bindings were built against (see relay_jetson.py)
[ -f "$HOME/relay_env/bin/activate" ] && . "$HOME/relay_env/bin/activate"
PY="${PY:-python}"
KEY="${KEY:-$HOME/.go2_relay_key}"
OUT="$HOME/relay_logs/$RUN"

[ -f "$KEY" ] || { echo "no key at $KEY (create it: python src/deploy/relay_jetson.py --init_key $KEY)" >&2; exit 2; }
[ -d "$OUT" ] && { echo "$OUT already exists; use a new RUN_ID" >&2; exit 2; }
avail_mb=$(df -Pm "$HOME" | awk 'NR==2 {print $4}')
[ "${avail_mb:-0}" -ge 300 ] || { echo "only ${avail_mb} MB free in $HOME; need 300 MB" >&2; exit 2; }
mkdir -p "$OUT/relay" "$OUT/fake_robot"

{
  echo "== date";       date
  echo "== uname";      uname -a
  echo "== uptime";     uptime
  echo "== nproc";      nproc
  echo "== memory";     free -m
  echo "== nvpmodel";   (nvpmodel -q 2>&1 || true)
  echo "== governors";  (cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c)
  echo "== cur freq kHz"; (cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | tr '\n' ' '); echo
  echo "== iwconfig wlan0"; (iwconfig wlan0 2>&1 || true)
  echo "== wireless"; (cat /proc/net/wireless 2>&1 || true)
  echo "== top 8 by cpu"; (ps -eo pid,pcpu,pmem,comm --sort=-pcpu | head -9)
  echo "== file sha256"; (sha256sum deploy/*.py src/deploy/configs/*.yaml 2>/dev/null)
} > "$OUT/jetson_env.txt" 2>&1

FAKE=""; RELAY=""
cleanup() {
  trap - INT TERM
  [ -n "$RELAY" ] && kill -INT "$RELAY" 2>/dev/null
  sleep 2
  [ -n "$FAKE" ] && kill -INT "$FAKE" 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10; do
    { [ -n "$FAKE" ] && kill -0 "$FAKE" 2>/dev/null; } || { [ -n "$RELAY" ] && kill -0 "$RELAY" 2>/dev/null; } || break
    sleep 1
  done
  [ -n "$FAKE" ] && kill -9 "$FAKE" 2>/dev/null
  [ -n "$RELAY" ] && kill -9 "$RELAY" 2>/dev/null
}
trap 'echo; echo "aborting: stopping the relay and the fake robot"; cleanup; exit 130' INT TERM

echo "[soak] run $RUN, logs in $OUT, loopback only, libddsc dir $DDSC"
PYTHONUNBUFFERED=1 $PY src/deploy/fake_robot.py --network_interface lo --responsive --rate_hz 200 --print_every 500 \
    --log_dir "$OUT/fake_robot" --run_id "$RUN" --duration_s $((MAXMIN * 60 + 120)) > "$OUT/fake_robot.stdout" 2>&1 &
FAKE=$!
sleep 6
kill -0 "$FAKE" 2>/dev/null || { echo "the fake robot died at start-up:"; tail -15 "$OUT/fake_robot.stdout"; cleanup; exit 1; }

PYTHONUNBUFFERED=1 $PY src/deploy/relay_jetson.py --network_interface lo --key_file "$KEY" --require_ddsc_dir "$DDSC" \
    --exit_after_session --log_level "${LOG_LEVEL:-full}" --log_dir "$OUT/relay" --run_id "$RUN" --wifi_iface wlan0 > "$OUT/relay.stdout" 2>&1 &
RELAY=$!
sleep 6
kill -0 "$RELAY" 2>/dev/null || { echo "the relay died at start-up:"; tail -15 "$OUT/relay.stdout"; cleanup; exit 1; }

echo "[soak] READY. Start the desktop run now (it takes about 15 minutes). Waiting up to $MAXMIN minutes..."
deadline=$((SECONDS + MAXMIN * 60))
while kill -0 "$RELAY" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
  sleep 5
  tail -n 1 "$OUT/relay.stdout" | cut -c1-150
done
if kill -0 "$RELAY" 2>/dev/null; then
  echo "[soak] the time limit passed before the session ended; stopping."
fi
cleanup
echo "[soak] finished. Files:"
ls -l "$OUT" "$OUT/relay" "$OUT/fake_robot" | sed 's/^/  /'
echo "[soak] fetch from the desktop with:"
echo "  scp -r unitree@192.168.152.215:relay_logs/$RUN logs/relay_runs/$RUN/jetson"
