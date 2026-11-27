#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Health-check helper: run ONE deploy_real.py session against the synthetic robot (src/deploy/fake_robot.py) on the
# loopback interface, then stop the fake robot. Both run in the throwaway testing container's own network namespace.
#
#   fake_robot_case.sh <out_dir> "<fake_robot.py extra args>" [deploy_real.py extra args ...]
#   HC_TOOL=src/deploy/preflight_check.py fake_robot_case.sh <out_dir> "<fake args>" [tool args ...]   (another client)
#   HC_WRAP="<words>"  put these words in front of the client command (e.g. pty_run.py ... -- to type into it)
#
# Leaves <out_dir>/{fake.log,deploy.log,fake/} behind; exits with deploy_real.py's exit code (124 = it hung and was
# killed by the 120 s guard). The fake robot runs in its own process group and that group is killed at the end, so
# nothing is left publishing on the interface (killing by name is avoided: it would also match this shell).
set -u
OUT="$1"; FAKE_ARGS="$2"; shift 2
PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
mkdir -p "$OUT"

setsid "$PY" src/deploy/fake_robot.py --network_interface lo --responsive --rate_hz 200 \
    --log_dir "$OUT/fake" $FAKE_ARGS >"$OUT/fake.log" 2>&1 &
FAKE_PGID=$!
cleanup() { kill -TERM -- "-$FAKE_PGID" 2>/dev/null; sleep 1; kill -KILL -- "-$FAKE_PGID" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT

for _ in $(seq 1 30); do grep -q "FAKE_ROBOT" "$OUT/fake.log" 2>/dev/null && break; sleep 0.5; done
sleep 3
if [ -n "${HC_TOOL:-}" ]; then
    timeout 120 "$PY" "$HC_TOOL" --network_interface lo "$@" >"$OUT/deploy.log" 2>&1
else
    WRAP=(); [ -n "${HC_WRAP:-}" ] && eval "WRAP=($HC_WRAP)"
    EXTRA=(--no_pause_for_hoist); [ -n "${HC_KEEP_HOIST_PAUSE:-}" ] && EXTRA=()
    timeout 120 "${WRAP[@]}" "$PY" src/deploy/deploy_real.py --network_interface lo --skip_release "${EXTRA[@]}" \
        --startup_ramp_s 2 --no_live_dashboard "$@" >"$OUT/deploy.log" 2>&1
fi
RC=$?
tail -n 400 "$OUT/deploy.log"
exit "$RC"
