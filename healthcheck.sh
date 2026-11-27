#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Repo health check: starts a SEPARATE container (quadruped_go2_locomotion_testing, never the main one), runs
# src/healthcheck/run.py inside it, then removes it. See src/healthcheck/README.md.
#
#   ./healthcheck.sh                       full suite on this checkout
#   ./healthcheck.sh --ref <commit>        full suite on that commit (checked out into a throwaway git worktree)
#   ./healthcheck.sh --only training,sim2sim   only these groups / case ids (comma separated)
#   ./healthcheck.sh --list                list the cases and exit
#   ./healthcheck.sh --keep                keep the container and training outputs afterwards (debugging)
#
# Any other argument is passed through to run.py (e.g. --gpu_wait_min 30, --no_coverage).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE=quadruped-go2-testing
CONTAINER=quadruped_go2_locomotion_testing
INNER_REPO=/workspace/quadruped_go2_locomotion
INNER_PY=/workspace/isaaclab/_isaac_sim/python.sh

REF=""
KEEP=0
RUN_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --ref) REF="${2:?--ref needs a commit}"; shift 2 ;;
        --keep) KEEP=1; RUN_ARGS+=(--keep); shift ;;
        -h|--help) sed -n '8,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) RUN_ARGS+=("$1"); shift ;;
    esac
done

cd "${REPO_ROOT}" || exit 2
SRC_DIR="${REPO_ROOT}"
WORKTREE=""
if [ -n "${REF}" ]; then
    SHA="$(git rev-parse --short "${REF}")" || { echo "unknown ref: ${REF}" >&2; exit 2; }
    WORKTREE="/tmp/go2_healthcheck_${SHA}"
    rm -rf "${WORKTREE}"
    git worktree add --detach "${WORKTREE}" "${SHA}" >/dev/null || exit 2
    SRC_DIR="${WORKTREE}"
    echo "[healthcheck] testing commit ${SHA} in ${WORKTREE}"
fi
export HEALTHCHECK_REPO="${SRC_DIR}"

cleanup() {
    if [ "${KEEP}" -eq 0 ]; then
        docker compose --profile testing rm -sf "${SERVICE}" >/dev/null 2>&1
        [ -n "${WORKTREE}" ] && git worktree remove --force "${WORKTREE}" >/dev/null 2>&1
    else
        echo "[healthcheck] --keep: container ${CONTAINER} left running (docker rm -f ${CONTAINER} to remove)"
    fi
}
trap cleanup EXIT

# pre-commit lives in a host venv, not in the image: run it here and hand the result to the runner.
PC_RC=skip
PC_BIN="${HOME}/.venvs/pre-commit/bin/pre-commit"
[ -x "${PC_BIN}" ] || PC_BIN="$(command -v pre-commit || true)"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$(git -C "${SRC_DIR}" rev-parse --short HEAD)"
OUT_DIR="${SRC_DIR}/logs/healthcheck/${RUN_ID}"
PC_LOG=""
if [ -n "${PC_BIN}" ] && [ -z "${REF}" ]; then
    PC_FILE="$(mktemp)"
    (cd "${SRC_DIR}" && "${PC_BIN}" run --all-files) >"${PC_FILE}" 2>&1
    PC_RC=$?
    PC_LOG="$(tail -c 4000 "${PC_FILE}")"
    rm -f "${PC_FILE}"
fi

echo "[healthcheck] starting ${CONTAINER} (main container untouched)"
docker rm -f "${CONTAINER}" >/dev/null 2>&1
docker compose --profile testing up -d "${SERVICE}" || exit 2
waited=0
until docker exec "${CONTAINER}" test -f /tmp/.entrypoint_ready 2>/dev/null; do
    sleep 1; waited=$((waited + 1))
    [ "${waited}" -ge 180 ] && { echo "[healthcheck] container did not become ready" >&2; exit 2; }
done

docker exec -e HEALTHCHECK_PRECOMMIT_RC="${PC_RC}" -e HEALTHCHECK_PRECOMMIT_LOG="${PC_LOG}" \
    -e HEALTHCHECK_HOST_UID="$(id -u)" -e HEALTHCHECK_HOST_GID="$(id -g)" -e HEALTHCHECK_RUN_ID="${RUN_ID}" \
    -e HEALTHCHECK_SHA="$(git -C "${SRC_DIR}" rev-parse --short HEAD)" -w "${INNER_REPO}" \
    "${CONTAINER}" "${INNER_PY}" src/healthcheck/run.py "${RUN_ARGS[@]}"
RC=$?
echo "[healthcheck] report: ${OUT_DIR}/report.md (exit ${RC})"
exit "${RC}"
