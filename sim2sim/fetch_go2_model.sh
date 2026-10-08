#!/usr/bin/env bash
# Fetches the Unitree Go2 model from MuJoCo Menagerie into sim2sim/mujoco/menagerie/ (gitignored).
# Pinned to a commit so the sim2sim model does not change underneath recorded results.
set -euo pipefail

MENAGERIE_COMMIT="0059d4335f8156206f63a35662313385f7ad6d74"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mujoco/menagerie"

if [ -f "$DEST/unitree_go2/go2.xml" ]; then
  echo "[INFO] $DEST/unitree_go2 already present, nothing to do."
  exit 0
fi

rm -rf "$DEST"
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
git -C "$DEST" sparse-checkout set unitree_go2
git -C "$DEST" checkout "$MENAGERIE_COMMIT"
echo "[INFO] Fetched Menagerie unitree_go2 at $MENAGERIE_COMMIT into $DEST"
