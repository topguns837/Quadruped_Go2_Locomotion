#!/usr/bin/env bash
# Run a phased training profile (src/training/profiles/<name>.yaml) by chaining src/training/train_batched.sh invocations,
# one per phase, in the same run folder (so curriculum state, terrain levels and the optimiser carry across phases
# via --resume). Thin wrapper around src/training/train_profile.py, which holds the schema, validation and command
# assembly (and is unit-tested in src/training/tests/test_train_profile.py).
#
# Run INSIDE the training container, from the repo root:
#   bash src/training/train.sh --profile iter4_realistic              # real run (headless)
#   bash src/training/train.sh --profile iter4_realistic --dry_run    # print the per-phase commands, launch none
#   bash src/training/train.sh --profile aggressive --num_envs 1024   # override num_envs for all phases
#   bash src/training/train.sh --profile iter4_realistic --start_phase 2   # resume at phase 2
#   bash src/training/train.sh --profile example agent.algorithm.learning_rate=5e-4   # trailing Hydra override
#
# --profile takes a name (src/training/profiles/<name>.yaml) or a path. Any trailing key=value is a Hydra override
# applied to EVERY phase and beats the profile's own value. See `python src/training/train_profile.py --help`.
set -uo pipefail
cd "$(dirname "$0")/../.."

# train_profile.py needs PyYAML. A host python3 usually has it, but the training container's /usr/bin/python3 does
# NOT -- PyYAML ships only in Isaac Sim's bundled Python. Pick python3 when it can import yaml, else fall back to the
# Isaac Python next to ISAACLAB_SH (same env/convention as src/training/train_batched.sh). Override with TRAIN_PROFILE_PYTHON.
PYTHON="${TRAIN_PROFILE_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if python3 -c 'import yaml' >/dev/null 2>&1; then
        PYTHON=python3
    else
        ISAAC_PY="$(dirname "${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}")/_isaac_sim/python.sh"
        [ -x "$ISAAC_PY" ] && PYTHON="$ISAAC_PY" || PYTHON=python3
    fi
fi
exec "$PYTHON" src/training/train_profile.py "$@"
