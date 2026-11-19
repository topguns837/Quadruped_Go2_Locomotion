#!/usr/bin/env bash
# Run a phased training profile (training_profiles/<name>.yaml) by chaining scripts/train_batched.sh invocations,
# one per phase, in the same run folder (so curriculum state, terrain levels and the optimiser carry across phases
# via --resume). Thin wrapper around scripts/train_profile.py, which holds the schema, validation and command
# assembly (and is unit-tested in scripts/tests/test_train_profile.py).
#
# Run INSIDE the training container, from the repo root:
#   bash scripts/train_phased.sh --profile iter4_realistic              # real run (headless)
#   bash scripts/train_phased.sh --profile iter4_realistic --dry_run    # print the per-phase commands, launch none
#   bash scripts/train_phased.sh --profile aggressive --num_envs 1024   # override num_envs for all phases
#   bash scripts/train_phased.sh --profile iter4_realistic --start_phase 2   # resume at phase 2
#   bash scripts/train_phased.sh --profile example agent.algorithm.learning_rate=5e-4   # trailing Hydra override
#
# --profile takes a name (training_profiles/<name>.yaml) or a path. Any trailing key=value is a Hydra override
# applied to EVERY phase and beats the profile's own value. See `python scripts/train_profile.py --help`.
set -uo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/train_profile.py "$@"
