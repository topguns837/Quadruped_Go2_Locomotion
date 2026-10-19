# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delete old training-run folders under logs/rsl_rl/<experiment_name>/ to reclaim disk space.

Pure Python, no Isaac Sim needed -- run with plain `python3`, not the `isaaclab -p` launcher.

Each run folder (`logs/rsl_rl/<experiment_name>/<timestamp>/`) holds its own checkpoints, TensorBoard events, and
`step_diagnostics.log`; these accumulate indefinitely across training sessions (measured: 24 folders, 5.4 GB, under
the project's default experiment directory). This script never touches `models/` (the curated, committed
checkpoints) or anything in git -- `logs/` is gitignored.

Safe by default: lists candidates and exits WITHOUT deleting anything unless `--yes` is also passed. Always keeps
the `--keep` most-recent folders (by name, which is a sortable timestamp) regardless of size, so the run you just
finished or are about to resume is never a candidate.

Usage:
    python3 scripts/cleanup_old_runs.py                        # dry run, default logdir, keep 5 newest
    python3 scripts/cleanup_old_runs.py --keep 3                # dry run, keep only the 3 newest
    python3 scripts/cleanup_old_runs.py --keep 3 --yes           # actually delete the rest
    python3 scripts/cleanup_old_runs.py --logdir logs/rsl_rl/<other_experiment>
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOGDIR = "logs/rsl_rl/go2_with_pitch_lean_and_height_control_ppo"


@dataclass
class RunFolder:
    path: Path
    size_bytes: int
    latest_checkpoint: str | None


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _latest_checkpoint(path: Path) -> str | None:
    checkpoints = sorted(
        path.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1
    )
    return checkpoints[-1].name if checkpoints else None


def _human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


def list_run_folders(logdir: Path) -> list[RunFolder]:
    """Sorted oldest-first by folder name (the timestamp format sorts chronologically)."""
    if not logdir.is_dir():
        return []
    folders = sorted(p for p in logdir.iterdir() if p.is_dir())
    return [RunFolder(p, _dir_size(p), _latest_checkpoint(p)) for p in folders]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR, help="Experiment directory to clean up.")
    parser.add_argument(
        "--keep", type=int, default=5, help="Always keep this many of the newest run folders (default: 5)."
    )
    parser.add_argument(
        "--yes", action="store_true", default=False,
        help="Actually delete the candidates. Without this flag, nothing is deleted (dry run).",
    )
    args = parser.parse_args()

    if args.keep < 1:
        raise ValueError("--keep must be at least 1 -- refusing to risk deleting every run, including the latest.")

    logdir = Path(args.logdir)
    runs = list_run_folders(logdir)
    if not runs:
        print(f"[CLEANUP] No run folders found under {logdir}.")
        return

    keep_count = min(args.keep, len(runs))
    to_keep, to_delete = runs[-keep_count:] if keep_count else [], runs[: len(runs) - keep_count]

    print(f"[CLEANUP] {logdir}")
    print(f"[CLEANUP] {len(runs)} run folder(s), {_human(sum(r.size_bytes for r in runs))} total.\n")

    print(f"KEEP (newest {len(to_keep)}, never a candidate):")
    for r in to_keep:
        print(f"  {r.path.name:24s} {_human(r.size_bytes):>10s}  latest={r.latest_checkpoint or '(none)'}")

    print(f"\n{'DELETE' if args.yes else 'WOULD DELETE'} ({len(to_delete)}):")
    if not to_delete:
        print("  (none -- nothing older than the kept set)")
    for r in to_delete:
        print(f"  {r.path.name:24s} {_human(r.size_bytes):>10s}  latest={r.latest_checkpoint or '(none)'}")

    freed = sum(r.size_bytes for r in to_delete)
    print(f"\n[CLEANUP] {'Freeing' if args.yes else 'Would free'} {_human(freed)} from {len(to_delete)} folder(s).")

    if not args.yes:
        print("[CLEANUP] Dry run only -- nothing deleted. Re-run with --yes to actually delete the folders above.")
        return

    for r in to_delete:
        shutil.rmtree(r.path)
    print(f"[CLEANUP] Deleted {len(to_delete)} folder(s), freed {_human(freed)}.")


if __name__ == "__main__":
    main()
