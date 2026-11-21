# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative, phased training profiles for the Go2 locomotion task.

A training profile is a YAML file (``training_profiles/<name>.yaml``) describing an ordered list of training
*phases*. Each phase runs for a number of iterations, optionally in batches, with its own set of config overrides
expressed as Hydra dotted paths (``env.events.*``, ``env.rewards.*``, ``env.curriculum.difficulty.params.*``,
``agent.*``). Phases chain by *resume*: phase k continues from phase k-1's final checkpoint in the same run folder,
so the curriculum, terrain levels and optimiser state carry over (exactly how the iter4 clean -> realistic re-ramp
was run by hand).

This module is PURE Python (no torch / Isaac Sim imports) so it can be unit-tested on any interpreter. It builds
``scripts/train_batched.sh`` command lines and, as a CLI (``python scripts/train_profile.py --profile <name>``),
runs them in sequence -- reusing all of train_batched.sh's batched-resume, curriculum save/restore and
memory-release machinery.

Scope (by design): overrides may only target Hydra-reachable fields on ``env.*`` / ``agent.*``. Module-level
constants baked into the actuator/observation config at class-definition time -- kp, kd, JOINT_ARMATURE,
PHYSX_JOINT_SPEED_LIMIT and the observation history length -- are NOT per-phase here; a profile that tries to set
one fails fast (see ``_BAKED_IN_SEGMENTS``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import yaml

# ----------------------------------------------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------------------------------------------

DEFAULT_TASK = "Quadruped-Locomotion-Go2"
TRAIN_BATCHED = "scripts/train_batched.sh"

_PHASE_KEYS = {"name", "iterations", "batched", "batch_iterations", "num_envs", "seed", "overrides"}
_TOPLEVEL_KEYS = {
    "experiment_name", "run_dir", "task", "num_envs", "seed", "dashboard", "viewer", "dashboard_output",
    "overrides", "phases",
}
DEFAULT_DASHBOARD_OUTPUT = "/tmp/dashboard.png"

# Override heads that Hydra can reach on this task's configs.
_ALLOWED_HEADS = {"env", "agent"}

# Path segments that identify a value BAKED INTO the Python config at class-definition time (not Hydra-reachable,
# or changing it silently desyncs training from deploy). A profile may not target any of these. Note "armature"
# is the ACTUATOR armature; the domain-randomisation range lives under env.events.joint_armature.params.
# armature_distribution_params (segments "joint_armature" / "armature_distribution_params"), which is allowed.
_BAKED_IN_SEGMENTS = {
    "actuators",
    "stiffness",
    "damping",
    "armature",
    "effort_limit",
    "kp",
    "kd",
    "history_length",
    "flatten_history_dim",
}


class ProfileError(ValueError):
    """A training profile is malformed or targets an out-of-scope field."""


@dataclass
class Phase:
    name: str
    iterations: int
    batched: bool = False
    batch_iterations: int | None = None
    num_envs: int | None = None
    seed: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingProfile:
    experiment_name: str
    run_dir: str
    phases: list[Phase]
    task: str = DEFAULT_TASK
    num_envs: int | None = None
    seed: int | None = None
    # Live graphs while training: train_batched.sh (re)starts scripts/live_dashboard.py -- which writes a PNG -- at
    # the start of every batch, and opens feh on it. No separate tmux panes needed.
    dashboard: bool = True  # run the live dashboard
    viewer: bool = True  # open feh on the dashboard PNG (only meaningful when dashboard is on)
    dashboard_output: str = DEFAULT_DASHBOARD_OUTPUT  # PNG the dashboard writes / feh shows
    overrides: dict[str, Any] = field(default_factory=dict)  # applied to every phase (phase wins on conflict)


# ----------------------------------------------------------------------------------------------------------------
# Loading & validation
# ----------------------------------------------------------------------------------------------------------------

def load_profile(path: str) -> TrainingProfile:
    """Parse and validate a training-profile YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ProfileError(f"{path}: top level must be a mapping")
    unknown = set(raw) - _TOPLEVEL_KEYS
    if unknown:
        raise ProfileError(f"{path}: unknown top-level key(s) {sorted(unknown)}; allowed {sorted(_TOPLEVEL_KEYS)}")
    for req in ("experiment_name", "run_dir"):
        if not raw.get(req):
            raise ProfileError(f"{path}: '{req}' is required")

    raw_phases = raw.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ProfileError(f"{path}: 'phases' must be a non-empty list")

    phases: list[Phase] = []
    seen_names: set[str] = set()
    for i, rp in enumerate(raw_phases):
        phases.append(_parse_phase(path, i, rp, seen_names))

    profile = TrainingProfile(
        experiment_name=str(raw["experiment_name"]),
        run_dir=str(raw["run_dir"]),
        task=str(raw.get("task", DEFAULT_TASK)),
        num_envs=_opt_int(path, "num_envs", raw.get("num_envs")),
        seed=_opt_int(path, "seed", raw.get("seed")),
        dashboard=_opt_bool(path, "dashboard", raw.get("dashboard"), True),
        viewer=_opt_bool(path, "viewer", raw.get("viewer"), True),
        dashboard_output=_opt_str(path, "dashboard_output", raw.get("dashboard_output"), DEFAULT_DASHBOARD_OUTPUT),
        overrides=raw.get("overrides") or {},
        phases=phases,
    )
    # validate every override path (global + per-phase)
    for src, ov in [("<global>", profile.overrides)] + [(p.name, p.overrides) for p in profile.phases]:
        for dotted in _flatten_map(ov):
            _check_override_path(path, src, dotted)
    return profile


def _parse_phase(path: str, idx: int, rp: Any, seen_names: set[str]) -> Phase:
    if not isinstance(rp, dict):
        raise ProfileError(f"{path}: phase #{idx} must be a mapping")
    unknown = set(rp) - _PHASE_KEYS
    if unknown:
        raise ProfileError(f"{path}: phase #{idx} has unknown key(s) {sorted(unknown)}; allowed {sorted(_PHASE_KEYS)}")
    name = rp.get("name")
    if not name or not isinstance(name, str):
        raise ProfileError(f"{path}: phase #{idx} needs a string 'name'")
    if name in seen_names:
        raise ProfileError(f"{path}: duplicate phase name '{name}'")
    seen_names.add(name)

    iterations = rp.get("iterations")
    if not _is_int(iterations) or iterations <= 0:
        raise ProfileError(f"{path}: phase '{name}' needs a positive integer 'iterations' (got {iterations!r})")

    batched = bool(rp.get("batched", False))
    batch_iterations = rp.get("batch_iterations")
    if batched:
        if not _is_int(batch_iterations) or batch_iterations <= 0:
            raise ProfileError(
                f"{path}: phase '{name}' is batched but 'batch_iterations' is missing/invalid ({batch_iterations!r})"
            )
        if batch_iterations > iterations:
            raise ProfileError(
                f"{path}: phase '{name}' batch_iterations ({batch_iterations}) exceeds its iterations ({iterations})"
            )
    return Phase(
        name=name,
        iterations=int(iterations),
        batched=batched,
        batch_iterations=int(batch_iterations) if batched else None,
        num_envs=_opt_int(path, f"phase '{name}'.num_envs", rp.get("num_envs")),
        seed=_opt_int(path, f"phase '{name}'.seed", rp.get("seed")),
        overrides=rp.get("overrides") or {},
    )


def _check_override_path(path: str, src: str, dotted: str) -> None:
    segments = dotted.split(".")
    head = segments[0]
    if head not in _ALLOWED_HEADS:
        raise ProfileError(
            f"{path}: override '{dotted}' in phase {src} must start with 'env.' or 'agent.' "
            f"(scene.num_envs is set with --num_envs; other heads are not Hydra-reachable here)"
        )
    baked = _BAKED_IN_SEGMENTS.intersection(segments)
    if baked:
        raise ProfileError(
            f"{path}: override '{dotted}' in phase {src} targets a baked-in constant {sorted(baked)} "
            f"(kp/kd/armature/effort/obs-history are defined in Python, not per-phase overridable -- see "
            f"scripts/train_profile.py module docstring)"
        )


# ----------------------------------------------------------------------------------------------------------------
# Override flattening & value formatting
# ----------------------------------------------------------------------------------------------------------------

def _flatten_map(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested/pre-dotted override dict -> {dotted_key: raw_value}. Pre-dotted keys pass through unchanged."""
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_map(v, key + "."))
        else:
            out[key] = v
    return out


def _format_value(v: Any) -> str:
    if isinstance(v, bool):  # before int: bool is an int subclass
        return "True" if v else "False"  # matches the repo's documented usage (env...debug_vis=False)
    if v is None:
        return "null"
    if isinstance(v, (int, float, str)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_format_value(x) for x in v) + "]"
    raise ProfileError(f"cannot format override value {v!r} (type {type(v).__name__})")


def flatten_overrides(d: dict[str, Any]) -> list[str]:
    """Nested/pre-dotted override dict -> sorted list of Hydra 'key=value' strings."""
    flat = _flatten_map(d)
    return sorted(f"{k}={_format_value(v)}" for k, v in flat.items())


# ----------------------------------------------------------------------------------------------------------------
# Command assembly
# ----------------------------------------------------------------------------------------------------------------

def phase_boundaries(profile: TrainingProfile) -> list[tuple[int, int]]:
    """Cumulative (start, end) iteration counts per phase: phase k resumes at the previous end."""
    out: list[tuple[int, int]] = []
    start = 0
    for p in profile.phases:
        end = start + p.iterations
        out.append((start, end))
        start = end
    return out


def build_batched_command(
    profile: TrainingProfile,
    phase_idx: int,
    cli_overrides: list[str] | None = None,
    cli_num_envs: int | None = None,
    cli_seed: int | None = None,
    cli_dashboard: bool | None = None,
    cli_viewer: bool | None = None,
    cli_dashboard_output: str | None = None,
    cli_headless: bool = True,
) -> list[str]:
    """Assemble the train_batched.sh argv for one phase.

    CLI values (cli_*) beat per-phase beat global-profile. Trailing Hydra overrides from cli_overrides are
    appended LAST so Hydra's last-wins makes an explicit CLI override win over the phase's own value.
    """
    phase = profile.phases[phase_idx]
    _, end = phase_boundaries(profile)[phase_idx]
    batch = phase.batch_iterations if phase.batched else phase.iterations
    num_envs = _first_not_none(cli_num_envs, phase.num_envs, profile.num_envs)
    seed = _first_not_none(cli_seed, phase.seed, profile.seed)

    merged = _flatten_map(profile.overrides)
    merged.update(_flatten_map(phase.overrides))  # phase wins over global
    phase_args = sorted(f"{k}={_format_value(v)}" for k, v in merged.items())

    argv = [
        "bash", TRAIN_BATCHED,
        "--task", profile.task,
        "--experiment_name", profile.experiment_name,
        "--run_dir", profile.run_dir,
        "--total_iterations", str(end),
        "--batch_iterations", str(batch),
    ]
    if num_envs is not None:
        argv += ["--num_envs", str(num_envs)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if cli_headless:
        argv += ["--headless"]
    # Live graphs: CLI > profile. train_batched.sh restarts the dashboard each batch; with it off there is nothing
    # for a viewer or an output path to do, so those flags are only emitted while the dashboard is on.
    dashboard = _first_not_none(cli_dashboard, profile.dashboard)
    viewer = _first_not_none(cli_viewer, profile.viewer)
    if not dashboard:
        argv += ["--no_dashboard"]
    else:
        argv += ["--dashboard_output", _first_not_none(cli_dashboard_output, profile.dashboard_output)]
        if not viewer:
            argv += ["--no_viewer"]
    argv += phase_args
    if cli_overrides:
        argv += list(cli_overrides)
    return argv


# ----------------------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------------------

def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _opt_int(path: str, field_name: str, v: Any) -> int | None:
    if v is None:
        return None
    if not _is_int(v):
        raise ProfileError(f"{path}: {field_name} must be an integer (got {v!r})")
    return int(v)


def _opt_bool(path: str, field_name: str, v: Any, default: bool) -> bool:
    if v is None:
        return default
    if not isinstance(v, bool):  # YAML true/false only: '1', 'yes_please' and quoted strings are mistakes
        raise ProfileError(f"{path}: {field_name} must be true or false (got {v!r})")
    return v


def _opt_str(path: str, field_name: str, v: Any, default: str) -> str:
    if v is None:
        return default
    if not isinstance(v, str) or not v:
        raise ProfileError(f"{path}: {field_name} must be a non-empty string (got {v!r})")
    return v


def _first_not_none(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def resolve_profile_path(profile: str) -> str:
    """A path ending in .yaml is used as-is; otherwise look up training_profiles/<name>.yaml at the repo root."""
    if profile.endswith((".yaml", ".yml")):
        return profile
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "training_profiles", f"{profile}.yaml")


# ----------------------------------------------------------------------------------------------------------------
# CLI orchestrator
# ----------------------------------------------------------------------------------------------------------------

def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the orchestrator's CLI. Unset live-graph flags are None so the profile decides (CLI > profile)."""
    parser = argparse.ArgumentParser(
        description="Run a phased training profile by chaining scripts/train_batched.sh invocations.",
    )
    parser.add_argument("--profile", required=True, help="Profile name (training_profiles/<name>.yaml) or a path.")
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved per-phase commands; launch nothing.")
    parser.add_argument("--start_phase", type=int, default=1, help="1-based phase to start at (resume a run).")
    parser.add_argument("--only_phase", type=int, default=None, help="1-based phase to run alone.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs for all phases.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for all phases.")
    # Explicit underscore pairs (BooleanOptionalAction would emit --no-dashboard, the repo uses --no_dashboard).
    parser.add_argument("--dashboard", dest="dashboard", action="store_true", default=None,
                        help="Force the live dashboard ON, overriding the profile's `dashboard`.")
    parser.add_argument("--no_dashboard", dest="dashboard", action="store_false",
                        help="Force the live dashboard OFF, overriding the profile's `dashboard`.")
    parser.add_argument("--viewer", dest="viewer", action="store_true", default=None,
                        help="Force the feh viewer ON, overriding the profile's `viewer`.")
    parser.add_argument("--no_viewer", dest="viewer", action="store_false",
                        help="Force the feh viewer OFF, overriding the profile's `viewer`.")
    parser.add_argument("--dashboard_output", type=str, default=None, help="Override the profile's dashboard PNG path.")
    parser.add_argument("--gui", action="store_true",
                        help="Run with the Isaac Sim GUI (omit the default --headless). Heavy -- meant for WATCHING "
                             "with a small --num_envs, not a full-throughput run; the window relaunches each "
                             "batch (every batch_iterations). Needs the container's X11 passthrough (docker-compose.yml).")
    parser.add_argument(
        "overrides", nargs="*",
        help="Trailing Hydra key=value overrides applied to EVERY phase (win over the profile's own values).",
    )
    args = parser.parse_args(argv)
    args._parser = parser  # so main() can report a bad --start_phase/--only_phase through argparse
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    parser = args._parser
    profile = load_profile(resolve_profile_path(args.profile))
    n = len(profile.phases)
    if args.only_phase is not None:
        indices = [args.only_phase - 1]
    else:
        indices = list(range(args.start_phase - 1, n))
    for i in indices:
        if not 0 <= i < n:
            parser.error(f"phase index {i + 1} out of range 1..{n}")

    for i in indices:
        cmd = build_batched_command(
            profile, i, cli_overrides=args.overrides or None, cli_num_envs=args.num_envs, cli_seed=args.seed,
            cli_dashboard=args.dashboard, cli_viewer=args.viewer, cli_dashboard_output=args.dashboard_output,
            cli_headless=not args.gui,
        )
        print(f"\n=== phase {i + 1}/{n}: {profile.phases[i].name} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        result = subprocess.run(cmd, cwd=repo_root)
        if result.returncode != 0:
            print(f"[train_profile] phase '{profile.phases[i].name}' exited {result.returncode}; stopping.")
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
