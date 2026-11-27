# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Python checks a health-check case can name in its ``checks:`` list (see cases.yaml).

Each check is ``fn(ctx, **args) -> list[str]`` and returns a list of problems (empty = OK). They only read files
the case produced, so they stay cheap and never import Isaac Sim.
"""

from __future__ import annotations

import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Context:
    repo: Path
    run_dir: Path
    log: str
    py: str

    def path(self, p: str) -> Path:
        p = p.replace("{run}", str(self.run_dir))
        q = Path(p)
        return q if q.is_absolute() else self.repo / q


def _glob(ctx: Context, pattern: str) -> list[str]:
    return sorted(glob.glob(str(ctx.path(pattern))))


def file_contains(ctx: Context, path: str, pattern: str) -> list[str]:
    """``path`` exists and its text matches the regex ``pattern``."""
    p = ctx.path(path)
    if not p.exists():
        return [f"{path} does not exist"]
    return [] if re.search(pattern, p.read_text(errors="replace")) else [f"{path} lacks /{pattern}/"]


def log_count(ctx: Context, pattern: str, exactly: int | None = None, at_least: int | None = None) -> list[str]:
    """The case log has ``exactly`` / ``at_least`` matches of ``pattern``."""
    n = len(re.findall(pattern, ctx.log))
    out = []
    if exactly is not None and n != exactly:
        out.append(f"/{pattern}/ appears {n}x in the log (expected {exactly})")
    if at_least is not None and n < at_least:
        out.append(f"/{pattern}/ appears {n}x in the log (expected >= {at_least})")
    return out


def no_nan_in_log(ctx: Context, allow_inf: bool = False) -> list[str]:
    """No ``nan``/``inf`` token in numeric output (word-bounded so e.g. 'infer' or 'Information' do not match).
    ``allow_inf``: only ``nan`` is an error (gym spaces legitimately print ``Box(-inf, inf)``)."""
    pat = r"(?<![A-Za-z_])nan(?![A-Za-z_])" if allow_inf else r"(?<![A-Za-z_])(nan|-?inf)(?![A-Za-z_])"
    m = re.search(pat, ctx.log, flags=re.IGNORECASE)
    return [f"non-finite value in the log: {m.group(0)!r}"] if m else []


def files_exist(ctx: Context, patterns: list[str], min_each: int = 1) -> list[str]:
    """Every glob in ``patterns`` matches at least ``min_each`` files."""
    return [f"{p}: {len(_glob(ctx, p))} match(es), expected >= {min_each}" for p in patterns
            if len(_glob(ctx, p)) < min_each]


def finite_numbers_in_file(ctx: Context, path: str, min_numbers: int = 1, allow_nan: bool = False) -> list[str]:
    """A text/markdown report contains numbers and none of them is inf (nor nan unless ``allow_nan``: a metric
    bucket with no samples, e.g. 'standing' in a very short random schedule, is legitimately nan)."""
    p = ctx.path(path)
    if not p.exists():
        return [f"{path} does not exist"]
    text = p.read_text(errors="replace")
    out = []
    bad = r"(?<![A-Za-z_])(inf|-inf)(?![A-Za-z_])" if allow_nan else r"(?<![A-Za-z_])(nan|-?inf)(?![A-Za-z_])"
    if re.search(bad, text, flags=re.IGNORECASE):
        out.append(f"{path} contains a non-finite value")
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+", text) if math.isfinite(float(x))]
    if len(nums) < min_numbers:
        out.append(f"{path} has only {len(nums)} numbers (expected >= {min_numbers})")
    return out


# --------------------------------------------------------------------------------------------------- training / policy
def training_run(ctx: Context, experiment: str, run: str, last: int, resumes: int = 0) -> list[str]:
    """A finished (batched) training run: checkpoint ``model_<last>.pt`` + ``curriculum_<last>.pt`` + params, the
    checkpoint is loadable and says it is iteration ``last``, and the curriculum was restored ``resumes`` times."""
    base = ctx.repo / "logs" / "rsl_rl" / experiment / run
    errs = []
    for name in (f"model_{last}.pt", f"curriculum_{last}.pt", "params/env.yaml", "params/agent.yaml"):
        if not (base / name).exists():
            errs.append(f"{experiment}/{run}/{name} is missing")
    if errs:
        return errs
    import torch

    ck = torch.load(base / f"model_{last}.pt", map_location="cpu", weights_only=False)
    if ck.get("iter") != last:
        errs.append(f"model_{last}.pt says iter={ck.get('iter')}")
    cur = torch.load(base / f"curriculum_{last}.pt", map_location="cpu", weights_only=False)
    if "difficulty_scale" not in cur or "terrain_levels" not in cur:
        errs.append(f"curriculum_{last}.pt lacks difficulty_scale/terrain_levels (has {sorted(cur)})")
    n_restored = len(re.findall(r"Restored curriculum from", ctx.log))
    if n_restored < resumes:
        errs.append(f"curriculum restored {n_restored}x, expected >= {resumes}")
    if re.search(r"No curriculum state next to", ctx.log):
        errs.append("a resume found no curriculum state (difficulty/terrain restarted from scratch)")
    return errs


def env_param(ctx: Context, experiment: str, run: str, key: str, value: float, file: str = "env") -> list[str]:
    """``params/<file>.yaml`` (env or agent) of the run holds ``value`` at the dotted ``key``
    (e.g. events.push_robot.params.max_velocity)."""
    import yaml

    p = ctx.repo / "logs" / "rsl_rl" / experiment / run / "params" / f"{file}.yaml"
    if not p.exists():
        return [f"{p} is missing"]
    node = yaml.unsafe_load(p.read_text())
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return [f"{key}: '{part}' not found in {file}.yaml"]
        node = node[part]
    return [] if abs(float(node) - value) < 1e-9 else [f"{key} = {node} in {file}.yaml, expected {value}"]


def jit_input_dim(ctx: Context, path: str, dim: int) -> list[str]:
    """The exported TorchScript policy at ``path`` (glob ok) takes ``dim`` inputs; its ONNX twin exists next to it."""
    import torch

    hits = _glob(ctx, path)
    if not hits:
        return [f"{path}: no file"]
    m = torch.jit.load(hits[-1], map_location="cpu")
    first = next(p for _n, p in m.named_parameters() if p.ndim == 2)
    errs = [] if int(first.shape[1]) == dim else [f"{hits[-1]}: input dim {int(first.shape[1])}, expected {dim}"]
    if not Path(hits[-1]).with_name("policy.onnx").exists():
        errs.append(f"policy.onnx missing next to {hits[-1]}")
    return errs


def npz_fields(ctx: Context, path: str, fields: list[str], min_rows: int = 10) -> list[str]:
    """A .npz trace has the named arrays, with at least ``min_rows`` rows and only finite numbers."""
    import numpy as np

    hits = _glob(ctx, path)
    if not hits:
        return [f"{path}: no file"]
    d = np.load(hits[-1], allow_pickle=True)
    errs = []
    for f in fields:
        if f not in d.files:
            errs.append(f"{hits[-1]} lacks '{f}' (has {sorted(d.files)[:12]}...)")
            continue
        a = d[f]
        if a.ndim >= 1 and a.shape[0] < min_rows:
            errs.append(f"{f}: only {a.shape[0]} rows")
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            errs.append(f"{f} contains nan/inf")
    return errs


# ------------------------------------------------------------------------------------------------------- fake robot
def fake_robot_summary(ctx: Context, out_dir: str, min_lowcmds: int = 100) -> list[str]:
    """The synthetic robot saw ``min_lowcmds`` LowCmd frames from deploy_real.py, none with a bad CRC/header."""
    import json

    p = ctx.path(out_dir) / "fake" / "summary.json"
    if not p.exists():
        return [f"{p} missing (fake robot did not finish cleanly)"]
    d = json.loads(p.read_text())
    errs = []
    if d.get("lowcmd_total", 0) < min_lowcmds:
        errs.append(f"fake robot received {d.get('lowcmd_total')} LowCmd frames (expected >= {min_lowcmds})")
    c = d.get("checks_total", {})
    for k in ("crc_bad", "header_bad", "level_bad", "mode_bad"):
        if c.get(k, 0):
            errs.append(f"fake robot: {k} = {c[k]}")
    if c.get("crc_checked", 0) < 1:
        errs.append("fake robot never checked a CRC")
    return errs


def curriculum_difficulty(ctx: Context, experiment: str, run: str, last: int, above: float = 0.0) -> list[str]:
    """``curriculum_<last>.pt`` holds a difficulty_scale strictly above ``above`` (the curriculum really moved and was
    saved) and a plausible 0..1 value."""
    import torch

    p = ctx.repo / "logs" / "rsl_rl" / experiment / run / f"curriculum_{last}.pt"
    if not p.exists():
        return [f"{p} is missing"]
    d = float(torch.load(p, map_location="cpu", weights_only=False)["difficulty_scale"])
    errs = []
    if not above < d <= 1.0:
        errs.append(f"difficulty_scale = {d} in {p.name}, expected in ({above}, 1.0]")
    return errs
