# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static repo-consistency checks used by the health check (no simulator needed).

    python src/healthcheck/check_repo.py links      # relative links in README.md / docs / src READMEs resolve
    python src/healthcheck/check_repo.py paths      # files named by profiles, scenarios and docs exist
    python src/healthcheck/check_repo.py policies   # committed models/*/exported/policy.pt load; dims match profiles
    python src/healthcheck/check_repo.py profiles   # every shipped training profile loads and is self-consistent

Exit code 1 and one line per problem on failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "training"))

LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def md_files() -> list[Path]:
    files = [REPO / "README.md", REPO / "CLAUDE.md"]
    files += sorted((REPO / "docs").glob("*.md"))
    files += sorted((REPO / "src").rglob("README.md"))
    files += sorted((REPO / "models").glob("*/SUMMARY.md")) + [REPO / "models" / "README.md"]
    return [f for f in files if f.exists() and "menagerie" not in str(f)]


def check_links() -> list[str]:
    errs = []
    for f in md_files():
        text = FENCE_RE.sub("", COMMENT_RE.sub("", f.read_text(errors="replace")))
        for target in LINK_RE.findall(text):
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            rel = target.split("#")[0]
            if rel and not (f.parent / rel).resolve().exists():
                errs.append(f"{f.relative_to(REPO)}: broken link -> {target}")
    return errs


def check_paths() -> list[str]:
    errs = []
    for prof in sorted((REPO / "src" / "deploy" / "profiles").glob("*.yaml")):
        cfg = yaml.safe_load(prof.read_text())
        pp = cfg.get("policy_path")
        if pp and not (REPO / pp).exists():
            errs.append(f"{prof.relative_to(REPO)}: policy_path {pp} does not exist")
    safety = yaml.safe_load((REPO / "src/deploy/configs/safety_profiles.yaml").read_text())
    for name, body in (safety.get("profiles", safety) or {}).items():
        if isinstance(body, dict) and "policy_path" in body:
            errs.append(f"safety profile {name} must not carry policy_path (it belongs to --deploy_profile)")
    for sc in sorted((REPO / "src" / "sim2sim" / "scenarios").glob("*.yaml")):
        data = yaml.safe_load(sc.read_text())
        segs = data if isinstance(data, list) else data.get("segments", data.get("scenario", []))
        if not segs:
            errs.append(f"{sc.relative_to(REPO)}: no segments")
    for rel in ("src/training/train.sh", "src/training/train_batched.sh", "src/sim2sim/run_sim2sim_eval.sh",
                "src/sim2sim/fetch_go2_model.sh", "start.sh", "healthcheck.sh"):
        p = REPO / rel
        if not p.exists():
            errs.append(f"missing {rel}")
        elif not p.stat().st_mode & 0o111:
            errs.append(f"{rel} is not executable")
    for stale in ("run_random_eval.sh", "train_phased.sh", "scripts/rsl_rl"):
        for f in [REPO / "README.md", REPO / "CLAUDE.md", REPO / "docker/isaaclab-shell.sh"]:
            if stale in f.read_text():
                errs.append(f"{f.relative_to(REPO)} still mentions the old name {stale}")
    return errs


def policy_input_dim(path: Path) -> int:
    import torch

    m = torch.jit.load(str(path), map_location="cpu")
    first = next(p for n, p in m.named_parameters() if p.ndim == 2)
    return int(first.shape[1])


def check_policies() -> list[str]:
    errs = []
    profiles = {p.name: yaml.safe_load(p.read_text()) for p in (REPO / "src/deploy/profiles").glob("*.yaml")}
    for prof_name, cfg in profiles.items():
        pp = cfg.get("policy_path")
        if not pp or not (REPO / pp).exists():
            continue
        dim = policy_input_dim(REPO / pp)
        hist = int(cfg.get("history_length", 1))
        if dim % hist:
            errs.append(f"{prof_name}: policy input {dim} not divisible by history_length {hist}")
        else:
            print(f"{prof_name}: {pp} input {dim} = {dim // hist} x {hist} frames")
    for pt in sorted((REPO / "models").glob("*/exported/policy.pt")):
        try:
            dim = policy_input_dim(pt)
        except Exception as e:
            errs.append(f"{pt.relative_to(REPO)} does not load: {e}")
            continue
        if dim <= 0:
            errs.append(f"{pt.relative_to(REPO)}: input dim {dim}")
    return errs


def check_profiles() -> list[str]:
    import train_profile as tp

    errs = []
    names = set()
    for f in sorted((REPO / "src/training/profiles").glob("*.yaml")):
        try:
            prof = tp.load_profile(str(f))
        except Exception as e:
            errs.append(f"{f.name}: {e}")
            continue
        if prof.num_phases != len(prof.phases):
            errs.append(f"{f.name}: num_phases {prof.num_phases} != {len(prof.phases)} phases")
        if prof.experiment_name in names:
            errs.append(f"{f.name}: experiment_name {prof.experiment_name} is used by another profile")
        names.add(prof.experiment_name)
    return errs


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"links": check_links, "paths": check_paths, "policies": check_policies, "profiles": check_profiles}.get(which)
    if fn is None:
        print(__doc__)
        return 2
    errs = fn()
    for e in errs:
        print(f"FAIL: {e}")
    print(f"{which}: {'OK' if not errs else str(len(errs)) + ' problem(s)'}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
