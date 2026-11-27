# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""cli_args: RSL-RL CLI arguments override the agent config; Hydra's run dir is redirected under logs/."""

import argparse
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
cli_args = pytest.importorskip("cli_args")


def agent(**kw):
    base = dict(seed=1, resume=False, load_run=".*", load_checkpoint="model_.*.pt", run_name="", logger="tensorboard",
                wandb_project="", neptune_project="")
    base.update(kw)
    return SimpleNamespace(**base)


def args(**kw):
    base = dict(seed=None, resume=None, load_run=None, checkpoint=None, run_name=None, logger=None,
                log_project_name=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_no_arguments_leave_the_config_untouched():
    cfg = cli_args.update_rsl_rl_cfg(agent(), args())
    assert (cfg.seed, cfg.resume, cfg.logger) == (1, False, "tensorboard")


def test_every_override_is_applied():
    cfg = cli_args.update_rsl_rl_cfg(
        agent(), args(seed=7, resume=True, load_run="run_a", checkpoint="model_5.pt", run_name="x", logger="neptune",
                      log_project_name="proj"))
    assert (cfg.seed, cfg.resume, cfg.load_run, cfg.load_checkpoint, cfg.run_name) == (7, True, "run_a", "model_5.pt", "x")
    assert cfg.logger == "neptune" and cfg.wandb_project == "proj" and cfg.neptune_project == "proj"


def test_seed_minus_one_draws_a_random_seed():
    a = args(seed=-1)
    cfg = cli_args.update_rsl_rl_cfg(agent(), a)
    assert 0 <= cfg.seed <= 10000 and a.seed == cfg.seed


def test_project_name_is_ignored_for_other_loggers():
    cfg = cli_args.update_rsl_rl_cfg(agent(), args(log_project_name="proj"))
    assert cfg.wandb_project == ""


def test_add_rsl_rl_args_registers_the_training_flags():
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    ns = parser.parse_args(["--resume", "--load_run", "r", "--checkpoint", "c.pt", "--logger", "tensorboard"])
    assert ns.resume and ns.load_run == "r" and ns.checkpoint == "c.pt" and ns.logger == "tensorboard"


def test_hydra_run_dir_is_redirected_unless_given():
    out = cli_args.with_hydra_log_dir(["env.a=1"])
    assert out[0] == "env.a=1" and out[-1].startswith("hydra.run.dir=logs/hydra/")
    explicit = ["hydra.run.dir=/tmp/x", "env.a=1"]
    assert cli_args.with_hydra_log_dir(explicit) == explicit
