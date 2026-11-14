# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export the vision actor to JIT/ONNX.

The stock isaaclab_rl exporter bakes a single flat input into `policy.actor`. The vision actor takes two
inputs -- the proprio vector and the depth stack -- and runs the depth through its CNN encoder first, so it
needs its own exporter. The exported module is self-contained (encoder + actor, no normalizer, since the
vision policy trains on raw observations) and is what the deploy/sim2sim side loads.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn


class _VisionActorExporter(nn.Module):
    """Deployable forward: proprio (B, P) + depth (B, K, H, W) -> actions (B, A).

    Mirrors VisionActorCritic.get_actor_obs + actor exactly: encode the depth stack, concatenate the latent
    AFTER the proprio vector, run the actor MLP. No observation normalizer (the vision policy uses raw obs).
    """

    def __init__(self, policy):
        super().__init__()
        self.encoder = copy.deepcopy(policy.depth_encoder)
        self.actor = copy.deepcopy(policy.actor)
        self.state_dependent_std = bool(getattr(policy, "state_dependent_std", False))

    def forward(self, proprio: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        z = self.encoder(depth)
        out = self.actor(torch.cat([proprio, z], dim=-1))
        if self.state_dependent_std:
            out = out[..., 0, :]  # drop the std row, keep the mean (matches act_inference)
        return out


def _example_inputs(policy):
    k, h, w = policy.depth_input_shape
    return torch.zeros(1, policy.num_proprio_obs), torch.zeros(1, k, h, w)


def export_vision_policy_as_jit(policy, path: str, filename: str = "policy.pt") -> str:
    """Trace the vision actor to a TorchScript file. Returns the written path."""
    os.makedirs(path, exist_ok=True)
    module = _VisionActorExporter(policy).to("cpu").eval()
    traced = torch.jit.trace(module, _example_inputs(policy))
    out_path = os.path.join(path, filename)
    traced.save(out_path)
    return out_path


def export_vision_policy_as_onnx(policy, path: str, filename: str = "policy.onnx", verbose: bool = False) -> str:
    """Export the vision actor to ONNX with named inputs `proprio`, `depth` and output `actions`."""
    os.makedirs(path, exist_ok=True)
    module = _VisionActorExporter(policy).to("cpu").eval()
    out_path = os.path.join(path, filename)
    torch.onnx.export(
        module,
        _example_inputs(policy),
        out_path,
        export_params=True,
        opset_version=18,
        input_names=["proprio", "depth"],
        output_names=["actions"],
        dynamic_axes={"proprio": {0: "batch"}, "depth": {0: "batch"}, "actions": {0: "batch"}},
        verbose=verbose,
    )
    return out_path
