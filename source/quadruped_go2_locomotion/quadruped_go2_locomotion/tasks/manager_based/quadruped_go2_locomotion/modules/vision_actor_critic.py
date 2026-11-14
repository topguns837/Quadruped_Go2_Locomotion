# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic with a depth-image encoder, warm-startable from the blind proprioceptive policy.

rsl_rl's stock `ActorCritic` only accepts 1D observation groups (it asserts so) and concatenates all actor
groups into one flat vector. This subclass keeps everything that machinery gives us (PPO, the Gaussian
head, the std parameter, the value network) and changes exactly one thing: one actor observation group is a
stacked depth image (N, K, H, W) that is first passed through a small CNN encoder, whose latent is then
concatenated AFTER the proprioceptive vector before the actor MLP.

Putting the depth latent after the proprio vector is deliberate: it makes the actor's first linear layer
`[proprio_columns | depth_latent_columns]`, so the blind policy's weights drop straight into the proprio
columns and the depth-latent columns start at zero (see `load_blind_weights`). At iteration 0 the network
therefore reproduces the blind policy exactly, and vision is learned strictly on top of it.

The critic is asymmetric: it reads a separate privileged `critic` group (clean proprio + ground-truth
height scan), all 1D, so it uses the stock concatenation path unchanged and never sees the depth image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_nn_activation


class DepthEncoder(nn.Module):
    """Small 3-conv encoder: (N, in_channels, H, W) depth stack -> (N, out_dim) latent.

    Deliberately tiny -- it runs for every env every step at 1000+ envs, and perceptive-locomotion work
    (ANYmal, Extreme Parkour) uses encoders of about this size. The flattened conv size is computed from a
    dummy forward so the head's input dim follows H/W automatically.
    """

    def __init__(self, in_channels: int, height: int, width: int, out_dim: int = 32, activation: str = "elu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            resolve_nn_activation(activation),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            resolve_nn_activation(activation),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            resolve_nn_activation(activation),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels, height, width)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 128),
            resolve_nn_activation(activation),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


class VisionActorCritic(ActorCritic):
    """ActorCritic whose actor consumes one depth-image group through `DepthEncoder`.

    Extra cfg params (passed through the RslRlPpoActorCriticCfg subclass -> runner -> here as kwargs):
        depth_group: name of the 4D image group inside obs_groups["policy"]. Defaults to "depth".
        depth_latent_dim: width of the encoder latent appended to the proprio vector. Defaults to 32.
        depth_encoder_activation: activation for the encoder. Defaults to "elu".
    """

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        depth_group: str = "depth",
        depth_latent_dim: int = 32,
        depth_encoder_activation: str = "elu",
        **kwargs: dict[str, Any],
    ) -> None:
        # Split the actor's groups into proprio (1D) and the single image group (4D), by inspecting the
        # actual observation tensors handed in at construction time.
        policy_groups = obs_groups["policy"]
        image_groups = [g for g in policy_groups if obs[g].dim() == 4]
        proprio_groups = [g for g in policy_groups if obs[g].dim() == 2]
        if len(image_groups) != 1 or image_groups[0] != depth_group:
            raise ValueError(
                f"VisionActorCritic expects exactly one 4D image group named '{depth_group}' in "
                f"obs_groups['policy']={policy_groups}; found image groups {image_groups}."
            )

        # Build the encoder before super().__init__ so we know the latent width; it is assigned to the
        # module (and so moved to device / registered) only after nn.Module is initialized below.
        _, k, h, w = obs[depth_group].shape
        encoder = DepthEncoder(k, h, w, depth_latent_dim, depth_encoder_activation)

        # Hand the parent a proxy where the 4D depth group is replaced by a flat (N, latent) placeholder, so
        # it sizes the actor MLP's input as (sum of proprio dims + latent) and its 1D-only assert passes.
        num_envs = obs[proprio_groups[0]].shape[0] if proprio_groups else obs[depth_group].shape[0]
        proxy = {}
        for g in set(policy_groups) | set(obs_groups["critic"]):
            proxy[g] = obs[g].new_zeros((num_envs, depth_latent_dim)) if g == depth_group else obs[g]
        super().__init__(proxy, obs_groups, num_actions, **kwargs)

        self.depth_group = depth_group
        self.proprio_groups = proprio_groups
        self.depth_latent_dim = depth_latent_dim
        self.depth_input_shape = (k, h, w)
        self.num_proprio_obs = sum(obs[g].shape[-1] for g in proprio_groups)
        self.depth_encoder = encoder

    def get_actor_obs(self, obs) -> torch.Tensor:
        # proprio columns first, depth latent last -- the column order the warm start relies on.
        z = self.depth_encoder(obs[self.depth_group])
        if not self.proprio_groups:
            return z
        proprio = torch.cat([obs[g] for g in self.proprio_groups], dim=-1)
        return torch.cat([proprio, z], dim=-1)

    def load_blind_weights(self, path: str, map_location: str = "cpu") -> None:
        """Warm-start from a blind (proprio-only) ActorCritic checkpoint.

        Copies the blind actor/critic MLP weights into the proprio input columns and leaves the new depth-
        latent columns (actor) and height-scan columns (critic) at zero, so the policy starts identical to
        the blind one and the encoder/privileged inputs are learned from there. The optimizer and iteration
        count are intentionally NOT restored (that's `runner.load`'s job for a true resume); this only sets
        weights, so training starts fresh with these as the initial parameters.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        self._warm_start_mlp(self.actor, sd, "actor")
        self._warm_start_mlp(self.critic, sd, "critic")
        if self.noise_std_type == "log" and "log_std" in sd:
            self.log_std.data.copy_(sd["log_std"])
        elif self.noise_std_type == "scalar" and "std" in sd:
            self.std.data.copy_(sd["std"])

    def _warm_start_mlp(self, mlp: nn.Module, sd: dict, prefix: str) -> None:
        new_sd = {}
        for name, p in mlp.state_dict().items():
            key = f"{prefix}.{name}"
            if key not in sd:
                raise KeyError(f"Blind checkpoint is missing '{key}'; is it a plain ActorCritic checkpoint?")
            blind = sd[key]
            if p.shape == blind.shape:
                new_sd[name] = blind.clone()
            elif (
                name.endswith(".weight")
                and p.dim() == 2
                and blind.dim() == 2
                and p.shape[0] == blind.shape[0]
                and p.shape[1] > blind.shape[1]
            ):
                # First layer only: new input is wider (proprio + extra). Copy the blind proprio columns,
                # zero the rest so those inputs contribute nothing until training moves them off zero.
                w = torch.zeros_like(p)
                w[:, : blind.shape[1]] = blind
                new_sd[name] = w
            else:
                raise ValueError(
                    f"Cannot warm-start '{key}': own shape {tuple(p.shape)} vs blind {tuple(blind.shape)}."
                )
        mlp.load_state_dict(new_sd, strict=True)

    def describe(self, save_dir: str | None = None) -> None:
        """Print and (optionally) render the network architecture.

        Text summary (layer shapes + parameter counts) via torchinfo; a rendered graph diagram via torchview
        when `save_dir` is given. Both are optional deps installed in the Docker image (see docker/Dockerfile
        app stage); if missing, this degrades to a plain module print plus a parameter count, so it never
        hard-fails. The actor graph is built from the deployable encoder+actor module (which has a concrete
        two-input forward), so it shows exactly what runs on the robot.
        """
        from .exporter import _VisionActorExporter

        proprio = torch.zeros(1, self.num_proprio_obs)
        depth = torch.zeros(1, *self.depth_input_shape)
        actor_module = _VisionActorExporter(self).eval()

        n_total = sum(p.numel() for p in self.parameters())
        n_encoder = sum(p.numel() for p in self.depth_encoder.parameters())
        print(f"[VisionActorCritic] params: total={n_total:,}  depth_encoder={n_encoder:,}")
        print(f"  proprio dim={self.num_proprio_obs}  depth input={tuple(self.depth_input_shape)}  "
              f"latent={self.depth_latent_dim}  actor input={self.actor[0].in_features}  "
              f"critic input={self.critic[0].in_features}")

        try:
            from torchinfo import summary

            print("\n=== Vision actor (depth encoder + actor MLP) ===")
            summary(actor_module, input_data=(proprio, depth), depth=4,
                    col_names=("input_size", "output_size", "num_params"))
            print("\n=== Critic MLP ===")
            summary(self.critic, input_size=(1, self.critic[0].in_features), depth=2,
                    col_names=("input_size", "output_size", "num_params"))
        except ImportError:
            print("[describe] torchinfo not installed; plain module dump follows.")
            print(actor_module)
            print(self.critic)

        if save_dir is not None:
            try:
                import os

                from torchview import draw_graph

                os.makedirs(save_dir, exist_ok=True)
                draw_graph(
                    actor_module,
                    input_data=(proprio, depth),
                    expand_nested=True,
                    save_graph=True,
                    directory=save_dir,
                    filename="vision_actor",
                    graph_name="vision_actor",
                )
                print(f"[describe] saved actor graph to {os.path.join(save_dir, 'vision_actor.png')}")
            except Exception as exc:  # noqa: BLE001
                print(f"[describe] graph rendering skipped ({exc}); torchview/graphviz may be missing.")
