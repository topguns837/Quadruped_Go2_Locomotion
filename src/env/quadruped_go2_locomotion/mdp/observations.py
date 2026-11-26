from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def depth_array(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg) -> torch.Tensor:
    """Get the depth array from the robot's perspective.

    This function can be used to create a curriculum term based on the depth array, for example by computing the
    mean depth or the number of pixels with a depth smaller than a certain threshold.

    Returns:
        The depth array from the robot's perspective.
    """
    camera = env.scene[asset_cfg.name]

    depth = camera.data.output["distance_to_camera"]

    depth_sampled = depth[:, ::20, ::20]

    depth_flat = depth_sampled.flatten(start_dim=1)

    return depth_flat
