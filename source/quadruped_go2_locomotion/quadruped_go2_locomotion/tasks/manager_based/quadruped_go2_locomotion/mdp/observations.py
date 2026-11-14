# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Depth-camera observation terms for the stairs task.

`DepthImage` is the term the vision policy consumes: it turns the TiledCamera's raw depth output into a
normalized, frame-stacked tensor of shape (num_envs, num_frames, H, W), the layout a CNN encoder expects.
It is a class term (ManagerTermBase) rather than a plain function because it holds a per-env ring buffer of
recent frames across steps (so it can stack frames spaced in time) and must clear that buffer on episode
reset to avoid leaking one episode's frames into the next.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DepthImage(ManagerTermBase):
    """Normalized, frame-stacked depth image for the vision policy.

    Pipeline per call, matched to how the real D435 depth is handled on the robot (see deploy-side notes in
    vision_staircase_plan.md):

    1. Read the sensor's depth output, shape (N, H, W, 1), and drop the channel -> (N, H, W).
    2. Treat inf, NaN, non-positive, and out-of-[near, far] pixels as *invalid* (the real D435 returns 0 for
       no-measurement pixels); set them to `far`.
    3. Normalize to [-0.5, 0.5] via (d - near) / (far - near) - 0.5. Invalid pixels therefore land at +0.5.
    4. Push into a per-env ring buffer and return `num_frames` frames spaced `frame_stride_steps` control
       steps apart, newest first: shape (N, num_frames, H, W).

    The spacing lines up with the camera's own update rate (update_period=0.1 s = 5 control steps at 50 Hz),
    so consecutive stacked frames are genuinely distinct captures, not the same frame repeated.

    Params (set via ObservationTermCfg.params):
        sensor_cfg: the depth camera. Defaults to SceneEntityCfg("depth_camera").
        data_type: sensor output key. Defaults to "distance_to_image_plane" (z-depth, what the real D435
            reports -- NOT "distance_to_camera", which is Euclidean ray length).
        near, far: clip/normalize range in metres. `far` doubles as the invalid-pixel fill value.
        num_frames: how many time-stacked frames to return (K).
        frame_stride_steps: control-step spacing between stacked frames (S).
        augment: reserved hook for the training-time depth augmentations (step 4); identity for now.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._sensor_cfg: SceneEntityCfg = p.get("sensor_cfg", SceneEntityCfg("depth_camera"))
        self._data_type: str = p.get("data_type", "distance_to_image_plane")
        self._near: float = float(p.get("near", 0.28))
        self._far: float = float(p.get("far", 2.0))
        self._num_frames: int = int(p.get("num_frames", 2))
        self._stride: int = int(p.get("frame_stride_steps", 5))
        # Ring buffer deep enough to hold the oldest frame we sample: (K-1)*S + 1 slots, newest at index 0.
        self._buffer_len = (self._num_frames - 1) * self._stride + 1
        self._sample_idx = [i * self._stride for i in range(self._num_frames)]
        self._buffer: torch.Tensor | None = None  # allocated lazily once H, W, device are known
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Mark these envs' history as stale; the next __call__ refills it from the first new frame, so no
        # frames from the previous episode leak across the reset boundary.
        if self._buffer is None:
            return
        if env_ids is None:
            self._initialized[:] = False
        else:
            self._initialized[env_ids] = False

    def _normalize(self, depth: torch.Tensor) -> torch.Tensor:
        invalid = ~torch.isfinite(depth) | (depth <= 0.0) | (depth < self._near) | (depth > self._far)
        depth = torch.where(invalid, torch.full_like(depth, self._far), depth)
        return (depth - self._near) / (self._far - self._near) - 0.5

    def _augment(self, depth: torch.Tensor) -> torch.Tensor:
        # Hook for the training-time sensor augmentations (noise, dropout, holes, blur, latency, ...) added
        # in the step-4 increment. Identity until then.
        return depth

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
        data_type: str = "distance_to_image_plane",
        near: float = 0.28,
        far: float = 2.0,
        num_frames: int = 2,
        frame_stride_steps: int = 5,
        augment: bool = True,
    ) -> torch.Tensor:
        camera = env.scene[sensor_cfg.name]
        raw = camera.data.output[data_type][..., 0]  # (N, H, W)
        frame = self._normalize(raw)
        if augment:
            frame = self._augment(frame)

        if self._buffer is None:
            n, h, w = frame.shape
            self._buffer = frame.new_zeros((n, self._buffer_len, h, w))

        # Shift history back by one and write the new frame at the front.
        self._buffer = torch.roll(self._buffer, shifts=1, dims=1)
        self._buffer[:, 0] = frame
        # For envs whose history is stale (just reset / first ever call), fill every slot with the current
        # frame so the stack is K copies of "now" rather than leftover frames from a previous episode.
        stale = ~self._initialized
        if stale.any():
            self._buffer[stale] = frame[stale].unsqueeze(1)
            self._initialized[stale] = True

        return self._buffer[:, self._sample_idx].clone()
