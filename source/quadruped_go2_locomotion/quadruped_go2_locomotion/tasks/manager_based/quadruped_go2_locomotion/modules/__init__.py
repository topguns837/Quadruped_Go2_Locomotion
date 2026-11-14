# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom RSL-RL policy modules for this project (vision actor-critic + its exporter)."""

from .exporter import export_vision_policy_as_jit, export_vision_policy_as_onnx  # noqa: F401
from .vision_actor_critic import DepthEncoder, VisionActorCritic  # noqa: F401

# RSL-RL's OnPolicyRunner builds the policy with `eval(class_name)` evaluated in the on_policy_runner
# module's own namespace, which only knows the stock classes. Inject VisionActorCritic there so a config
# with class_name="VisionActorCritic" resolves, without having to subclass the runner. Importing this
# package (done from the task __init__) is enough to register it.
import rsl_rl.runners.on_policy_runner as _on_policy_runner  # noqa: E402

_on_policy_runner.VisionActorCritic = VisionActorCritic
