# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "go2_with_pitch_lean_and_height_control_ppo"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[256, 512, 128],
        critic_hidden_dims=[256, 512, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        normalize_advantage_per_mini_batch=False,
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0075, #NOTE 0.00275 is stable value  # increased from 0.00275 to 0.0075 to encourage more exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )