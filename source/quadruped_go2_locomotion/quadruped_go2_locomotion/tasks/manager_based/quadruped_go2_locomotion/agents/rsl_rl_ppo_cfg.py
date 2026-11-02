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
    save_interval = 100
    # New experiment dir: this policy drops base_lin_vel from the actor (47-dim), trains at kp 40 and ramps the mass /
    # COM / push randomisation with a curriculum (CurriculumCfg.difficulty), so it is not resume-compatible with the
    # old 50-dim / kp-25 checkpoints or with go2_kp40_no_lin_vel_ppo (full randomisation from iteration 0).
    experiment_name = "go2_kp40_curriculum_ppo"
    empirical_normalization = True
    # Asymmetric actor-critic: the actor sees only the hardware-available `policy` group (no base_lin_vel);
    # the critic sees the privileged `critic` group (includes base_lin_vel). The critic is discarded at
    # deploy. See ObservationsCfg.PolicyCfg / CriticCfg in quadruped_go2_locomotion_env_cfg.py.
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
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
        entropy_coef=0.0075, #NOTE 0.00275 is stable value
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )