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
    # New experiment dir: this policy drops base_lin_vel from the actor and STACKS the last HISTORY_LENGTH (=6) frames,
    # so the actor input is 48*6=288 and the critic 51*6=306. The changed input dim makes it NOT resume-compatible with
    # the single-frame iter3/iter4 checkpoints (nor the old 50-dim / kp-25 ones); it must train from scratch.
    experiment_name = "go2_obs_stacking_with_height"
    empirical_normalization = True
    # Asymmetric actor-critic: the actor sees only the hardware-available `policy` group (no base_lin_vel);
    # the critic sees the privileged `critic` group (includes base_lin_vel). The critic is discarded at
    # deploy. See ObservationsCfg.PolicyCfg / CriticCfg in env_cfg.py.
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


@configclass
class VisionActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic cfg for the stairs vision policy. Adds the depth-encoder knobs (consumed as **kwargs by
    VisionActorCritic) and pins class_name to our custom module. Observation normalization is explicitly OFF
    so the warm-started blind weights keep seeing raw observations (the blind policy was trained that way)."""

    class_name: str = "VisionActorCritic"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    # extra kwargs forwarded to VisionActorCritic.__init__
    depth_group: str = "depth"
    depth_latent_dim: int = 32
    depth_encoder_activation: str = "elu"


@configclass
class VisionPPORunnerCfg(PPORunnerCfg):
    """Stairs vision-policy training. Same PPO machinery as the blind task, but: a separate experiment
    folder, the depth+critic observation groups wired via obs_groups, the VisionActorCritic policy, and a
    lower learning rate (we're fine-tuning from blind weights; the adaptive-KL schedule still protects it)."""

    experiment_name = "go2_stairs_vision"
    # empirical_normalization is deprecated upstream; leave it off and rely on the explicit per-group flags
    # in VisionActorCriticCfg so the warm start sees raw obs.
    empirical_normalization = None
    obs_groups = {"policy": ["policy", "depth"], "critic": ["critic"]}
    policy = VisionActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[256, 512, 128],
        critic_hidden_dims=[256, 512, 128],
        activation="elu",
        depth_latent_dim=32,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        normalize_advantage_per_mini_batch=False,
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0075,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
