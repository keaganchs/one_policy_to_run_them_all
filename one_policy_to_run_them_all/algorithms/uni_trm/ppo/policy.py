from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from one_policy_to_run_them_all.algorithms.uni_trm.models.trm_policy import TRMPolicy


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        # Extract TRM config from config.algorithm if available, else use defaults
        # Assuming config.algorithm might have these keys in the future
        
        return (
            TRMPolicy(
                hidden_size=getattr(config.algorithm, 'trm_hidden_size', 256),
                expansion=getattr(config.algorithm, 'trm_expansion', 2.0),
                num_heads=getattr(config.algorithm, 'trm_num_heads', 4),
                mlp_t=getattr(config.algorithm, 'trm_mlp_t', False), # Default to Attention
                H_cycles=getattr(config.algorithm, 'trm_h_cycles', 2),
                L_cycles=getattr(config.algorithm, 'trm_l_cycles', 2),
                max_seq_len=getattr(config.algorithm, 'trm_max_seq_len', 64),
                std_dev=config.algorithm.std_dev,
                softmax_temperature=config.algorithm.softmax_temperature,
                softmax_temperature_min=config.algorithm.softmax_temperature_min,
                stability_epsilon=config.algorithm.stability_epsilon,
                policy_mean_abs_clip=config.algorithm.policy_mean_abs_clip,
                policy_std_min_clip=config.algorithm.policy_std_min_clip,
                policy_std_max_clip=config.algorithm.policy_std_max_clip
            ), 
            get_processed_action_function()
        )
    else:
        raise ValueError(f"Unsupported action/observation space type: {action_space_type}, {observation_space_type}")


def get_processed_action_function():
    def get_clipped_and_scaled_action(action):
        return action
    return jax.jit(get_clipped_and_scaled_action)
