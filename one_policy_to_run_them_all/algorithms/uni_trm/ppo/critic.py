import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.observation_space_type import ObservationSpaceType
from one_policy_to_run_them_all.algorithms.uni_trm.models.trm_critic import TRMCritic


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return TRMCritic(
            hidden_size=getattr(config.algorithm, 'trm_hidden_size', 256),
            expansion=getattr(config.algorithm, 'trm_expansion', 2.0),
            num_heads=getattr(config.algorithm, 'trm_num_heads', 4),
            mlp_t=getattr(config.algorithm, 'trm_mlp_t', False),
            H_cycles=getattr(config.algorithm, 'trm_h_cycles', 2),
            L_cycles=getattr(config.algorithm, 'trm_l_cycles', 2),
            max_seq_len=getattr(config.algorithm, 'trm_max_seq_len', 64)
        )
    else:
        raise ValueError(f"Unsupported observation space type: {observation_space_type}")
