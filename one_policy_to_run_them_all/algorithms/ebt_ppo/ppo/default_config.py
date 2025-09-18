from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    
    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e9
    config.start_learning_rate = 3e-4
    config.end_learning_rate = 3e-4
    config.nr_steps = 2048
    config.nr_epochs = 10
    config.minibatch_size = 64
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.softmax_temperature = 1.0
    config.softmax_temperature_min = 0.015
    config.stability_epsilon = 1e-8
    config.missing_value = 0.0
    config.entropy_coef = 0.0
    config.critic_coef = 0.5
    config.max_grad_norm = 0.5
    config.std_dev = 1.0
    config.policy_mean_abs_clip = 10.0
    config.policy_std_min_clip = 1e-8
    config.policy_std_max_clip = 2.0
    config.nr_hidden_units = 256
    config.evaluation_frequency = 204800  # -1 to disable
    config.evaluation_episodes = 10
    config.save_latest_frequency = 204800
    config.determine_fastest_cpu_for_gpu = False

    # EBT-specific configuration
    config.ebt_dim = 256  # Transformer hidden dimension
    config.ebt_n_layers = 4  # Number of transformer layers
    config.ebt_n_heads = 8  # Number of attention heads
    config.ebt_max_seq_len = 64  # Maximum sequence length for state history
    config.ebt_dropout = 0.1  # Dropout rate
    config.ebt_use_energy_attention = True  # Use energy-based attention mechanism
    config.ebt_energy_scale = 1.0  # Scale factor for energy computation
    config.ebt_state_embedding_dim = 128  # State embedding dimension

    return config