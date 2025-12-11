import gymnasium as gym

from one_policy_to_run_them_all.environments.talos.environment import Talos
from one_policy_to_run_them_all.environments.talos.wrappers import RLXInfo, RecordEpisodeStatistics
from one_policy_to_run_them_all.environments.talos.general_properties import GeneralProperties


def create_env(config):
    def make_env(seed, render):
        def thunk():
            env = Talos(
                seed=seed,
                render=render,
                mode=config.environment.mode,
                control_type=config.environment.control_type,
                command_type=config.environment.command_type,
                command_sampling_type=config.environment.command_sampling_type,
                initial_state_type=config.environment.initial_state_type,
                reward_type=config.environment.reward_type,
                termination_type=config.environment.termination_type,
                domain_randomization_sampling_type=config.environment.domain_randomization_sampling_type,
                domain_randomization_action_delay_type=config.environment.domain_randomization_action_delay_type,
                domain_randomization_mujoco_model_type=config.environment.domain_randomization_mujoco_model_type,
                domain_randomization_control_type=config.environment.domain_randomization_control_type,
                domain_randomization_seen_robot_type=config.environment.domain_randomization_seen_robot_type,
                domain_randomization_unseen_robot_type=config.environment.domain_randomization_unseen_robot_type,
                domain_randomization_perturbation_type=config.environment.domain_randomization_perturbation_type,
                domain_randomization_perturbation_sampling_type=config.environment.domain_randomization_perturbation_sampling_type,
                observation_noise_type=config.environment.observation_noise_type,
                observation_dropout_type=config.environment.observation_dropout_type,
                terrain_type=config.environment.terrain_type,
                missing_value=config.environment.missing_value,
                add_goal_arrow=config.environment.add_goal_arrow,
                timestep=config.environment.timestep,
                episode_length_in_seconds=config.environment.episode_length_in_seconds,
                total_nr_envs=config.environment.nr_envs,
            )
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    vector_environment_class = gym.vector.SyncVectorEnv if config.environment.nr_envs == 1 else gym.vector.AsyncVectorEnv
    train_env = vector_environment_class([make_env(config.environment.seed + i, config.environment.render) for i in range(config.environment.nr_envs)])
    train_env = RLXInfo(train_env)
    train_env.general_properties = GeneralProperties
    train_env.reset(seed=config.environment.seed)

    eval_vector_environment_class = gym.vector.SyncVectorEnv if config.environment.nr_eval_envs == 1 else gym.vector.AsyncVectorEnv
    eval_env = eval_vector_environment_class([make_env(config.environment.seed + config.environment.nr_envs + i, False) for i in range(config.environment.nr_eval_envs)])
    eval_env = RLXInfo(eval_env)
    eval_env.general_properties = GeneralProperties
    eval_env.reset(seed=config.environment.seed + config.environment.nr_envs)

    class EnvContainer:
        def __init__(self, train_env, eval_env):
            self.train_env = train_env
            self.eval_env = eval_env
        
        def close(self):
            self.train_env.close()
            self.eval_env.close()

    return EnvContainer(train_env, eval_env)
