from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from one_policy_to_run_them_all.algorithms.uni_trm.ppo.ppo import PPO
from one_policy_to_run_them_all.algorithms.uni_trm.ppo.default_config import get_config
from one_policy_to_run_them_all.algorithms.uni_trm.ppo.general_properties import GeneralProperties


TRM_PPO = extract_algorithm_name_from_file(__file__)
register_algorithm(TRM_PPO, get_config, PPO, GeneralProperties)
