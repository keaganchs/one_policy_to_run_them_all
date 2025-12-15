from rl_x.runner.runner import Runner
import sys

if __name__ == "__main__":

    defaults = [
        "--algorithm.name=uni_trm.ppo",
        "--algorithm.total_timesteps=100000000",
        "--algorithm.nr_steps=20000",
        "--algorithm.minibatch_size=10000",
        "--algorithm.nr_epochs=10",
        "--algorithm.start_learning_rate=0.0004",
        "--algorithm.end_learning_rate=0.0",
        "--algorithm.entropy_coef=0.0",
        "--algorithm.gae_lambda=0.9",
        "--algorithm.critic_coef=1.0",
        "--algorithm.max_grad_norm=5.0",
        "--algorithm.clip_range=0.1",
        "--algorithm.evaluation_episodes=50",
        "--algorithm.evaluation_frequency=128000",
        "--algorithm.save_latest_frequency=128000",
        "--algorithm.determine_fastest_cpu_for_gpu=False",
        "--algorithm.device=gpu",
        
        # TRM Config
        # "--algorithm.trm_hidden_size=64",
        # "--algorithm.trm_expansion=2.0",
        # "--algorithm.trm_num_heads=4",
        # "--algorithm.trm_mlp_t=True", # False for Attention, True for MLP
        # "--algorithm.trm_h_cycles=2",
        # "--algorithm.trm_l_cycles=2",
        # "--algorithm.trm_max_seq_len=64",

        "--environment.name=talos", # ENV
        "--environment.nr_envs=1", # NR_ENVS
        "--environment.render=False", # RENDER
        "--environment.seed=0",
        "--runner.mode=train",
        "--runner.track_console=False",
        "--runner.track_tb=True",
        "--runner.track_wandb=False", # WANDB
        "--runner.save_model=True",
        "--runner.wandb_entity=keagan",
        "--runner.project_name=trm",
        "--runner.exp_name=uni_trm_E0", # EXP
        "--runner.notes=Default params" # NOTES
    ]

    # Overwrite defaults with command line args
    for arg in sys.argv:
        if arg.startswith("--"):
            key = arg.split("=")[0]
            defaults = [d for d in defaults if not d.startswith(key + "=")]

    sys.argv[1:1] = defaults

    runner = Runner(implementation_package_names=["rl_x", "one_policy_to_run_them_all"])
    runner.run()