#!/bin/bash

#SBATCH --job-name=onep_talos_trm
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=2000
#SBATCH --time=23:59:59

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate onep

# export JAX_TRACEBACK_FILTERING="off"

python experiment.py \
    --algorithm.name="uni_ppo.ppo" \
    --algorithm.total_timesteps=100000000 \
    --algorithm.nr_steps=500 \
    --algorithm.minibatch_size=250 \
    --algorithm.nr_epochs=10 \
    --algorithm.start_learning_rate=0.0004 \
    --algorithm.end_learning_rate=0.0 \
    --algorithm.entropy_coef=0.0 \
    --algorithm.gae_lambda=0.9 \
    --algorithm.critic_coef=1.0 \
    --algorithm.max_grad_norm=5.0 \
    --algorithm.clip_range=0.1 \
    --algorithm.evaluation_episodes=50 \
    --algorithm.evaluation_frequency=400000 \
    --algorithm.save_latest_frequency=400000 \
    --algorithm.determine_fastest_cpu_for_gpu=True \
    --algorithm.device="gpu" \
    --environment.name="talos" \
    --environment.nr_envs=20 \
    --environment.seed=0 \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="keagan" \
    --runner.project_name="trm_talos" \
    --runner.exp_name="trm_talos_PPO_BASELINE_E0" \
    --runner.notes="Same params as TRM models"