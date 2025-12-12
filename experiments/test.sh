#!/bin/bash

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate onep

python experiment.py \
    --algorithm.name=uni_trm.ppo \
    --environment.name="talos" \
    --runner.track_console=True \
    --algorithm.nr_steps=500 \
    --algorithm.minibatch_size=250 \
    --algorithm.nr_epochs=10 \
    --algorithm.evaluation_frequency=-1 \
    --algorithm.save_latest_frequency=-1 \
    --runner.save_model=False \
    --runner.mode=test \
    --runner.load_model=/home/holmes/projects/thesis/ebt/one_policy_to_run_them_all/experiments/runs/trm_talos/trm_talos_E0/1765542203/models/model_best_jax \
    --algorithm.determine_fastest_cpu_for_gpu=False \
    --environment.mode=test \
    --environment.add_goal_arrow=True \
    --environment.nr_envs=1 \
    --environment.multi_render=False \
    --environment.render=True
