#!/bin/bash

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate onep

python experiment.py \
    --algorithm.name=uni_trm.ppo \
    --environment.name="talos" \
    --runner.track_console=True \
    --runner.load_model=model_best_jax \
    --algorithm.determine_fastest_cpu_for_gpu=False \
    --runner.mode=test \
    --environment.mode=test \
    --environment.add_goal_arrow=True \
    --environment.nr_envs=1 \
    --environment.multi_render=False \
    --environment.render=True
