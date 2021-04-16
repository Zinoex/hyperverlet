#!/usr/bin/env bash
#SBATCH --job-name Hypersolver
#SBATCH --partition batch
#SBATCH --mail-type=ALL  # Possible values: NONE, BEGIN, END, FAIL, REQUEUE, ALL, STAGE_OUT, TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80, TIME_LIMIT_50, ARRAY_TASKS
#SBATCH --mail-user=MI1012F21@cs.aau.dk
#SBATCH --time 62:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=allgpus #normal  # Possible values: normal, short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

srun singularity run -B /user/share/projects:/user/share/projects --nv hyperverlet.sif python3 -m hyperverlet.main --config-path configurations/integrator_experiments/three_body_spring_mass/hyperverlet.json plot
srun singularity run -B /user/share/projects:/user/share/projects --nv hyperverlet.sif python3 -m hyperverlet.main --config-path configurations/integrator_experiments/three_body_spring_mass/velocityverlet.json plot
