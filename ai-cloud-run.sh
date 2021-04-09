#!/usr/bin/env bash
#SBATCH --job-name hyperverlet
#SBATCH --partition batch
#SBATCH --mail-type=ALL  # Possible values: NONE, BEGIN, END, FAIL, REQUEUE, ALL, STAGE_OUT, TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80, TIME_LIMIT_50, ARRAY_TASKS
#SBATCH --mail-user=mi1012f21@cs.aau.dk
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=normal  # Possible values: normal, short, allgpus
#SBATCH --mem=32G

srun singularity run -B /user/share/projects:/user/share/projects --nv hyperverlet.sif python3 scripts/run_experiment_parallel.py --experiement=integrator_comparison --system=three_body_spring_system --num-processes=1




