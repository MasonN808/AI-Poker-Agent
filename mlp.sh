#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=mlp           # Name of the job
#SBATCH --cpus-per-task=5         # Number of CPUs per task
#SBATCH --mem=20G                # Memory allocated
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --time=1-00:00:00           # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

BASE_SCRIPT="/nas/ucb/mason/AI-Poker-Agent/MCTS_poker/mlp.py"

srun -N1 -n1 python3 $BASE_SCRIPT

wait
