#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=populate-50M           # Name of the job
#SBATCH --cpus-per-task=10         # Number of CPUs per task
#SBATCH --mem=2G                # Memory allocated
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --time=2-00:00:00           # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source env/bin/activate

BASE_SCRIPT="/nas/ucb/mason/AI-Poker-Agent/MCTS_poker/populate.py"

srun -N1 -n1 python3 $BASE_SCRIPT

wait
