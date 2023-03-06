#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS0027
#SBATCH --output=experiment_logs/test.%j

ml miniconda3
ml cuda/11.6.2
source activate implicit-env

python src/main.py 