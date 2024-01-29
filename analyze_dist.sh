#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=analyze_dist
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS0027
#SBATCH --output=experiment_logs/analyze_dist.%j

ml miniconda3
ml cuda/11.6.2
source activate implicit-env

python src/analyze_hist.py