#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=time
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS0027
#SBATCH --output=experiment_logs/time.%j

ml miniconda3
ml cuda/11.6.2
source activate implicit-env

python src/main_up_time.py 