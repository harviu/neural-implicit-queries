#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=test_network_size
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS0027
#SBATCH --output=experiment_logs/test_network_size.%j

ml miniconda3
ml cuda/11.6.2
source activate implicit-env

python src/main_network_size.py 