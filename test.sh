#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS0027
#SBATCH --output=experiment_logs/test.%j

ml miniconda3
ml cuda/11.6.2
source activate implicit-env

python src/main.py -t 0 -d 8
python src/main.py -t 3 -d 8
python src/main.py -t 2 -d 9
python src/main.py -t 4 -d 10