#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=fit_volume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS0027
#SBATCH --output=osc_outputs/fit_volume.%j

ml miniconda3
ml cuda/11.6.2
source activate implicit-env

N=8
W=32
A=elu
n=1000

# python src/main_fit_volume.py ../data/99_500_v02.bin ./sample_inputs/v02_${A}_${N}_${W}.npz --n_epochs ${n} --data_type asteroid --activation $A --n_layers $N --layer_width $W --batch_size 25000 --lr 1e-3
python src/main_fit_volume.py ../data/vorts01.data ./sample_inputs/vorts_${A}_${N}_${W}.npz --n_epochs ${n} --data_type vorts --activation $A --n_layers $N --layer_width $W --batch_size 16384 --lr 1e-3
# python src/main_fit_volume.py ../data/jet_chi_0054.dat ./sample_inputs/jet_${A}_${N}_${W}.npz --n_epochs ${n} --data_type combustion --activation $A --n_layers $N --layer_width $W --batch_size 57600 --lr 1e-3
# python src/main_fit_volume.py ../data/ethanediol.bin ./sample_inputs/eth_${A}_${N}_${W}.npz --n_epochs ${n} --data_type ethanediol --activation $A --n_layers $N --layer_width $W --batch_size 13340 --lr 1e-3
# python src/main_fit_volume.py ../data/Isotropic.nc ./sample_inputs/iso_${A}_${N}_${W}.npz --n_epochs ${n} --data_type isotropic --activation $A --n_layers $N --layer_width $W --batch_size 65536 --lr 1e-3