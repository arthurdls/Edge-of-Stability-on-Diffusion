#!/bin/bash

# Note, for active submissions: srun -p mit_normal_gpu --gres=gpu:1 -t 00:30:00 --pty /bin/bash

#SBATCH -p mit_normal_gpu   
#SBATCH --gres=gpu:1 
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16GB

module load miniforge/24.3.0-0

python train.py --lr $1 $2