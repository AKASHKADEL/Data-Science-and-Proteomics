#!/bin/bash
#
#SBATCH --job-name=capstone_job
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --mem=90GB
#SBATCH --output=slurm_%A.out
#SBATCH --error=slurm_%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.6/0.2.0_3
module load torchvision/python3.5/0.1.9

# python3 -m pip install comet_ml --user

python3 -u main_model_gru_gpu.py --USE_CUDA --n_epochs 5 --learning_rate 0.001 --main_data_dir "/scratch/ak6201/Capstone/data/" --out_dir "/scratch/ak6201/Capstone/ModelOutputs/"
