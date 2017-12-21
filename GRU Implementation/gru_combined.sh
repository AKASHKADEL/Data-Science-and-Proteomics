#!/bin/bash
#
#SBATCH --job-name=comb_gru
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=90GB
#SBATCH --output=slurm_%A.out
#SBATCH --error=slurm_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=da1933@nyu.edu

module purge
module load python3/intel/3.5.3
module load pytorch/python3.6/0.2.0_3
module load torchvision/python3.5/0.1.9

# python3 -m pip install comet_ml --user

python3 -u gru_combined.py \
--USE_CUDA \
--hidden_size 100 \
--emb_dim 100 \
--n_epochs 5 \
--batch_size 50 \
--learning_rate 0.001 \
--main_data_dir "/scratch/da1933/capstone/data/" \
--out_dir "/scratch/da1933/capstone/gru_output/baseline/" \
--k 1 \
--l2_penalty .0005
