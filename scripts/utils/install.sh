#!/bin/bash
#SBATCH -J pretrain
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:1

cd /projects/slmreasoning/yifang/

module load Miniforge3
module load cuda12.6/toolkit/12.6.2
export CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6.2
source activate /projects/slmreasoning/yifang/envs/NAS

pip install mamba-ssm