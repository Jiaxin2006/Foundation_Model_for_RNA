#!/bin/bash
#SBATCH -J protein_dataset
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:1

cd /projects/slmreasoning/yifang/core/data_utils

module load Miniforge3
source activate /projects/slmreasoning/yifang/envs/protein


python -m protein_dataset > "/projects/slmreasoning/yifang/outs/process/protein_dataset.out" 2>&1
