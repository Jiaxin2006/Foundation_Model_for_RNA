#!/bin/bash
#SBATCH -J toy
#SBATCH --account=begl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x.out

module purge
module load anaconda3_gpu/23.7.4

source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate renv
echo "Job started at $(date)"
python toy.py
echo "Job finished at $(date) with exit code $?"
