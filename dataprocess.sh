#!/bin/bash
#SBATCH -J data_processing
#SBATCH --account=begl-delta-gpu
#SBATCH --partition=gpuMI100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%j.err

# 加载模块和环境
module purge
module load anaconda3_gpu/23.7.4

source $(conda info --base)/etc/profile.d/conda.sh
conda activate renv
# pip install jsonlines

# 运行数据处理脚本
python preprocess.py --input RNAcentral.json --output processed_data.jsonl