#!/bin/bash
#SBATCH -J preprocess
#SBATCH --account=slmreasoning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-1:00:00

cd /projects/slmreasoning/yifang/

module load Miniforge3
source activate /projects/slmreasoning/yifang/envs/NAS

mkdir -p outs
python -m core.data_utils.preprocess-bench > outs/process/benchdata_preprocess.out 2>&1
