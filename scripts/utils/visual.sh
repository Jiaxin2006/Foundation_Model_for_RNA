#!/bin/bash
#SBATCH -J res_visual
#SBATCH --account=slmreasoning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-1:00:00

cd /projects/slmreasoning/yifang/

module load Miniforge3
source activate /projects/slmreasoning/yifang/envs/NAS

mkdir -p outs/process
python -m core.visual > outs/process/visual.out 2>&1
