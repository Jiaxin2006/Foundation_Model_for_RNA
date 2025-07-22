#!/bin/bash
#SBATCH -J preprocess
#SBATCH --account=slmreasoning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=0-5:00:00

cd /projects/slmreasoning/yifang/

module load Miniforge3
source activate /projects/slmreasoning/yifang/envs/NAS

mkdir -p outs
python -m core.data_utils.create_kmer_tokenizer > outs/process/kmer_tokenizers.out 2>&1
