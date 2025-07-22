#!/bin/bash
#SBATCH -J pretrain
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:1

cd /projects/slmreasoning/yifang/

module load Miniforge3
source activate /projects/slmreasoning/yifang/envs/NAS

#CASE=contrastive
CASE=mask
EXPERIMENT_NAME=transformer-kmer1

mkdir -p outs/pretrain/$EXPERIMENT_NAME
if [[ "$CASE" == "contrastive" ]]; then
    python -m core.training.pretrain.contrastive-pretrain --experiment_name $EXPERIMENT_NAME > outs/pretrain/$EXPERIMENT_NAME/contrastive.out 2>&1
elif [[ "$CASE" == "mask" ]]; then
    python -m core.training.pretrain.mask-pretrain --experiment_name $EXPERIMENT_NAME > "outs/pretrain/$EXPERIMENT_NAME/mask.out" 2>&1
else
    echo "Invalid CASE: $CASE. Please use one of: contrastive, mask"
    exit 1
fi