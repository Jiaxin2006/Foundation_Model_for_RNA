#!/bin/bash
#SBATCH -J repretrain
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-23:00:00
#SBATCH --gres=gpu:1

cd /projects/slmreasoning/yifang/

module load Miniforge3
source activate /projects/slmreasoning/yifang/envs/NAS




#CASE=con
CASE=mask
EXPERIMENT_NAME=transformer-bpe512

mkdir -p outs/re-pretrain/$EXPERIMENT_NAME
if [[ "$CASE" == "con" ]]; then
    python -m core.training.re-pretrain.contrastive-repre --experiment_name $EXPERIMENT_NAME > outs/re-pretrain/$EXPERIMENT_NAME/contrastive.out 2>&1
elif [[ "$CASE" == "mask" ]]; then
    python -m core.training.re-pretrain.mask-repre --experiment_name $EXPERIMENT_NAME > "outs/re-pretrain/$EXPERIMENT_NAME/mask.out" 2>&1
else
    echo "Invalid CASE: $CASE. Please use one of: con, mask"
    exit 1
fi