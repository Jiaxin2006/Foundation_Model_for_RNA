#!/bin/bash
#SBATCH -J experiment
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



# mask-mask con-con only-ft only-mask only-con mask-ft con-ft
CASE=mask-ft
EXPERIMENT_NAME=mix-kmer1

mkdir -p outs/finetune/$EXPERIMENT_NAME

if [[ "$CASE" == "mask-mask" ]]; then
  python -m core.training.finetune.continue-pre --experiment_name $EXPERIMENT_NAME --strategy mask > "outs/finetune/$EXPERIMENT_NAME/mask-mask.out" 2>&1

elif [[ "$CASE" == "con-con" ]]; then
  python -m core.training.finetune.continue-pre --experiment_name $EXPERIMENT_NAME --strategy contrastive > "outs/finetune/$EXPERIMENT_NAME/con-con.out" 2>&1

elif [[ "$CASE" == "only-ft" ]]; then
  #python -m core.training.finetune.only-ft --experiment_name $EXPERIMENT_NAME > "outs/finetune/$EXPERIMENT_NAME/only-ft.out" 2>&1
  python -m core.training.finetune.only-ft-pickup --experiment_name $EXPERIMENT_NAME > "outs/finetune/$EXPERIMENT_NAME/only-ft.out" 2>&1

elif [[ "$CASE" == "only-mask" ]]; then
  python -m core.training.finetune.only-pre --experiment_name $EXPERIMENT_NAME --strategy mask > "outs/finetune/$EXPERIMENT_NAME/only-mask.out" 2>&1

elif [[ "$CASE" == "only-con" ]]; then
  python -m core.training.finetune.only-pre --experiment_name $EXPERIMENT_NAME --strategy contrastive > "outs/finetune/$EXPERIMENT_NAME/only-con.out" 2>&1

elif [[ "$CASE" == "mask-ft" ]]; then
  #python -m core.training.finetune.pre-ft --experiment_name $EXPERIMENT_NAME --strategy mask > "outs/finetune/$EXPERIMENT_NAME/mask-ft.out" 2>&1
  python -m core.training.finetune.pre-ft-pickup --experiment_name $EXPERIMENT_NAME --strategy mask > "outs/finetune/$EXPERIMENT_NAME/mask-ft.out" 2>&1

elif [[ "$CASE" == "con-ft" ]]; then
  python -m core.training.finetune.pre-ft --experiment_name $EXPERIMENT_NAME --strategy contrastive > "outs/finetune/$EXPERIMENT_NAME/con-ft.out" 2>&1
  #python -m core.training.finetune.pre-ft-pickup --experiment_name $EXPERIMENT_NAME --strategy contrastive > "outs/finetune/$EXPERIMENT_NAME/con-ft.out" 2>&1
else
  echo "Invalid CASE: $CASE. Please use one of: pre-ft, only-pre, only-ft"
  exit 1
fi