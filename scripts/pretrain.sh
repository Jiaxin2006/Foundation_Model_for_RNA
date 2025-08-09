#!/bin/bash
#SBATCH -J pretrain
#SBATCH --account=begl-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G 

module purge
module load anaconda3_gpu/23.7.4

unset PYTHONHOME
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate renv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


CASE=contrastive
# CASE=mask
EXPERIMENT_NAME=hyena-kmer1
# bimamba-kmer1 hyena-bpe512 hyena-bpe4096 hyena-kmer1 hyena-kmer6 lstm-kmer6 mamba-bpe512 mamba-kmer1 mix-bpe512 mix-kmer1 transformer-kmer1 transformer-bpe512 transformer-bpe4096
# bimamba-kmer1 hyena-bpe512 hyena-kmer1 hyena-kmer6 mix-kmer1 transformer-kmer1 transformer-kmer6


mkdir -p outs/pretrain/$EXPERIMENT_NAME
if [[ "$CASE" == "contrastive" ]]; then
    python -m core.training.pretrain.contrastive-pretrain --experiment_name $EXPERIMENT_NAME > outs/pretrain/$EXPERIMENT_NAME/contrastive.out 2>&1
elif [[ "$CASE" == "mask" ]]; then
    python -m core.training.pretrain.mask-pretrain --experiment_name $EXPERIMENT_NAME > "outs/pretrain/$EXPERIMENT_NAME/mask.out" 2>&1
else
    echo "Invalid CASE: $CASE. Please use one of: contrastive, mask"
    exit 1
fi