#!/bin/bash
#SBATCH -J auto-pretrain
#SBATCH --account=begl-delta-gpu
#SBATCH --partition=gpuH200x8  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G  # 增加到160G内存
#SBATCH --output=%x_%j.out

module purge
module load anaconda3_gpu/23.7.4

unset PYTHONHOME
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate renv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CONFIG_DIR="configs"
OUT_DIR="outs/pretrain"
PRETRAIN_BASE="/u/yfang4/projects/jiaxin/NAS-for-Bio/nni_pre_logs"  # 预训练模型基础路径

# 手动排除这些实验

SKIP_MASK=("cnn-kmer1","cnn-kmer6")
SKIP_CONTRASTIVE=("lstm-kmer1")

# 帮助函数：检查是否在跳过列表里
should_skip() {
    local exp_name=$1
    shift
    local skip_list=("$@")
    for skip_exp in "${skip_list[@]}"; do
        if [[ "$skip_exp" == "$exp_name" ]]; then
            return 0  # yes, skip
        fi
    done
    return 1  # no, don't skip
}

# 检查预训练模型是否已存在
model_exists() {
    local exp_name=$1
    local pretrain_type=$2
    
    # 构建模型路径
    local model_path="$PRETRAIN_BASE/$pretrain_type/$exp_name/Pretrain-epoch=0.ckpt"
    
    # 检查文件是否存在
    if [[ -f "$model_path" ]]; then
        echo "Found existing pretrained model at: $model_path"
        return 0  # 存在
    else
        echo "No existing pretrained model found at: $model_path"
        return 1  # 不存在
    fi
}

# 遍历所有 configs 子目录
for dir in "$CONFIG_DIR"/*/; do
    EXPERIMENT_NAME=$(basename "$dir")
    echo ">>> Starting for experiment: $EXPERIMENT_NAME"

    # ----------- MASK PRETRAINING -----------
    if should_skip "$EXPERIMENT_NAME" "${SKIP_MASK[@]}"; then
        echo "Skipping MASK for $EXPERIMENT_NAME (in skip list)"
    else
        # 检查模型是否已存在
        if model_exists "$EXPERIMENT_NAME" "mask"; then
            echo "Skipping MASK pretraining for $EXPERIMENT_NAME (model already exists)"
        else
            echo "Running MASK pretraining for $EXPERIMENT_NAME"
            mkdir -p "$OUT_DIR/$EXPERIMENT_NAME"
            python -m core.training.pretrain.mask-pretrain \
                --experiment_name "$EXPERIMENT_NAME" \
                > "$OUT_DIR/$EXPERIMENT_NAME/mask.out" 2>&1
        fi
    fi

    # --------- CONTRASTIVE PRETRAINING ----------
    if should_skip "$EXPERIMENT_NAME" "${SKIP_CONTRASTIVE[@]}"; then
        echo "Skipping CONTRASTIVE for $EXPERIMENT_NAME (in skip list)"
    else
        # 检查模型是否已存在
        if model_exists "$EXPERIMENT_NAME" "contrastive"; then
            echo "Skipping CONTRASTIVE pretraining for $EXPERIMENT_NAME (model already exists)"
        else
            echo "Running CONTRASTIVE pretraining for $EXPERIMENT_NAME"
            mkdir -p "$OUT_DIR/$EXPERIMENT_NAME"
            python -m core.training.pretrain.contrastive-pretrain \
                --experiment_name "$EXPERIMENT_NAME" \
                > "$OUT_DIR/$EXPERIMENT_NAME/contrastive.out" 2>&1
        fi
    fi

    echo "<<< Done for $EXPERIMENT_NAME"
    echo ""
done

echo "=== ALL EXPERIMENTS COMPLETED ==="