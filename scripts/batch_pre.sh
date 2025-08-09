#!/bin/bash
#SBATCH -J auto-pretrain
#SBATCH --account=begl-delta-gpu
#SBATCH --partition=gpuH200x8  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=%x_%j.out

module purge
module load anaconda3_gpu/23.7.4

unset PYTHONHOME
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate renv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 设置环境变量解决显存碎片化问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_DIR="configs"
OUT_DIR="outs/pretrain"
PRETRAIN_BASE="/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/nni_pre_logs"

# 定义需要运行的实验列表 "hyena-kmer1"
CONTRASTIVE_EXPERIMENTS=(
    "hyena-kmer6"
    "mix-kmer1"
    "transformer-kmer6"
)

MASK_EXPERIMENTS=(
    "hyena-kmer1"
    "hyena-kmer6"
    "mix-kmer1"
    "transformer-kmer1"
    "transformer-kmer6"
)

# 检查预训练模型是否已存在
model_exists() {
    local exp_name=$1
    local pretrain_type=$2
    
    # 构建模型路径
    local model_path="$PRETRAIN_BASE/$pretrain_type/$exp_name/Pretrain-epoch=4.ckpt"
    
    # 检查文件是否存在
    if [[ -f "$model_path" ]]; then
        echo "Found existing pretrained model at: $model_path"
        return 0  # 存在
    else
        echo "No existing pretrained model found at: $model_path"
        return 1  # 不存在
    fi
}

# 运行CONTRASTIVE预训练
echo "=== STARTING CONTRASTIVE PRETRAINING ==="
for exp_name in "${CONTRASTIVE_EXPERIMENTS[@]}"; do
    echo ">>> Starting CONTRASTIVE for: $exp_name"
    
    # 检查模型是否已存在
    if model_exists "$exp_name" "contrastive"; then
        echo "Skipping CONTRASTIVE pretraining for $exp_name (model already exists)"
    else
        echo "Running CONTRASTIVE pretraining for $exp_name"
        mkdir -p "$OUT_DIR/$exp_name"
        python -m core.training.pretrain.contrastive-pretrain \
            --experiment_name "$exp_name" \
            > "$OUT_DIR/$exp_name/contrastive.out" 2>&1
    fi
    
    echo "<<< Done for $exp_name"
    echo ""
done

# 运行MASK预训练
echo "=== STARTING MASK PRETRAINING ==="
for exp_name in "${MASK_EXPERIMENTS[@]}"; do
    echo ">>> Starting MASK for: $exp_name"
    
    # 检查模型是否已存在
    if model_exists "$exp_name" "mask"; then
        echo "Skipping MASK pretraining for $exp_name (model already exists)"
    else
        echo "Running MASK pretraining for $exp_name"
        mkdir -p "$OUT_DIR/$exp_name"
        python -m core.training.pretrain.mask-pretrain \
            --experiment_name "$exp_name" \
            > "$OUT_DIR/$exp_name/mask.out" 2>&1
    fi
    
    echo "<<< Done for $exp_name"
    echo ""
done

echo "=== ALL SPECIFIED EXPERIMENTS COMPLETED ==="