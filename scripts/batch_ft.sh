#!/bin/bash
#SBATCH -J auto-finetune
#SBATCH --account=begl-delta-gpu
#SBATCH --partition=gpuH200x8    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --output=%x_%j.out

module purge
module load anaconda3_gpu/23.7.4
unset PYTHONHOME

source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate renv

CONFIG_DIR="configs"
OUT_DIR="outs/finetune"
RESULTS_BASE="/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/results"
TASK_INDEX=(0 1 2)

# ---------- 跳过配置 ----------
SKIP_EXPERIMENTS=("cnn-bpe512" "bimamba-kmer1")
SKIP_CASES=("")

# ---------- 帮助函数 ----------
should_skip_experiment() {
    local exp=$1
    for skip in "${SKIP_EXPERIMENTS[@]}"; do
        [[ "$skip" == "$exp" ]] && return 0
    done
    return 1
}

should_skip_case() {
    local exp=$1
    local case=$2
    for skip in "${SKIP_CASES[@]}"; do
        [[ "$skip" == "$exp:$case" ]] && return 0
    done
    return 1
}

# 检查结果文件是否存在且有效（每列至少有一行有数据）
result_exists_and_valid() {
    local exp_name=$1
    local case_name=$2
    local result_file="$RESULTS_BASE/$exp_name/${case_name}_results.csv"
    
    # 检查文件是否存在
    if [[ ! -f "$result_file" ]]; then
        echo "Result file not found: $result_file"
        return 1
    fi
    
    # 检查文件是否有数据行（不只是标题行）
    local line_count=$(wc -l < "$result_file")
    if [[ $line_count -le 1 ]]; then
        echo "Result file has only header or is empty: $result_file"
        return 1
    fi
    
    # 检查每一列是否至少有一行有数据
    local num_columns=0
    local col_has_data=()
    
    # 读取标题行获取列数
    IFS=',' read -r -a headers < <(head -n 1 "$result_file")
    num_columns=${#headers[@]}
    
    # 初始化列数据状态数组
    for ((i=0; i<num_columns; i++)); do
        col_has_data[$i]=0
    done
    
    # 处理数据行（跳过标题行）
    local line_num=0
    while IFS= read -r line; do
        ((line_num++))
        [[ $line_num -eq 1 ]] && continue  # 跳过标题行
        
        # 分割行
        IFS=',' read -r -a fields <<< "$line"
        
        # 检查每一列
        for ((i=0; i<num_columns; i++)); do
            # 清理字段值
            field_value=$(echo "${fields[$i]}" | tr -d '"' | tr -d ' ' | tr -d '\r')
            
            # 如果字段有值，标记该列有数据
            if [[ -n "$field_value" ]]; then
                col_has_data[$i]=1
            fi
        done
    done < "$result_file"
    
    # 检查是否有列没有数据
    local all_columns_have_data=1
    for ((i=0; i<num_columns; i++)); do
        if [[ ${col_has_data[$i]} -eq 0 ]]; then
            echo "Column ${headers[$i]} has no data in file: $result_file"
            all_columns_have_data=0
        fi
    done
    
    if [[ $all_columns_have_data -eq 1 ]]; then
        echo "All columns have data in result file: $result_file"
        return 0
    else
        echo "Some columns are missing data in result file: $result_file"
        return 1
    fi
}

# ---------- 主循环 ----------
for dir in "$CONFIG_DIR"/*/; do
    EXPERIMENT_NAME=$(basename "$dir")
    echo ">>> Starting experiments for: $EXPERIMENT_NAME"

    if should_skip_experiment "$EXPERIMENT_NAME"; then
        echo "Skipping entire experiment: $EXPERIMENT_NAME"
        continue
    fi

    mkdir -p "$OUT_DIR/$EXPERIMENT_NAME"

    # 需要跑的所有 CASE mask-mask con-con  only-mask only-con mask-ft con-ft only-ft
    CASES=(mask-ft con-ft only-ft)

    for CASE in "${CASES[@]}"; do
        if should_skip_case "$EXPERIMENT_NAME" "$CASE"; then
            echo "  Skipping CASE $CASE for $EXPERIMENT_NAME (manual skip)"
            continue
        fi
        
        # 检查结果文件是否已存在且有效
        if result_exists_and_valid "$EXPERIMENT_NAME" "$CASE"; then
            echo "  Skipping CASE $CASE for $EXPERIMENT_NAME (results already exist and are valid)"
            continue
        fi

        echo "  Running CASE $CASE for $EXPERIMENT_NAME"
        OUT_FILE="$OUT_DIR/$EXPERIMENT_NAME/${CASE}.out"

        # 根据CASE类型运行不同的Python脚本
        if [[ "$CASE" == "mask-mask" ]]; then
            python -m core.training.finetune.continue-pre \
                --experiment_name "$EXPERIMENT_NAME" \
                --task_index ${TASK_INDEX[@]} \
                --strategy mask \
                > "$OUT_FILE" 2>&1

        elif [[ "$CASE" == "con-con" ]]; then
            python -m core.training.finetune.continue-pre \
                --experiment_name "$EXPERIMENT_NAME" \
                --task_index ${TASK_INDEX[@]} \
                --strategy contrastive \
                > "$OUT_FILE" 2>&1

        elif [[ "$CASE" == "only-ft" ]]; then
            python -m core.training.finetune.only-ft \
                --experiment_name "$EXPERIMENT_NAME" \
                --task_index ${TASK_INDEX[@]} \
                > "$OUT_FILE" 2>&1

        elif [[ "$CASE" == "only-mask" ]]; then
            python -m core.training.finetune.only-pre \
                --experiment_name "$EXPERIMENT_NAME" \
                --task_index ${TASK_INDEX[@]} \
                --strategy mask \
                > "$OUT_FILE" 2>&1

        elif [[ "$CASE" == "only-con" ]]; then
            python -m core.training.finetune.only-pre \
                --experiment_name "$EXPERIMENT_NAME" \
                --task_index ${TASK_INDEX[@]} \
                --strategy contrastive \
                > "$OUT_FILE" 2>&1

        elif [[ "$CASE" == "mask-ft" ]]; then
            python -m core.training.finetune.pre-ft \
                --experiment_name "$EXPERIMENT_NAME" \
                --strategy mask \
                --task_index ${TASK_INDEX[@]} \
                > "$OUT_FILE" 2>&1

        elif [[ "$CASE" == "con-ft" ]]; then
            python -m core.training.finetune.pre-ft \
                --experiment_name "$EXPERIMENT_NAME" \
                --strategy contrastive \
                --task_index ${TASK_INDEX[@]} \
                > "$OUT_FILE" 2>&1

        else
            echo "Unknown CASE: $CASE"
            exit 1
        fi
    done

    echo "<<< Done for $EXPERIMENT_NAME"
    echo ""
done

echo "=== ALL FINETUNE EXPERIMENTS COMPLETED ==="