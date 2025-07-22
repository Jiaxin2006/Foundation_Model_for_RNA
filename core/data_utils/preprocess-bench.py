import os
import csv
import json
from pathlib import Path

# 每个任务对应的目录
TASK_PATHS = {
    0: '/projects/slmreasoning/yifang/datasets/GUE/Human/tf/0',
    1: '/projects/slmreasoning/yifang/datasets/GUE/Human/tf/1',
    2: '/projects/slmreasoning/yifang/datasets/GUE/Human/tf/2',
    3: '/projects/slmreasoning/yifang/datasets/GUE/Human/tf/3',
    4: '/projects/slmreasoning/yifang/datasets/GUE/Human/tf/4',
    5: '/projects/slmreasoning/yifang/datasets/GUE/Human/prom/prom_core_all',
    6: '/projects/slmreasoning/yifang/datasets/GUE/Human/prom/prom_core_notata',
    7: '/projects/slmreasoning/yifang/datasets/GUE/Human/prom/prom_core_tata',
    8: '/projects/slmreasoning/yifang/datasets/GUE/Human/prom/prom_300_all',
    9: '/projects/slmreasoning/yifang/datasets/GUE/Human/prom/prom_300_notata',
    10: '/projects/slmreasoning/yifang/datasets/GUE/Human/prom/prom_300_tata',
    11: '/projects/slmreasoning/yifang/datasets/GUE/Human/splice/reconstructed'
}

def convert_csv_to_jsonl(task_index):
    dataset_dir = Path(TASK_PATHS[task_index])
    csv_file = dataset_dir / "train.csv"
    jsonl_file = dataset_dir / "train.jsonl"

    if not csv_file.exists():
        print(f"CSV file not found for task {task_index}: {csv_file}")
        return

    print(f"Converting task {task_index} -> {jsonl_file}")

    with open(csv_file, "r") as f_in, open(jsonl_file, "w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            sequence = row["sequence"].upper()
            json_obj = {"text": sequence}
            f_out.write(json.dumps(json_obj) + "\n")

    print(f"Saved JSONL for task {task_index}: {jsonl_file}")

# 主程序：对所有 task_index 进行处理
for task_index in TASK_PATHS:
    convert_csv_to_jsonl(task_index)
