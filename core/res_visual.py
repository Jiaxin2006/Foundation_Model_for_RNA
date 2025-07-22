import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    '''
    base_path = "/projects/slmreasoning/yifang/results/"

    # 文件前缀
    prefixes = [
        "con-con", "con-ft", "con-mask",
        "mask-con", "mask-ft", "mask-mask",
        "only-con", "only-ft", "only-mask"
    ]


    task_index_name_map = {
        0: "Transcription Factor Prediction-0",
        1: "Transcription Factor Prediction-1",
        2: "Transcription Factor Prediction-2",
        3: "Transcription Factor Prediction-3",
        4: "Transcription Factor Prediction-4",
        5: "Core Prompter Detection-all",
        6: "Core Prompter Detection-notata",
        7: "Core Prompter Detection-tata",
        8: "Prompter Detection-all",
        9: "Prompter Detection-notata",
        10: "Prompter Detection-tata",
        11: "Splice Site Detection"
    }

    # 用于存储所有DataFrame的字典
    dfs = {}

    # 读取每个文件
    for prefix in prefixes:
        file_name = f"{prefix}_results.csv"
        file_path = os.path.join(base_path, file_name)
        dfs[prefix] = pd.read_csv(file_path)
    
    
    def plot_avg_acc_vs_epoch(name_list, epoch_list):
        plt.figure(figsize=(10, 8))
        
        for name in name_list:
            df = dfs[name]
            df_filtered = df[df["epoch_num"].isin(epoch_list)]
            df_avg = df_filtered.groupby("epoch_num")["acc"].mean().reset_index()
            plt.plot(df_avg["epoch_num"], df_avg["acc"], marker='o', label=name)

        plt.xlim(min(epoch_list) - 1, max(epoch_list) + 1)
        plt.xlabel("Epoch Number")
        plt.ylabel("Average Accuracy")
        plt.title("Average Accuracy vs Epoch Number")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        save_path = "/projects/slmreasoning/yifang/results/figures/acc-epoch.png"
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.close()
        logger.info(f"{save_path} DONE.")

    name_list = ["con-ft", "mask-ft", "only-con", "only-mask"]
    epoch_list = [1, 3, 5]
    plot_avg_acc_vs_epoch(name_list, epoch_list)
    '''
    '''
    ### Pretrain Data Size
    previous_only_con_path = "/projects/slmreasoning/yifang/results/previous results/only-conpre_results.csv"
    previous_con_ft_path = "/projects/slmreasoning/yifang/results/previous results/conpre-ft_results.csv"

    df_previous_only_con = pd.read_csv(previous_only_con_path)
    df_previous_con_ft = pd.read_csv(previous_con_ft_path)

    df_only_con = dfs["only-con"].copy()
    df_con_ft = dfs["con-ft"].copy()

    def compare_pretrain_data_size(num_of_epoch):
        plot_data = {
            "only-con (30%)": df_only_con,
            "con-ft (30%)": df_con_ft,
            "only-con (100%)": df_previous_only_con,
            "con-ft (100%)": df_previous_con_ft
        }

        # 存储横轴和纵轴
        task_ids = sorted(task_index_name_map.keys())
        plt.figure(figsize=(10, 6))

        for label, df in plot_data.items():
            # 筛选 epoch 对应的数据
            df_filtered = df[df["epoch_num"] == num_of_epoch]

            # 计算每个 task 的平均 acc
            avg_acc_per_task = []
            for task_id in task_ids:
                task_name = task_index_name_map[task_id]
                acc_values = df_filtered[df_filtered["task_index"] == task_name]["acc"]
                avg_acc = acc_values.mean() if not acc_values.empty else None
                avg_acc_per_task.append(avg_acc)

            plt.plot(task_ids, avg_acc_per_task, marker='o', label=label)

        plt.xticks(task_ids, task_ids)
        plt.xlabel("Task Index")
        plt.ylabel("Average Accuracy")
        plt.title(f"Average Accuracy per Task (epoch {num_of_epoch})")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("/projects/slmreasoning/yifang/results/figures/pretrain_data_size.png", dpi=1200, bbox_inches='tight')
        plt.close()
        logger.info("/projects/slmreasoning/yifang/results/figures/pretrain_data_size.png Done")

    compare_pretrain_data_size(1)
    '''

    '''
    ### ACC-Num_layers
    def plot_avg_acc_vs_num_layers(name_list, selected_epochs):
        plt.figure(figsize=(10, 8))

        for name in name_list:
            df = dfs[name]
            df_filtered = df[df["epoch_num"].isin(selected_epochs)].copy()

            # 提取 num_of_layers
            def get_num_layers(model_name):
                try:
                    params = ast.literal_eval(model_name.split('_')[-1])
                    return len(params)
                except Exception as e:
                    print(f"Error parsing model_name: {model_name} -> {e}")
                    return None

            df_filtered['num_of_layers'] = df_filtered['model_name'].apply(get_num_layers)
            df_avg = df_filtered.groupby("num_of_layers")["acc"].mean().reset_index()

            plt.plot(df_avg["num_of_layers"], df_avg["acc"], marker='o', label=name)

        plt.xlabel("Number of Layers")
        plt.ylabel("Average Accuracy")
        plt.title(f"Average Accuracy vs Number of Layers (Epochs: {selected_epochs})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("/projects/slmreasoning/yifang/results/figures/acc-num_layers.png", dpi=1200, bbox_inches='tight')
        plt.close()
        logger.info(f"/projects/slmreasoning/yifang/results/figures/acc-num_layers.png Done")

    name_list = ["con-con", "mask-con", "con-mask", "mask-mask", "only-con", "only-mask"]
    selected_epochs = [1, 3, 5]
    plot_avg_acc_vs_num_layers(name_list, selected_epochs)
    '''
    '''


    ## comparison with 1 epoch pretraining


    dnabert2_base = [71.99, 76.06, 66.52, 58.54, 77.43, 69.37, 68.04, 74.17, 86.77, 94.27, 71.59, 84.99]
    dnabert2_fp = [69.12, 71.87, 62.96, 55.35, 74.94, 67.50, 69.53, 76.18, 88.31, 94.34, 68.79, 85.93]
    num_tasks = len(task_index_name_map)



    def compare_with_baseline(name_list, num_of_epoch, title):
        plt.figure(figsize=(14, 6))
        
        for name in name_list:
            df = dfs[name]
            best_accs = []
            for i in range(num_tasks):
                task_name = task_index_name_map[i]
                # 只保留当前任务和epoch的记录
                df_task = df[(df['task_index'] == task_name) & (df['epoch_num'] == num_of_epoch)]
                if not df_task.empty:
                    best_acc = df_task['acc'].max()
                else:
                    best_acc = float('nan')  # 缺失数据用NaN表示
                best_accs.append(best_acc)
            
            plt.plot(range(num_tasks), best_accs, marker='o', label=name)

        # 加入 baseline
        plt.plot(range(num_tasks), dnabert2_base, linestyle='--', marker='x', color='black', label='DNABERT2 Base')
        plt.plot(range(num_tasks), dnabert2_fp, linestyle='--', marker='s', color='gray', label='DNABERT2 FP')

        # 图表设置
        plt.xticks(range(num_tasks))
        plt.xlabel("Task Index")
        plt.ylabel("Best Accuracy")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"/projects/slmreasoning/yifang/results/figures/{title}.png", dpi=1200, bbox_inches='tight')
        logger.info(f"/projects/slmreasoning/yifang/results/figures/{title}.png  Done")

    ### Training Strategy Comparison
    num_of_epoch = 1
    title_1 = 'Without Supervise Tuning'
    name_list_1 = ['only-mask', 'only-con']
    compare_with_baseline(name_list_1, num_of_epoch, title_1)
    
    num_of_epoch = 1
    title_2 = 'With Supervise Tuning'
    name_list_2 = ['mask-ft', 'con-ft', 'only-ft']
    compare_with_baseline(name_list_2, num_of_epoch, title_2)

    num_of_epoch = 1
    title_3 = 'Contrastive Pretraining'
    name_list_3 = ['con-ft', 'only-con', 'con-mask', 'con-con']
    compare_with_baseline(name_list_3, num_of_epoch, title_3)

    num_of_epoch = 1
    title_4 = 'Mask Modeling Pretraining'
    name_list_4 = ['mask-ft', 'only-mask', 'mask-mask', 'mask-con']
    compare_with_baseline(name_list_4, num_of_epoch, title_4)
    
    '''
    '''
    # compare tokenizers

    task_index_name_map = {
        "Transcription Factor Prediction-0": 0,
        "Transcription Factor Prediction-1": 1,
        "Transcription Factor Prediction-2": 2,
        "Transcription Factor Prediction-3": 3,
        "Transcription Factor Prediction-4": 4,
        "Core Prompter Detection-all": 5,
        "Core Prompter Detection-notata": 6,
        "Core Prompter Detection-tata": 7,
        "Prompter Detection-all": 8,
        "Prompter Detection-notata": 9,
        "Prompter Detection-tata": 10,
        "Splice Site Detection": 11
    }
    index_task_name_map = {v: k for k, v in task_index_name_map.items()}

    base_path = "/projects/slmreasoning/yifang/results/"
    tokenizer_files = {
        "kmer1": [base_path + "base_experiment/only-ft_results.csv", base_path + "base_experiment/con-ft_results.csv"],
        "kmer3": [base_path + "base-kmer3/only-ft_results.csv", base_path + "base-kmer3/con-ft_results.csv"],
        "kmer6": [base_path + "base-kmer6/only-ft_results.csv", base_path + "base-kmer6/con-ft_results.csv"],
        "bpe512": [base_path + "base-bpe512/only-ft_results.csv", base_path + "base-bpe512/con-ft_results.csv"],
    }

    color_map = {
        "kmer1": "#1f77b4",
        "kmer3": "#ff7f0e",
        "kmer6": "#2ca02c",
        "bpe512": "#d62728"
    }

    tokenizer_results = {k: [0]*12 for k in tokenizer_files}

    for tokenizer, (only_ft_path, con_ft_path) in tokenizer_files.items():
        df_only = pd.read_csv(only_ft_path)
        df_con = pd.read_csv(con_ft_path)
        df = pd.concat([df_only, df_con])
        
        # 确保task_index列是字符串形式的任务名
        df = df[df['epoch_num'] == 1]
        df = df[df['task_index'].isin(task_index_name_map.keys())]  # 过滤不合法的行

        grouped = df.groupby('task_index')['acc'].mean()

        for task_name, acc in grouped.items():
            task_idx = task_index_name_map[task_name]
            tokenizer_results[tokenizer][task_idx] = acc

    # 绘图
    plt.figure(figsize=(14, 6))
    x = list(range(12))  # task indices

    for tokenizer, accs in tokenizer_results.items():
        plt.plot(x, accs, marker='o', label=tokenizer, color=color_map[tokenizer])

    plt.xticks(x, x)  # 用index作横坐标
    plt.xlabel("Task Index")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison Across Tasks (epoch_num=0)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"/projects/slmreasoning/yifang/results/figures/compare_tokenizers.png", dpi=1200, bbox_inches='tight')
    '''


    # architectures
    task_index_name_map = {
        "Transcription Factor Prediction-0": 0,
        "Transcription Factor Prediction-1": 1,
        "Transcription Factor Prediction-2": 2,
        "Transcription Factor Prediction-3": 3,
        "Transcription Factor Prediction-4": 4,
        "Core Prompter Detection-all": 5,
        "Core Prompter Detection-notata": 6,
        "Core Prompter Detection-tata": 7,
        "Prompter Detection-all": 8,
        "Prompter Detection-notata": 9,
        "Prompter Detection-tata": 10,
        "Splice Site Detection": 11
    }
    index_task_name_map = {v: k for k, v in task_index_name_map.items()}

    base_path = "/projects/slmreasoning/yifang/results/"
    tokenizer_files = {
        "base-bpe512": [base_path + "base-bpe512/only-ft_results.csv", base_path + "base_experiment/only-ft_results.csv"],
        "three-bpe512": [base_path + "three-bpe512/only-ft_results.csv", base_path + "base-kmer3/only-ft_results.csv"],
    }

    color_map = {
        "base-bpe512": "#1f77b4",
        "three-bpe512": "#ff7f0e",
    }

    tokenizer_results = {k: [0]*12 for k in tokenizer_files}

    for tokenizer, (only_ft_path, con_ft_path) in tokenizer_files.items():
        df_only = pd.read_csv(only_ft_path)
        df_con = pd.read_csv(con_ft_path)
        df = pd.concat([df_only, df_con])
        
        # 确保task_index列是字符串形式的任务名
        df = df[df['epoch_num'] == 1]
        df = df[df['task_index'].isin(task_index_name_map.keys())]  # 过滤不合法的行

        grouped = df.groupby('task_index')['acc'].mean()

        for task_name, acc in grouped.items():
            task_idx = task_index_name_map[task_name]
            tokenizer_results[tokenizer][task_idx] = acc

    # 绘图
    plt.figure(figsize=(14, 6))
    x = list(range(12))  # task indices

    for tokenizer, accs in tokenizer_results.items():
        plt.plot(x, accs, marker='o', label=tokenizer, color=color_map[tokenizer])

    plt.xticks(x, x)  # 用index作横坐标
    plt.xlabel("Task Index")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison Across Architectures")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"/projects/slmreasoning/yifang/results/figures/compare_modules.png", dpi=1200, bbox_inches='tight')