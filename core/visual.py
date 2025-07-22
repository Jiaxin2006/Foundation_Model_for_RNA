import pandas as pd
import os
import numpy as np
import json
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from core.data_utils.mytokenizers import MyTokenizer

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
extra_models = {
    "DNABERT-kmer3": [67.95, 70.90, 60.51, 53.03, 69.76, 70.92, 69.82, 78.15, 90.44, 93.61, 69.83, 84.14],
    "DNABERT2-bpe": [71.99, 76.06, 66.52, 58.54, 77.43, 77.43, 68.04, 74.17, 86.77, 94.27, 71.59, 84.99],
    "VQDNA-HRQ": [72.48, 76.43, 66.85, 58.92, 78.10, 71.02, 70.58, 78.50, 90.75, 94.48, 74.52, 89.53]
}

def plot_tokenizer_vs_task_acc(
    df,
    architecture,
    training_strategy,
    epoch=1,
    base_save_path=None
):
    import matplotlib.pyplot as plt
    import os

    df_filtered = df[
        (df["architecture"] == architecture) &
        (df["training_strategy"] == training_strategy) &
        (df["epoch_num"] == epoch)
    ]

    agg_df = df_filtered.groupby(
        ["training_strategy", "tokenizer", "task_index"]
    )["acc"].max().reset_index()

    linestyle = "-"
    tokenizer_colors = {
        tokenizer: color
        for tokenizer, color in zip(
            agg_df["tokenizer"].unique(),
            plt.cm.tab10.colors
        )
    }

    plt.figure(figsize=(12, 6))

    for tokenizer, group in agg_df.groupby("tokenizer"):
        group_sorted = group.sort_values("task_index")
        color = tokenizer_colors.get(tokenizer, "gray")
        label = f"{training_strategy} + {tokenizer}"
        plt.plot(
            group_sorted["task_index"],
            group_sorted["acc"],
            label=label,
            linestyle=linestyle,
            color=color,
            marker='o'
        )

    task_indices = list(range(12))
    colors = plt.cm.tab10.colors
    for i, (model_name, acc_values) in enumerate(extra_models.items()):
        plt.plot(
            task_indices,
            acc_values,
            label=model_name,
            linestyle="--",
            color=colors[(i + 5) % len(colors)],
            marker='x'
        )

    plt.xlabel("Task Index")
    plt.ylabel("Max Accuracy at Epoch 1")
    plt.title(f"Tokenizer Comparison for architecture={architecture}, strategy={training_strategy}")
    plt.grid(True)
    plt.legend(title="Training Strategy + Tokenizer", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    save_path = base_save_path + f"{architecture}-{training_strategy}-tokenizers.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=800)
    plt.close()
    print(f"图像已保存到：{save_path}")

    

def plot_architecture_vs_task_acc(
    df,
    tokenizer,
    training_strategy,
    epoch=1,
    base_save_path=None
):
    # 过滤数据
    df_filtered = df[
        (df["tokenizer"] == tokenizer) &
        (df["training_strategy"] == training_strategy) &
        (df["epoch_num"] == epoch)
    ]
    #print("过滤后的数据行数:", len(df_filtered))

    # 聚合
    agg_df = df_filtered.groupby(
        ["architecture", "task_index"]
    )["acc"].max().reset_index()
    #print("聚合后数据预览:")
    #print(agg_df.head())

    linestyle = "-"
    architecture_colors = {
        arch: color
        for arch, color in zip(
            agg_df["architecture"].unique(),
            plt.cm.tab10.colors
        )
    }

    plt.figure(figsize=(12, 6))

    for arch, group in agg_df.groupby("architecture"):
        group_sorted = group.sort_values("task_index")
        color = architecture_colors.get(arch, "gray")
        label = f"{arch}"
        plt.plot(
            group_sorted["task_index"],
            group_sorted["acc"],
            label=label,
            linestyle=linestyle,
            color=color,
            marker='o'
        )

    task_indices = list(range(12))
    colors = plt.cm.tab10.colors
    for i, (model_name, acc_values) in enumerate(extra_models.items()):
        plt.plot(
            task_indices,
            acc_values,
            label=model_name,
            linestyle="--",
            color=colors[(i + 5) % len(colors)],
            marker='x'
        )

    plt.xlabel("Task Index")
    plt.ylabel("Max Accuracy at Epoch 1")
    plt.title(f"Architecture Comparison for tokenizer={tokenizer}, strategy={training_strategy}")
    plt.grid(True)
    plt.legend(title="Architecture + Baselines", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    save_path = base_save_path + f"{tokenizer}-{training_strategy}-architectures.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=800)
    plt.close()
    print(f"图像已保存到：{save_path}")

def plot_strategy_vs_task_acc(
    df,
    tokenizer,
    architecture,
    epoch=1,
    base_save_path=None,
    extra_models=None  # 保留这个参数以便额外模型对比
):
    # 过滤数据
    df_filtered = df[
        (df["tokenizer"] == tokenizer) &
        (df["architecture"] == architecture) &
        (df["epoch_num"] == epoch)
    ]

    # 聚合：同一个 training_strategy 下取每个 task 的最高准确率
    agg_df = df_filtered.groupby(
        ["training_strategy", "task_index"]
    )["acc"].max().reset_index()

    linestyle = "-"
    strategy_colors = {
        strat: color
        for strat, color in zip(
            agg_df["training_strategy"].unique(),
            plt.cm.tab10.colors
        )
    }

    plt.figure(figsize=(12, 6))

    for strat, group in agg_df.groupby("training_strategy"):
        group_sorted = group.sort_values("task_index")
        color = strategy_colors.get(strat, "gray")
        label = f"{strat}"
        plt.plot(
            group_sorted["task_index"],
            group_sorted["acc"],
            label=label,
            linestyle=linestyle,
            color=color,
            marker='o'
        )

    task_indices = list(range(12))
    colors = plt.cm.tab10.colors
    if extra_models:
        for i, (model_name, acc_values) in enumerate(extra_models.items()):
            plt.plot(
                task_indices,
                acc_values,
                label=model_name,
                linestyle="--",
                color=colors[(i + 5) % len(colors)],
                marker='x'
            )

    plt.xlabel("Task Index")
    plt.ylabel("Max Accuracy at Epoch 1")
    plt.title(f"Strategy Comparison for tokenizer={tokenizer}, architecture={architecture}")
    plt.grid(True)
    plt.legend(title="Training Strategies + Baselines", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    save_path = base_save_path + f"{tokenizer}-{architecture}-strategies.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=800)
    plt.close()
    print(f"图像已保存到：{save_path}")



def analyze_tokenizer(name):
    print(f"Analyzing tokenizer: {name}")
    tokenizer = MyTokenizer(name)

    special_tokens = {"[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"}  # 特殊token列表

    if "bpe" in name:
        vocab = tokenizer.tokenizer.get_vocab()
        tokens = list(vocab.keys())
    elif "kmer" in name:
        tokens = list(tokenizer.tokenizer.vocab.keys())
    else:
        raise ValueError("Unsupported tokenizer type")

    # 长度处理：特殊 token 视为长度1
    token_lengths = [1 if token in special_tokens else len(token) for token in tokens]

    print(f" - Vocab size: {len(tokens)}")
    print(f" - Avg token length: {sum(token_lengths)/len(token_lengths):.2f}")
    print(f" - Max token length: {max(token_lengths)}")
    print(f" - Min token length: {min(token_lengths)}")

    return {
        "name": name,
        "vocab_size": len(tokens),
        "token_lengths": token_lengths,
    }


def compute_avg_tokenized_length(tokenizer_names, sequences):
    results = []

    for name in tokenizer_names:
        tokenizer = MyTokenizer(name)
        token_counts = []

        for seq in sequences:
            encoding = tokenizer.encode(seq)
            token_counts.append(len(encoding.tokens))  # 或 len(encoding.ids)

        avg_len = sum(token_counts) / len(token_counts)
        results.append({
            "name": name,
            "avg_token_count": avg_len,
            "num_sequences": len(sequences)
        })

        print(f"{name}: Avg tokenized length = {avg_len:.2f} (over {len(sequences)} sequences)")

    return results




def plot_within_one_arc_heatmap(
    df, 
    architecture,
    tokenizer,
    training_strategy,
    epoch=1,
    base_save_path=None):
    """
    使用热力图展示在特定架构、epoch 和训练策略下，
    不同模型在不同任务上的表现。

    参数:
    df (pd.DataFrame): 包含所有实验数据的DataFrame。
                            应包含 'architecture', 'epoch_num', 'training_strategy',
                            'task_index', 'model_name', 'acc' 列。
    architecture (str): 要筛选的架构名称 (例如 'cnn').
    epoch (int): 要筛选的 epoch 数量。
    training_strategy (str): 要筛选的训练策略 (例如 'random_init').
    """

    # 1. 筛选数据
    filtered_df = df[
        (df["architecture"] == architecture) &
        (df["epoch_num"] == epoch) &
        (df["training_strategy"] == training_strategy)
    ].copy() # 使用 .copy() 避免SettingWithCopyWarning

    if filtered_df.empty:
        print(f"在 architecture='{architecture}', epoch={epoch}, training_strategy='{training_strategy}' 下没有找到数据。")
        return

    # 2. 准备热力图数据
    # 使用 pivot_table 将数据重塑为热力图所需的格式
    # index: 行 - 模型的名称
    # columns: 列 - 任务的名称
    # values: 单元格中的值 - 准确率
    heatmap_data = filtered_df.pivot_table(index="model_name", columns="task_index", values="acc")

    # 可选：根据模型的平均准确率对行进行排序，这样表现好的模型会聚在一起
    # 您也可以根据特定任务的准确率排序，或不排序
    heatmap_data = heatmap_data.reindex(heatmap_data.mean(axis=1).sort_values(ascending=False).index)


    # 3. 绘制热力图
    plt.figure(figsize=(15, 12)) # 调整图表大小以适应更多模型和任务
    sns.heatmap(
        heatmap_data,
        annot=True,     # 在每个单元格中显示数值
        fmt=".1f",      # 格式化数值，保留一位小数
        cmap="viridis", # 选择颜色方案。'viridis' 通常是从紫色到黄色，数值越大颜色越亮/黄
                        # 其他好的选择包括 'YlGnBu' (黄绿蓝) 或 'coolwarm' (用于对比正负值)
        linewidths=.5,  # 添加单元格之间的线条，增加视觉分隔
        linecolor='black', # 线条颜色
        cbar_kws={'label': 'Accuracy (%)'} # 颜色条的标签
    )

    plt.title(f"Model Performance Heatmap for Architecture: {architecture}\nEpoch: {epoch}, Strategy: {training_strategy}", fontsize=16)
    plt.xlabel("Task Index", fontsize=12)
    plt.ylabel("Model Name", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10) # 旋转X轴标签，防止重叠，并右对齐
    plt.yticks(rotation=0, fontsize=10) # Y轴标签保持水平

    plt.tight_layout() # 自动调整布局，防止标签被裁剪
    save_path = base_save_path + f"within-{architecture}-{tokenizer}-{training_strategy}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=800)
    plt.close()
    print(f"图像已保存到：{save_path}")





if __name__ == "__main__":
    all_data = []
    base_path = "/projects/slmreasoning/yifang/results"

    for experiment_name in os.listdir(base_path):
        experiment_path = os.path.join(base_path, experiment_name)
        if not os.path.isdir(experiment_path):
            continue
        for csv_file in os.listdir(experiment_path):
            if csv_file.endswith(".csv"):
                strategy = os.path.splitext(csv_file)[0].replace("_results", "")
                df = pd.read_csv(os.path.join(experiment_path, csv_file))
                df["training_strategy"] = strategy
                df["experiment_name"] = experiment_name
                all_data.append(df)

    full_df = pd.concat(all_data)
    full_df[["architecture", "tokenizer"]] = full_df["experiment_name"].str.split("-", n=1, expand=True)

    task_name_to_index = {v: k for k, v in task_index_name_map.items()}
    full_df["task_index"] = full_df["task_index"].map(task_name_to_index)
    
    #print(full_df.head())

    '''
    # 比较不同tokenizer的效果, 通过折线图展示cnn和transformer不同的architecture下, 不同的tokenizer的效果
    plot_tokenizer_vs_task_acc(
        full_df,
        architecture="hyena",
        training_strategy="only-ft",
        epoch=1,
        base_save_path="/projects/slmreasoning/yifang/results/figures/"
    )
    '''
    '''
    ###tokenizer 分析
    datapath = "/projects/slmreasoning/yifang/datasets/GRCh38/processed_data/filtered_sentences.jsonl"
    jsonl_file = pathlib.Path(datapath)
    sequences = []
    with open(jsonl_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            sequences.append(obj["text"].upper())
    sequences = sequences[:100000]

    tokenizer_names = ["bpe-512", "kmer-6", "kmer-1", "bpe-4096"]
    results = compute_avg_tokenized_length(tokenizer_names, sequences)
    
    results = [analyze_tokenizer(name) for name in tokenizer_names]

    plt.figure(figsize=(10, 6))

    for result in results:
        # 计算 histogram 数据（密度）
        counts, bin_edges = np.histogram(
            result["token_lengths"],
            bins=range(1, max(result["token_lengths"]) + 2),
            density=True
        )
        # 获取 bin 中心点作为 x 坐标
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 用 plt.plot() 绘制折线图
        plt.plot(
            bin_centers,
            counts,
            label=f'{result["name"]} (Vocab: {result["vocab_size"]})',
            linewidth=2,
            marker="o"  # 可选，加上小圆点更清晰
        )

    plt.title("Token Length Distribution Comparison")
    plt.xlabel("Token Length (nucleotides)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/projects/slmreasoning/yifang/results/figures/tokenzier_analysis.png", dpi=800)
    plt.show()
    '''

    # 比较不同architecture的效果, 在同一种tokenizer和training_strategy下, 比较每个任务的acc
    '''
    plot_architecture_vs_task_acc(full_df,
        tokenizer="bpe512",
        training_strategy="only-ft",
        epoch=1,
        base_save_path="/projects/slmreasoning/yifang/results/figures/"
    )
    '''

    ## 比较同一个架构内，比如cnn内，不同模型的表现

    plot_within_one_arc_heatmap(full_df,
        architecture="mix",
        tokenizer="kmer1",
        training_strategy="only-ft",
        epoch=1,
        base_save_path="/projects/slmreasoning/yifang/results/figures/"
    )

    '''
    ## 比较同一个架构和同一个tokenizer内，比如cnn kmer1内，不同training strategy的表现
    plot_strategy_vs_task_acc(full_df,
        tokenizer="kmer1",
        architecture="lstm",
        epoch=1,
        base_save_path="/projects/slmreasoning/yifang/results/figures/"
    )
    '''