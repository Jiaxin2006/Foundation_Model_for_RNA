# -*- coding: utf-8 -*-
"""
绘图脚本（已适配 results/{exp_name}/{method}_results.csv 结构）
用法：python plot_results.py
注意：根据需要修改 base_path、use_max、figure_dir
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ====================
# 配置（根据实际调整）
# ====================
base_path = "/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/results"  # ← 改成你的根目录
figure_dir = os.path.join("/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio", "figures-new")
os.makedirs(figure_dir, exist_ok=True)

# True：跨 path/model 同 epoch 取数值列的最大值（你偏好的方式）
# False：取均值（保留用于比较）
use_max = True

# 任务-指标映射（与你原代码保持一致）
task_index_name_map = {
    0: "5'UTR MLR Prediction",
    1: "RNA Clustering Prediction",
    2: "Secondary Structure Alignment",
    # 其他任务如果有请补进来或取消注释
}
# 反向映射：任务名 -> index
task_name_to_index = {v: k for k, v in task_index_name_map.items()}

# 每个任务关注的主要指标（脚本会依据这些指标绘图）
task_metrics = {
    "5'UTR MLR Prediction": ["acc"],
    "RNA Clustering Prediction": ["ARI", "Homogeneity", "Completeness"],
    "Secondary Structure Alignment": ["F1", "SEN", "PPV"]
}
# RNA 时间列（如果存在）
time_metrics_rna = ["embed_time", "cluster_time"]

# 可选基线模型（如果长度与任务数不匹配会智能跳过）
# extra_models = {
#     "DNABERT-kmer3": [67.95, 70.90, 60.51],
#     "DNABERT2-bpe": [71.99, 76.06, 66.52],
#     "VQDNA-HRQ": [72.48, 76.43, 66.85]
# }

# ====================
# 1) 自动读取 results/*/*_results.csv （采用你给出的读取逻辑）
# ====================
all_data = []
for experiment_name in os.listdir(base_path):
    experiment_path = os.path.join(base_path, experiment_name)
    if not os.path.isdir(experiment_path):
        continue
    for csv_file in os.listdir(experiment_path):
        if not csv_file.endswith(".csv"):
            continue
        strategy = os.path.splitext(csv_file)[0].replace("_results", "")
        csv_path = os.path.join(experiment_path, csv_file)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"跳过无法读取的文件: {csv_path}，错误：{e}")
            continue
        df["training_strategy"] = strategy
        df["experiment_name"] = experiment_name
        all_data.append(df)

if len(all_data) == 0:
    raise SystemExit(f"在 {base_path} 未发现任何结果 CSV。请确认路径是否正确。")

full_df = pd.concat(all_data, ignore_index=True)
print(f"读取到 {len(full_df)} 条记录，来自 {len(all_data)} 个文件。")

# ====================
# 2) 从 experiment_name 拆出 architecture 与 tokenizer（与你原代码一致）
# ====================
# 期望 experiment_name 格式： "<architecture>-<tokenizer>"（只按第一个 '-' 拆分）
full_df[["architecture", "tokenizer"]] = full_df["experiment_name"].str.split("-", n=1, expand=True)

# ====================
# 3) 处理 task_index（支持两种情况：原来就是数字，或是任务名称字符串）
#    将 task_index 统一变为 int（任务编号）
# ====================
# 先尝试把字符串任务名映射为索引；若行内容已是数字则保留数字
mapped = full_df["task_index"].map(task_name_to_index)  # 对 task 名称做映射，失败则为 NaN
# 把那些原来已经是数字字符串或数字的，尝试转为数字并用它们补上
fallback_numeric = pd.to_numeric(full_df["task_index"], errors="coerce")
full_df["task_index"] = mapped.where(mapped.notnull(), fallback_numeric)

# 丢弃仍然无法识别的行（如果有）
bad_task_rows = full_df["task_index"].isnull().sum()
if bad_task_rows > 0:
    print(f"警告：有 {bad_task_rows} 条记录 task_index 无法识别（既不是已知任务名，也不是数字），这些行将被丢弃。")
    full_df = full_df[full_df["task_index"].notnull()]

# 转为整型索引
full_df["task_index"] = full_df["task_index"].astype(int)

# ====================
# 4) 统一 epoch 列名与类型 （原数据里是 epoch_num）
# ====================
if "epoch_num" not in full_df.columns:
    if "epoch" in full_df.columns:
        full_df["epoch_num"] = full_df["epoch"]
    else:
        # 如果不存在 epoch 列，创建默认值 1
        full_df["epoch_num"] = 1
full_df["epoch_num"] = pd.to_numeric(full_df["epoch_num"], errors="coerce").fillna(1).astype(int)

# 确保存在 model_name 列（如果你数据中列名不同请替换）
if "model_name" not in full_df.columns:
    if "model" in full_df.columns:
        full_df["model_name"] = full_df["model"]
    else:
        raise SystemExit("错误：输入 CSV 中缺少 'model_name' 或 'model' 列，请检查数据格式。")

# ====================
# 5) 聚合函数（按 model_name + epoch_num 聚合，支持 max/mean）
# ====================
def aggregate_model_epoch(task_df, use_max=True):
    """
    返回按 model_name & epoch_num 聚合后的数值表。
    use_max=True -> numeric 列取 max；False -> 取 mean
    """
    if use_max:
        # numeric_only=True 保证只对数值列做聚合
        return task_df.groupby(["model_name", "epoch_num"], as_index=False).max(numeric_only=True)
    else:
        return task_df.groupby(["model_name", "epoch_num"], as_index=False).mean(numeric_only=True)

# ====================
# 6) 常用绘图函数（训练曲线、柱状最终比较、箱线、时间曲线、相关矩阵、直方图）
#    注意：均使用 model_name / epoch_num / task_df
# ====================
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

def plot_training_curves(agg_df, metrics, task_name):
    for metric in metrics:
        if metric not in agg_df.columns:
            print(f"跳过 {task_name} 的 metric={metric}（列不存在）")
            continue
        plt.figure(figsize=(8,4.5))
        for model in sorted(agg_df["model_name"].unique()):
            mdf = agg_df[agg_df["model_name"] == model].sort_values("epoch_num")
            plt.plot(mdf["epoch_num"], mdf[metric], label=model, marker="o")
        plt.xlabel("epoch_num")
        plt.ylabel(metric)
        plt.title(f"{task_name} - {metric} training curve ({'max' if use_max else 'mean'})")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
        plt.tight_layout()
        save_path = os.path.join(figure_dir, f"{task_name.replace(' ', '_')}_{metric}_curve.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"保存：{save_path}")

def plot_final_bar(task_df, metrics, task_name):
    # 先按 model+epoch 聚合，再取每个 model 的最大 epoch 作为“最终表现”
    agg = aggregate_model_epoch(task_df, use_max=use_max)
    if agg.empty:
        return
    # 找每个 model 的最大 epoch 行
    last_idx = agg.groupby("model_name")["epoch_num"].idxmax()
    final_df = agg.loc[last_idx].copy()
    # 以第一个 metric 排序作图（若不存在则跳过）
    primary = metrics[0]
    if primary not in final_df.columns:
        print(f"{task_name} 无主指标 {primary}，跳过 final bar")
        return
    final_df = final_df.sort_values(primary, ascending=False)
    plt.figure(figsize=(8,4))
    sns.barplot(x="model_name", y=primary, data=final_df)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{task_name} - Final {primary} ({'Max' if use_max else 'Mean'})")
    plt.tight_layout()
    save_path = os.path.join(figure_dir, f"{task_name.replace(' ', '_')}_final_{primary}_bar.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存：{save_path}")

def plot_box(task_df, metrics, task_name):
    for metric in metrics:
        if metric not in task_df.columns:
            print(f"跳过箱线图：{task_name} 的 {metric} 不存在")
            continue
        plt.figure(figsize=(12,6))
        sns.boxplot(x="model_name", y=metric, data=task_df)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{task_name} - {metric} Distribution ({'Max' if use_max else 'Mean'})")
        plt.tight_layout()
        save_path = os.path.join(figure_dir, f"{task_name.replace(' ', '_')}_{metric}_box.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"保存：{save_path}")

def plot_time_curves(agg_df, task_name):
    for metric in time_metrics_rna:
        if metric not in agg_df.columns:
            continue
        plt.figure(figsize=(10,6))
        for model in sorted(agg_df["model_name"].unique()):
            mdf = agg_df[agg_df["model_name"] == model].sort_values("epoch_num")
            plt.plot(mdf["epoch_num"], mdf[metric], label=model, marker="o")
        plt.xlabel("epoch_num")
        plt.ylabel(metric)
        plt.title(f"{task_name} - {metric} Curve")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        plt.tight_layout()
        save_path = os.path.join(figure_dir, f"{task_name.replace(' ', '_')}_{metric}_curve.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"保存：{save_path}")

def plot_corr_heatmap(task_df, task_name):
    numeric = task_df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        print(f"{task_name} 无足够数值列用于相关性矩阵，跳过")
        return
    corr = numeric.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.3)
    plt.title(f"{task_name} - 指标相关性")
    plt.tight_layout()
    save_path = os.path.join(figure_dir, f"{task_name.replace(' ', '_')}_corr.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存：{save_path}")

def plot_histograms(task_df, metrics, task_name):
    for metric in metrics:
        if metric not in task_df.columns:
            continue
        plt.figure(figsize=(6,4))
        sns.histplot(task_df[metric].dropna(), kde=True)
        plt.title(f"{task_name} - {metric} 直方图")
        plt.tight_layout()
        save_path = os.path.join(figure_dir, f"{task_name.replace(' ', '_')}_{metric}_hist.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"保存：{save_path}")

# ====================
# 7) 针对原始代码中（含被注释掉的）函数保留热力图 / tokenizer/architecture/strategy 比较（可选）
#    便于你沿用原有分析流程
# ====================
def plot_within_one_arc_heatmap(
    df,
    architecture,
    tokenizer,
    training_strategy,
    epoch=1,
    base_save_path=None
):
    filtered_df = df[
        (df["architecture"] == architecture) &
        (df["epoch_num"] == epoch) &
        (df["training_strategy"] == training_strategy)
    ].copy()
    if filtered_df.empty:
        print(f"heatmap: 在 arch={architecture}, tokenizer={tokenizer}, strategy={training_strategy}, epoch={epoch} 未找到数据")
        return
    # pivot: index model_name, columns task_index, values acc
    if "acc" not in filtered_df.columns:
        print("heatmap: 数据中不存在 'acc' 列，跳过 heatmap")
        return
    heatmap_data = filtered_df.pivot_table(index="model_name", columns="task_index", values="acc")
    heatmap_data = heatmap_data.reindex(heatmap_data.mean(axis=1).sort_values(ascending=False).index)
    plt.figure(figsize=(12, max(4, 0.3 * len(heatmap_data))))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, linecolor='black', cbar_kws={'label': 'Accuracy (%)'})
    plt.title(f"Model Performance Heatmap for Arch: {architecture} | Strategy: {training_strategy} | epoch={epoch}")
    plt.xlabel("Task Index")
    plt.ylabel("Model Name")
    plt.tight_layout()
    save_path = os.path.join(base_save_path or figure_dir, f"within-{architecture}-{tokenizer}-{training_strategy}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存 heatmap: {save_path}")

# ====================
# 8) 主流程：按预定义 task_metrics 逐任务绘图
# ====================
for task_name, metrics in task_metrics.items():
    # 确认此任务在数据中是否存在（通过索引）
    if task_name not in task_name_to_index:
        print(f"警告：脚本未配置任务名 {task_name} 的索引映射，跳过该任务")
        continue
    idx = task_name_to_index[task_name]
    task_df = full_df[full_df["task_index"] == idx].copy()
    if task_df.empty:
        print(f"数据中无任务 {task_name}（索引 {idx}），跳过")
        continue

    print(f"\n=== 绘制任务：{task_name} (records={len(task_df)}) ===")
    # 聚合后用于训练曲线 & 时间曲线
    agg_df = aggregate_model_epoch(task_df, use_max=use_max)

    # 1) 训练曲线（每个 model 一条线）
    plot_training_curves(agg_df, metrics, task_name)

    # 2) 最终 epoch 柱状对比
    plot_final_bar(task_df, metrics, task_name)

    # 3) 箱线（不同 path/model 的分布）
    plot_box(task_df, metrics, task_name)

    # 4) RNA 专属：绘制 embed_time / cluster_time
    if task_name == "RNA Clustering Prediction":
        plot_time_curves(agg_df, task_name)

    # 5) 指标相关性热力图
    plot_corr_heatmap(task_df, task_name)

    # 6) 直方图
    plot_histograms(task_df, metrics, task_name)

print("\n全部绘图完成，图片已保存在：", figure_dir)

# ====================
# 额外示例：调用原来注释掉的 heatmap（如果需要请取消下面调用）
# ====================
# 示例（按你原本脚本的调用）：
plot_within_one_arc_heatmap(full_df,
    architecture="mix",
    tokenizer="kmer1",
    training_strategy="only-ft",
    epoch=1,
    base_save_path=figure_dir
)
