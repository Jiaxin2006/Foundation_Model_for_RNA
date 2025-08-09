import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取 TSV 文件
file_path = "/u/yfang4/projects/jiaxin/NAS-for-Bio/data/UTR/GSM3130435_egfp_unmod_1.csv"
df = pd.read_csv(file_path)

print(df.columns.tolist())

# 检查包含 MRL 的列是哪一个，这里我们根据你提供的图像，用 `rl` 列
mrl_column = "rl"
sequence_column = "utr"

# 计算 MRL 的分位数，确定前 25% 和后 25%
q_low = df[mrl_column].quantile(0.25)
q_high = df[mrl_column].quantile(0.75)

# 筛选前 25% 和后 25% 的序列
df_low = df[df[mrl_column] <= q_low].copy()
df_high = df[df[mrl_column] >= q_high].copy()

# 添加标签
df_low["label"] = 0
df_high["label"] = 1

# 合并两类数据
df_binary = pd.concat([df_low, df_high], ignore_index=True)

print(len(df_binary))

# 保证每类样本数一致
min_class_size = min(len(df_low), len(df_high))
df_balanced = pd.concat([
    df_low.sample(min_class_size, random_state=42),
    df_high.sample(min_class_size, random_state=42)
])

# 打乱顺序
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 拆分 train / val / test（60%/20%/20%），保持标签均衡
X = df_balanced[[sequence_column]]
y = df_balanced["label"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

# 合并回 DataFrame
train_df = X_train.copy()
train_df["label"] = y_train.values

val_df = X_val.copy()
val_df["label"] = y_val.values

test_df = X_test.copy()
test_df["label"] = y_test.values

# 重命名列
train_df = train_df.rename(columns={"utr": "sequence"})
val_df   = val_df.rename(columns={"utr": "sequence"})
test_df  = test_df.rename(columns={"utr": "sequence"})

# 保存 CSV 文件
train_path = "/u/yfang4/projects/jiaxin/NAS-for-Bio/data/UTR/train.csv"
val_path = "/u/yfang4/projects/jiaxin/NAS-for-Bio/data/UTR/dev.csv"
test_path = "/u/yfang4/projects/jiaxin/NAS-for-Bio/data/UTR/test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

