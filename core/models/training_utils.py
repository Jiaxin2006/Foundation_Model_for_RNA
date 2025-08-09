import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
# gai
import os
import time
from pathlib import Path
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
from core.data_utils.mytokenizers import MyTokenizer

from core.data_utils.dataset import DNAContrastiveDataset,FTDNADataset,DNAMaskDataset
from core.models.model import CLS_FixedModel
from core.models.utils import calculate_contrastive_loss
from core.models.modules import build_module
import random
from torch.utils.data import DataLoader
import logging
from transformers import get_linear_schedule_with_warmup 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_rna_clustering(data_dir):
    """
    从 data_dir 下的每个文件读取 RNA 序列，
    假设每个文件对应一个 family，文件名就是类别名。
    返回：
      sequences: List[str]，所有序列
      labels:    List[int]，对应的真是 family 索引
      label_names: List[str]，family 名称列表
    """
    sequences = []
    labels = []
    label_names = []
    for idx, fasta_path in enumerate(sorted(Path(data_dir).glob("*.fa*"))):
        label = fasta_path.stem
        label_names.append(label)
        for rec in SeqIO.parse(str(fasta_path), "fasta"):
            sequences.append(str(rec.seq))
            labels.append(idx)
    # print(sequences)
    # print(labels)
    # print(label_names)
    return sequences, labels, label_names


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_collate_fn(pad_idx):
    def collate_fn(batch):
        xs, labels, lengths = zip(*batch)
        target_max_len = max(lengths)

        padded_xs = []
        padded_labels = []
        attention_mask = torch.zeros(len(batch), target_max_len, dtype=torch.bool)

        for i, (x, label, length) in enumerate(zip(xs, labels, lengths)):
            if x.size(0) < target_max_len:
                padded_x = torch.full((target_max_len,), pad_idx, dtype=torch.long)
                padded_x[:x.size(0)] = x
            else:
                padded_x = x
            padded_xs.append(padded_x)

            attention_mask[i, :length] = True

        x_batch = torch.stack(padded_xs)
        label_batch = torch.stack(labels)

        return x_batch, label_batch, attention_mask
    return collate_fn



def get_continual_pretrain_dataset_dir(task_index):
    path_dict = {
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
    return path_dict[task_index]






def build_all_models(num_classes, model_space):
    """
    枚举 model_space 中所有 cnn_arch 架构，构造完整模型。
    返回: List of (arch_name, model)
    """
    all_models = []
    #logger.info(f"model_space.paths_candidates is {model_space.paths_candidates}")
    for arch_name, path in model_space.paths_candidates.items():
        model = CLS_FixedModel(num_classes=num_classes,
            embedding=model_space.embedding,
            path=path,
            mask_pred_head=getattr(model_space, "pred_head", None)
        )
        all_models.append((arch_name, model))

    return all_models

# gai
def align_train_epoch(model, device, train_loader, optimizer, scheduler):
    model.train()
    for idsA, idsB, contact in train_loader:
        idsA, idsB, contact = idsA.to(device), idsB.to(device), contact.to(device)
        optimizer.zero_grad()
        # 用 embedding 做打分矩阵
        emA = model.embedding(idsA)
        emB = model.embedding(idsB)
        score = torch.einsum("bid,bjd->bij", emA, emB)  # [B, L1, L2]
        loss = F.binary_cross_entropy_with_logits(score, contact)
        loss.backward()
        optimizer.step()
        scheduler.step()

def align_eval_epoch(model, device, val_loader):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for idsA, idsB, contact in val_loader:
            idsA, idsB, contact = idsA.to(device), idsB.to(device), contact.to(device)
            emA = model.embedding(idsA)   # [B, L1, d]
            emB = model.embedding(idsB)   # [B, L2, d]
            score = torch.einsum("bid,bjd->bij", emA, emB)  # [B, L1, L2]
            pred = (torch.sigmoid(score) > 0.5).int()
            contact = contact.int() 

            # 逐样本累加
            for b in range(contact.size(0)):
                c_true = contact[b]      # [L1, L2]
                c_pred = pred[b]         # [L1, L2]
                tp += (c_pred & c_true).sum().item()
                fp += (c_pred & (~c_true)).sum().item()
                fn += ((~c_pred) & c_true).sum().item()

    sen = tp / (tp + fn + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    f1  = 2 * sen * ppv / (sen + ppv + 1e-8)
    return f1, sen, ppv

# TODO: 写其他任务的训练
# 这里已经把预训练好的参数固定好成为一个单独的模型（也就是说对当前任务的finetune不会影响到其他任务）
def cls_train_epoch(model, device, train_loader, optimizer, scheduler):
    set_seed()
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, labels, attention_mask) in enumerate(train_loader):
        optimizer.zero_grad()
        data, labels, attention_mask = data.to(device), labels.to(device), attention_mask.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(data, attention_mask)
        #logger.info(f"Logits shape: {logits.shape}")
        #logger.info(f"Labels shape: {labels.shape}, Labels dtype: {labels.dtype}")
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()



def cls_val_epoch(model, device, val_loader):
    set_seed()
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, attention_mask in val_loader:
            data, target, attention_mask = data.to(device), target.to(device), attention_mask.to(device)
            output, _, _ = model(data, attention_mask)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)

    return val_loss, val_acc



def cls_test_epoch(model, device, test_loader):
    set_seed()
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, attention_mask in test_loader:
            data, target, attention_mask = data.to(device), target.to(device), attention_mask.to(device)
            output, _, _ = model(data, attention_mask)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


# 整体的finetune评估流程
def cls_evaluate_model(experiment_name, freeze_flag, model, task_index, tokenizer_name):
    set_seed()
    if 'mamba' in experiment_name:
        lr = 1e-4
    else:
        lr = 3e-5
    #lr = 3e-5
    # 参照之前文章设置的预训练参数
    if task_index in [0,1,2,3,4]:
        num_epochs = 3
    elif task_index in [7,10]:
        num_epochs = 10
    elif task_index in [5,6,8,9]:
        num_epochs = 4
    elif task_index == 11:
        num_epochs = 5
    train_dataset, val_dataset, test_dataset = FTDNADataset(task_index,'train',tokenizer_name=tokenizer_name), FTDNADataset(task_index,'dev',tokenizer_name=tokenizer_name), FTDNADataset(task_index,'test',tokenizer_name=tokenizer_name)
    my_collate_fn = build_collate_fn(train_dataset.pad_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn = my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn = my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn = my_collate_fn)

    for param in model.parameters():
        param.requires_grad = True
        
    if freeze_flag == True:
        for param in model.embedding.parameters():
            param.requires_grad = False

        for param in model.path.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = 50

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    for epoch in range(num_epochs):
        cls_train_epoch(model, device, train_loader, optimizer, scheduler)
        val_loss, val_acc = cls_val_epoch(model, device, val_loader)
        
    final_accuracy = cls_test_epoch(model, device, test_loader)

    return final_accuracy

# gai
def embed_sequences(sequences, model, tokenizer_name):
    """
    对一批序列生成 embedding。
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    all_feats = []
    start = time.time()
    batch_size = 1 # ns
    tokenizer = MyTokenizer(tokenizer_name)
    pad_idx = tokenizer.token_to_id("[PAD]")

    with torch.no_grad():
        all_encodings = [tokenizer.encode(seq) for seq in sequences]
        all_ids = [enc.ids for enc in all_encodings]

        max_len = max(len(ids) for ids in all_ids)
        # print(max_len)
        # print("First 3 sequences:")
        # for s in sequences[:3]:
        #     print(s)

  
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]

            # print(batch)
            # print(batch_ids)
            
            # 创建填充后的 input_ids 和 attention_mask
            input_ids_batch = []
            attention_mask_batch = []

            for ids in batch_ids:
                # 填充序列
                pad_len = max_len - len(ids)
                padded_ids = ids + [pad_idx] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
                input_ids_batch.append(padded_ids)
                attention_mask_batch.append(mask)
                # print(input_ids_batch)
            
            input_ids = torch.tensor(input_ids_batch, device=device)
            attention_mask = torch.tensor(attention_mask_batch, device=device)

            _, feats, _ = model(input_ids, attention_mask)
            all_feats.append(feats.cpu())
    embeddings = torch.cat(all_feats, dim=0).numpy()
    embed_time = time.time() - start
    return embeddings, embed_time

def cluster_and_evaluate(embeddings, true_labels, n_clusters):
    """
    用 KMeans 做聚类，并计算 ARI、Homogeneity、Completeness，以及聚类耗时。
    """
    start = time.time()
    km = KMeans(n_clusters=n_clusters, random_state=42)
    pred = km.fit_predict(embeddings)
    cluster_time = time.time() - start

    ari = adjusted_rand_score(true_labels, pred)
    hom = homogeneity_score(true_labels, pred)
    comp = completeness_score(true_labels, pred)
    return {
        "ARI": ari,
        "Homogeneity": hom,
        "Completeness": comp,
        "Time (s)": cluster_time
    }


def continual_contrastive_pretrain(model, task_index, tokenizer_name, data_usage_rate, epoch_num=5):
    train_data_jsonl_path = get_continual_pretrain_dataset_dir(task_index) + '/train.jsonl'
    dataset = DNAContrastiveDataset(train_data_jsonl_path, augment="random_substitution", data_usage_rate=data_usage_rate, tokenizer_name=tokenizer_name)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epoch_num):
        for batch in dataloader:
            optimizer.zero_grad()
            aug_1, aug_2 = batch  # batch should be a tuple of (aug_1, aug_2)
            aug_1 = aug_1.to(device)
            aug_2 = aug_2.to(device)
            _, emb_1, _ = model(aug_1)
            _, emb_2, _ = model(aug_2)
            loss = calculate_contrastive_loss(emb_1, emb_2)
            loss.backward()
            optimizer.step()
    return model



def continual_mask_pretrain(model, task_index, tokenizer_name, data_usage_rate):
    train_data_jsonl_path = get_continual_pretrain_dataset_dir(task_index) + '/train.jsonl'
    dataset = DNAMaskDataset(train_data_jsonl_path, data_usage_rate=data_usage_rate, tokenizer_name=tokenizer_name)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=3)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            masked_x, target = batch
            masked_x = masked_x.to(device)
            target = target.to(device)
            _, _, logits = model(masked_x)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
    return model




def find_best_architecture(experiment_name: str, training_strategy: str, epoch: int):
    csv_file_path = f"/projects/slmreasoning/yifang/results/{experiment_name}/{training_strategy}-ft_results.csv"

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return {}, None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return {}, None

    df_filtered = df[df['epoch_num'] == epoch]
    if df_filtered.empty:
        print(f"No data found for epoch_num = {epoch} in {csv_file_path}")
        return {}, None

    # 不转换 task_index 类型，保留字符串形式
    try:
        best_rows = df_filtered.loc[df_filtered.groupby('task_index')['acc'].idxmax()]
    except Exception as e:
        print(f"Error during groupby-idxmax: {e}")
        return {}, None

    # 字典形式保存每个任务的最佳模型编号
    best_model_indices_per_task = {}

    for _, row in best_rows.iterrows():
        task_key = row['task_index']  # 保留为字符串
        model_name = row['model_name']
        match = re.search(r'path_(\d+)', model_name)
        if match:
            best_model_indices_per_task[task_key] = int(match.group(1))
        else:
            print(f"Warning: Could not extract number from model_name: {model_name} for task: {task_key}")

    # 所有任务平均 acc 最佳模型
    overall_best_model_index = None
    overall_avg = df_filtered.groupby('model_name')['acc'].mean().reset_index()
    if not overall_avg.empty:
        best_row = overall_avg.loc[overall_avg['acc'].idxmax()]
        model_name = best_row['model_name']
        match = re.search(r'path_(\d+)', model_name)
        if match:
            overall_best_model_index = int(match.group(1))
        else:
            print(f"Warning: Could not extract number from overall best model_name: {model_name}")

    return best_model_indices_per_task, overall_best_model_index
