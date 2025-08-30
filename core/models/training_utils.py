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
from core.models.model import CLS_FixedModel, SecondaryStructureAlignmentModel
from core.models.utils import calculate_contrastive_loss
from core.models.modules import build_module
import random
from torch.utils.data import DataLoader
import logging
from transformers import get_linear_schedule_with_warmup 
import alignment_C as Aln_C

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def align_eval_epoch_correct(model, device, val_loader, use_dp=True, verbose=True):
    """
    正确使用Aln_C.global_aln的版本
    """
    model.eval()
    total_TP = 0
    total_pred_match = 0
    total_ref_match = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            idsA, idsB, contact, seqA_list, seqB_list, seq_lensA, seq_lensB, pairA_list, pairB_list = batch
            idsA = idsA.to(device)
            idsB = idsB.to(device)
            contact = contact.to(device)

            emA = model.get_encoded_sequence(idsA)
            emB = model.get_encoded_sequence(idsB)

            z0_list = [emA[b] for b in range(emA.size(0))]
            z1_list = [emB[b] for b in range(emB.size(0))]

            bert_scores, _ = model.match(z0_list, z1_list)


            B = idsA.size(0)
            for b in range(B):
                bert_score = bert_scores[b]  # [L1, L2]
                
                # 获取实际序列长度
                sequence_a = seqA_list[b]
                sequence_b = seqB_list[b]
                seq_len_a = int(seq_lensA[b].item())
                seq_len_b = int(seq_lensB[b].item())

                print(f"Match score statistics: min={bert_score.min():.4f}, max={bert_score.max():.4f}, std={bert_score.std():.4f}")

                # 测试simple alignment
                alignment = model._simple_alignment(bert_score, seq_len_a, seq_len_b)
                print(f"Simple alignment: {alignment}")

                exit(1)
                    
                # 确保使用实际长度的bert_score
                actual_bert_score = bert_score[:seq_len_a, :seq_len_b]
                
                # 按照C++代码，flatten应该是按行优先（C风格）
                # 即 score[i*width + j] 对应 score[i][j]
                bert_score_flat = actual_bert_score.flatten().tolist()  
                
                # 准备margin score（根据C++代码，这些参数也被使用）
                margin_score_FP = [0.0] * len(bert_score_flat)  # 可以调整这些值
                margin_score_FN = [0.0] * len(bert_score_flat)
                
                # 调用global_aln，参数顺序要正确
                try:
                    common_index_A_B_list = safe_global_aln(
                        bert_score_flat,      # match_score 列表
                        sequence_a,           # seq1 字符串
                        sequence_b,           # seq2 字符串
                        seq_len_a,            # lengthA 整数
                        seq_len_b,            # lengthB 整数
                    )
                    
                    # 根据C++代码，返回的是长度为 max_length*2 的列表
                    # 前 max_length 个是序列A的匹配标记，后 max_length 个是序列B的
                    max_length = 1024
                    
                    # 提取匹配标记
                    match_A = common_index_A_B_list[:max_length]
                    match_B = common_index_A_B_list[max_length:]
                    
                    # 计算预测匹配数（只统计实际序列长度内的）
                    len_pred_match = sum(match_A[:seq_len_a])
                    
                    if verbose and batch_idx == 0 and b == 0:
                        print(f"Aln_C success: pred_matches={len_pred_match}")
                        print(f"Match_A[:10]: {match_A[:10]}")
                        print(f"Match_B[:10]: {match_B[:10]}")
                
                except Exception as e:
                    if verbose and batch_idx == 0 and b == 0:
                        print(f"Aln_C failed: {e}")
                    
                    # Fallback到简单阈值方法
                    threshold = 0.5
                    matches = (actual_bert_score > threshold).nonzero()
                    len_pred_match = len(matches)
                    
                    # 创建匹配标记
                    match_A = [0] * seq_len_a
                    match_B = [0] * seq_len_b
                    for match in matches:
                        match_A[match[0]] = 1
                        match_B[match[1]] = 1
                
                # 获取参考配对
                if isinstance(pairA_list[b], torch.Tensor) and len(pairA_list[b]) > 0:
                    refA_indices = pairA_list[b].cpu().numpy()
                    refB_indices = pairB_list[b].cpu().numpy()
                    len_ref_match = len(refA_indices)
                    
                    if len_pred_match > 0 and len_ref_match > 0:
                        # 计算TP：按照参考代码的方式
                        try:
                            # 从匹配标记中提取预测的配对位置
                            pred_a_indices = [i for i in range(seq_len_a) if match_A[i] == 1]
                            pred_b_indices = [i for i in range(seq_len_b) if match_B[i] == 1]
                            
                            # 如果长度匹配，按顺序配对；否则使用阈值方法的配对
                            if len(pred_a_indices) == len(pred_b_indices):
                                pred_pairs = set((pred_a_indices[i] * 10000 + pred_b_indices[i]) 
                                                for i in range(len(pred_a_indices)))
                            else:
                                # 使用阈值方法的直接配对
                                matches = (actual_bert_score > 0.5).nonzero()
                                pred_pairs = set((int(match[0]) * 10000 + int(match[1])) 
                                               for match in matches)
                                len_pred_match = len(pred_pairs)
                            
                            ref_pairs = set((int(refA_indices[i]) * 10000 + int(refB_indices[i])) 
                                          for i in range(len(refA_indices)))
                            
                            len_TP = len(pred_pairs & ref_pairs)
                        except:
                            len_TP = 0
                    else:
                        len_TP = 0
                else:
                    len_ref_match = 0
                    len_TP = 0
                
                total_TP += len_TP
                total_pred_match += len_pred_match  
                total_ref_match += len_ref_match
                
                if verbose and batch_idx == 0 and b < 2:
                    print(f"Sample {b}: pred={len_pred_match}, ref={len_ref_match}, TP={len_TP}")

    # 计算指标
    sens = total_TP / (total_ref_match + 1e-8)
    ppv = total_TP / (total_pred_match + 1e-8)
    f1 = 2 * sens * ppv / (sens + ppv + 1e-8)
    
    if verbose:
        print(f"\nTotal - TP: {total_TP}, pred: {total_pred_match}, ref: {total_ref_match}")
        print(f"Sensitivity: {sens:.4f}, PPV: {ppv:.4f}, F1: {f1:.4f}")
    
    return f1, sens, ppv


def safe_global_aln(bert_score, sequence_a, sequence_b, seq_len_a, seq_len_b, 
                    gap_open=-1.0, gap_extend=-0.1, verbose=True):
    """
    安全的global_aln包装器，处理所有可能的错误情况
    """
    try:
        # 1. 参数验证
        max_length = 1024  # C++代码中的常数
        
        if seq_len_a <= 0 or seq_len_b <= 0:
            if verbose:
                print("Invalid sequence lengths")
            return None
            
        if seq_len_a > max_length or seq_len_b > max_length:
            if verbose:
                print(f"Sequence too long: {seq_len_a}, {seq_len_b} > {max_length}")
            return None
        
        # 2. 确保bert_score是正确的大小和类型
        expected_size = seq_len_a * seq_len_b
        if len(bert_score) != expected_size:
            if verbose:
                print(f"Score size mismatch: {len(bert_score)} != {expected_size}")
            return None
        
        # 3. 转换为float列表，确保类型正确
        match_score = [float(x) for x in bert_score]
        margin_score_FP = [0.0] * expected_size
        margin_score_FN = [0.0] * expected_size
        
        # 4. 确保序列是字符串
        seq_a_str = str(sequence_a)[:seq_len_a]  # 截断到实际长度
        seq_b_str = str(sequence_b)[:seq_len_b]
        
        # 5. 确保序列长度匹配
        if len(seq_a_str) != seq_len_a or len(seq_b_str) != seq_len_b:
            if verbose:
                print(f"String length mismatch: {len(seq_a_str)}!={seq_len_a} or {len(seq_b_str)}!={seq_len_b}")
            return None
        
        # 6. 调用C++函数
        result = Aln_C.global_aln(
            match_score,        # list of float
            margin_score_FP,    # list of float
            margin_score_FN,    # list of float
            seq_a_str,          # string
            seq_b_str,          # string
            int(seq_len_a),     # int
            int(seq_len_b),     # int
            float(gap_open),    # float
            float(gap_extend),  # float
            1 if verbose else 0, # int
            0                   # int
        )
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"safe_global_aln error: {e}")
        return None


# 改进的训练函数，确保模型真正学习
def align_train_epoch_improved(model, device, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            idsA, idsB, contact = batch[:3]
            idsA, idsB, contact = idsA.to(device), idsB.to(device), contact.to(device)

            optimizer.zero_grad()

            # 确保模型输出正确
            emA = model.embedding(idsA)  # [B, L1, d]
            emB = model.embedding(idsB)  # [B, L2, d]

            z0_list = [emA[b] for b in range(emA.size(0))]
            z1_list = [emB[b] for b in range(emB.size(0))]

            match_scores, _ = model.match(z0_list, z1_list)
            
            # 检查match_scores是否需要梯度
            if not any(score.requires_grad for score in match_scores):
                print("Warning: match_scores do not require gradients!")
            
            # 将match_scores转换为tensor并确保形状匹配
            max_len_a = emA.size(1)
            max_len_b = emB.size(1)
            
            score_tensor = torch.zeros(len(match_scores), max_len_a, max_len_b, 
                                     device=device, requires_grad=True)
            
            for i, score in enumerate(match_scores):
                h, w = score.shape
                score_tensor[i, :h, :w] = score

            # 使用BCE loss
            loss = F.binary_cross_entropy_with_logits(score_tensor, contact)
            
            # 检查loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}: {loss}")
                continue
                
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            
            # 定期输出进度
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)

def load_rna_clustering(data_dir):
    """
    从 data_dir 下的每个文件读取 RNA 序列，
    每个文件对应一个 family，文件名就是类别名。
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

def build_all_alignment_models(model_space):
    """
    枚举 model_space 中所有 cnn_arch 架构，构造完整模型。
    返回: List of (arch_name, model)
    """
    all_models = []
    #logger.info(f"model_space.paths_candidates is {model_space.paths_candidates}")
    for arch_name, path in model_space.paths_candidates.items():
        alignment_model = SecondaryStructureAlignmentModel(
            embedding=model_space.embedding,
            encoder=path,
            config=None
        )
        all_models.append((arch_name, alignment_model))

    return all_models

# gai
def align_train_epoch(model, device, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        # unpack collate output (training collate can be same)
        idsA, idsB, contact, *_ = batch[:3]  # we only need first 3 for training
        idsA, idsB, contact = idsA.to(device), idsB.to(device), contact.to(device)

        optimizer.zero_grad()

        # get embeddings via model (or encoded layers then module.em if needed)
        # BUT match expects z0_list, z1_list (list of [L,d] per sample)
        emA = model.embedding(idsA)  # [B, L1, d] TODO
        emB = model.embedding(idsB)  # [B, L2, d]

        # convert to lists for match: z0_list = [emA[b] for b in range(B)], etc.
        z0_list = [emA[b] for b in range(emA.size(0))]
        z1_list = [emB[b] for b in range(emB.size(0))]

        # get match matrices (list of [L1,L2]) and logits (B)
        match_scores, _ = model.match(z0_list, z1_list)  # match_scores: list of [L1,L2]
        # stack match_scores into tensor [B, L1, L2] for loss
        # Note: samples may have different lengths due to padding; but we padded ids => emA/emB have padding rows
        score_tensor = torch.stack([
            ms if ms.shape == (emA.size(1), emB.size(1)) else F.pad(ms, (0, emB.size(1)-ms.shape[1], 0, emA.size(1)-ms.shape[0]))
            for ms in match_scores
        ], dim=0).to(device)

        # Use BCE with logits (score_tensor can be treated as logits)
        loss = F.binary_cross_entropy_with_logits(score_tensor, contact)
        # print(type(loss), loss.requires_grad, loss)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def align_train_epoch_MUL(model, device, train_loader, optimizer, scheduler, tokenizer):
    model.train()
    total_loss = 0.0
    epoch_mul_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        # 解包批次数据 - 先打印看看batch的结构
        if batch_idx == 0:
            print(f"Batch structure: {len(batch)} items")
            for i, item in enumerate(batch):
                print(f"Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'no shape')}")
        
        idsA, idsB, contact = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        
        # 从contact matrix中提取common_index（简化版本）
        batch_size = idsA.size(0)
        common_index_0 = []
        common_index_1 = []
        
        for b in range(batch_size):
            # 直接从contact矩阵中推导common_index
            contact_b = contact[b]
            indices_0, indices_1 = torch.where(contact_b > 0.5)
            pad_token_id = tokenizer.token_to_id("[PAD]")
            
            # print("len(indices_0):",len(indices_0))
            if len(indices_0) == 0:
                # 如果没有正例，创建一些基于序列长度的对齐
                # 获取实际序列长度（排除padding）
                print("wrong")
                actual_len_A = (idsA[b] != pad_token_id).sum().item()
                actual_len_B = (idsB[b] != pad_token_id).sum().item()
                
                # 创建均匀分布的对齐点
                num_align = min(5, min(actual_len_A, actual_len_B))  # 最多5个对齐点
                if num_align > 0:
                    indices_0 = torch.linspace(0, actual_len_A-1, num_align).long()
                    indices_1 = torch.linspace(0, actual_len_B-1, num_align).long()
                else:
                    indices_0 = torch.tensor([0])
                    indices_1 = torch.tensor([0])
            
            common_index_0.append(indices_0.to(device))
            common_index_1.append(indices_1.to(device))
        
        optimizer.zero_grad()
        
        # 获取编码层
        attention_mask_A = (idsA != pad_token_id).int()
        attention_mask_B = (idsB != pad_token_id).int()
        
        encoded_layers0 = model.get_encoded_sequence(idsA, attention_mask_A)
        encoded_layers1 = model.get_encoded_sequence(idsB, attention_mask_B)

        # print(idsA)
        # print(encoded_layers0)
        # print(idsB)
        # print(encoded_layers1)
        
        # 获取实际序列长度
        seq_len_0 = attention_mask_A.sum(dim=1).tolist()
        seq_len_1 = attention_mask_B.sum(dim=1).tolist()
        # print(seq_len_0, seq_len_1)
        # 转换为列表格式
        z0_list = model.em(encoded_layers0, seq_len_0)
        z1_list = model.em(encoded_layers1, seq_len_1)

        # print(z0_list)
        # print(z1_list)
        
        # MUL训练
        mul_loss = model.train_MUL(z0_list, z1_list, common_index_0, common_index_1, seq_len_0, seq_len_1)
        mul_loss = torch.tensor(0.0, device=device) if torch.isnan(mul_loss) else mul_loss
        # print(mul_loss)
        
        loss = mul_loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        epoch_mul_loss += mul_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
    
        print(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
        # exit(1)
    return total_loss / len(train_loader), epoch_mul_loss / len(train_loader)

def align_eval_epoch(model, device, val_loader, use_dp=True, verbose=False):
    """
    修复后的评估函数
    """
    model.eval()
    total_TP = 0
    total_pred_match = 0
    total_ref_match = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                # 解包batch
                idsA, idsB, contact, seqA_list, seqB_list, seq_lensA, seq_lensB, pairA_list, pairB_list = batch
                idsA = idsA.to(device)
                idsB = idsB.to(device)
                contact = contact.to(device)

                # 使用模型的编码方法，而不是直接embedding
                # 这样保持与训练时的一致性
                pad_token_id = 0  # 或者从tokenizer获取
                attention_mask_A = (idsA != pad_token_id).int()
                attention_mask_B = (idsB != pad_token_id).int()
                
                # 使用模型的get_encoded_sequence方法
                encoded_A = model.get_encoded_sequence(idsA, attention_mask_A)  # [B, L1, d]
                encoded_B = model.get_encoded_sequence(idsB, attention_mask_B)  # [B, L2, d]

                # 转换为列表格式（使用模型的em方法）
                seq_lens_A_list = seq_lensA.cpu().tolist()
                seq_lens_B_list = seq_lensB.cpu().tolist()
                
                z0_list = model.em(encoded_A, seq_lens_A_list)
                z1_list = model.em(encoded_B, seq_lens_B_list)

                # 获取match scores
                bert_scores, _ = model.match(z0_list, z1_list)
                
                B = idsA.size(0)
                for b in range(B):
                    try:
                        bert_score = bert_scores[b]  # [L1, L2]
                        
                        # 获取实际序列长度
                        seq_len_a = int(seq_lensA[b].item())
                        seq_len_b = int(seq_lensB[b].item())
                        
                        # 检查序列长度有效性
                        if seq_len_a <= 0 or seq_len_b <= 0:
                            if verbose:
                                print(f"Batch {batch_idx}, Sample {b}: Invalid sequence lengths A={seq_len_a}, B={seq_len_b}")
                            continue
                        
                        if seq_len_a > bert_score.shape[0] or seq_len_b > bert_score.shape[1]:
                            if verbose:
                                print(f"Batch {batch_idx}, Sample {b}: Sequence length exceeds bert_score shape")
                            continue
                        
                        # 使用simple_alignment（修复版本）
                        common_index_A_B = model._simple_alignment(bert_score, seq_len_a, seq_len_b)
                        
                        # 检查对齐结果
                        if common_index_A_B.shape[1] == 0:
                            if verbose:
                                print(f"Batch {batch_idx}, Sample {b}: No alignment found")
                            continue
                        
                        # 获取预测的配对
                        pred_a_indices = common_index_A_B[0].cpu()  # 预测的A序列位置
                        pred_b_indices = common_index_A_B[1].cpu()  # 预测的B序列位置
                        len_pred_match = len(pred_a_indices)
                        
                        # 获取参考配对
                        if isinstance(pairA_list[b], torch.Tensor):
                            refA_indices = pairA_list[b].cpu()
                            refB_indices = pairB_list[b].cpu()
                        elif isinstance(pairA_list[b], list):
                            refA_indices = torch.tensor(pairA_list[b])
                            refB_indices = torch.tensor(pairB_list[b])
                        else:
                            if verbose:
                                print(f"Batch {batch_idx}, Sample {b}: Unknown pair format {type(pairA_list[b])}")
                            continue
                        
                        len_ref_match = len(refA_indices)
                        
                        # 计算TP：预测配对与参考配对的交集
                        # 使用配对编码的方式：(i, j) -> i * 10000 + j
                        if len_pred_match > 0 and len_ref_match > 0:
                            pred_pairs = set((pred_a_indices * 10000 + pred_b_indices).tolist())
                            ref_pairs = set((refA_indices * 10000 + refB_indices).tolist())
                            len_TP = len(pred_pairs & ref_pairs)
                        else:
                            len_TP = 0
                        
                        # 累加统计
                        total_TP += len_TP
                        total_pred_match += len_pred_match
                        total_ref_match += len_ref_match
                        
                        if verbose and batch_idx < 3:  # 只打印前几个batch的详细信息
                            print(f"Batch {batch_idx}, Sample {b}: TP={len_TP}, Pred={len_pred_match}, Ref={len_ref_match}")
                            if len_pred_match > 0:
                                print(f"  First few pred pairs: {list(zip(pred_a_indices[:3].tolist(), pred_b_indices[:3].tolist()))}")
                            if len_ref_match > 0:
                                print(f"  First few ref pairs: {list(zip(refA_indices[:3].tolist(), refB_indices[:3].tolist()))}")
                        
                    except Exception as e:
                        if verbose:
                            print(f"Error processing sample {b} in batch {batch_idx}: {e}")
                        continue
                        
            except Exception as e:
                if verbose:
                    print(f"Error processing batch {batch_idx}: {e}")
                continue

    # 计算最终指标
    sens = total_TP / (total_ref_match + 1e-8)  # Sensitivity (Recall)
    ppv = total_TP / (total_pred_match + 1e-8)   # PPV (Precision)
    f1 = 2 * sens * ppv / (sens + ppv + 1e-8)
    
    if verbose:
        print(f"\n=== Final Results ===")
        print(f"Total TP: {total_TP}")
        print(f"Total Predicted: {total_pred_match}")
        print(f"Total Reference: {total_ref_match}")
        print(f"Sensitivity (Recall): {sens:.4f}")
        print(f"PPV (Precision): {ppv:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    return f1, sens, ppv

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
  
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            
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
