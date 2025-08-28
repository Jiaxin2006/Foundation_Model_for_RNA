import sys
sys.path.append("../")
from core.data_utils.mytokenizers import MyTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import json
import pathlib
import random
import pandas as pd
from torch.utils.data import Dataset
import os
import nni
import csv
import logging
from Bio import SeqIO
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_data_line_by_line(file_path):
    sequences = []
    labels = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row['sequence'])
            labels.append(int(row['label']))
    return sequences, labels


class FTDNADataset(Dataset):    # TODO: 改成RNA的数据集
    def __init__(self, task_index, split='train', tokenizer_name = "kmer-1"):
        dataset_dir = self.get_dataset_dir(task_index)
        self.seq_len = self.get_sequence_length(task_index)
        file_path = os.path.join(dataset_dir, f"{split}.csv")

        self.sequences, self.labels = load_data_line_by_line(file_path)

        self.num_classes = len(set(self.labels))

        self.tokenizer_name = tokenizer_name

        self.tokenizer = MyTokenizer(tokenizer_name)
        self.pad_idx = self.tokenizer.token_to_id("[PAD]")

    def get_dataset_dir(self, task_index):
        path_dict = {
            0: '/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/data/UTR',        # UTR/GSM3130435_egfp_unmod_1.csv
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

    def get_sequence_length(self, task_index):
        path_dict = {
            0: 100,
            1: 100,
            2: 100,
            3: 100,
            4: 100,
            5: 70,
            6: 70,
            7: 70,
            8: 300,
            9: 300,
            10: 300,
            11: 400,
        }
        return path_dict[task_index]

    def __len__(self):
        return len(self.sequences)

    def encode_sequence(self, seq):
        encoded = self.tokenizer.encode(seq)
        return encoded

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        x_ids = self.encode_sequence(sequence).ids
        original_len = len(x_ids) # Store original length
        x = torch.tensor(x_ids, dtype=torch.long)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, label, original_len
        
    def get_num_classes(self):
        return self.num_classes

# gai
# TODO
'''
def load_align_pairs(data_dir, max_pairs=1000):
    """
    读取 .ref.fa，输出 [(seqA, seqB, contact)]
    contact: L×L 矩阵，对齐列上双碱基非 gap 且互补配对 =>1
    """
    # print("here_1")
    ref_path = os.path.join(data_dir, "all_ref.fa")
    records = list(SeqIO.parse(ref_path, "fasta"))
    pairs = []
    # print(len(records))
    # 简单：相邻两条做配对
    for i in range(0, min(len(records), max_pairs*2), 2):
        if i+1 >= len(records): break
        refA = str(records[i].seq).upper().replace("T", "U")
        refB = str(records[i+1].seq).upper().replace("T", "U")
        seqA = refA.replace("-", "").replace('.', '')
        seqB = refB.replace("-", "").replace('.', '')
        mapA, mapB = [], []
        idxA = idxB = 0
        for a, b in zip(refA, refB):
            if a != "-":
                mapA.append(idxA)  # 原始坐标
                idxA += 1
            else:
                mapA.append(-1)  # gap
            if b != "-":
                mapB.append(idxB)
                idxB += 1
            else:
                mapB.append(-1)

        L1, L2 = len(seqA), len(seqB)
        contact = np.zeros((L1, L2), dtype=np.int8)
        for k, (a, b) in enumerate(zip(refA, refB)):
            if a == "-" or b == "-":
                continue
            if a in "AUGC" and b in "AUGC":
                i = mapA[k]
                j = mapB[k]
                contact[i, j] = 1
        # print(len(seqA),len(seqB))
        # print(contact.shape)
        pairs.append((seqA, seqB, contact))
    # print("第一条记录长度:", len(pairs[0]) if pairs else 0)
    # print("第一条记录:", pairs[0]) if pairs else None
    return pairs
'''
def load_align_pairs(data_dir, max_pairs=1000): # TODO
    """
    读取 .ref.fa，输出 [(seqA, seqB, contact)]
    contact: L1×L2 矩阵，表示序列间的对应关系（从比对中提取）
    """
    ref_path = os.path.join(data_dir, "all_ref.fa")
    records = list(SeqIO.parse(ref_path, "fasta"))
    pairs = []
    
    # 简单：相邻两条做配对
    for i in range(0, min(len(records), max_pairs*2), 2):
        if i+1 >= len(records): 
            break
            
        refA = str(records[i].seq).upper().replace("T", "U")
        refB = str(records[i+1].seq).upper().replace("T", "U")
        
        # 去除gap得到原始序列
        seqA = refA.replace("-", "").replace('.', '')
        seqB = refB.replace("-", "").replace('.', '')
        
        # 建立从比对位置到原始序列位置的映射
        mapA, mapB = [], []
        idxA = idxB = 0
        
        for a, b in zip(refA, refB):
            if a != "-" and a != '.':
                mapA.append(idxA)  # 原始序列中的位置
                idxA += 1
            else:
                mapA.append(-1)  # gap
                
            if b != "-" and b != '.':
                mapB.append(idxB)
                idxB += 1
            else:
                mapB.append(-1)

        L1, L2 = len(seqA), len(seqB)
        contact = np.zeros((L1, L2), dtype=np.int8)
        
        # 从比对中提取对应关系
        for k, (a, b) in enumerate(zip(refA, refB)):
            # 跳过任一方为gap的位置
            if a in "-." or b in "-.":
                continue
                
            # 只有当两边都是有效碱基时才建立联系 TODO
            # if a in "AUGC" and b in "AUGC":
            if (a == 'A' and b == 'U') or (a == 'U' and b == 'A') or (a == 'G' and b == 'C') or (a == 'C' and b == 'G'):
            # 只有互补碱基对才通过
                pos_a = mapA[k]  # seqA中的位置
                pos_b = mapB[k]  # seqB中的位置
                
                # 确保位置有效
                if pos_a >= 0 and pos_b >= 0 and pos_a < L1 and pos_b < L2:
                    contact[pos_a, pos_b] = 1
        
        pairs.append((seqA, seqB, contact))     
    return pairs

def get_offsets_from_encoding(enc):
    """
    从 tokenizers.Encoding 对象获取 offsets，格式为 list of (start, end)
    """
    if hasattr(enc, "offsets"):
        return enc.offsets
    elif hasattr(enc, "get_offsets"):
        return enc.get_offsets()
    else:
        raise RuntimeError("Encoding对象无offsets属性")

def aggregate_contact_to_tokens(contact_base: np.ndarray, offsetsA, offsetsB, agg='any'):
    Ta = len(offsetsA)
    Tb = len(offsetsB)
    out = np.zeros((Ta, Tb), dtype=np.float32)
    for i, (sa, ea) in enumerate(offsetsA):
        if sa >= ea:
            continue
        for j, (sb, eb) in enumerate(offsetsB):
            if sb >= eb:
                continue
            block = contact_base[sa:ea, sb:eb]
            if block.size == 0:
                val = 0.0
            else:
                if agg == 'any':
                    val = 1.0 if np.any(block) else 0.0
                elif agg == 'max':
                    val = float(np.max(block))
                elif agg == 'mean':
                    val = float(np.mean(block))
                else:
                    raise ValueError("agg must be 'any'|'max'|'mean'")
            out[i, j] = val
    return out


class AlignDataset(Dataset):
    def __init__(self, split, tokenizer_name, root_dir):
        self.pairs = load_align_pairs(root_dir)
        self.tokenizer = MyTokenizer(tokenizer_name)
        self.pad = self.tokenizer.token_to_id("[PAD]")
        #  8:1:1 划分
        n = len(self.pairs)
        if split == "train":
            self.pairs = self.pairs[:int(0.8*n)]
        elif split == "dev":
            self.pairs = self.pairs[int(0.8*n):int(0.9*n)]
        else:
            self.pairs = self.pairs[int(0.9*n):]
        # print(len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        返回：
        idsA_tensor: token ids of seqA (no padding)
        idsB_tensor: token ids of seqB (no padding)
        contact_tensor: float tensor shape [L1, L2], 0/1
        raw_seqA: original ungapped sequence string (e.g. "AUGC...")
        raw_seqB: same for B
        seq_lenA, seq_lenB: ints
        pair_idx_A, pair_idx_B: 1D LongTensor, lists of paired indices (same length K)
        """
        seqA, seqB, mat = self.pairs[idx]
        contact_base = np.array(mat, dtype=np.int8)  # L1 x L2 (base-level contact)

        # tokenization (may produce token offsets)
        encA = self.tokenizer.encode(seqA)
        encB = self.tokenizer.encode(seqB)
        idsA = encA.ids if hasattr(encA, "ids") else encA['input_ids']
        idsB = encB.ids if hasattr(encB, "ids") else encB['input_ids']

        offsetsA = get_offsets_from_encoding(encA)  # list of (start,end) per token
        offsetsB = get_offsets_from_encoding(encB)

        # Truncate offsets to token count to be safe
        offsetsA = offsetsA[:len(idsA)]
        offsetsB = offsetsB[:len(idsB)]

        # 把 base-level contact 聚合到 token 级别：token_contact shape [T1, T2]
        token_contact = aggregate_contact_to_tokens(contact_base, offsetsA, offsetsB, agg='any')
        # token_contact is 2D numpy array (T1 x T2) of 0/1

        # 构造成对索引（稀疏对）
        # pair_indices: list of (i,j) where token_contact[i,j]==1
        coords = np.array(np.nonzero(token_contact)).T  # shape [K,2]
        if coords.shape[0] == 0:
            # no matches: set empty tensors
            pair_idx_A = torch.empty(0, dtype=torch.long)
            pair_idx_B = torch.empty(0, dtype=torch.long)
        else:
            pair_idx_A = torch.tensor(coords[:, 0], dtype=torch.long)
            pair_idx_B = torch.tensor(coords[:, 1], dtype=torch.long)

        idsA_tensor = torch.tensor(idsA, dtype=torch.long)
        idsB_tensor = torch.tensor(idsB, dtype=torch.long)
        contact_tensor = torch.from_numpy(token_contact).float()  # float for BCE loss

        seq_lenA = idsA_tensor.size(0)
        seq_lenB = idsB_tensor.size(0)

        # 返回更多字段（collate 会负责批处理）
        return idsA_tensor, idsB_tensor, contact_tensor, seqA, seqB, seq_lenA, seq_lenB, pair_idx_A, pair_idx_B

class MSADataset(Dataset):
    def __init__(self, split, tokenizer_name, root_dir, max_pairs_per_family=50000):
        """
        修复版本的MSA数据集
        - 只在同family内配对
        - 基于MSA创建更准确的contact矩阵
        - 不重复序列，避免过拟合
        """
        self.tokenizer = MyTokenizer(tokenizer_name)
        self.pad = self.tokenizer.token_to_id("[PAD]")
        
        # 加载数据
        self.pairs = self._load_msa_pairs_fixed(root_dir, max_pairs_per_family)
        
        # 划分数据集
        n = len(self.pairs)
        if split == "train":
            self.pairs = self.pairs[:int(0.8*n)]
        elif split == "dev":
            self.pairs = self.pairs[int(0.8*n):int(0.9*n)]
        else:
            self.pairs = self.pairs[:int(n)]
        
        print(f"{split} dataset size: {len(self.pairs)}")
    
    def _load_msa_pairs_fixed(self, root_dir, max_pairs_per_family):
        """修复的MSA数据加载"""        
        all_pairs = []
        family_files = [f for f in os.listdir(root_dir) if f.endswith(('.fa', '.fasta', '.aln'))]
        
        for family_file in family_files:
            family_path = os.path.join(root_dir, family_file)
            
            try:
                # 读取该family的所有序列
                records = list(SeqIO.parse(family_path, "fasta"))
                if len(records) < 2:
                    continue
                
                family_pairs = []
                
                # 在同family内生成配对，但不重复序列
                for i in range(len(records)):
                    for j in range(i + 1, len(records)):
                        refA = str(records[i].seq).upper().replace("T", "U")
                        refB = str(records[j].seq).upper().replace("T", "U")
                        
                        # 去除gap得到原始序列
                        seqA = refA.replace('-', '').replace('.', '')
                        seqB = refB.replace('-', '').replace('.', '')
                        
                        # 建立从比对位置到原始序列位置的映射
                        mapA, mapB = [], []
                        idxA = idxB = 0
                        
                        for a, b in zip(refA, refB):
                            if a != "-" and a != '.':
                                mapA.append(idxA)  # 原始序列中的位置
                                idxA += 1
                            else:
                                mapA.append(-1)  # gap
                                
                            if b != "-" and b != '.':
                                mapB.append(idxB)
                                idxB += 1
                            else:
                                mapB.append(-1)

                        L1, L2 = len(seqA), len(seqB)
                        contact = np.zeros((L1, L2), dtype=np.int8)
                        
                        # 从比对中提取对应关系
                        for k, (a, b) in enumerate(zip(refA, refB)):
                            # 跳过任一方为gap的位置
                            if a in "-." or b in "-.":
                                continue
                                
                            # 只有当两边都是有效碱基时才建立联系 TODO
                            # if a in "AUGC" and b in "AUGC":
                            if (a == 'A' and b == 'U') or (a == 'U' and b == 'A') or (a == 'G' and b == 'C') or (a == 'C' and b == 'G'):
                            # 只有互补碱基对才通过
                                pos_a = mapA[k]  # seqA中的位置
                                pos_b = mapB[k]  # seqB中的位置
                                
                                # 确保位置有效
                                if pos_a >= 0 and pos_b >= 0 and pos_a < L1 and pos_b < L2:
                                    contact[pos_a, pos_b] = 1
                        
                        family_pairs.append((seqA, seqB, contact))    
                        
                        # 限制每个family的配对数
                        if len(family_pairs) >= max_pairs_per_family:
                            break
                    
                    if len(family_pairs) >= max_pairs_per_family:
                        break
                
                all_pairs.extend(family_pairs)
                print(f"Family {family_file}: {len(family_pairs)} valid pairs")
                
            except Exception as e:
                print(f"Error processing {family_file}: {e}")
                continue
        
        random.shuffle(all_pairs)
        print(f"Total valid pairs: {len(all_pairs)}")
        return all_pairs
    
    def _create_alignment_based_contact(self, gapped_seq1, gapped_seq2, seq1, seq2):
        """基于MSA比对创建contact矩阵，使用更严格的标准"""
        len1, len2 = len(seq1), len(seq2)
        contact_matrix = np.zeros((len1, len2), dtype=np.int8)
        
        # 建立位置映射
        pos1_to_ungapped = {}
        pos2_to_ungapped = {}
        
        ungapped_pos1 = 0
        for i, char in enumerate(gapped_seq1):
            if char not in ['-', '.']:
                pos1_to_ungapped[i] = ungapped_pos1
                ungapped_pos1 += 1
        
        ungapped_pos2 = 0
        for i, char in enumerate(gapped_seq2):
            if char not in ['-', '.']:
                pos2_to_ungapped[i] = ungapped_pos2
                ungapped_pos2 += 1
        
        # 只有当两个位置都不是gap时才建立contact
        # 而且要求有一定的序列相似性
        valid_contacts = 0
        total_aligned_positions = 0
        
        for i in range(min(len(gapped_seq1), len(gapped_seq2))):
            if (i in pos1_to_ungapped and i in pos2_to_ungapped):
                pos1 = pos1_to_ungapped[i]
                pos2 = pos2_to_ungapped[i]
                total_aligned_positions += 1
                
                # 可以根据需要添加额外的条件，比如要求残基相同或相似
                char1 = gapped_seq1[i]
                char2 = gapped_seq2[i]
                
                # 简单策略：比对位置就认为是contact
                if pos1 < len1 and pos2 < len2:
                    contact_matrix[pos1, pos2] = 1
                    valid_contacts += 1
        
        # 质量检查：如果contact太少可能是低质量比对
        if total_aligned_positions == 0:
            return None
        
        contact_density = valid_contacts / (len1 * len2)
        if contact_density < 0.01 or contact_density > 0.5:  # 过稀疏或过密集都可能有问题
            return None
        
        return contact_matrix
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """与原数据集相同的接口"""
        seqA, seqB, mat = self.pairs[idx]
        contact_base = np.array(mat, dtype=np.int8)

        # tokenization
        encA = self.tokenizer.encode(seqA)
        encB = self.tokenizer.encode(seqB)
        idsA = encA.ids if hasattr(encA, "ids") else encA['input_ids']
        idsB = encB.ids if hasattr(encB, "ids") else encB['input_ids']

        offsetsA = get_offsets_from_encoding(encA)
        offsetsB = get_offsets_from_encoding(encB)
        offsetsA = offsetsA[:len(idsA)]
        offsetsB = offsetsB[:len(idsB)]

        # 聚合到token级别
        token_contact = aggregate_contact_to_tokens(contact_base, offsetsA, offsetsB, agg='any')

        # 构造配对索引
        coords = np.array(np.nonzero(token_contact)).T
        if coords.shape[0] == 0:
            pair_idx_A = torch.empty(0, dtype=torch.long)
            pair_idx_B = torch.empty(0, dtype=torch.long)
        else:
            pair_idx_A = torch.tensor(coords[:, 0], dtype=torch.long)
            pair_idx_B = torch.tensor(coords[:, 1], dtype=torch.long)

        idsA_tensor = torch.tensor(idsA, dtype=torch.long)
        idsB_tensor = torch.tensor(idsB, dtype=torch.long)
        contact_tensor = torch.from_numpy(token_contact).float()

        seq_lenA = idsA_tensor.size(0)
        seq_lenB = idsB_tensor.size(0)

        return idsA_tensor, idsB_tensor, contact_tensor, seqA, seqB, seq_lenA, seq_lenB, pair_idx_A, pair_idx_B

# 渐进式训练策略
def progressive_training_strategy(tokenizer_name, msa_data_dir, eval_data_dir):
    """渐进式训练策略：先用少量高质量MSA数据，再用原数据微调"""
    
    # 阶段1：高质量MSA数据
    msa_train_ds = MSADataset("train", tokenizer_name, msa_data_dir, max_pairs_per_family=30)
    
    # 阶段2：原始数据（如果需要）
    orig_train_ds = AlignDataset("train", tokenizer_name, eval_data_dir)
    
    # 评估数据保持不变
    val_ds = AlignDataset("dev", tokenizer_name, eval_data_dir)
    test_ds = AlignDataset("test", tokenizer_name, eval_data_dir)
    
    return msa_train_ds, orig_train_ds, val_ds, test_ds

# 使用参考代码风格的训练
def create_reference_style_training(tokenizer_name, msa_data_dir):
    """创建参考代码风格的训练数据"""
    train_ds = MSADataset("train", tokenizer_name, msa_data_dir, 
                                       mag=8, max_length=1024, family_ratio=0.8)
    val_ds = MSADataset("dev", tokenizer_name, msa_data_dir, 
                                     mag=3, max_length=1024, family_ratio=0.8)  
    test_ds = MSADataset("test", tokenizer_name, msa_data_dir, 
                                      mag=3, max_length=1024, family_ratio=0.8)
    
    return train_ds, val_ds, test_ds

# gai
def pad2d(tensors, fill_value=0):
    """把一批二维张量统一填充到最大尺寸"""
    max_h = max(t.shape[0] for t in tensors)
    max_w = max(t.shape[1] for t in tensors)
    out = []
    for t in tensors:
        h, w = t.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded = torch.nn.functional.pad(
            t, (0, pad_w, 0, pad_h),  # (left, right, top, bottom)
            value=fill_value
        )
        out.append(padded)
    return torch.stack(out)


def align_collate(batch):
    """
    batch: list of tuples from __getitem__:
      (idsA, idsB, contact, seqA, seqB, seq_lenA, seq_lenB, pairA, pairB)
    返回：
      idsA_padded: [B, L1max]
      idsB_padded: [B, L2max]
      contact_padded: [B, L1max, L2max]
      seqA_list: list of raw seqA strings (len B)
      seqB_list: list of raw seqB strings
      seq_lensA: tensor shape [B]
      seq_lensB: tensor shape [B]
      pairA_list: list of 1D LongTensors (pair indices for each sample)
      pairB_list: list of 1D LongTensors
    """
    idsA, idsB, contact, seqA_list, seqB_list, seq_lensA, seq_lensB, pairA_list, pairB_list = zip(*batch)

    idsA_padded = pad_sequence(idsA, batch_first=True, padding_value=0)  # [B, L1max]
    idsB_padded = pad_sequence(idsB, batch_first=True, padding_value=0)  # [B, L2max]

    contact_padded = pad2d(contact, fill_value=0)                        # [B, L1max, L2max]

    seq_lensA = torch.tensor(seq_lensA, dtype=torch.long)
    seq_lensB = torch.tensor(seq_lensB, dtype=torch.long)

    # pair lists: keep as Python lists of tensors (variable-length)
    pairA_list = list(pairA_list)
    pairB_list = list(pairB_list)
    fixed_pairA, fixed_pairB = [], []
    for a, b in zip(pairA_list, pairB_list):
        if a.numel() != b.numel():
            print(f"[Warning] pairA shape {a.shape}, pairB shape {b.shape} — fixing.")
            min_len = min(a.numel(), b.numel())
            a = a[:min_len]
            b = b[:min_len]
        fixed_pairA.append(a)
        fixed_pairB.append(b)

    return idsA_padded, idsB_padded, contact_padded, list(seqA_list), list(seqB_list), seq_lensA, seq_lensB, fixed_pairA, fixed_pairB


class DNAContrastiveDataset(Dataset):
    def __init__(self, jsonl_file, data_usage_rate=1.0, tokenizer_name="kmer-1"):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = MyTokenizer(tokenizer_name)
        self.pad_idx = self.tokenizer.token_to_id("[PAD]")
        self.vocab_size = self.tokenizer.vocab_size

        jsonl_file = pathlib.Path(jsonl_file)
        self.sequences = []
        with open(jsonl_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                self.sequences.append(obj["text"].upper())

        if data_usage_rate < 1.0:
            total = len(self.sequences)
            num_samples = int(total * data_usage_rate)
            self.sequences = random.sample(self.sequences, num_samples)

    def __len__(self):
        return len(self.sequences)

    def random_substitution(self, x, vocab_size, sub_rate=0.15, seed=None):
        x_aug = x.clone()

        if seed is not None:
            with torch.random.fork_rng(devices=[]):  # 不影响全局
                torch.manual_seed(seed)
                mask = (torch.rand(x.shape) < sub_rate)
                special_token_count = 5
                random_bases = torch.randint(special_token_count, vocab_size, x.shape)

        x_aug[mask] = random_bases[mask]
        return x_aug

    def encode_sequence(self, seq):
        encoded = self.tokenizer.encode(seq)
        return encoded

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x_ids = self.encode_sequence(seq).ids
        original_len = len(x_ids) # Store original length
        x = torch.tensor(x_ids, dtype=torch.long)
        x1 = self.random_substitution(x, self.vocab_size, seed = 1)
        x2 = self.random_substitution(x, self.vocab_size, seed = 2)

        return x1, x2, original_len


def build_collate_fn(pad_idx):
    def collate_fn(batch):
        x1_list, x2_list, lengths = zip(*batch)
        target_max_len = max(lengths)
        # target_max_len = 1024   # gai

        padded_x1s = []
        padded_x2s = []
        attention_masks = torch.zeros(len(batch), target_max_len, dtype=torch.bool)

        for i, (x1, x2, length) in enumerate(zip(x1_list, x2_list, lengths)):
            # pad x1
            if x1.size(0) < target_max_len:
                padded_x1 = torch.full((target_max_len,), pad_idx, dtype=torch.long)
                padded_x1[:x1.size(0)] = x1
            else:
                padded_x1 = x1
            padded_x1s.append(padded_x1)

            # pad x2
            if x2.size(0) < target_max_len:
                padded_x2 = torch.full((target_max_len,), pad_idx, dtype=torch.long)
                padded_x2[:x2.size(0)] = x2
            else:
                padded_x2 = x2
            padded_x2s.append(padded_x2)

            attention_masks[i, :length] = True

        x1_batch = torch.stack(padded_x1s)
        x2_batch = torch.stack(padded_x2s)

        return x1_batch, x2_batch, attention_masks

    return collate_fn



class DNAMaskDataset(Dataset):
    def __init__(self, jsonl_file, mask_rate=0.15, data_usage_rate=1.0, tokenizer_name="kmer-1"):
        self.tokenizer_name = tokenizer_name

        self.mask_rate = mask_rate
        self.tokenizer = MyTokenizer(tokenizer_name)
        self.pad_idx = self.tokenizer.token_to_id("[PAD]")
        self.mask_idx = self.tokenizer.token_to_id("[MASK]")

        jsonl_file = pathlib.Path(jsonl_file)
        self.sequences = []
        with open(jsonl_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                self.sequences.append(obj["text"].upper())

        if data_usage_rate < 1.0:
            total = len(self.sequences)
            num_samples = int(total * data_usage_rate)
            self.sequences = random.sample(self.sequences, num_samples)

    def __len__(self):
        return len(self.sequences)

    def encode_sequence(self, seq):
        encoded = self.tokenizer.encode(seq)
        return encoded

    def mask_sequence(self, x):
        masked_x = x.clone()
        target = torch.full_like(x, fill_value=self.pad_idx)

        mask = (torch.rand(x.shape) < self.mask_rate) & (x != self.mask_idx)
        masked_x[mask] = self.mask_idx
        target[mask] = x[mask]

        return masked_x, target

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x_ids = self.encode_sequence(sequence).ids
        original_len = len(x_ids)
        x = torch.tensor(x_ids, dtype=torch.long)

        masked_x, target = self.mask_sequence(x)

        return masked_x, target, original_len