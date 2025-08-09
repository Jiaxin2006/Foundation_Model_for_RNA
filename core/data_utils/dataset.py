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
def load_align_pairs(data_dir, max_pairs=1000):
    """
    读取 .ref.fa，输出 [(seqA, seqB, contact)]
    contact: L×L 矩阵，对齐列上双碱基非 gap 且互补配对 =>1
    """
    ref_path = os.path.join(data_dir, "all_ref.fa")
    records = list(SeqIO.parse(ref_path, "fasta"))
    pairs = []
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
        # 简单 8:1:1 划分
        n = len(self.pairs)
        if split == "train":
            self.pairs = self.pairs[:int(0.8*n)]
        elif split == "dev":
            self.pairs = self.pairs[int(0.8*n):int(0.9*n)]
        else:
            self.pairs = self.pairs[int(0.9*n):]

    def __len__(self):
        return len(self.pairs)

    # def __getitem__(self, idx):
    #     seqA, seqB, mat = self.pairs[idx]
    #     idsA = self.tokenizer.encode(seqA).ids
    #     idsB = self.tokenizer.encode(seqB).ids
    #     contact = torch.from_numpy(mat)
    #     print(f"idx={idx} seqA_len={len(seqA)} idsA_len={len(idsA)} contact_dim0={contact.shape[0]}")
    #     print(f"idx={idx} seqB_len={len(seqB)} idsB_len={len(idsB)} contact_dim1={contact.shape[1]}")
    #     assert len(idsA) == contact.shape[0], f"len(idsA)={len(idsA)} vs contact.shape[0]={contact.shape[0]}"
    #     assert len(idsB) == contact.shape[1], f"len(idsB)={len(idsB)} vs contact.shape[1]={contact.shape[1]}"
        
    #     return torch.tensor(idsA), torch.tensor(idsB), contact
    def __getitem__(self, idx):
        seqA, seqB, mat = self.pairs[idx]
        contact_base = np.array(mat, dtype=np.int8)

        # encode 返回 Encoding 对象或类似结构
        encA = self.tokenizer.encode(seqA)
        encB = self.tokenizer.encode(seqB)

        idsA = encA.ids if hasattr(encA, "ids") else encA['input_ids']
        idsB = encB.ids if hasattr(encB, "ids") else encB['input_ids']

        offsetsA = get_offsets_from_encoding(encA)  # list of (start, end)
        offsetsB = get_offsets_from_encoding(encB)

        # offsets长度应该和ids长度匹配，若不匹配，简单截断
        offsetsA = offsetsA[:len(idsA)]
        offsetsB = offsetsB[:len(idsB)]

        token_contact = aggregate_contact_to_tokens(contact_base, offsetsA, offsetsB, agg='any')

        idsA_tensor = torch.tensor(idsA, dtype=torch.long)
        idsB_tensor = torch.tensor(idsB, dtype=torch.long)
        contact_tensor = torch.from_numpy(token_contact).float()

        assert idsA_tensor.size(0) == contact_tensor.size(0), f"idsA {idsA_tensor.size(0)} vs contact rows {contact_tensor.size(0)}"
        assert idsB_tensor.size(0) == contact_tensor.size(1), f"idsB {idsB_tensor.size(0)} vs contact cols {contact_tensor.size(1)}"

        return idsA_tensor, idsB_tensor, contact_tensor

# def align_collate(batch):
#     idsA, idsB, contact = zip(*batch)
#     pad = lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
#     return pad(idsA), pad(idsB), torch.stack(contact)

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
    idsA, idsB, contact = zip(*batch)
    pad = lambda x: pad_sequence(x, batch_first=True)
    return pad(idsA), pad(idsB), pad2d(contact)

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