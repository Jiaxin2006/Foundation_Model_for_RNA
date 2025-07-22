import sys
sys.path.append("../")
from core.data_utils.mytokenizers import MyTokenizer
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


class FTDNADataset(Dataset):
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