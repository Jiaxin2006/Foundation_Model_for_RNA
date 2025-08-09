import torch
import torch.nn as nn
import nni
from nni.nas.nn.pytorch import ModelSpace,ParametrizedModule,ValueChoice, Repeat, LayerChoice
from core.data_utils.mytokenizers import MyTokenizer
from nni.mutable.frozen import ensure_frozen
import nni.nas.evaluator.pytorch.lightning as pl
from nni.nas.evaluator.pytorch.lightning import LightningModule
from collections import OrderedDict
from itertools import product
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
import wandb
import logging
from tqdm import tqdm
from itertools import product
from sklearn.cluster import KMeans
import numpy as np
import json
from sklearn.metrics import pairwise_distances_argmin_min
from dataclasses import dataclass, field
from typing import List, Dict, Callable
import os

# IGNORE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "identity": nn.Identity
}

NORM_MAP = {
    "batchnorm": lambda dim: nn.BatchNorm1d(dim),
    "layernorm": lambda dim: nn.LayerNorm(dim),
    "identity": lambda dim: nn.Identity()
}

@dataclass
class ModelConfig:
    experiment_name: str = "debug"
    tokenizer_name: str = "kmer-1"
    data_usage_rate: float = 0.3
    hidden_dim_list: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256, 512])
    layer_nums_list: List[int] = field(default_factory=lambda: [15, 30, 50])
    embedding_dim: int = 4
    drop_out_list: List[float] = field(default_factory=lambda: [0.0, 0.5])
    kernel_size_list: List[int] = field(default_factory=lambda: [9,15])
    activation_list: List[str] = field(default_factory=lambda: ["relu"])
    norm_list: List[str] = field(default_factory=lambda: ["batchnorm"])

    activation_candidates: Dict[str, nn.Module] = field(init=False)
    norm_candidates: Dict[str, Callable[[int], nn.Module]] = field(init=False)
    dropout_candidates: Dict[str, nn.Module] = field(init=False)

    channel_config_list_done_flag: bool = False

    def __post_init__(self):
        self.activation_candidates = {
            name: ACTIVATION_MAP[name]() for name in self.activation_list
        }

        self.norm_candidates = {
            name: NORM_MAP[name] for name in self.norm_list
        }

        self.dropout_candidates = {
            str(p).replace('.', '_'): nn.Dropout(p) for p in self.drop_out_list
        }


    @staticmethod
    def from_json(path: str) -> "ModelConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        return ModelConfig(**data)


def generate_channel_configs(embedding_dim, hidden_dim_list, layer_nums_list, threshold=1.2, max_representatives_per_group=20):
    hidden_dim_list = sorted(set(hidden_dim_list))  # 去重 + 排序
    representatives_dict = {length: [] for length in layer_nums_list}

    def is_far_enough_log_scale(embedding_dim, candidate, reps, threshold):
        if not reps:
            return True
        candidate_full = [embedding_dim] + candidate
        reps_full = [[embedding_dim] + r for r in reps]
        candidate_log = np.log2(candidate_full)
        reps_log = np.log2(np.array(reps_full))
        dists = np.linalg.norm(reps_log - candidate_log, axis=1)
        return np.all(dists >= threshold)

    final_configs = []

    def dfs(path, depth):
        if len(path) == depth:
            # 将最后一层固定为hidden_dim_list的最大值
            full_seq = path + [max(hidden_dim_list)]
            length = len(full_seq)
            reps = representatives_dict[length]
            if is_far_enough_log_scale(embedding_dim, full_seq, reps, threshold):
                reps.append(full_seq)
            return

        start = 0 if not path else hidden_dim_list.index(path[-1])
        for i in range(start, len(hidden_dim_list)):
            dfs(path + [hidden_dim_list[i]], depth)

    for depth in tqdm(layer_nums_list, desc="Generating configs with pruning"):
        dfs([], depth - 1)

    for length, group in tqdm(representatives_dict.items(), desc="Clustering groups"):
        if len(group) == 0:
            continue
        if len(group) <= max_representatives_per_group:
            final_configs.extend(group)
        else:
            data = np.array([[embedding_dim] + config for config in group])
            k = min(max_representatives_per_group, len(group))
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(data)

            # 找每个聚类中心最近的原始序列作为代表
            closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
            representative_configs = [data[idx][1:].tolist() for idx in closest_indices]
            final_configs.extend(representative_configs)

    return final_configs



def calculate_contrastive_loss(emb_1, emb_2, temperature=0.5):
    """
    emb_1, emb_2: [batch_size, embed_dim] — Two augmented views
    temperature: Scaling factor for logits
    """
    batch_size = emb_1.shape[0]

    # Normalize embeddings
    emb_1 = F.normalize(emb_1, dim=1)
    emb_2 = F.normalize(emb_2, dim=1)

    # Concatenate for easier computation
    embeddings = torch.cat([emb_1, emb_2], dim=0)  # [2 * batch_size, embed_dim]

    # Cosine similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)  # [2*B, 2*B]
        
    # Remove self similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=embeddings.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    # Positive pairs: [i, i + batch_size] and [i + batch_size, i]
    positives = torch.cat([torch.arange(batch_size, device=embeddings.device),
                        torch.arange(batch_size, device=embeddings.device)]) + \
                torch.tensor([0, batch_size], device=embeddings.device).repeat_interleave(batch_size)

    # Get the positive similarities
    sim_pos = torch.sum(emb_1 * emb_2, dim=-1)
    sim_pos = torch.cat([sim_pos, sim_pos], dim=0)  # [2 * batch_size]

    # Compute logits
    logits = similarity_matrix / temperature
    sim_pos = sim_pos / temperature

    # Cross entropy loss: positives vs all others
    labels = torch.arange(2 * batch_size, device=embeddings.device)
    loss = F.cross_entropy(torch.cat([sim_pos.unsqueeze(1), logits], dim=1), labels)

    return loss






class CNN_ContrastiveLearning_ModelSpace(ModelSpace):
    def __init__(self, config: ModelConfig):
        super().__init__()
        logger.info(f"Initializing ModelSpace....")
        self.modelSpaceconfig = config
        self.tokenizer_name = self.modelSpaceconfig.tokenizer_name
        self.channel_config_list_done_flag = self.modelSpaceconfig.channel_config_list_done_flag
        self.experiment_name = self.modelSpaceconfig.experiment_name
        self.hidden_dim_list = self.modelSpaceconfig.hidden_dim_list
        max_channel = max(self.hidden_dim_list)
        self.layer_nums_list = self.modelSpaceconfig.layer_nums_list
        self.embedding_dim = self.modelSpaceconfig.embedding_dim
        self.drop_out_list = self.modelSpaceconfig.drop_out_list
        self.activation_candidates = self.modelSpaceconfig.activation_candidates
        self.norm_candidates = self.modelSpaceconfig.norm_candidates
        self.dropout_candidates = self.modelSpaceconfig.dropout_candidates

        self.tokenizer = MyTokenizer(self.tokenizer_name)
        pad_idx = self.tokenizer.token_to_id("[PAD]")
        logger.info(f"pad_idx is {pad_idx}")
        vocab_size = self.tokenizer.vocab_size

        # CNN config
        self.kernel_size = 9
        self.padding = (self.kernel_size - 1)//2

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim, padding_idx=pad_idx)

        channel_config_path = f"/projects/slmreasoning/yifang/configs/{self.experiment_name}/channel_configs.json"
        if not self.channel_config_list_done_flag:
            logger.info("Building channel_config_list....")
            self.channel_config_list = generate_channel_configs(self.embedding_dim, self.hidden_dim_list, self.layer_nums_list)
            os.makedirs(os.path.dirname(channel_config_path), exist_ok=True)
            with open(channel_config_path, "w") as f:
                json.dump({"channel_config_list": self.channel_config_list}, f, indent=4)
                
        else:
            logger.info("Reading channel_config_list....")
            if not os.path.exists(channel_config_path):
                raise FileNotFoundError(f"Expected channel config at {channel_config_path}, but it does not exist.")
            with open(channel_config_path, "r") as f:
                data = json.load(f)
                self.channel_config_list = data["channel_config_list"]

        # Shared CNN layers
        logger.info(f"Building shared conv layers....")
        self.conv_pool = nn.ModuleDict()
        combinations = [(in_ch, out_ch) for in_ch in self.hidden_dim_list for out_ch in self.hidden_dim_list if in_ch <= out_ch]
        for in_ch, out_ch in tqdm(combinations, desc="Building conv layers"):
            key = f"{in_ch}_{out_ch}"
            self.conv_pool[key] = nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size, padding=self.padding)
        max_value = max(self.hidden_dim_list)
        for in_ch in self.hidden_dim_list:
            key = f"{in_ch}_{max_value}"
            if key not in list(self.conv_pool.keys()):
                self.conv_pool[key] = nn.Conv1d(in_ch, max_value, kernel_size=self.kernel_size, padding=self.padding)


        logger.info(f"Creating LayerChoices....")
        self.cnn_block_candidates = OrderedDict()
        for i, config in enumerate(tqdm(self.channel_config_list, desc="Building LayerChoices")):
            layers = []
            in_ch = self.embedding_dim
            for idx, out_ch in enumerate(config):
                conv = self.conv_pool[f"{in_ch}_{out_ch}"]

                act_choice = LayerChoice(self.activation_candidates, label=f"act_{i}_{idx}")
                
                norm_layer_choice = LayerChoice(
                    {k: (v(out_ch) if k=="batchnorm" else v(out_ch)) for k,v in self.norm_candidates.items()}, 
                    label=f"norm_{i}_{idx}"
                )

                # 构建这个block
                block = nn.Sequential(
                    conv,
                    norm_layer_choice,
                    act_choice,
                )
                layers.append(block)
                in_ch = out_ch
            
            self.cnn_block_candidates[f"cnn_{i}_{config}"] = nn.Sequential(*layers)

        # 用LayerChoice选cnn结构
        self.cnn_block = LayerChoice(self.cnn_block_candidates, label="cnn_arch")
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop_choice = LayerChoice(self.dropout_candidates, label=f"dropout_{i}_{idx}")

    def forward(self, input_ids):
        x = self.embedding(input_ids).permute(0, 2, 1)
        x = self.cnn_block(x)
        x = self.drop_choice(x)
        x = self.pool(x).squeeze(-1)
        return x



class CNN_ContrastiveLearning_PretrainModel(LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        model = CNN_ContrastiveLearning_ModelSpace(config=config)
        self.set_model(model)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedding = self.model(x)  # [batch_size, embed_dim]

    def training_step(self, batch, batch_idx):
        aug_1, aug_2 = batch
        emb_1 = self.model(aug_1)
        emb_2 = self.model(aug_2)
        loss = calculate_contrastive_loss(emb_1, emb_2)
        logger.info(f"train loss {loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        aug_1, aug_2 = batch

        emb_1 = self.model(aug_1)
        emb_2 = self.model(aug_2)
        loss = calculate_contrastive_loss(emb_1, emb_2)
        logger.info(f"val loss {loss.item():.4f}")
        return loss


class CNN_MaskedModeling_ModelSpace(ModelSpace):
    def __init__(self, config: ModelConfig):
        super().__init__()
        logger.info(f"Initializing ModelSpace....")
        self.modelSpaceconfig = config
        self.tokenizer_name = self.modelSpaceconfig.tokenizer_name
        self.channel_config_list_done_flag = self.modelSpaceconfig.channel_config_list_done_flag
        self.experiment_name = self.modelSpaceconfig.experiment_name
        self.hidden_dim_list = self.modelSpaceconfig.hidden_dim_list
        max_channel = max(self.hidden_dim_list)
        self.layer_nums_list = self.modelSpaceconfig.layer_nums_list
        self.embedding_dim = self.modelSpaceconfig.embedding_dim
        self.drop_out_list = self.modelSpaceconfig.drop_out_list
        self.activation_candidates = self.modelSpaceconfig.activation_candidates
        self.norm_candidates = self.modelSpaceconfig.norm_candidates
        self.dropout_candidates = self.modelSpaceconfig.dropout_candidates

        self.tokenizer = MyTokenizer(self.tokenizer_name)
        self.pad_idx = self.tokenizer.token_to_id("[PAD]")
        logger.info(f"pad_idx is {self.pad_idx}")
        vocab_size = self.tokenizer.vocab_size

        # CNN config
        self.kernel_size = 9
        self.padding = (self.kernel_size - 1)//2

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim, padding_idx=self.pad_idx)

        channel_config_path = f"/projects/slmreasoning/yifang/configs/{self.experiment_name}/channel_configs.json"
        if not self.channel_config_list_done_flag:
            logger.info("Building channel_config_list....")
            self.channel_config_list = generate_channel_configs(self.embedding_dim, self.hidden_dim_list, self.layer_nums_list)
            os.makedirs(os.path.dirname(channel_config_path), exist_ok=True)
            with open(channel_config_path, "w") as f:
                json.dump({"channel_config_list": self.channel_config_list}, f, indent=4)
                
        else:
            logger.info("Reading channel_config_list....")
            if not os.path.exists(channel_config_path):
                raise FileNotFoundError(f"Expected channel config at {channel_config_path}, but it does not exist.")
            with open(channel_config_path, "r") as f:
                data = json.load(f)
                self.channel_config_list = data["channel_config_list"]

        # Shared CNN layers
        logger.info(f"Building shared conv layers....")
        self.conv_pool = nn.ModuleDict()
        combinations = [(in_ch, out_ch) for in_ch in self.hidden_dim_list for out_ch in self.hidden_dim_list if in_ch <= out_ch]
        for in_ch, out_ch in tqdm(combinations, desc="Building conv layers"):
            key = f"{in_ch}_{out_ch}"
            self.conv_pool[key] = nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size, padding=self.padding)
        max_value = max(self.hidden_dim_list)
        for in_ch in self.hidden_dim_list:
            key = f"{in_ch}_{max_value}"
            if key not in list(self.conv_pool.keys()):
                self.conv_pool[key] = nn.Conv1d(in_ch, max_value, kernel_size=self.kernel_size, padding=self.padding)


        logger.info(f"Creating LayerChoices....")
        self.cnn_block_candidates = OrderedDict()
        for i, config in enumerate(tqdm(self.channel_config_list, desc="Building LayerChoices")):
            layers = []
            in_ch = self.embedding_dim
            for idx, out_ch in enumerate(config):
                conv = self.conv_pool[f"{in_ch}_{out_ch}"]

                act_choice = LayerChoice(self.activation_candidates, label=f"act_{i}_{idx}")
                
                norm_layer_choice = LayerChoice(
                    {k: (v(out_ch) if k=="batchnorm" else v(out_ch)) for k,v in self.norm_candidates.items()}, 
                    label=f"norm_{i}_{idx}"
                )

                # 构建这个block
                block = nn.Sequential(
                    conv,
                    act_choice,
                    norm_layer_choice,
                )
                layers.append(block)
                in_ch = out_ch
            
            self.cnn_block_candidates[f"cnn_{i}_{config}"] = nn.Sequential(*layers)

        # 用LayerChoice选cnn结构
        self.cnn_block = LayerChoice(self.cnn_block_candidates, label="cnn_arch")
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop_choice = LayerChoice(self.dropout_candidates, label=f"dropout_{i}_{idx}")
        self.pred_head = nn.Linear(in_ch, vocab_size)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)     
        x = self.cnn_block(x)        
        x = self.drop_choice(x)
        B, C, S = x.shape
        x = x.reshape(B * S, C) 
        logits = self.pred_head(x) 
        logits = logits.reshape(B, -1, S)
        return logits



class CNN_MaskedModeling_PretrainModel(LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        model = CNN_MaskedModeling_ModelSpace(config=config)
        self.set_model(model)
        self.pad_idx = model.pad_idx
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, x):
        logits = self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        loss = self.__calculate_loss(logits, target)
        logger.info(f"train_loss: {loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        logits = self.model(x)
        
        loss = self.__calculate_loss(logits, target)
        logger.info(f"val_loss: {loss.item():.4f}")
        return loss

    def __calculate_loss(self, logits, target):
        loss = self.loss_fn(logits, target)
        return loss



