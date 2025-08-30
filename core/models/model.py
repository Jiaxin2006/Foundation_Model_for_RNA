import torch
import torch.nn as nn
import nni
from nni.nas.nn.pytorch import ModelSpace,ParametrizedModule,ValueChoice, Repeat, LayerChoice
from core.data_utils.mytokenizers import MyTokenizer
from core.models.utils import generate_dim_configs, generate_layer_type_configs, merge_configs, extract_layer_sets, calculate_contrastive_loss, build_model_with_index
from core.models.modules import CNNModule, MambaModule, HyenaModule, GRUModule, RNNModule, LSTMModule, TransformerModule, BiMambaModule, OnlyMambaModule, build_module
from nni.mutable.frozen import ensure_frozen
import nni.nas.evaluator.pytorch.lightning as pl
from nni.nas.evaluator.pytorch.lightning import LightningModule
from collections import OrderedDict
from itertools import product
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import itertools
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 针对每个实验所用的参数：模型、tokenizer、dimension等
@dataclass
class ModelConfig:
    experiment_name: str = "debug"

    ## data
    tokenizer_name: str = "kmer-1"
    data_usage_rate: float = 0.3

    ## dimension，从哪些维度中选
    embedding_dim: int = 64
    hidden_dim_list: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    max_representatives_per_group: int = 5
    
    ## layer_type，3层（不同模型架构的组合）的选5个
    architecture_list: List[str] = field(default_factory=lambda: ["cnn","gru","transformer","mamba","hyena"])
    layer_nums_dict: Dict[str, int] = field(default_factory=lambda: {"3": 5, "4": 10, "5": 10})
    
    ## flags，是否已经生成过这个架构
    architecture_config_flag: bool = False

    @staticmethod
    def from_json(path: str) -> "ModelConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        return ModelConfig(**data)


def register_shape_hooks(model):
    def hook_fn(module, input, output):
        class_name = module.__class__.__name__
        module_idx = len(shape_info)
        m_key = f"{module_idx:02d}-{class_name}"
        # 记录输入输出 shape
        shape_info[m_key] = {
            "input_shape": [tuple(t.shape) for t in input if hasattr(t, "shape")],
            "output_shape": tuple(output.shape) if hasattr(output, "shape") else str(type(output))
        }
        print(f"[ShapeTracer] {m_key} | in: {shape_info[m_key]['input_shape']} | out: {shape_info[m_key]['output_shape']}")

    shape_info = {}
    for name, module in model.named_modules():
        # 跳过容器类模块，防止信息过多
        if not list(module.children()):
            module.register_forward_hook(hook_fn)
    return shape_info

def match(self, z0_list, z1_list):
    """
    z0_list, z1_list: lists of tensors, 每个元素形状为 [L, d]（单样本的位置向量）
    返回:
      match_scores: list of [L1, L2] per-sample similarity matrices
      logits: tensor [B] per-sample scalar score (same as RNABERT)
    """
    match_scores = []
    logits = []
    for z0, z1 in zip(z0_list, z1_list):
        # cosine similarity over last dim -> [L0, L1]
        match_score = nn.CosineSimilarity(dim=2, eps=1e-6)(
            z0.unsqueeze(1).repeat(1, z1.shape[0], 1),
            z1.unsqueeze(0).repeat(z0.shape[0], 1, 1)
        )  # [L0, L1]
        s = 1.3 * match_score  # scale factor as in RNABERT
        # soft align aggregation
        a, b = F.softmax(s, 1), F.softmax(s, 0)
        c = a + b - a * b
        c = torch.sum(c * s) / (torch.sum(c) + 1e-12)  # scalar
        match_scores.append(match_score)  # keep the matrix (cosine), not collapsed
        logits.append(c.view(-1))
    logits = torch.stack(logits, 0)  # [B]
    return match_scores, logits

class ContrastiveLearning_ModelSpace(ModelSpace):
    def __init__(self, config: ModelConfig):
        super().__init__()
        logger.info(f"Initializing ModelSpace....")
        self.experiment_name = config.experiment_name

        # tokenizer
        self.tokenizer_name = config.tokenizer_name
        self.tokenizer = MyTokenizer(self.tokenizer_name)
        pad_idx = self.tokenizer.token_to_id("[PAD]")
        #logger.info(f"pad_idx is {pad_idx}")
        vocab_size = self.tokenizer.vocab_size

        # dimension
        self.embedding_dim = config.embedding_dim
        self.hidden_dim_list = config.hidden_dim_list
        max_channel = max(self.hidden_dim_list)
        self.max_representatives_per_group = config.max_representatives_per_group

        # architecture
        self.layer_nums_dict = config.layer_nums_dict
        self.architecture_list = config.architecture_list
        self.architecture_config_flag = config.architecture_config_flag

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim, padding_idx=pad_idx)

        # generate architecthre configs
        architecture_config_path = f"/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/configs/{self.experiment_name}/architecture_configs.json"
        if not self.architecture_config_flag:
            logger.info("Building architecture configs....")

            self.dim_config_list = generate_dim_configs(embedding_dim = self.embedding_dim, 
                                            hidden_dim_list = self.hidden_dim_list, 
                                            layer_nums_dict = self.layer_nums_dict, 
                                            max_representatives_per_group=self.max_representatives_per_group)

            self.layer_types_config_list = generate_layer_type_configs(self.architecture_list, self.layer_nums_dict)
            logger.info(f"dim_config_list length is {len(self.dim_config_list)}")
            logger.info(f"layer_types_config_list length is {len(self.layer_types_config_list)}")
            self.architecture_config_list = merge_configs(self.dim_config_list, self.layer_types_config_list, self.embedding_dim)
            

            os.makedirs(os.path.dirname(architecture_config_path), exist_ok=True)
            with open(architecture_config_path, "w") as f:
                json.dump({"architecture_config_list": self.architecture_config_list}, f, indent=4)
                
        else:
            logger.info("Reading architecture configs....")
            if not os.path.exists(architecture_config_path):
                raise FileNotFoundError(f"Expected channel config at {dim_config_path}, but it does not exist.")
            with open(architecture_config_path, "r") as f:
                data = json.load(f)
                self.architecture_config_list = data["architecture_config_list"]

        # Shared Modules，把每一个用到的module初始化
        logger.info(f"Building shared modules....")
        self.module_set = extract_layer_sets(self.architecture_config_list)
        logger.info(f"self.module_set is {self.module_set}")
        self.module_pools = nn.ModuleDict()
        
        for layer_frozenset in tqdm(self.module_set, desc="Building modules"):
            config = dict(layer_frozenset)
            layer_type = config["layer_type"]
            input_dim = config["input_dim"]
            output_dim = config["output_dim"]
            key = f"{layer_type}_{input_dim}_{output_dim}"
            try:
                module = build_module(layer_type, input_dim, output_dim)
                self.module_pools[key] = module     # 构建一个module的pool
            except ValueError as e:
                print(f"Skipping module due to error: {e} for config: {config}")
    
        #logger.info(f"self.module_pools is {self.module_pools}")
        # 用module组成的不同的path，是什么样子
        logger.info(f"Creating LayerChoices....")
        self.paths_candidates = OrderedDict()
        for i, configs in enumerate(tqdm(self.architecture_config_list, desc="Building Paths")):
            layers = []
            for idx, config in enumerate(configs):
                layer_type = config["layer_type"]
                input_dim = config["input_dim"]
                output_dim = config["output_dim"]
                key = f"{layer_type}_{input_dim}_{output_dim}"
                block = self.module_pools[key]      # 从pool中把每一个module用key筛选出来
                layers.append(block)
            
            self.paths_candidates[f"path_{i}"] = nn.Sequential(*layers)

        # 用LayerChoice选path结构
        self.path = LayerChoice(self.paths_candidates, label="path")
        last_layer = list(self.path.children())[-1]
        if isinstance(last_layer, (MambaModule, RNNModule, GRUModule, LSTMModule)):     # 单向模型取pooling不是很合理
            self.pooling_flag = False
        else:
            self.pooling_flag = True

    def masked_average_pooling(self, x, attention_mask):    # 通过attention 把padding的部分去掉 不影响后续计算
        mask_expanded = attention_mask.unsqueeze(-1).float() 
        x_masked = x * mask_expanded 
        lengths = attention_mask.sum(dim=1).float().unsqueeze(-1) 
        lengths = lengths.clamp(min=1e-9)
        pooled_output = x_masked.sum(dim=1) / lengths 

        return pooled_output

    def masked_last_token(self, x, attention_mask):
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = x.size(0)
            last_token = x[torch.arange(batch_size), sequence_lengths]
        else:
            last_token = x[:, -1, :]
        return last_token

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.path(x)
        if self.pooling_flag == True:
            x = self.masked_average_pooling(x, attention_mask)
        else:
            x = self.masked_last_token(x, attention_mask)
        return x


class MaskedModeling_ModelSpace(ModelSpace):
    def __init__(self, config: ModelConfig):
        super().__init__()
        logger.info(f"Initializing ModelSpace....")
        self.experiment_name = config.experiment_name

        # tokenizer
        self.tokenizer_name = config.tokenizer_name
        self.tokenizer = MyTokenizer(self.tokenizer_name)
        self.pad_idx = self.tokenizer.token_to_id("[PAD]")

        vocab_size = self.tokenizer.vocab_size

        # dimension
        self.embedding_dim = config.embedding_dim
        self.hidden_dim_list = config.hidden_dim_list
        max_channel = max(self.hidden_dim_list)
        self.max_representatives_per_group = config.max_representatives_per_group

        # architecture
        self.layer_nums_dict = config.layer_nums_dict
        self.architecture_list = config.architecture_list

        self.architecture_config_flag = config.architecture_config_flag

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim, padding_idx=self.pad_idx)

        # generate architecthre configs
        architecture_config_path = f"/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/configs/{self.experiment_name}/architecture_configs.json"
        if not self.architecture_config_flag:
            logger.info("Building architecture configs....")

            self.dim_config_list = generate_dim_configs(embedding_dim = self.embedding_dim, 
                                            hidden_dim_list = self.hidden_dim_list, 
                                            layer_nums_dict = self.layer_nums_dict, 
                                            max_representatives_per_group=self.max_representatives_per_group)

            self.layer_types_config_list = generate_layer_type_configs(self.architecture_list, self.layer_nums_dict)
            logger.info(f"dim_config_list length is {len(self.dim_config_list)}")
            logger.info(f"layer_types_config_list length is {len(self.layer_types_config_list)}")
            self.architecture_config_list = merge_configs(self.dim_config_list, self.layer_types_config_list, self.embedding_dim)

            os.makedirs(os.path.dirname(architecture_config_path), exist_ok=True)
            with open(architecture_config_path, "w") as f:
                json.dump({"architecture_config_list": self.architecture_config_list}, f, indent=4)
                
        else:
            logger.info("Reading architecture configs....")
            if not os.path.exists(architecture_config_path):
                raise FileNotFoundError(f"Expected channel config at {architecture_config_path}, but it does not exist.")
            with open(architecture_config_path, "r") as f:
                data = json.load(f)
                self.architecture_config_list = data["architecture_config_list"]


        
        # Shared Modules
        logger.info(f"Building shared modules....")
        self.module_set = extract_layer_sets(self.architecture_config_list)
        logger.info(f"self.module_set is {self.module_set}")
        self.module_pools = nn.ModuleDict()
        
        for layer_frozenset in tqdm(self.module_set, desc="Building modules"):
            config = dict(layer_frozenset)
            layer_type = config["layer_type"]
            input_dim = config["input_dim"]
            output_dim = config["output_dim"]
            key = f"{layer_type}_{input_dim}_{output_dim}"
            try:
                module = build_module(layer_type, input_dim, output_dim)
                self.module_pools[key] = module
            except ValueError as e:
                print(f"Skipping module due to error: {e} for config: {config}")
    
        #logger.info(f"self.module_pools is {self.module_pools}")

        logger.info(f"Creating LayerChoices....")
        self.paths_candidates = OrderedDict()
        for i, configs in enumerate(tqdm(self.architecture_config_list, desc="Building Paths")):
            layers = []
            for idx, config in enumerate(configs):
                layer_type = config["layer_type"]
                input_dim = config["input_dim"]
                output_dim = config["output_dim"]
                key = f"{layer_type}_{input_dim}_{output_dim}"
                block = self.module_pools[key]
                layers.append(block)
            
            self.paths_candidates[f"path_{i}"] = nn.Sequential(*layers)

        # 用LayerChoice选path结构
        self.path = LayerChoice(self.paths_candidates, label="path")
        
        self.pred_head = nn.Linear(max_channel, vocab_size)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        # print(f"输入形状: {input_ids.shape}")
        x = self.embedding(input_ids)
        # print(f"嵌入后: {x.shape}")
        
        x = self.path(x)
        # print(f"路径后: {x.shape}")
        
        logits = self.pred_head(x)
        # print(f"预测头后: {logits.shape}")
        
        return logits.permute(0, 2, 1)


class ContrastiveLearning_PretrainModel(LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        model = ContrastiveLearning_ModelSpace(config=config)
        self.set_model(model)
        # shape_info = register_shape_hooks(self.model)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedding = self.model(x)  # [batch_size, embed_dim]

    def training_step(self, batch, batch_idx):
        aug_1, aug_2, attention_mask = batch

        emb_1 = self.model(aug_1, attention_mask)
        emb_2 = self.model(aug_2, attention_mask)
        loss = calculate_contrastive_loss(emb_1, emb_2)
        logger.info(f"train loss {loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        aug_1, aug_2, attention_mask = batch

        emb_1 = self.model(aug_1, attention_mask)
        emb_2 = self.model(aug_2, attention_mask)
        loss = calculate_contrastive_loss(emb_1, emb_2)
        logger.info(f"val loss {loss.item():.4f}")
        return loss


class MaskedModeling_PretrainModel(LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        model = MaskedModeling_ModelSpace(config=config)
        self.set_model(model)
        self.pad_idx = model.pad_idx
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        # shape_info = register_shape_hooks(self.model)

    def forward(self, x):
        logits = self.model(x)

    def training_step(self, batch, batch_idx):
        x, target, _ = batch
        logits = self.model(x)
        loss = self.__calculate_loss(logits, target)
        logger.info(f"train_loss: {loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, target, _ = batch
        logits = self.model(x)
        loss = self.__calculate_loss(logits, target)
        logger.info(f"val_loss: {loss.item():.4f}")
        return loss

    def __calculate_loss(self, logits, target):
        loss = self.loss_fn(logits, target)
        return loss

    # new，TODO
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)


# 模型评估部分，把整个模型固定下来之后做聚类
# TODO：添加类似的 FixedModel
class CLS_FixedModel(nn.Module):
    def __init__(self, num_classes, embedding, path: nn.Sequential, mask_pred_head: nn.Linear = None):
        super().__init__()
        self.embedding = embedding
        self.path = path
        if mask_pred_head is None:  # 如果是masked modeling会有pred_head，相当于把它删掉重新训练
            self.mask_pred_head = nn.Identity()
        else:
            self.mask_pred_head = mask_pred_head

        self.cls_head = nn.LazyLinear(num_classes)  # 加入分类的cls_head; LazyLinear：不能确定输入的维度时，可以不写，会自动识别
        # 默认配置
        class DefaultConfig:
            attention_probs_dropout_prob = 0.0
            hidden_act = "gelu"
            hidden_dropout_prob = 0.0
            initializer_range = 0.02
            intermediate_size = 40
            max_position_embeddings = 440
            vocab_size = 6
            ss_size = 8
            type_vocab_size = 2
            margin_FP = 0.1
            margin_FN = 0.05
            num_attention_heads = 12
            multiple = 10
            num_hidden_layers = 6
            gap_extension = 0.1
            gap_opening = 1.0
            optimizer = "AdamW"
            adam_lr = 0.0010162239966782008

        # 使用方式
        self.config = DefaultConfig()

        
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        last_layer = list(self.path.children())[-1]
        if isinstance(last_layer, (MambaModule, RNNModule, GRUModule, LSTMModule)):
            self.pooling_flag = False
        else:
            self.pooling_flag = True

    def masked_average_pooling(self, x, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float() 
        x_masked = x * mask_expanded 
        lengths = attention_mask.sum(dim=1).float().unsqueeze(-1) 
        lengths = lengths.clamp(min=1e-9)
        pooled_output = x_masked.sum(dim=1) / lengths 

        return pooled_output

    def masked_last_token(self, x, attention_mask):
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = x.size(0)
            last_token = x[torch.arange(batch_size), sequence_lengths]
        else:
            last_token = x[:, -1, :]
        return last_token

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.path(x)
        if self.pooling_flag == True:
            feature = self.masked_average_pooling(x, attention_mask)
        else:
            feature = self.masked_last_token(x, attention_mask)
        mask_pred_logits = self.mask_pred_head(x).permute(0,2,1)    # 这里保留pred_head是因为之前做了repre
        cls_logit = self.cls_head(feature)      # 做classification

        return cls_logit, feature, mask_pred_logits
    
    def get_encoded_layers(self, input_ids, attention_mask):
        """获取编码层输出，用于MUL训练"""
        x = self.embedding(input_ids)
        encoded_layers = self.path(x)
        return encoded_layers
    
    def gapscore(self, index, seq_len):
        seq_len = int(seq_len)
        index = index[:seq_len].to('cpu').detach().numpy().copy()
        all_zeros = seq_len - np.count_nonzero(index)
        extend_zeros = seq_len - np.count_nonzero(np.insert(index[:-1], 0, 1) + index)
        open_zeros = all_zeros - extend_zeros
        return (-1*self.config.gap_opening + -1*self.config.gap_extension ) * open_zeros + -1*self.config.gap_extension * extend_zeros
    
    def em(self, encoded_layers, seq_lens):
        """将编码层输出转换为序列列表格式"""
        z_list = []
        for b in range(encoded_layers.size(0)):
            # 根据实际序列长度截取
            actual_len = seq_lens[b] if isinstance(seq_lens, list) else encoded_layers.size(1)
            z_list.append(encoded_layers[b, :actual_len, :])  # [seq_len, hidden_dim]
        return z_list
    
    def match(self, z0_list, z1_list):
        match_scores = []
        logits = []
        for z0, z1 in zip(z0_list, z1_list):
            match_score = nn.CosineSimilarity(dim=2, eps=1e-6)(z0.unsqueeze(1).repeat(1, z1.shape[0],1) , z1.unsqueeze(0).repeat(z0.shape[0], 1,1))
            s = 1.3 * match_score
			
            a, b = F.softmax(s, 1), F.softmax(s, 0)
			# soft align
            c = a + b - a*b
            c = torch.sum(c*s)/torch.sum(c)
            match_scores.append(match_score)
            logits.append(c.view(-1))

        logits = torch.stack(logits, 0)
        return match_scores, logits
    
    def structural_learning(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """结构学习损失计算"""
        reference_alignment_score = self.calc_reference_alignment_score(
            bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1
        )
        prediction_alignment_score, _ = self.calc_prediction_alignment_score(
            bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1
        )
        return prediction_alignment_score - reference_alignment_score
    
    def match_bert_score(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """
        bert_score: tensor [L0, L1] for this sample
        common_index_0/1: can be
        - 1D LongTensor of paired indices (len K)  (e.g. pair_idx_A)
        - 1D binary mask of length L0/L1 (0/1)
        - empty tensor
        返回: omega: 1D tensor of bert_score for each paired (i,j). If no pairs -> empty tensor.
        """
        device = bert_score.device

        # empty guard
        if common_index_0 is None or common_index_1 is None:
            return torch.tensor([], device=device)
        if common_index_0.numel() == 0 or common_index_1.numel() == 0:
            return torch.tensor([], device=device)

        # Helper: convert mask->index list
        def to_index_list(x):
            # if already LongTensor of indices (likely): return flattened
            if x.dtype in (torch.int64, torch.long) and x.dim() == 1:
                return x.view(-1).long()
            # if bool / byte mask or 0/1 mask
            if x.dim() == 1:
                # works for ByteTensor / BoolTensor / LongTensor containing 0/1
                idx = (x == 1).nonzero(as_tuple=False).view(-1).long()
                return idx
            # fallback: try nonzero
            idx = x.nonzero(as_tuple=False).view(-1).long()
            return idx

        idx0 = to_index_list(common_index_0)
        idx1 = to_index_list(common_index_1)

        # If lengths mismatch, warn and truncate to min length (safe fallback)
        if idx0.numel() != idx1.numel():
            print(f"[Warning][match_bert_score] unequal pair lengths: idx0={idx0.numel()}, idx1={idx1.numel()} -> truncating to min.")
            minlen = min(idx0.numel(), idx1.numel())
            if minlen == 0:
                return torch.tensor([], device=device)
            idx0 = idx0[:minlen]
            idx1 = idx1[:minlen]

        # now advanced-index bert_score with paired coordinates
        # idx0, idx1 are 1D tensors of same length K
        omega = bert_score[idx0.long(), idx1.long()]  # returns 1D length K
        return omega


    def calc_reference_alignment_score(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """计算参考对齐分数"""
        reference_alignment_score = 0.0
        
        # 添加gap分数
        reference_alignment_score += self.gapscore(common_index_0, seq_len_0)
        reference_alignment_score += self.gapscore(common_index_1, seq_len_1)
        
        if common_index_0.numel() == 0 or common_index_1.numel() == 0:
            return torch.tensor(0.0, device=bert_score.device)

        # 添加匹配分数
        omega = self.match_bert_score(bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1)
        reference_alignment_score += torch.sum(omega)
        
        return reference_alignment_score

    def margin_score(self, common_index_A_B, common_index_0, common_index_1):
        """
        compute FP/FN counts robustly. Accepts pair-lists or masks.
        """
        # get predicted pairs count and reference pairs count robustly
        def list_from_mask_or_indices(x):
            if x is None or x.numel() == 0:
                return []
            if x.dtype in (torch.int64, torch.long) and x.dim() == 1:
                return x.view(-1).tolist()
            return (x == 1).nonzero(as_tuple=False).view(-1).tolist()

        predA = list_from_mask_or_indices(common_index_A_B[0])
        predB = list_from_mask_or_indices(common_index_A_B[1])
        refA = list_from_mask_or_indices(common_index_0)
        refB = list_from_mask_or_indices(common_index_1)

        # form pair-keys as we did before (i*10000 + j) to compute intersections
        a = set([int(i) * 10000 + int(j) for i, j in zip(predA, predB)])
        b = set([int(i) * 10000 + int(j) for i, j in zip(refA, refB)])
        len_pred_match = len(predA)
        len_ref_match = len(refA)
        len_TP = len(a & b)
        len_FP = len_pred_match - len_TP
        len_FN = len_ref_match - len_TP
        return len_FP * self.config.margin_FP + len_FN * self.config.margin_FN

    def margin_matrix(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """
        Create margin matrices of same shape as bert_score.
        Accepts index-list or mask as common_index_0/1.
        """
        L0, L1 = bert_score.shape
        device = bert_score.device

        # default full-FP margin and zero FN margin
        margin_mat_FP = torch.ones((L0, L1), device=device) * self.config.margin_FP
        margin_mat_FN = torch.zeros((L0, L1), device=device)

        # if either index empty -> nothing to zero-out / set
        if common_index_0 is None or common_index_1 is None:
            return margin_mat_FP, margin_mat_FN
        if common_index_0.numel() == 0 or common_index_1.numel() == 0:
            return margin_mat_FP, margin_mat_FN

        # reuse match_bert_score helper logic: get index lists
        def to_index_list(x):
            if x.dtype in (torch.int64, torch.long) and x.dim() == 1:
                return x.view(-1).long()
            idx = (x == 1).nonzero(as_tuple=False).view(-1).long()
            return idx

        idx0 = to_index_list(common_index_0)
        idx1 = to_index_list(common_index_1)

        # make sure same length
        if idx0.numel() != idx1.numel():
            print(f"[Warning][margin_matrix] unequal pair lengths: idx0={idx0.numel()}, idx1={idx1.numel()} -> truncating.")
            minlen = min(idx0.numel(), idx1.numel())
            idx0 = idx0[:minlen]
            idx1 = idx1[:minlen]

        if idx0.numel() > 0:
            # advanced indexing to set specific (i,j) entries
            margin_mat_FP[idx0.long(), idx1.long()] = 0.0
            margin_mat_FN[idx0.long(), idx1.long()] = self.config.margin_FN

        return margin_mat_FP, margin_mat_FN

    
    def calc_prediction_alignment_score(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """计算预测对齐分数"""
        # 创建虚拟序列（用于对齐算法）
        sequence_a = "N" * seq_len_0
        sequence_b = "N" * seq_len_1
        
        # 计算边际矩阵
        margin_mat_FP, margin_mat_FN = self.margin_matrix(
            bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1
        )
        
        # 使用简化的全局对齐算法（你需要根据实际情况实现或导入Aln_C）
        try:
            # 如果有Aln_C模块
            import alignment_C as Aln_C
            common_index_A_B = Aln_C.global_aln(
                torch.flatten(bert_score.T).tolist(), 
                torch.flatten(margin_mat_FP.T).tolist(), 
                torch.flatten(margin_mat_FN.T).tolist(), 
                sequence_a, sequence_b, seq_len_0, seq_len_1, 
                self.config.gap_opening, self.config.gap_extension, 0, 0
            )
            common_index_A_B = torch.tensor(common_index_A_B).to(self.device).view(2, -1)
        except ImportError:
            # 简化版本的对齐算法
            print("wrong")
        
        prediction_alignment_score = self.calc_reference_alignment_score(
            bert_score, common_index_A_B[0], seq_len_0, common_index_A_B[1], seq_len_1
        ) + self.margin_score(common_index_A_B, common_index_0, common_index_1)
        
        return prediction_alignment_score, common_index_A_B
    
    def train_MUL(self, z0_list, z1_list, common_index_0, common_index_1, seq_len_0, seq_len_1):   
        bert_scores, _ = self.match(z0_list, z1_list)
        loss = 0.0
        for i, bert_score in enumerate(bert_scores):
            loss += self.structural_learning(bert_score, common_index_0[i], seq_len_0[i], common_index_1[i], seq_len_1[i])
        return loss


class Cluster_FixedModel(nn.Module):
    def a():
        return 1

class SecondaryStructureAlignmentModel(nn.Module):
    def __init__(self, embedding, encoder, config=None):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder  # 原来的path部分
        
        # 配置参数
        if config is None:
            class DefaultConfig:
                margin_FP = 0.1
                margin_FN = 0.05
                gap_extension = 0.1
                gap_opening = 1.0
            self.config = DefaultConfig()
        else:
            self.config = config
            
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, input_ids_1, input_ids_2, attention_mask_1=None, attention_mask_2=None):
        """
        前向传播，返回两个序列的编码表示
        """
        # 获取两个序列的编码
        encoded_1 = self.get_encoded_sequence(input_ids_1, attention_mask_1)
        encoded_2 = self.get_encoded_sequence(input_ids_2, attention_mask_2)
        
        return encoded_1, encoded_2

    def get_encoded_sequence(self, input_ids, attention_mask=None):
        """获取单个序列的编码表示"""
        x = self.embedding(input_ids)
        encoded = self.encoder(x)
        return encoded
    
    def align_sequences(self, input_ids_1, input_ids_2, 
                       attention_mask_1=None, attention_mask_2=None,
                       seq_len_1=None, seq_len_2=None):
        """
        对齐两个序列并返回对齐分数和对齐结果
        """
        # 获取编码表示
        encoded_1, encoded_2 = self.forward(input_ids_1, input_ids_2, 
                                          attention_mask_1, attention_mask_2)
        
        # 转换为列表格式
        if seq_len_1 is None:
            seq_len_1 = attention_mask_1.sum().item() if attention_mask_1 is not None else encoded_1.size(1)
        if seq_len_2 is None:
            seq_len_2 = attention_mask_2.sum().item() if attention_mask_2 is not None else encoded_2.size(1)
            
        z1_list = [encoded_1[0, :seq_len_1, :]]  # 假设batch_size=1
        z2_list = [encoded_2[0, :seq_len_2, :]]
        
        # 计算匹配分数
        match_scores, alignment_logits = self.match(z1_list, z2_list)
        
        return match_scores[0], alignment_logits[0]

    def compute_alignment_loss(self, input_ids_1, input_ids_2,
                             common_index_1, common_index_2,
                             seq_len_1, seq_len_2,
                             attention_mask_1=None, attention_mask_2=None):
        """
        计算对齐损失（用于训练）
        """
        # 获取编码表示
        encoded_1, encoded_2 = self.forward(input_ids_1, input_ids_2, 
                                          attention_mask_1, attention_mask_2)
        
        # 转换为列表格式进行批处理
        z1_list = self.em(encoded_1, seq_len_1 if isinstance(seq_len_1, list) else [seq_len_1])
        z2_list = self.em(encoded_2, seq_len_2 if isinstance(seq_len_2, list) else [seq_len_2])
        
        # 计算MUL损失
        loss = self.train_MUL(z1_list, z2_list, 
                             [common_index_1], [common_index_2],
                             [seq_len_1], [seq_len_2])
        
        return loss

    # 以下方法从原模型中复制，专门用于对齐任务
    
    def gapscore(self, index, seq_len):
        seq_len = int(seq_len)
        index = index[:seq_len].to('cpu').detach().numpy().copy()
        all_zeros = seq_len - np.count_nonzero(index)
        extend_zeros = seq_len - np.count_nonzero(np.insert(index[:-1], 0, 1) + index)
        open_zeros = all_zeros - extend_zeros
        return (-1*self.config.gap_opening + -1*self.config.gap_extension ) * open_zeros + -1*self.config.gap_extension * extend_zeros
    
    def em(self, encoded_layers, seq_lens):
        """将编码层输出转换为序列列表格式"""
        z_list = []
        for b in range(encoded_layers.size(0)):
            actual_len = seq_lens[b] if isinstance(seq_lens, list) else encoded_layers.size(1)
            z_list.append(encoded_layers[b, :actual_len, :])
        return z_list
    
    def match(self, z0_list, z1_list):
        match_scores = []
        logits = []
        for z0, z1 in zip(z0_list, z1_list):
            match_score = nn.CosineSimilarity(dim=2, eps=1e-6)(
                z0.unsqueeze(1).repeat(1, z1.shape[0], 1), 
                z1.unsqueeze(0).repeat(z0.shape[0], 1, 1)
            )
            s = 1.3 * match_score
            
            a, b = F.softmax(s, 1), F.softmax(s, 0)
            # soft align
            c = a + b - a*b
            c = torch.sum(c*s)/torch.sum(c)
            match_scores.append(match_score)
            logits.append(c.view(-1))

        logits = torch.stack(logits, 0)
        return match_scores, logits
    
    def structural_learning(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """结构学习损失计算"""
        reference_alignment_score = self.calc_reference_alignment_score(
            bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1
        )
        prediction_alignment_score, _ = self.calc_prediction_alignment_score(
            bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1
        )
        return prediction_alignment_score - reference_alignment_score
    
    def match_bert_score(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """从原模型复制的匹配分数计算方法"""
        device = bert_score.device

        if common_index_0 is None or common_index_1 is None:
            return torch.tensor([], device=device)
        if common_index_0.numel() == 0 or common_index_1.numel() == 0:
            return torch.tensor([], device=device)

        def to_index_list(x):
            if x.dtype in (torch.int64, torch.long) and x.dim() == 1:
                return x.view(-1).long()
            if x.dim() == 1:
                idx = (x == 1).nonzero(as_tuple=False).view(-1).long()
                return idx
            idx = x.nonzero(as_tuple=False).view(-1).long()
            return idx

        idx0 = to_index_list(common_index_0)
        idx1 = to_index_list(common_index_1)

        if idx0.numel() != idx1.numel():
            print(f"[Warning] unequal pair lengths: idx0={idx0.numel()}, idx1={idx1.numel()}")
            minlen = min(idx0.numel(), idx1.numel())
            if minlen == 0:
                return torch.tensor([], device=device)
            idx0 = idx0[:minlen]
            idx1 = idx1[:minlen]

        omega = bert_score[idx0.long(), idx1.long()]
        return omega

    def calc_reference_alignment_score(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """计算参考对齐分数"""
        reference_alignment_score = 0.0
        
        reference_alignment_score += self.gapscore(common_index_0, seq_len_0)
        reference_alignment_score += self.gapscore(common_index_1, seq_len_1)
        
        if common_index_0.numel() == 0 or common_index_1.numel() == 0:
            return torch.tensor(0.0, device=bert_score.device)

        omega = self.match_bert_score(bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1)
        reference_alignment_score += torch.sum(omega)
        
        return reference_alignment_score

    def margin_score(self, common_index_A_B, common_index_0, common_index_1):
        """计算边际分数"""
        def list_from_mask_or_indices(x):
            if x is None or x.numel() == 0:
                return []
            if x.dtype in (torch.int64, torch.long) and x.dim() == 1:
                return x.view(-1).tolist()
            return (x == 1).nonzero(as_tuple=False).view(-1).tolist()

        predA = list_from_mask_or_indices(common_index_A_B[0])
        predB = list_from_mask_or_indices(common_index_A_B[1])
        refA = list_from_mask_or_indices(common_index_0)
        refB = list_from_mask_or_indices(common_index_1)

        a = set([int(i) * 10000 + int(j) for i, j in zip(predA, predB)])
        b = set([int(i) * 10000 + int(j) for i, j in zip(refA, refB)])
        len_pred_match = len(predA)
        len_ref_match = len(refA)
        len_TP = len(a & b)
        len_FP = len_pred_match - len_TP
        len_FN = len_ref_match - len_TP
        return len_FP * self.config.margin_FP + len_FN * self.config.margin_FN

    def margin_matrix(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """创建边际矩阵"""
        L0, L1 = bert_score.shape
        device = bert_score.device

        margin_mat_FP = torch.ones((L0, L1), device=device) * self.config.margin_FP
        margin_mat_FN = torch.zeros((L0, L1), device=device)

        if common_index_0 is None or common_index_1 is None:
            return margin_mat_FP, margin_mat_FN
        if common_index_0.numel() == 0 or common_index_1.numel() == 0:
            return margin_mat_FP, margin_mat_FN

        def to_index_list(x):
            if x.dtype in (torch.int64, torch.long) and x.dim() == 1:
                return x.view(-1).long()
            idx = (x == 1).nonzero(as_tuple=False).view(-1).long()
            return idx

        idx0 = to_index_list(common_index_0)
        idx1 = to_index_list(common_index_1)

        if idx0.numel() != idx1.numel():
            print(f"[Warning] unequal pair lengths: idx0={idx0.numel()}, idx1={idx1.numel()}")
            minlen = min(idx0.numel(), idx1.numel())
            idx0 = idx0[:minlen]
            idx1 = idx1[:minlen]

        if idx0.numel() > 0:
            margin_mat_FP[idx0.long(), idx1.long()] = 0.0
            margin_mat_FN[idx0.long(), idx1.long()] = self.config.margin_FN

        return margin_mat_FP, margin_mat_FN
    
    def calc_prediction_alignment_score(self, bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1):
        """计算预测对齐分数"""
        sequence_a = "N" * seq_len_0
        sequence_b = "N" * seq_len_1
        
        margin_mat_FP, margin_mat_FN = self.margin_matrix(
            bert_score, common_index_0, seq_len_0, common_index_1, seq_len_1
        )
        
        # alnC版本
        # try:
        #     import alignment_C as Aln_C
        #     common_index_A_B = Aln_C.global_aln(
        #         torch.flatten(bert_score.T).tolist(), 
        #         torch.flatten(margin_mat_FP.T).tolist(), 
        #         torch.flatten(margin_mat_FN.T).tolist(), 
        #         sequence_a, sequence_b, seq_len_0, seq_len_1, 
        #         self.config.gap_opening, self.config.gap_extension, 0, 0
        #     )
        #     common_index_A_B = torch.tensor(common_index_A_B).to(self.device).view(2, -1)
        # except ImportError:
        #     print("[Warning] alignment_C not found, using simplified alignment")
        
        # 简化版本
        common_index_A_B = self._simple_alignment(bert_score, seq_len_0, seq_len_1)
        
        prediction_alignment_score = self.calc_reference_alignment_score(
            bert_score, common_index_A_B[0], seq_len_0, common_index_A_B[1], seq_len_1
        ) + self.margin_score(common_index_A_B, common_index_0, common_index_1)
        
        return prediction_alignment_score, common_index_A_B
    
    def _simple_alignment(self, bert_score, seq_len_0, seq_len_1):
        """简化的对齐算法（当alignment_C不可用时）"""
        # 贪婪对齐：选择最高分数的匹配
        L0, L1 = bert_score.shape
        aligned_0, aligned_1 = [], []
        
        # 简单的贪婪匹配
        used_0, used_1 = set(), set()
        for _ in range(min(L0, L1)):
            best_score = float('-inf')
            best_i, best_j = -1, -1
            
            for i in range(L0):
                if i in used_0:
                    continue
                for j in range(L1):
                    if j in used_1:
                        continue
                    if bert_score[i, j] > best_score:
                        best_score = bert_score[i, j]
                        best_i, best_j = i, j
            
            if best_i != -1 and best_j != -1:
                aligned_0.append(best_i)
                aligned_1.append(best_j)
                used_0.add(best_i)
                used_1.add(best_j)
        
        if not aligned_0:
            return torch.zeros((2, 0), device=bert_score.device, dtype=torch.long)
        
        return torch.tensor([aligned_0, aligned_1], device=bert_score.device).long()
    
    def train_MUL(self, z0_list, z1_list, common_index_0, common_index_1, seq_len_0, seq_len_1):   
        """MUL训练损失"""
        bert_scores, _ = self.match(z0_list, z1_list)
        loss = 0.0
        for i, bert_score in enumerate(bert_scores):
            loss += self.structural_learning(bert_score, common_index_0[i], seq_len_0[i], common_index_1[i], seq_len_1[i])
        return loss

# 从头开始做预训练过程
class Contrastive_FixedModel(nn.Module):
    def __init__(self, experiment_name, model_index, tokenizer_name, embedding_dim):
        super().__init__()
        self.experiment_name = experiment_name
        self.model_index = model_index
        self.path = build_model_with_index(experiment_name, model_index)    # 从前面的架构中挑选表现较好的模型做评估
        self.tokenizer_name = tokenizer_name
        self.tokenizer = MyTokenizer(self.tokenizer_name)
        pad_idx = self.tokenizer.token_to_id("[PAD]")
        vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx)

    def masked_average_pooling(self, x, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float() 
        x_masked = x * mask_expanded 
        lengths = attention_mask.sum(dim=1).float().unsqueeze(-1) 
        lengths = lengths.clamp(min=1e-9)
        pooled_output = x_masked.sum(dim=1) / lengths 
        return pooled_output

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.path(x)
        x_permuted = x.permute(0, 2, 1)
        feature = self.masked_average_pooling(x, attention_mask)
        return feature



class MaskedModeling_FixedModel(nn.Module):
    def __init__(self, experiment_name, model_index, tokenizer_name, embedding_dim):
        super().__init__()
        self.experiment_name = experiment_name
        self.model_index = model_index
        self.model = build_model_with_index(experiment_name, model_index)
        self.tokenizer_name = tokenizer_name
        self.tokenizer = MyTokenizer(self.tokenizer_name)
        self.pad_idx = self.tokenizer.token_to_id("[PAD]")
        vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=self.pad_idx)

        dummy_input_ids = torch.randint(0, vocab_size, (1, 10))
        dummy_embedded_output = self.embedding(dummy_input_ids)
        with torch.no_grad():
            dummy_model_output = self.model(dummy_embedded_output)
        max_channel = dummy_model_output.shape[-1] 
        logger.info(f"max_channel is {max_channel}")
        logger.info(f"vocab_size is {vocab_size}")
        self.mask_pred_head = nn.Linear(max_channel, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.model(x)
        x_permuted = x.permute(0, 2, 1)
        mask_pred_logits = self.mask_pred_head(x).permute(0,2,1)
        return mask_pred_logits