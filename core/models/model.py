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
                raise FileNotFoundError(f"Expected channel config at {dim_config_path}, but it does not exist.")
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

class Cluster_FixedModel(nn.Module):
    def a():
        return 1

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