import sys
sys.path.append("../")
from torch.utils.data import random_split
from core.data_utils.dataset import DNAContrastiveDataset, build_collate_fn
from core.models.model import ContrastiveLearning_PretrainModel, ModelConfig, ContrastiveLearning_ModelSpace
import torch 
from nni.nas.strategy import RandomOneShot
import nni.nas.evaluator.pytorch.lightning as pl
from nni.nas.experiment import NasExperiment
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import os
import json
from typing import List, Tuple
import argparse
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验参数配置")
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="base_experiment", 
    )

    args = parser.parse_args()
    experiment_name = args.experiment_name

    config_path = f"/projects/slmreasoning/yifang/configs/{experiment_name}"
    architecture_config_flie = os.path.join(config_path, "architecture_configs.json")
    config = ModelConfig.from_json(os.path.join(config_path, "searchSpace_configs.json"))
    if experiment_name != config.experiment_name:
        raise ValueError(f"Mismatch between experiment_name ('{experiment_name}') and config.experiment_name ('{config.experiment_name}')")
    # 检查是否存在 architecture_config_flie.json
    if os.path.exists(architecture_config_flie):
        config.architecture_config_flag = True
    data_usage_rate = config.data_usage_rate
    tokenizer_name = config.tokenizer_name



    jsonl_file = "/projects/slmreasoning/yifang/datasets/GRCh38/processed_data/filtered_sentences.jsonl"
    dataset = DNAContrastiveDataset(jsonl_file, data_usage_rate=data_usage_rate, tokenizer_name=config.tokenizer_name)

    total_size = len(dataset)
    val_size = int(total_size * 0.05)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    Contrastive_Pretrain_Model = ContrastiveLearning_PretrainModel(config)
    
    log_dir = f"/projects/slmreasoning/yifang/nni_pre_logs/contrastive/{experiment_name}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        save_top_k=-1,                      # 保存所有 checkpoint（不只是最好的）
        every_n_epochs=1, 
        save_weights_only=False,           # 是否只保存权重
        filename="Pretrain-{epoch}",            # 文件名格式
    )

    
    trainer = pl.Trainer(
        max_epochs=10,
        log_every_n_steps=1000,
        callbacks=[checkpoint_callback],
        default_root_dir=log_dir,
    )

    collate_fn = build_collate_fn(pad_idx=dataset.pad_idx)

    if "kmer" in experiment_name:
        batch_size = 256
        if 'mamba' in experiment_name:
            batch_size = 128
    else:
        batch_size=512

    evaluator = pl.Lightning(Contrastive_Pretrain_Model,
    trainer,
    train_dataloaders=pl.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn),
    val_dataloaders=pl.DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn),
    )

    exploration_strategy = RandomOneShot()
    experiment = NasExperiment(Contrastive_Pretrain_Model.model, evaluator, exploration_strategy)
    experiment.config.debug = True
    experiment.config.training_service.debug = True
    experiment.run(port=10086)
    