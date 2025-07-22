import torch
import torch.nn as nn
import sys
sys.path.append("../")
import csv
from torch.utils.data import random_split
from core.data_utils.dataset import DNAContrastiveDataset, build_collate_fn
from core.models.model import Contrastive_FixedModel, ModelConfig, CLS_FixedModel
from core.models.training_utils import set_seed, cls_evaluate_model, find_best_architecture
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup 
from core.models.utils import calculate_contrastive_loss
import wandb
import os
from tqdm import tqdm
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

    set_seed(42)

    task_index_name_map = {
        0: "Transcription Factor Prediction-0",
        1: "Transcription Factor Prediction-1",
        2: "Transcription Factor Prediction-2",
        3: "Transcription Factor Prediction-3",
        4: "Transcription Factor Prediction-4",
        5: "Core Prompter Detection-all",
        6: "Core Prompter Detection-notata",
        7: "Core Prompter Detection-tata",
        8: "Prompter Detection-all",
        9: "Prompter Detection-notata",
        10: "Prompter Detection-tata",
        11: "Splice Site Detection"
    }

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
    
    best_of_each, best_overall = find_best_architecture(experiment_name, training_strategy="con", epoch=1)
    logger.info(f"best_of_each is {best_of_each}")
    logger.info(f"best_overall is {best_overall}")

    pre_epoch_num = 5
    batch_size = 1024

    jsonl_file = "/projects/slmreasoning/yifang/datasets/GRCh38/processed_data/filtered_sentences.jsonl"
    dataset = DNAContrastiveDataset(jsonl_file, data_usage_rate=data_usage_rate, tokenizer_name=config.tokenizer_name)
    total_size = len(dataset)
    val_size = int(total_size * 0.05)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    collate_fn = build_collate_fn(pad_idx=dataset.pad_idx)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    
    model = Contrastive_FixedModel(experiment_name=experiment_name, model_index=best_overall, tokenizer_name=tokenizer_name, embedding_dim=config.embedding_dim)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    num_warmup_steps = 50
    num_training_steps = len(dataloader) * pre_epoch_num
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    model_save_path = f'/projects/slmreasoning/yifang/re-pretrained/contrastive/{experiment_name}/'
    os.makedirs(model_save_path, exist_ok=True)
    '''
    ### pre-train ###
    for epoch in range(pre_epoch_num):
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            aug_1, aug_2, attention_mask = batch
            attention_mask = attention_mask.to(device)
            aug_1 = aug_1.to(device)
            aug_2 = aug_2.to(device)
            emb_1 = model(aug_1, attention_mask)
            emb_2 = model(aug_2, attention_mask)
            loss = calculate_contrastive_loss(emb_1, emb_2)
            loss.backward()
            optimizer.step()
            scheduler.step()
        torch.save(model.state_dict(), model_save_path+f'RePretrain-epoch={epoch}.ckpt')

    '''
    csv_file_path = f"/projects/slmreasoning/yifang/results/{experiment_name}/re-con-ft.csv"
    if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["experiment_name", "task_index", "epoch_num", "model_name", "acc"])
    ### finetune ###
    for epoch_num in [0,4]:
        repre_model_path = model_save_path + f'RePretrain-epoch={epoch_num}.ckpt'
        repre_model = Contrastive_FixedModel(experiment_name=experiment_name, model_index=best_overall, tokenizer_name=tokenizer_name, embedding_dim=config.embedding_dim)
        repre_model.load_state_dict(torch.load(repre_model_path))
        for task_index in tqdm(range(12), desc=f"Tasks for epoch={epoch_num}", leave=False):
            num_classes = 3 if task_index == 11 else 2
            task_name = task_index_name_map[task_index]

            model = CLS_FixedModel(num_classes=num_classes,
                embedding=repre_model.embedding,
                path=repre_model.path,
                mask_pred_head=getattr(repre_model, "pred_head", None)
                )

            acc = cls_evaluate_model(freeze_flag=False, model=model, task_index=task_index, tokenizer_name=tokenizer_name)
            acc = round(acc, 2)
            
            with open(csv_file_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([experiment_name, task_name, epoch_num+1, f"path_{best_overall}", acc])