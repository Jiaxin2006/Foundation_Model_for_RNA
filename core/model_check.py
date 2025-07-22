import sys
sys.path.append("../")
import os
import torch 
from torch.utils.data import random_split
from core.data_utils.dataset import FTDNADataset
from core.models.cnn import CNN_ContrastiveLearning_ModelSpace,ModelConfig
import nni.nas.evaluator.pytorch.lightning as pl
from nni.nas.experiment import NasExperiment
from torch.utils.data import DataLoader
import torch.nn as nn
from itertools import product
import csv
import random
import numpy as np
from tqdm import tqdm
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用了GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def my_collate_fn(batch):
    xs, ys = zip(*batch)  # 解开成两个元组
    return torch.stack(xs), torch.tensor(ys)  # 返回 list[str], tensor


def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

def val_epoch(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)

    return val_loss, val_acc

def test_epoch(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy

def evaluate_model(model, task_index):
    train_dataset, val_dataset, test_dataset = FTDNADataset(task_index,'train'), FTDNADataset(task_index,'dev'), FTDNADataset(task_index,'test')
    train_loader = DataLoader(train_dataset, batch_size=512, collate_fn = my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=512, collate_fn = my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn = my_collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        train_epoch(model, device, train_loader, optimizer, epoch)
        val_loss, val_acc = val_epoch(model, device, val_loader)

    final_accuracy = test_epoch(model, device, test_loader)
    return final_accuracy


def build_all_models(num_classes, model_space: CNN_ContrastiveLearning_ModelSpace):
    """
    枚举 model_space 中所有 cnn_arch 架构，构造完整模型。
    返回: List of (arch_name, model)
    """
    all_models = []

    for arch_name, cnn_seq in model_space.cnn_candidates.items():
        model = CNN_CLS_FixedModel(num_classes=num_classes,
            embedding=model_space.embedding,
            cnn_block=cnn_seq
        )
        all_models.append((arch_name, model))

    return all_models



class CNN_CLS_FixedModel(nn.Module):
    def __init__(self, num_classes, embedding, cnn_block: nn.Sequential):
        super().__init__()
        self.embedding = embedding
        self.conv_blocks = nn.Sequential()

        block_id = 0
        for layer in cnn_block:
            conv = None
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Conv1d):
                        conv = sublayer
                        break
            elif isinstance(layer, nn.Conv1d):
                conv = layer

            if conv is not None:
                # 固定组件：ReLU + BatchNorm1d + Dropout(0.1)
                block = nn.Sequential(
                    conv,
                    nn.ReLU(),
                    nn.BatchNorm1d(conv.out_channels),
                    nn.Dropout(p=0.3)
                )
                self.conv_blocks.add_module(f"block_{block_id}", block)
                block_id += 1

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cls_head = nn.Linear(128, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids).permute(0, 2, 1)
        x = self.conv_blocks(x)
        x = self.pool(x).squeeze(-1)
        logit = self.cls_head(x)
        return logit

def count_parameters(model_name, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} Total parameters: {total_params:,}")
    print(f"{model_name} Trainable parameters: {trainable_params:,}")
    print(f"{model_name} Total size (in MB): {total_params * 4 / 1024**2:.2f} MB")  # 假设每个参数4字节(float32)



if __name__ == "__main__":
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
    ckpt_step_num_list = [8000, 12000, 16000, 20000, 24000, 40000, 60000]
    config = ModelConfig.from_json("/projects/slmreasoning/yifang/configs/test_run/searchSpace_configs.json")
    config.channel_config_list_done_flag = True
    CNN_Pretrained_ModelSpace = CNN_ContrastiveLearning_ModelSpace(config)
    csv_file_path = "/projects/slmreasoning/yifang/results/eval_results.csv"
    if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["task_index", "step_num", "model_name", "acc"])


    for step_num in tqdm(ckpt_step_num_list, desc="Checkpoint Steps"):
        pretrained_path = f"/projects/slmreasoning/yifang/nni_pre_logs/Pretrain-step={step_num}.ckpt"
        state_dict = torch.load(pretrained_path)['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("training_module._model.", "") 
            new_state_dict[new_key] = v
        CNN_Pretrained_ModelSpace.load_state_dict(new_state_dict)

        for task_index in tqdm(range(1), desc=f"Tasks for step={step_num}", leave=False):
            log_dir = f"/projects/slmreasoning/yifang/nni_ft_logs/task_{task_index}"
            os.makedirs(log_dir, exist_ok=True)

            num_classes = 3 if task_index == 11 else 2
            task_name = task_index_name_map[task_index]
            
            models = build_all_models(num_classes, CNN_Pretrained_ModelSpace)
            for tuple_model in tqdm(models, desc=f"Models for task {task_index}", leave=False):
                model_name = tuple_model[0].strip("")
                model=tuple_model[1]
                count_parameters(model_name, model)

                # acc = evaluate_model(model=tuple_model[1], task_index=task_index)
                # acc = round(acc, 2)
                # with open(csv_file_path, mode='a', newline='') as csv_file:
                #     writer = csv.writer(csv_file)
                #     writer.writerow([task_name, step_num, model_name, acc])