import sys
sys.path.append("../")
import os
import torch 
from core.models.model import ContrastiveLearning_ModelSpace,ModelConfig,MaskedModeling_ModelSpace
import nni.nas.evaluator.pytorch.lightning as pl
import csv
from tqdm import tqdm
import argparse
# gai
from core.models.training_utils import set_seed, cls_evaluate_model, continual_mask_pretrain, continual_contrastive_pretrain, build_all_models, load_rna_clustering, embed_sequences, cluster_and_evaluate
from core.models.training_utils import align_train_epoch, align_eval_epoch
from core.data_utils.dataset import AlignDataset, align_collate
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验参数配置")
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="base_experiment", 
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="contrastive", 
    )
    # gai
    parser.add_argument(
        "--task_index", 
        type=int, 
        nargs='+', 
        choices=[0, 1, 2, 3, 4], 
        default=[0],
        help="List of task indices to run: 0 (5'UTR MLR), 1 (RNA Clustering)"
    )

    args = parser.parse_args()
    experiment_name = args.experiment_name
    strategy = args.strategy

    set_seed(42)

    # gai
    task_index_name_map = {
        0: "5'UTR MLR Prediction",
        1: "RNA Clustering Prediction",
        2: "Secondary Structure Alignment",
        # 3: "Transcription Factor Prediction-3",
        # 4: "Transcription Factor Prediction-4",
        # 5: "Core Prompter Detection-all",
        # 6: "Core Prompter Detection-notata",
        # 7: "Core Prompter Detection-tata",
        # 8: "Prompter Detection-all",
        # 9: "Prompter Detection-notata",
        # 10: "Prompter Detection-tata",
        # 11: "Splice Site Detection"
    }

    #ckpt_epoch_num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ckpt_epoch_num_list = [0, 4]
    config = ModelConfig.from_json(f"/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/configs/{experiment_name}/searchSpace_configs.json")
    config.channel_config_list_done_flag = True
    tokenizer_name = config.tokenizer_name
    data_usage_rate = config.data_usage_rate

    output_dir = f"/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/results/{experiment_name}/"
    os.makedirs(output_dir, exist_ok=True)
    if strategy == 'contrastive':
        continual_pretrain = continual_contrastive_pretrain
        Pretrained_ModelSpace = ContrastiveLearning_ModelSpace(config)
        csv_file_path = output_dir + "con-ft_results.csv"

    elif strategy == 'mask':
        Pretrained_ModelSpace = MaskedModeling_ModelSpace(config)
        csv_file_path = output_dir + "mask-ft_results.csv"
    # gai
    csv_header = [
        "experiment_name", "task_index", "epoch_num", "model_name",
        "acc",                       # task 0 专用
        "embed_time",                # task 1 专用
        "ARI", "Homogeneity", "Completeness", "cluster_time",
        "F1", "SEN", "PPV"           # task 2 专用
    ]
    
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

    for epoch_num in tqdm(ckpt_epoch_num_list, desc="Checkpoint Steps"):
        pretrained_path = f"/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/nni_pre_logs/{strategy}/{experiment_name}/Pretrain-epoch={epoch_num}.ckpt"
        state_dict = torch.load(pretrained_path)['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("training_module._model.", "") 
            new_state_dict[new_key] = v
        Pretrained_ModelSpace.load_state_dict(new_state_dict)

        for task_index in tqdm(args.task_index, desc=f"Tasks for epoch={epoch_num}", leave=False):   
            num_classes = 3 if task_index == 11 else 2
            task_name = task_index_name_map[task_index]
            
            models = build_all_models(num_classes, Pretrained_ModelSpace)
            for tuple_model in tqdm(models, desc=f"Models for task {task_index}", leave=False):
                model_name = tuple_model[0].strip("")
                model = tuple_model[1]

                # gai
                if task_index == 0:
                    acc = cls_evaluate_model(experiment_name=experiment_name, freeze_flag=False, model=model, task_index=task_index, tokenizer_name=tokenizer_name)
                    acc = round(acc, 2)
                    with open(csv_file_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([experiment_name, task_name, epoch_num+1, model_name, acc, "", "", "", "", "","","",""])


                if task_index == 1:
                    data_dir = "/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/data/family"
                    seqs, labels, family_names = load_rna_clustering(data_dir)
                    embeddings, t_embed = embed_sequences(seqs, model, tokenizer_name)
                    results = cluster_and_evaluate(embeddings, labels, n_clusters=len(family_names))
                    row = [
                        experiment_name, task_name, epoch_num + 1, model_name,
                        "",                                  # acc 留空
                        round(t_embed, 2),
                        round(results["ARI"], 4),
                        round(results["Homogeneity"], 4),
                        round(results["Completeness"], 4),
                        round(results["Time (s)"], 2),
                        "","",""
                    ]
                    with open(csv_file_path, mode='a', newline='') as f:
                        csv.writer(f).writerow(row)

                if task_index == 2:
                    data_dir = "/work/hdd/begl/yfang4/projects/jiaxin/NAS-for-Bio/data/k2"  # 放 .ref.fa 的目录
                    train_ds = AlignDataset("train", tokenizer_name, data_dir)
                    val_ds   = AlignDataset("dev",   tokenizer_name, data_dir)
                    test_ds  = AlignDataset("test",  tokenizer_name, data_dir)
                    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=align_collate)
                    val_loader   = DataLoader(val_ds,   batch_size=8, collate_fn=align_collate)
                    test_loader  = DataLoader(test_ds,  batch_size=8, collate_fn=align_collate)

                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    num_steps = len(train_loader) * 3   # 3 epoch 快速验证
                    scheduler = get_linear_schedule_with_warmup(optimizer, 50, num_steps)

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.to(device)
                    for epoch in range(3):
                        align_train_epoch(model, device, train_loader, optimizer, scheduler)
                        val_f1, _, _ = align_eval_epoch(model, device, val_loader)
                    test_f1, sen, ppv = align_eval_epoch(model, device, test_loader)
                    row = [
                        experiment_name, task_name, epoch_num + 1, model_name,
                        "",                                 # acc 留空
                        "",                                 # embed_time 留空
                        "",                                 # ARI 留空
                        "",                                 # Homogeneity 留空
                        "",                                 # Completeness 留空
                        "",                                 # cluster_time 留空
                        round(test_f1, 4),                  # task2 的 F1
                        round(sen, 4),                      # SEN
                        round(ppv, 4)                       # PPV
                    ]
                    with open(csv_file_path, mode='a', newline='') as f:
                        csv.writer(f).writerow(row)