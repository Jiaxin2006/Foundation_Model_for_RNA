import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import OneHotEncoder
import itertools
from collections import defaultdict
import logging
from core.models.modules import build_module
import json
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def calculate_contrastive_loss(emb_1, emb_2, temperature=0.5):
    """
    Args:
        emb_1, emb_2: [batch_size, embed_dim] 
    """
    batch_size = emb_1.shape[0]
    device = emb_1.device

    embeddings = torch.cat([emb_1, emb_2], dim=0) # -> [2 * batch_size, embed_dim]

    embeddings = F.normalize(embeddings, p=2, dim=1)

    similarity_matrix = torch.matmul(embeddings, embeddings.T)

    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)

    similarity_matrix.masked_fill_(self_mask, -9e15)

    logits = similarity_matrix / temperature # -> [2 * batch_size, 2 * batch_size]

    labels = torch.arange(batch_size, device=device) # [0, 1, ..., B-1]
    labels = torch.cat([labels + batch_size, labels]) # -> [B, B+1, ..., 2B-1, 0, 1, ..., B-1]

    loss = F.cross_entropy(logits, labels)

    return loss


def generate_dim_configs(embedding_dim, hidden_dim_list, layer_nums_dict, threshold=1.2, max_representatives_per_group=5):
    layer_nums_list = [int(k) for k in layer_nums_dict.keys()]
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



def generate_layer_type_configs(architecture_list, layer_nums_dict):
    final_configs = []
    
    # 注意这里 keys 可能是字符串，转换成整数使用
    for num_layers_str, target_num in layer_nums_dict.items():
        num_layers = int(num_layers_str)  # 确保是整数
        
        combos = list(itertools.product(architecture_list, repeat=num_layers))
        # 目标采样数不能超过所有组合数
        target_num = min(target_num, len(combos))

        # OneHot encode each architecture string in the sequence
        combos_array = np.array(combos)
        encoder = OneHotEncoder(sparse_output=False)
        combos_encoded = encoder.fit_transform(combos_array)
        
        # Clustering
        kmeans = KMeans(n_clusters=target_num, random_state=42)
        labels = kmeans.fit_predict(combos_encoded)
        centers = kmeans.cluster_centers_
        
        # Select the closest to each cluster center
        for i in range(target_num):
            cluster_indices = np.where(labels == i)[0]
            cluster_points = combos_encoded[cluster_indices]
            dists = np.linalg.norm(cluster_points - centers[i], axis=1)
            closest_idx = cluster_indices[np.argmin(dists)]
            final_configs.append(list(combos[closest_idx]))
    
    return final_configs




def merge_configs(dim_config_list, layer_type_config_list, embedding_dim):
    dim_buckets = defaultdict(list)
    type_buckets = defaultdict(list)

    for dims in dim_config_list:
        dim_buckets[len(dims)].append(dims)
    
    for types in layer_type_config_list:
        type_buckets[len(types)].append(types)
    
    merged_configs = []
    for length in dim_buckets:
        dim_group = dim_buckets[length]
        type_group = type_buckets.get(length, [])
        
        if not type_group:
            raise ValueError(f"No matching layer_type_config found for dim_config with length {length}")

        for dims, types in itertools.product(dim_group, type_group):
            config = []
            input_dim = embedding_dim
            for output_dim, layer_type in zip(dims, types):
                config.append({
                    'layer_type': layer_type,
                    'input_dim': input_dim,
                    'output_dim': output_dim
                })
                input_dim = output_dim
            merged_configs.append(config)
    
    return merged_configs


def extract_layer_sets(config_list):
    unique_layers = set()
    for config in config_list:
        for layer in config:
            layer_set = frozenset(layer.items())
            unique_layers.add(layer_set)
    return unique_layers



def build_model_with_index(experiment_name, model_index):
    architecture_config_path = f"/projects/slmreasoning/yifang/configs/{experiment_name}/architecture_configs.json"
    with open(architecture_config_path, "r") as f:
        data = json.load(f)
        architecture_config_list = data["architecture_config_list"]
    layers = []
    best_architecture = architecture_config_list[model_index]
    for idx, config in enumerate(best_architecture):
        layer_type = config["layer_type"]
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        module = build_module(layer_type, input_dim, output_dim)
        layers.append(module)
    main_model = nn.Sequential(*layers)
    return main_model