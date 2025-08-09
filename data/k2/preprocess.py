"""
把目录下所有 *.ref.fa 的成对序列整合成单个 all_ref.fa
格式：>pair_0_A
      UGAACCG...
      >pair_0_B
      UGACCG...
      ...
"""
import os
from Bio import SeqIO
import numpy as np
'''
def collect_all_pairs(root_dir, max_len=1024):
    pairs = []
    for fname in sorted(os.listdir(root_dir)):
        if not fname.endswith(".ref.fa"):
            continue
        path = os.path.join(root_dir, fname)
        recs = list(SeqIO.parse(path, "fasta"))
        if len(recs) != 2:
            continue
        seqA = str(recs[0].seq).upper()[:max_len]
        seqB = str(recs[1].seq).upper()[:max_len]
        pairs.append((seqA, seqB))
    return pairs

def save_all_ref(pairs, out_path):
    with open(out_path, "w") as f:
        for idx, (seqA, seqB) in enumerate(pairs):
            f.write(f">pair_{idx}_A\n{seqA}\n")
            f.write(f">pair_{idx}_B\n{seqB}\n")
    print(f"已写入 {len(pairs)} 对序列 → {out_path}")

if __name__ == "__main__":
    src_dir   = "/u/yfang4/projects/jiaxin/NAS-for-Bio/data/k2/tRNA"      # 原始 *.ref.fa 目录
    dst_file  = "/u/yfang4/projects/jiaxin/NAS-for-Bio/data/k2/tRNA/all_ref.fa" # 目标统一文件
    pairs = collect_all_pairs(src_dir)
    save_all_ref(pairs, dst_file)

'''
"""
save_all_ref_recursive.py
递归扫描目录下所有 ***.ref.fa**，整合成单个 all_ref.fa
格式：>pair_0_A
      UGAACCG...
      >pair_0_B
      UGACCG...
"""

def collect_all_pairs_recursive(root_dir, max_len=512):
    """
    root_dir: 顶层目录
    返回 [(seqA, seqB), ...]
    """
    pairs = []
    # 递归遍历
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if not fname.endswith(".ref.fa"):
                continue
            path = os.path.join(dirpath, fname)
            recs = list(SeqIO.parse(path, "fasta"))
            if len(recs) != 2:
                continue
            seqA = str(recs[0].seq).upper()[:max_len]
            seqB = str(recs[1].seq).upper()[:max_len]
            pairs.append((seqA, seqB))
    return pairs

def save_all_ref(pairs, out_path):
    with open(out_path, "w") as f:
        for idx, (seqA, seqB) in enumerate(pairs):
            f.write(f">pair_{idx}_A\n{seqA}\n")
            f.write(f">pair_{idx}_B\n{seqB}\n")
    print(f"已写入 {len(pairs)} 对序列 → {out_path}")

if __name__ == "__main__":
    src_root  = "/u/yfang4/projects/jiaxin/NAS-for-Bio/data/k2"   # 顶层目录（含多层子目录）
    dst_file  = "/u/yfang4/projects/jiaxin/NAS-for-Bio/data/k2/all_ref.fa"
    pairs = collect_all_pairs_recursive(src_root)
    save_all_ref(pairs, dst_file)