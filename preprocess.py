import json
import jsonlines
import subprocess
from collections import defaultdict

input_file = 'RNAcentral.json'
output_jsonl = 'cleaned_rna.jsonl'
fasta_file = 'cdhit_input.fasta'
cdhit_output_fasta = 'cdhit_output.fasta'
final_output_jsonl = 'deduplicated_rna.jsonl'

# 新增函数：检查是否包含非标准碱基
def contains_invalid_bases(sequence):
    valid_bases = {'A', 'C', 'G', 'U', 'T'}  # 包含T是为了处理可能未转换的序列
    return any(base.upper() not in valid_bases for base in sequence)

# 修改后的统计函数（增加无效序列统计）
def count_sequence_lengths(sequences, stage_name=""):
    length_bins = {
        "<512": 0,
        "512-1024": 0,
        "1024-2048": 0,
        ">2048": 0
    }
    invalid_count = 0  # 新增：统计无效序列
    
    for seq in sequences:
        if contains_invalid_bases(seq):
            invalid_count += 1
            continue
        length = len(seq)
        if length < 512:
            length_bins["<512"] += 1
        elif 512 <= length < 1024:
            length_bins["512-1024"] += 1
        elif 1024 <= length < 2048:
            length_bins["1024-2048"] += 1
        else:
            length_bins[">2048"] += 1
    
    print(f"\n[STATS] {stage_name} Sequence length distribution:")
    for bin_name, count in length_bins.items():
        print(f"  {bin_name} nt: {count} sequences ({count/len(sequences)*100:.1f}%)")
    if invalid_count > 0:
        print(f"  [FILTERED] Sequences with invalid bases: {invalid_count}")
    return length_bins

# Step 1: 读取原始数据并过滤
with open(input_file, 'r') as f:
    data = json.load(f)
    records = data.get("results", [])

sequences = []
invalid_sequences = 0  # 记录被过滤的序列数
for entry in records:
    seq = entry.get("sequence", "")
    if seq:
        seq_rna = seq.upper().replace("T", "U")
        # 新增过滤条件：长度>1024且含非ACGUT的序列将被排除
        if contains_invalid_bases(seq_rna) or len(seq_rna) > 1024:
            invalid_sequences += 1
            continue
        sequences.append(seq_rna)

print(f"[INFO] Total raw sequences parsed: {len(records)}")
print(f"[FILTER] Removed {invalid_sequences} sequences (length>1024 with invalid bases)")
_ = count_sequence_lengths(sequences, "Raw input after filtering")

# Step 2: 去重（保持不变）
unique_sequences = list(set(sequences))
print(f"\n[INFO] Unique sequences after initial deduplication: {len(unique_sequences)}")
_ = count_sequence_lengths(unique_sequences, "After initial deduplication")

# Step 3: 保存为 fasta（增加有效性检查）
with open(fasta_file, 'w') as f:
    valid_count = 0
    for idx, seq in enumerate(unique_sequences):
        if contains_invalid_bases(seq):
            continue
        f.write(f">seq{idx}\n{seq}\n")
        valid_count += 1
    print(f"[INFO] Saved {valid_count} valid sequences to FASTA file: {fasta_file}")

# Step 4: 运行 cd-hit-est
cdhit_cmd = [
    "cd-hit-est",
    "-i", fasta_file,
    "-o", cdhit_output_fasta,
    "-c", "1.0",
    "-n", "10"
]

try:
    subprocess.run(cdhit_cmd, check=True)
    print("[INFO] cd-hit-est finished successfully.")
except FileNotFoundError:
    print("[WARNING] cd-hit-est not found. Skipping clustering.")
    with jsonlines.open(output_jsonl, mode='w') as writer:
        for seq in unique_sequences:
            writer.write({"text": seq})
    print(f"[INFO] Output written to {output_jsonl}")
    print(f"[INFO] Total output sequences: {len(unique_sequences)}")
    _ = count_sequence_lengths(unique_sequences, "Final output (no CD-HIT)")
    exit(0)

# Step 5: 读取 cd-hit-est 输出
deduplicated_sequences = []
with open(cdhit_output_fasta, 'r') as f:
    seq = ''
    for line in f:
        if line.startswith('>'):
            if seq:
                deduplicated_sequences.append(seq)
                seq = ''
        else:
            seq += line.strip()
    if seq:
        deduplicated_sequences.append(seq)

print(f"\n[INFO] Final deduplicated sequences after cd-hit-est: {len(deduplicated_sequences)}")

# 最终输出前再次检查（安全措施）
final_valid_sequences = [seq for seq in deduplicated_sequences if not contains_invalid_bases(seq)]
print(f"\n[FILTER] Removed {len(deduplicated_sequences)-len(final_valid_sequences)} invalid sequences from final output")

# 修改最终写入逻辑
with jsonlines.open(final_output_jsonl, mode='w') as writer:
    for seq in final_valid_sequences:
        writer.write({
            "text": seq,
            "length": len(seq),
            "valid": True  # 新增验证标记
        })

print(f"[INFO] Final VALID output written to {final_output_jsonl} (count: {len(final_valid_sequences)})")