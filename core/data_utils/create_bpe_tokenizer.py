from pathlib import Path
import json
import click
from tokenizers import Tokenizer, models, trainers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import random
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)





if __name__ == "__main__":
    vocab_size = 4096
    sample_ratio = 0.05
    random.seed(42)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                  special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                                  show_progress=True)

    jsonl_file = Path("/projects/slmreasoning/yifang/datasets/GRCh38/processed_data/filtered_sentences.jsonl")
    sequences = []
    with open(jsonl_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            sequences.append(obj["text"].upper())

    num_samples = int(len(sequences) * sample_ratio)
    sampled_sequences = random.sample(sequences, num_samples)

    start_time = time.time()
    logger.info(f"Start to train BPE...")
    tokenizer.train_from_iterator(sampled_sequences, trainer)

    output_path = f"/projects/slmreasoning/yifang/tokenizers/bpe-{vocab_size}.json"
    tokenizer.save(output_path)

    # 打印训练时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Tokenizer training completed in {elapsed_time:.2f} seconds.")

    tokenizer = Tokenizer.from_file("/projects/slmreasoning/yifang/tokenizers/bpe-512.json")
    tokens = tokenizer.encode("ACTGCTAGCAGT").tokens
    ids = tokenizer.encode("ACTGCTAGCAGT").ids
    logger.info(f"Tokenized tokens: {tokens}")
    logger.info(f"Tokenized ids: {ids}")