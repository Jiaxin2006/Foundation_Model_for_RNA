import json
import os
from itertools import product
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_kmer_vocab_dict(k, alphabet):
    special_tokens = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[PAD]": 3,
        "[MASK]": 4
    }

    kmers = [''.join(p) for p in product(alphabet, repeat=k)]

    vocab = dict(special_tokens)
    current_id = len(special_tokens)

    for kmer in kmers:
        if kmer not in vocab:
            vocab[kmer] = current_id
            current_id += 1

    return {"vocab": vocab}

def save_vocab_to_json(vocab_dict, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, indent=2, ensure_ascii=False)


alphabet = ['A', 'C', 'G', 'U']

for k in [1, 3, 4, 5, 6]:
    vocab_dict = create_kmer_vocab_dict(k, alphabet)
    save_vocab_to_json(vocab_dict, f'tokenizers/kmer-{k}.json')
    logger.info(f"词表已保存到 /tokenizers/kmer-{k}.json")
