from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import PreTokenizer, Whitespace
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers import normalizers
from typing import List, Tuple
import json
import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class KmerDecoder:
    def decode(self, tokens):
        return "".join(tokens)


class KMERTokenizer:
    def __init__(self, name: str):
        if not name.startswith("kmer-"):
            raise ValueError(f"Invalid tokenizer name format: {name}")
        try:
            self.k = int(name.split("-")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Failed to extract k from tokenizer name: {name}")

        base_path = "/projects/slmreasoning/yifang/tokenizers"
        file_path = os.path.join(base_path, f"{name}.json")

        with open(file_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)["vocab"]
        self.vocab_size = len(self.vocab)
        model = WordLevel(vocab=self.vocab, unk_token="[UNK]") 
        self.tokenizer = Tokenizer(model)

        self.tokenizer.pre_tokenizer = Whitespace()

        self.tokenizer.decoder = decoders.Decoder.custom(KmerDecoder())

    def generate_kmer_str(self, sequence: str, k: int) -> str:
        """Generate k-mer string from DNA sequence."""
        k = self.k
        return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

    def encode(self, text: str):
        kmer_text = self.generate_kmer_str(text, k=self.k)
        return self.tokenizer.encode(kmer_text)

    def decode(self, ids: list[int]):
        return self.tokenizer.decode(ids)

    def batch_encode(self, texts: list[str]):
        return self.tokenizer.encode_batch(texts)

    def save(self, path: str):
        self.tokenizer.save(path)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("[UNK]", 0))


class MyTokenizer:
    def __init__(self, name):
        base_path="/projects/slmreasoning/yifang/tokenizers"
        file_path = os.path.join(base_path, f"{name}.json")
        if "bpe" in name:
            self.tokenizer = Tokenizer.from_file(file_path)
            self.vocab_size = self.tokenizer.get_vocab_size()
        elif "kmer" in name:
            self.tokenizer = KMERTokenizer(name)
            self.vocab_size = self.tokenizer.vocab_size

        else:
            raise ValueError("Unknown tokenizer type")

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def batch_encode(self, texts):
        return self.tokenizer.batch_encode(texts)

    def save(self, path):
        return self.tokenizer.save(path)

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)