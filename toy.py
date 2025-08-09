import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from core.models.model import MaskedModeling_PretrainModel, ModelConfig
from core.data_utils.dataset import DNAMaskDataset, build_collate_fn
import torch.nn as nn
import pytorch_lightning as pl

class ToyModel(pl.LightningModule):
    def __init__(self, vocab_size=5, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.linear(self.embedding(x))

    def training_step(self, batch, batch_idx):
        x, y, _ = batch 
        logits = self(x)
        return self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    # 1) 加载 config & model
    cfg = ModelConfig.from_json("/u/yfang4/projects/jiaxin/NAS-for-Bio/configs/cnn-kmer1/searchSpace_configs.json")
    model = ToyModel()

    # 2) 数据集 + collate_fn
    ds = DNAMaskDataset("/u/yfang4/projects/jiaxin/NAS-for-Bio/cleaned_rna.jsonl",
                        data_usage_rate=cfg.data_usage_rate,
                        tokenizer_name=cfg.tokenizer_name)
    collate_fn = build_collate_fn(pad_idx=ds.pad_idx)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # 3) Lightning trainer：只跑 1 个 train batch + 1 个 val batch，且 fast_dev_run 会做一遍 train+val
    trainer = Trainer(
        fast_dev_run=True,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
    )

    # 4) 直接把 train/val 都指向同一个 loader，先看它能不能跑
    trainer.fit(model, loader, loader)
    print(111)