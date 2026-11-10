"""A small model API in the spirit of bio foundation model toolkits: load a pretrained backbone, get embeddings,
fine-tune with a chosen strategy, predict. The same calls work for the Transformer and the state-space backbone.

    fm = GenomicFM.from_pretrained("transformer")
    emb = fm.get_embeddings(["ACGT...", ...])                  # (n, d) numpy array
    fm.fine_tune(train_sequences, train_labels, mode="lora")
    probs = fm.predict(test_sequences)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import Classifier, add_lora, batch_tokens, pool
from .train import cosine, load_backbone


class GenomicFM:
    def __init__(self, backbone):
        self.backbone = backbone
        self.model: Classifier | None = None

    @classmethod
    def from_pretrained(cls, name: str = "transformer", d: int = 64, layers: int = 4):
        return cls(load_backbone(name, d, layers, pretrained=True))

    @torch.no_grad()
    def get_embeddings(self, sequences: list[str], batch: int = 128) -> np.ndarray:
        self.backbone.eval()
        out = []
        for i in range(0, len(sequences), batch):
            h, pad = self.backbone(batch_tokens(sequences[i:i + batch]))
            out.append(pool(h, pad))
        return torch.cat(out).numpy()

    def fine_tune(self, sequences: list[str], labels, mode: str = "lora", steps: int = 300, batch: int = 32,
                  lr: float | None = None, seed: int = 0):
        torch.manual_seed(seed)
        self.model = Classifier(self.backbone, n_classes=int(max(labels)) + 1)
        if mode in ("lora", "probe"):
            for p in self.backbone.parameters():
                p.requires_grad = False
            if mode == "lora":
                add_lora(self.backbone)
        params = [p for p in self.model.parameters() if p.requires_grad]
        lr = lr or (3e-4 if mode == "full" else 1e-3)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        ids, y = batch_tokens(sequences), torch.tensor(labels)
        g = torch.Generator().manual_seed(seed)
        self.model.train()
        for step in range(steps):
            idx = torch.randint(0, len(ids), (min(batch, len(ids)),), generator=g)
            loss = F.cross_entropy(self.model(ids[idx]), y[idx])
            for grp in opt.param_groups:
                grp["lr"] = cosine(step, steps, lr, warm=20)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict(self, sequences: list[str], batch: int = 256) -> np.ndarray:
        assert self.model is not None, "call fine_tune first"
        self.model.eval()
        ids = batch_tokens(sequences)
        return torch.cat([self.model(ids[i:i + batch]).softmax(-1) for i in range(0, len(ids), batch)]).numpy()
