"""Pretraining (masked language modelling), fine-tuning in four modes, evaluation and attribution.

Every run writes its config, a metrics stream and a summary into runs/<run_id>/ (see tracking.py), so any figure in
the README can be traced back to the runs that produced it.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_task, pretraining_corpus
from .models import MASK, PAD, STRIDE, Classifier, MLM, add_lora, batch_tokens, build_backbone, n_params
from .tracking import Tracker

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


@dataclass
class PretrainConfig:
    backbone: str = "transformer"
    d: int = 64
    layers: int = 4
    steps: int = 1500
    batch: int = 64
    lr: float = 1e-3
    mask_prob: float = 0.15
    corpus: int = 20000
    seed: int = 0


@dataclass
class FinetuneConfig:
    backbone: str = "transformer"
    mode: str = "full"              # scratch | probe | lora | full | probe_random (control: frozen random weights)
    task: str = "promoter"
    n_labels: int = 1000
    steps: int = 300
    batch: int = 32
    lr: float | None = None          # default: 3e-4 for full fine-tuning, 1e-3 otherwise
    lora_r: int = 8
    seed: int = 0
    n_test: int = 2000
    extra: dict = field(default_factory=dict)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def mask_tokens(ids: torch.Tensor, p: float, g: torch.Generator):
    """Mask whole tokens (3 bp at a time, the stride of the stem) and ask for the hidden bases."""
    b, n = ids.shape
    tok = (torch.rand(b, n // STRIDE, generator=g) < p) & (ids[:, ::STRIDE] != PAD)
    sel = tok.repeat_interleave(STRIDE, dim=1) & (ids < 4)
    labels = torch.where(sel, ids, torch.full_like(ids, -100))
    inp = ids.masked_fill(sel, MASK)
    return inp, labels


def cosine(step, total, lr, warm=50):
    if step < warm:
        return lr * (step + 1) / warm
    return lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def pretrain(cfg: PretrainConfig, tracker: Tracker | None = None) -> Path:
    set_seed(cfg.seed)
    tracker = tracker or Tracker("pretrain", asdict(cfg))
    corpus = pretraining_corpus(cfg.corpus, seed=1000 + cfg.seed)
    ids_all = batch_tokens(corpus)
    model = MLM(build_backbone(cfg.backbone, d=cfg.d, layers=cfg.layers))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(cfg.seed)
    t0 = time.time()
    for step in range(cfg.steps):
        idx = torch.randint(0, len(ids_all), (cfg.batch,), generator=g)
        inp, labels = mask_tokens(ids_all[idx], cfg.mask_prob, g)
        loss = F.cross_entropy(model(inp).reshape(-1, 4), labels.reshape(-1), ignore_index=-100)
        for grp in opt.param_groups:
            grp["lr"] = cosine(step, cfg.steps, cfg.lr)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == cfg.steps - 1:
            tracker.log(step=step, loss=loss.item(), bases_per_s=(step + 1) * cfg.batch * inp.shape[1]
                        / (time.time() - t0))
    CKPT.mkdir(exist_ok=True)
    path = CKPT / f"{cfg.backbone}_d{cfg.d}_l{cfg.layers}_s{cfg.seed}.pt"
    torch.save({"config": asdict(cfg), "backbone": model.backbone.state_dict()}, path)
    tracker.finish(checkpoint=str(path), params=n_params(model), minutes=(time.time() - t0) / 60)
    return path


def load_backbone(name: str, d=64, layers=4, pretrained: bool = True, seed: int = 0):
    bb = build_backbone(name, d=d, layers=layers)
    if pretrained:
        state = torch.load(CKPT / f"{name}_d{d}_l{layers}_s{seed}.pt", weights_only=False)["backbone"]
        bb.load_state_dict(state)
    return bb


def mcc(y, p):
    tp, tn = int(((p == 1) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    fp, fn = int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


@torch.no_grad()
def evaluate(model, ids, y, batch=128):
    model.eval()
    logits = torch.cat([model(ids[i:i + batch]) for i in range(0, len(ids), batch)])
    p = logits.argmax(-1).numpy()
    return {"accuracy": float((p == y).mean()), "mcc": mcc(y, p)}


def attribution_precision(model, examples, top_k: int = 4) -> float:
    """Share of the k most attributed tokens (gradient x input on the embedding) that fall on a planted motif.
    Only positives carry a motif mask; a model that looks at the right bases scores close to 1."""
    model.eval()
    pos = [e for e in examples if e.label == 1][:200]
    ids = batch_tokens([e.seq for e in pos])
    emb = model.backbone.emb
    hits = []
    store = {}

    def hook(_, __, out):                       # works for frozen embeddings too (probe, LoRA)
        out = out.detach().requires_grad_(True)
        store["e"] = out
        return out
    h = emb.register_forward_hook(hook)
    logits = model(ids)
    logits[:, 1].sum().backward()
    h.remove()
    score = (store["e"].grad * store["e"]).sum(-1).abs()           # (b, tokens)
    for i, e in enumerate(pos):
        tok_mask = e.mask[: len(e.mask) // 3 * 3].reshape(-1, 3).max(1)
        top = score[i, : len(tok_mask)].topk(top_k).indices.numpy()
        hits.append(tok_mask[top].mean())
    model.zero_grad()
    return float(np.mean(hits))


_DATA_CACHE: dict = {}


def task_data(task, n_labels, n_test, seed):
    key = (task, n_labels, n_test, seed)
    if key not in _DATA_CACHE:
        train = make_task(task, n_labels, seed=10_000 + seed)
        test = make_task(task, n_test, seed=99)                # same test set for every run
        _DATA_CACHE[key] = (train, test, batch_tokens([e.seq for e in train]), np.array([e.label for e in train]),
                            batch_tokens([e.seq for e in test]), np.array([e.label for e in test]))
    return _DATA_CACHE[key]


def finetune(cfg: FinetuneConfig, d=64, layers=4, tracker: Tracker | None = None) -> dict:
    set_seed(cfg.seed)
    tracker = tracker or Tracker("finetune", asdict(cfg))
    train, test, xtr, ytr, xte, yte = task_data(cfg.task, cfg.n_labels, cfg.n_test, cfg.seed)
    pretrained = cfg.mode not in ("scratch", "probe_random") and cfg.backbone != "cnn"
    model = Classifier(load_backbone(cfg.backbone, d, layers, pretrained))
    if cfg.mode in ("probe", "probe_random"):
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif cfg.mode == "lora":
        for p in model.backbone.parameters():
            p.requires_grad = False
        add_lora(model.backbone, cfg.lora_r)
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = cfg.lr or (3e-4 if cfg.mode == "full" else 1e-3)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(cfg.seed)
    ytr_t = torch.tensor(ytr)
    t0 = time.time()
    best, best_state = -2.0, None
    for step in range(cfg.steps):
        model.train()
        idx = torch.randint(0, len(xtr), (min(cfg.batch, len(xtr)),), generator=g)
        loss = F.cross_entropy(model(xtr[idx]), ytr_t[idx])
        for grp in opt.param_groups:
            grp["lr"] = cosine(step, cfg.steps, lr, warm=20)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == cfg.steps - 1:
            tracker.log(step=step, loss=loss.item())
    train_min = (time.time() - t0) / 60
    res = evaluate(model, xte, yte)
    res.update(trainable_params=sum(p.numel() for p in trainable), total_params=n_params(model),
               train_minutes=round(train_min, 3))
    if cfg.task in ("promoter", "long_range"):
        res["attribution_precision"] = attribution_precision(model, test)
    tracker.finish(**res)
    return res


@torch.no_grad()
def mlm_motif_recovery(name: str, d=64, layers=4, n=400, seed=0) -> dict:
    """Mask one token at a time and ask the pretrained model for its 3 bases: accuracy on motif tokens versus
    background tokens. Background is close to random (about 30 % is the ceiling); motifs are predictable only if
    the model learned them and their context."""
    state = torch.load(CKPT / f"{name}_d{d}_l{layers}_s{seed}.pt", weights_only=False)
    model = MLM(build_backbone(name, d=d, layers=layers))
    model.backbone.load_state_dict(state["backbone"])
    model.eval()
    ex = [e for e in make_task("promoter", 2 * n, seed=777) if e.label == 1]
    ids = batch_tokens([e.seq for e in ex])
    rng = np.random.default_rng(0)
    res = {"motif": [], "background": []}
    for kind in res:
        inp, tgt, rows, cols = ids.clone(), [], [], []
        for i, e in enumerate(ex):
            tok = e.mask[: len(e.mask) // STRIDE * STRIDE].reshape(-1, STRIDE).min(1)    # fully inside a motif
            cand = np.where(tok == 1)[0] if kind == "motif" else np.where(
                e.mask[: len(e.mask) // STRIDE * STRIDE].reshape(-1, STRIDE).max(1) == 0)[0]
            if len(cand) == 0:
                continue
            t = int(rng.choice(cand))
            sl = slice(t * STRIDE, (t + 1) * STRIDE)
            rows.append(i)
            cols.append(sl)
            tgt.append(ids[i, sl].clone())
            inp[i, sl] = MASK
        logits = model(inp)
        for r, sl, y in zip(rows, cols, tgt):
            res[kind].append(float((logits[r, sl].argmax(-1) == y).float().mean()))
    return {k: float(np.mean(v)) for k, v in res.items()}


def saliency_tracks(model, examples, n: int = 3) -> list[dict]:
    """Per-base attribution (gradient x input at the stem output, spread over the 3 bases of each token) for a few
    positive examples, with the planted motif mask, for plotting."""
    model.eval()
    pos = [e for e in examples if e.label == 1][:n]
    ids = batch_tokens([e.seq for e in pos])
    store = {}

    def hook(_, __, out):
        out = out.detach().requires_grad_(True)
        store["e"] = out
        return out
    h = model.backbone.emb.register_forward_hook(hook)
    model(ids)[:, 1].sum().backward()
    h.remove()
    score = (store["e"].grad * store["e"]).sum(-1).abs().detach().numpy()
    out = []
    for i, e in enumerate(pos):
        s = np.repeat(score[i], STRIDE)[: len(e.seq)]
        out.append({"saliency": (s / s.max()).round(4).tolist(), "mask": e.mask.tolist()})
    return out
