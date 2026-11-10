"""Tokenizer, two backbones (Transformer and a diagonal state-space model), a CNN baseline, LoRA and heads.

Both backbones expose the same interface, so pretraining, fine-tuning, LoRA, benchmarking and the embedding API
work on either: the comparison isolates the sequence mixer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = "ACGTN"                      # N doubles as the mask symbol
PAD = 5
STOI = {c: i for i, c in enumerate(CHARS)}
STRIDE = 3                           # the stem turns 300 bp into 100 tokens
VOCAB = list(CHARS)                  # what the MLM head predicts per base (A, C, G, T)
MASK = STOI["N"]


def tokenize(seq: str) -> list[int]:
    return [STOI.get(c, MASK) for c in seq.upper()]


def batch_tokens(seqs: list[str]) -> torch.Tensor:
    """Base-level ids, padded to a multiple of the stem stride."""
    ids = [tokenize(s) for s in seqs]
    n = max(map(len, ids))
    n += (-n) % STRIDE
    return torch.tensor([x + [PAD] * (n - len(x)) for x in ids])


class Stem(nn.Module):
    """One-hot bases -> strided convolution: every token sees 9 bp, so a motif is recognised whatever its frame,
    unlike fixed k-mer tokens."""

    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv1d(len(CHARS) + 1, d, kernel_size=9, stride=STRIDE, padding=3)

    def forward(self, ids):
        x = F.one_hot(ids, len(CHARS) + 1).float()
        x[..., PAD] = 0
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


def token_pad(ids):
    return ids[:, ::STRIDE] == PAD


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x, with W frozen and only A, B trained (Hu et al., 2021)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


def add_lora(model: nn.Module, r: int = 8, targets=("qkv", "out", "fc1", "fc2", "in_proj", "out_proj")) -> int:
    """Swap the target Linear layers for LoRA versions; returns the number of trainable LoRA parameters."""
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(mod, child_name, LoRALinear(child, r))
    return sum(p.numel() for n, p in model.named_parameters() if ".A" in n or ".B" in n)


def sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, pad):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=~pad[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class S4DMixer(nn.Module):
    """Bidirectional diagonal state-space layer (S4D-Lin initialisation, Gu et al., 2022) applied as a long
    convolution through the FFT: cost O(n log n) in the sequence length instead of O(n^2) for attention."""

    def __init__(self, d, state=32):
        super().__init__()
        self.in_proj, self.out_proj = nn.Linear(d, 2 * d), nn.Linear(d, d)
        h = 2 * d                                     # forward and backward kernels
        self.log_dt = nn.Parameter(torch.rand(h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.log_a_re = nn.Parameter(torch.log(0.5 * torch.ones(h, state // 2)))
        self.a_im = nn.Parameter(math.pi * torch.arange(state // 2).float().repeat(h, 1))
        self.c = nn.Parameter(torch.randn(h, state // 2, 2) * 0.5)
        self.D = nn.Parameter(torch.ones(d))
        self._cache: tuple[int, torch.Tensor] | None = None

    def kernel(self, n):
        dt = self.log_dt.exp()[:, None]
        a = -self.log_a_re.exp() + 1j * self.a_im
        dta = a * dt
        c = torch.view_as_complex(self.c) * (dta.exp() - 1) / a
        vand = torch.exp(dta[:, :, None] * torch.arange(n, device=a.device))
        return 2 * torch.einsum("hs,hsl->hl", c, vand).real   # (2d, n)

    def forward(self, x, pad):
        b, n, d = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = u.masked_fill(pad[..., None], 0).transpose(1, 2)          # (b, d, n)
        if self.training or self._cache is None or self._cache[0] != n:
            k = self.kernel(n)                     # the kernel does not depend on the input: cache it for inference
            self._cache = None if self.training else (n, k.detach())
        else:
            k = self._cache[1]
        kf, kb = k[:d], k[d:]
        L = 2 * n
        y = torch.fft.irfft(torch.fft.rfft(u, L) * torch.fft.rfft(kf, L), L)[..., :n]
        y = y + torch.fft.irfft(torch.fft.rfft(u.flip(-1), L) * torch.fft.rfft(kb, L), L)[..., :n].flip(-1)
        y = y.transpose(1, 2) + u.transpose(1, 2) * self.D
        return self.out_proj(F.gelu(y) * torch.sigmoid(gate))


class Block(nn.Module):
    def __init__(self, d, mixer: str, heads=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mix = Attention(d, heads) if mixer == "transformer" else S4DMixer(d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, pad):
        x = x + self.mix(self.n1(x), pad)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class Backbone(nn.Module):
    def __init__(self, mixer: str = "transformer", d: int = 96, layers: int = 4, max_len: int = 4096):
        super().__init__()
        self.mixer, self.d = mixer, d
        self.emb = Stem(d)
        self.register_buffer("pe", sinusoidal(max_len, d), persistent=False)
        self.blocks = nn.ModuleList([Block(d, mixer) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        pad = token_pad(ids)
        x = self.emb(ids)
        if self.mixer == "transformer":
            x = x + self.pe[: x.shape[1]]
        for blk in self.blocks:
            x = blk(x, pad)
        return self.norm(x), pad


class CNNBackbone(nn.Module):
    """Dilated residual CNN, the usual supervised baseline for regulatory genomics."""

    def __init__(self, d: int = 96, layers: int = 6, max_len: int | None = None):
        super().__init__()
        self.mixer, self.d = "cnn", d
        self.emb = Stem(d)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2 * 2 ** i, dilation=2 ** i) for i in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        pad = token_pad(ids)
        x = self.emb(ids).transpose(1, 2)
        for c in self.convs:
            x = x + F.gelu(c(x))
        return self.norm(x.transpose(1, 2)), pad


def pool(h, pad):
    keep = (~pad).unsqueeze(-1).float()
    return (h * keep).sum(1) / keep.sum(1).clamp(min=1)


class MLM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.d, STRIDE * 4)          # the 3 bases of each token

    def forward(self, ids):
        h = self.head(self.backbone(ids)[0])
        return h.view(h.shape[0], -1, 4)                         # (b, bases, A/C/G/T)


class Classifier(nn.Module):
    def __init__(self, backbone, n_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(backbone.d, backbone.d), nn.GELU(), nn.Linear(backbone.d, n_classes))

    def forward(self, ids):
        return self.head(pool(*self.backbone(ids)))


def build_backbone(name: str, **kw):
    return CNNBackbone(**kw) if name == "cnn" else Backbone(name, **kw)


def n_params(m: nn.Module, trainable: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)
