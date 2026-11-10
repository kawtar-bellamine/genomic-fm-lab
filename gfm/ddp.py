"""Distributed data-parallel pretraining with torch.distributed (gloo on CPU, nccl on GPUs).

    torchrun --nproc_per_node 2 -m gfm.ddp --steps 100       # 2 workers, global batch 64
    python -m gfm.ddp --steps 100                              # 1 worker, same global batch

With the same global batch and the same seed, both runs see the same samples in the same order, so their loss
curves must match up to floating-point noise: this is the check that the distributed setup is correct before it is
used to go faster.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import pretraining_corpus
from .models import MLM, batch_tokens, build_backbone
from .train import mask_tokens

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--backbone", default="transformer")
    a = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("gloo")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // world))
    torch.manual_seed(0)
    ids_all = batch_tokens(pretraining_corpus(4000, seed=1000))
    model = MLM(build_backbone(a.backbone, d=64, layers=4))
    if world > 1:
        model = DDP(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    local = a.global_batch // world
    losses, t0 = [], time.time()
    for step in range(a.steps):
        idx = torch.randint(0, len(ids_all), (a.global_batch,), generator=g)   # same draw on every rank
        inp, labels = mask_tokens(ids_all[idx], 0.15, g)
        inp, labels = inp[rank * local:(rank + 1) * local], labels[rank * local:(rank + 1) * local]
        loss = F.cross_entropy(model(inp).reshape(-1, 4), labels.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()                                    # DDP all-reduces (averages) the gradients here
        opt.step()
        if world > 1:
            t = loss.detach().clone()
            dist.all_reduce(t)
            losses.append(t.item() / world)
        else:
            losses.append(loss.item())
    seconds = time.time() - t0
    if rank == 0:
        out = ROOT / "results" / f"ddp_world{world}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"world_size": world, "global_batch": a.global_batch, "steps": a.steps,
                                   "seconds": seconds, "bases_per_s": a.steps * a.global_batch * ids_all.shape[1]
                                   / seconds, "losses": losses}))
        print(f"world={world} {seconds:.1f}s final loss {losses[-1]:.4f}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
