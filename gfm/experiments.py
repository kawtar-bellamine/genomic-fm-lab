"""The experiment grid behind the README. Each cell is one tracked run; results land in results/*.json.

    label efficiency   promoter task, 3 label budgets x 4 fine-tuning modes x 2 backbones x 2 seeds, plus a CNN
    long range         1,536 bp task, Transformer vs state-space model vs CNN
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .tracking import Tracker
from .train import FinetuneConfig, finetune

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _run(cfg: FinetuneConfig, group: str) -> dict:
    tr = Tracker("finetune", asdict(cfg), group=group)
    res = finetune(cfg, tracker=tr)
    return {**asdict(cfg), **res, "run": tr.id}


def label_efficiency(seeds=(0, 1, 2), budgets=(250, 1000, 4000), steps=400, out="label_efficiency.json",
                     backbones=("transformer", "s4d", "cnn")):
    RES.mkdir(exist_ok=True)
    path = RES / out
    rows = json.loads(path.read_text()) if path.exists() else []
    seen = {(r["backbone"], r["mode"], r["n_labels"], r["seed"]) for r in rows}
    grid = [(bb, m) for bb in ("transformer", "s4d") for m in ("scratch", "probe", "lora", "full", "probe_random")] + [("cnn", "scratch")]
    grid = [(bb, m) for bb, m in grid if bb in backbones]
    for n in budgets:
        for seed in seeds:
            for bb, mode in grid:
                if (bb, mode, n, seed) in seen:
                    continue
                cfg = FinetuneConfig(backbone=bb, mode=mode, task="promoter", n_labels=n, steps=steps, seed=seed,
                                     n_test=1000)
                rows.append(_run(cfg, "label_efficiency"))
                path.write_text(json.dumps(rows, indent=1))
                print(bb, mode, n, seed, round(rows[-1]["mcc"], 3), flush=True)
    return rows


def long_range(seeds=(0, 1), n_labels=2000, steps=400, out="long_range.json"):
    RES.mkdir(exist_ok=True)
    path = RES / out
    rows = json.loads(path.read_text()) if path.exists() else []
    seen = {(r["backbone"], r["mode"], r["seed"]) for r in rows}
    for seed in seeds:
        for bb, mode in (("transformer", "full"), ("s4d", "full"), ("cnn", "scratch")):
            if (bb, mode, seed) in seen:
                continue
            cfg = FinetuneConfig(backbone=bb, mode=mode, task="long_range", n_labels=n_labels, steps=steps,
                                 seed=seed, n_test=600, batch=16)
            rows.append(_run(cfg, "long_range"))
            path.write_text(json.dumps(rows, indent=1))
            print(bb, mode, seed, round(rows[-1]["mcc"], 3), flush=True)
    return rows
