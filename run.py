"""Whole study, stage by stage. Each stage is resumable: finished runs are read back from results/.

    python run.py                  # everything (about 1.5 h on 2 CPU cores)
    python run.py --stage grid     # one stage: pretrain | grid | longrange | bench | ddp | report
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def stage_pretrain():
    from gfm.tracking import Tracker
    from gfm.train import CKPT, PretrainConfig, pretrain
    for bb in ("transformer", "s4d"):
        cfg = PretrainConfig(backbone=bb, steps=4000, corpus=40000)
        if not (CKPT / f"{bb}_d64_l4_s0.pt").exists():
            pretrain(cfg, Tracker("pretrain", asdict(cfg), group="pretrain"))


def stage_probe():
    from gfm.bench import save
    from gfm.train import mlm_motif_recovery
    save("motif_recovery", {bb: mlm_motif_recovery(bb) for bb in ("transformer", "s4d")})


def stage_grid():
    from gfm.experiments import label_efficiency
    label_efficiency(seeds=(0, 1), budgets=(500, 2000), steps=800)


def stage_longrange():
    from gfm.experiments import long_range
    long_range(seeds=(0, 1), n_labels=2000, steps=500)


def stage_bench():
    import numpy as np
    import torch
    from gfm.bench import optimisations, save, scaling
    from gfm.models import batch_tokens
    from gfm.train import FinetuneConfig, finetune, task_data
    if not (ROOT / "results" / "scaling.json").exists():
        save("scaling", scaling())
    out = {}
    for bb in ("transformer", "s4d"):
        import gfm.train as T
        captured = {}
        orig = T.evaluate

        def keep(model, ids, y, batch=128):
            captured["model"] = model
            return orig(model, ids, y, batch)
        T.evaluate = keep
        finetune(FinetuneConfig(backbone=bb, mode="full", n_labels=2000, steps=800, n_test=1000))
        T.evaluate = orig
        _, test, _, _, xte, yte = task_data("promoter", 2000, 1000, 0)
        out[bb] = optimisations(captured["model"], xte, np.array(yte), ROOT / f"{bb}.onnx")
        print(bb, out[bb], flush=True)
    save("optimisations", out)


def stage_ddp():
    py = sys.executable
    subprocess.run([py, "-m", "gfm.ddp", "--steps", "100"], cwd=ROOT, check=True)
    subprocess.run([py, "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m", "gfm.ddp", "--steps", "100"],
                   cwd=ROOT, check=True)


def stage_report():
    from gfm import report
    report.build(ROOT)


STAGES = {"pretrain": stage_pretrain, "grid": stage_grid, "longrange": stage_longrange,
          "bench": stage_bench, "ddp": stage_ddp, "report": stage_report}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES))
    a = ap.parse_args()
    for name, fn in STAGES.items():
        if a.stage in (None, name):
            print(f"== {name}", flush=True)
            fn()
    if a.stage is None:
        print(json.dumps({"done": list(STAGES)}))
