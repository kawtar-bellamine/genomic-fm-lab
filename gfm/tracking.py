"""Experiment tracking: every run gets a folder with its config, a JSONL metric stream and a summary.

Local by default, so results are reproducible without an account. If `wandb` is installed and WANDB_API_KEY (or
WANDB_MODE=offline) is set, the same calls are mirrored to Weights & Biases.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "no-git"


class Tracker:
    def __init__(self, kind: str, config: dict, group: str = "default"):
        cfg_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        self.id = f"{kind}-{cfg_hash}"
        self.dir = RUNS / group / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = {"kind": kind, "config": config, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git": _git_commit(), "python": platform.python_version()}
        (self.dir / "config.json").write_text(json.dumps(meta, indent=2))
        self.stream = open(self.dir / "metrics.jsonl", "w")
        self.wandb = None
        if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline":
            try:
                import wandb
                self.wandb = wandb.init(project="genomic-fm-lab", group=group, name=self.id, config=config,
                                        reinit=True)
            except ImportError:
                pass

    def log(self, **metrics):
        self.stream.write(json.dumps(metrics) + "\n")
        self.stream.flush()
        if self.wandb:
            self.wandb.log(metrics)

    def finish(self, **summary):
        self.stream.close()
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if self.wandb:
            self.wandb.summary.update(summary)
            self.wandb.finish()
