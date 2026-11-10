"""Inference benchmarks: how latency and throughput scale with sequence length for attention vs state-space mixing,
and what int8 quantization and ONNX Runtime buy on CPU, with the accuracy checked after each optimisation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import background
from .models import Classifier, batch_tokens, build_backbone

ROOT = Path(__file__).resolve().parent.parent


def _time(fn, repeat=5, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


@torch.no_grad()
def scaling(lengths=(300, 1536, 6144, 12288, 24576, 49152), batch=4, d=64, layers=4, max_seconds=30.0):
    """Median latency of one forward pass for a batch of `batch` sequences; stops a backbone once a call
    exceeds `max_seconds`."""
    import random
    rng = random.Random(0)
    rows = []
    for name in ("transformer", "s4d", "cnn"):
        model = Classifier(build_backbone(name, d=d, layers=layers, max_len=max(lengths) // 3 + 8)).eval()
        for n in lengths:
            ids = batch_tokens(["".join(background(n, rng)) for _ in range(batch)])
            try:
                t = _time(lambda: model(ids), repeat=3, warmup=1)
            except RuntimeError as e:                                # out of memory on long attention
                rows.append({"backbone": name, "length": n, "error": str(e)[:80]})
                break
            rows.append({"backbone": name, "length": n, "latency_s": t, "bases_per_s": batch * n / t})
            print(name, n, round(t, 4), flush=True)
            if t > max_seconds:
                break
    return rows


@torch.no_grad()
def optimisations(model: Classifier, ids: torch.Tensor, y: np.ndarray, onnx_path: Path):
    """fp32 eager vs dynamic int8 vs ONNX Runtime on the same fine-tuned model and the same test set."""
    model.eval()
    out = {}

    def acc(logits):
        return float((logits.argmax(-1).numpy() == y).mean())

    batch = ids[:64]
    out["fp32"] = {"latency_ms_per_seq": 1000 * _time(lambda: model(batch)) / len(batch),
                   "accuracy": acc(torch.cat([model(ids[i:i + 256]) for i in range(0, len(ids), 256)])),
                   "size_mb": sum(p.numel() * 4 for p in model.parameters()) / 1e6}
    q = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    out["int8_dynamic"] = {"latency_ms_per_seq": 1000 * _time(lambda: q(batch)) / len(batch),
                           "accuracy": acc(torch.cat([q(ids[i:i + 256]) for i in range(0, len(ids), 256)]))}
    try:
        import onnxruntime as ort
        torch.onnx.export(model, (batch,), str(onnx_path), input_names=["ids"], output_names=["logits"],
                          dynamic_axes={"ids": {0: "batch"}}, opset_version=17, dynamo=False)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        run = lambda x: sess.run(None, {"ids": x.numpy()})[0]  # noqa: E731
        logits = np.concatenate([run(ids[i:i + 256]) for i in range(0, len(ids), 256)])
        out["onnxruntime"] = {"latency_ms_per_seq": 1000 * _time(lambda: run(batch)) / len(batch),
                              "accuracy": float((logits.argmax(-1) == y).mean()),
                              "size_mb": onnx_path.stat().st_size / 1e6}
    except Exception as e:                                           # export unsupported for this mixer
        out["onnxruntime"] = {"error": str(e)[:120]}
    return out


def save(name: str, obj):
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(obj, indent=1))
