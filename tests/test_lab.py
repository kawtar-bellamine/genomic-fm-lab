import numpy as np
import torch

from gfm.data import MOTIFS, make_task, pretraining_corpus, revcomp
from gfm.models import (Classifier, LoRALinear, MLM, add_lora, batch_tokens, build_backbone, n_params)
from gfm.train import FinetuneConfig, PretrainConfig, finetune, mask_tokens


def test_motif_masks_mark_the_planted_motifs():
    for e in make_task("promoter", 20, seed=1):
        if e.label:
            assert e.mask.sum() >= 13
            marked = "".join(c for c, m in zip(e.seq, e.mask) if m)
            assert any(t in marked for t in MOTIFS["TATA"])


def test_long_range_classes_share_the_same_motifs():
    """Only the order differs, so a bag of k-mers cannot separate the classes."""
    pos, neg = make_task("long_range", 2, seed=3)
    for e in (pos, neg):
        assert any(c in e.seq for c in MOTIFS["CTCF"]) and any(a in e.seq for a in MOTIFS["AP1"])


def test_revcomp():
    assert revcomp("AACG") == "CGTT"


def test_masking_hides_whole_tokens_and_keeps_targets():
    ids = batch_tokens(pretraining_corpus(8, seed=0))
    inp, lab = mask_tokens(ids, 0.3, torch.Generator().manual_seed(0))
    sel = lab != -100
    assert sel.any() and (inp[sel] == 4).all() and (lab[sel] == ids[sel]).all()


def test_backbones_share_one_interface():
    ids = batch_tokens(["ACGT" * 75, "GGCC" * 60])
    for name in ("transformer", "s4d", "cnn"):
        bb = build_backbone(name, d=32, layers=2)
        h, pad = bb(ids)
        assert h.shape == (2, 100, 32) and pad[1, -1] and not pad[0, -1]
        assert Classifier(bb)(ids).shape == (2, 2)
        assert MLM(bb)(ids).shape == (2, 300, 4)


def test_lora_starts_as_the_identity_and_trains_few_parameters():
    bb = build_backbone("transformer", d=32, layers=2)
    ids = batch_tokens(["ACGT" * 75])
    before = bb(ids)[0].detach()
    for p in bb.parameters():
        p.requires_grad = False
    added = add_lora(bb, r=4)
    assert torch.allclose(before, bb(ids)[0], atol=1e-6)            # B starts at zero
    assert any(isinstance(m, LoRALinear) for m in bb.modules())
    assert added < 0.25 * n_params(bb)


def test_s4d_runs_in_linear_memory_on_long_sequences():
    bb = build_backbone("s4d", d=16, layers=1, max_len=8000)
    h, _ = bb(batch_tokens(["ACGT" * 3000]))
    assert h.shape == (1, 4000, 16) and torch.isfinite(h).all()


def test_tiny_pipeline_end_to_end(tmp_path, monkeypatch):
    import gfm.train as T
    import gfm.tracking as TR
    monkeypatch.setattr(T, "CKPT", tmp_path)
    monkeypatch.setattr(TR, "RUNS", tmp_path / "runs")
    T.pretrain(PretrainConfig(steps=5, batch=8, corpus=64, d=32, layers=2))
    res = finetune(FinetuneConfig(mode="lora", n_labels=32, steps=5, n_test=40), d=32, layers=2)
    assert 0 <= res["accuracy"] <= 1 and res["trainable_params"] < res["total_params"]
    assert 0 <= res["attribution_precision"] <= 1
