"""Figures and the results block of the README, generated from results/*.json only."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODE_C = {"scratch": "#9a9893", "probe": "#eda100", "lora": "#2a78d6", "full": "#1baf7a", "probe_random": "#d9d6cf"}
BB_C = {"transformer": "#2a78d6", "s4d": "#eb6834", "cnn": "#52514e"}
BB_L = {"transformer": "Transformer", "s4d": "State-space (S4D)", "cnn": "CNN (supervised only)"}
MODE_L = {"scratch": "from scratch", "probe": "linear probe", "lora": "LoRA (r = 8)", "full": "full fine-tuning",
          "probe_random": "probe on random weights (control)"}
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "legend.frameon": False})


def _load(res: Path, name):
    p = res / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _agg(rows, keys, metric="mcc"):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r[metric])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in out.items()}


def pipeline_diagram(fig_dir: Path):
    def card(title, lines, bg, ink):
        rows = "".join(f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#3a3935">{x}</FONT></TD></TR>' for x in lines)
        return (f'<<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="5" BGCOLOR="{bg}" STYLE="ROUNDED">'
                f'<TR><TD ALIGN="LEFT"><B><FONT POINT-SIZE="12" COLOR="{ink}">{title}</FONT></B></TD></TR>{rows}</TABLE>>')
    n = {
        "data": card("Regulatory genome", ["TF motifs + grammar", "answer key per base"], "#f1efe9", "#6b6862"),
        "pre": card("Pretraining", ["masked bases (MLM)", "Transformer | S4D", "DDP across workers"], "#e3eefb", "#2a5fa8"),
        "ck": card("Checkpoints", ["one per backbone", "tracked config + metrics"], "#f1efe9", "#6b6862"),
        "ft": card("Fine-tuning", ["scratch | probe | LoRA | full", "2 label budgets, 1 to 2 seeds"], "#e2f4ec", "#137a56"),
        "ev": card("Evaluation", ["MCC on a fixed test set", "attribution on motifs", "long-range task"], "#fdf0e6", "#a8501d"),
        "srv": card("Serving", ["latency vs length", "int8, ONNX Runtime", "embeddings API"], "#f6eccb", "#8a6d00"),
        "trk": card("Tracking", ["runs/*: config, metrics, summary", "optional Weights &amp; Biases"], "#efe9f6", "#5b3f8a"),
    }
    src = ['digraph G {rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.45; pad=0.15;',
           'node [shape=plain fontname="Helvetica"]; edge [color="#9a9893" penwidth=1.4 arrowsize=0.7];']
    src += [f"{k} [label={v}];" for k, v in n.items()]
    src += ["data -> pre -> ck -> ft -> ev; ck -> srv; ft -> srv;", 'pre -> trk [style=dashed]; ft -> trk [style=dashed];',
            "{rank=same; ev; srv}", "}"]
    subprocess.run(["dot", "-Tpng", "-Gdpi=170", "-o", str(fig_dir / "pipeline.png")], input="\n".join(src).encode(),
                   check=True)


def build(root: Path):
    res, fig_dir = root / "results", root / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pipeline_diagram(fig_dir)
    lines = []

    rec = _load(res, "motif_recovery")
    grid = _load(res, "label_efficiency")
    if grid:
        agg = _agg(grid, ("backbone", "mode", "n_labels"))
        budgets = sorted({r["n_labels"] for r in grid})
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
        for ax, bb in zip(axes, ("transformer", "s4d")):
            for mode in ("full", "lora", "probe", "scratch", "probe_random"):
                m = [agg.get((bb, mode, b), (np.nan, 0, 0)) for b in budgets]
                ax.errorbar(budgets, [x[0] for x in m], yerr=[x[1] for x in m], color=MODE_C[mode], marker="o",
                            linewidth=2.2, capsize=3, label=MODE_L[mode])
            c = [agg.get(("cnn", "scratch", b), (np.nan, 0, 0)) for b in budgets]
            ax.plot(budgets, [x[0] for x in c], color=BB_C["cnn"], linestyle="--", marker="s", label=BB_L["cnn"])
            ax.set_xscale("log")
            ax.set_xticks(budgets, [f"{b:,}" for b in budgets])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_title(BB_L[bb])
            ax.set_xlabel("labelled training sequences")
        axes[0].set_ylabel("MCC on 1,000 test sequences")
        axes[0].legend(fontsize=8, loc="upper left")
        seeds = {bb: len({r["seed"] for r in grid if r["backbone"] == bb}) for bb in ("transformer", "s4d")}
        fig.suptitle(f"Label efficiency on the promoter task (Transformer: {seeds['transformer']} seeds, "
                     f"S4D: {seeds['s4d']} seed; bars show the spread)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "label_efficiency.png", dpi=160)
        plt.close(fig)

        # trainable parameters vs score at the middle budget
        mid = budgets[len(budgets) // 2]
        fig, ax = plt.subplots(figsize=(8.4, 3.6))
        for r in [r for r in grid if r["n_labels"] == mid]:
            ax.scatter(r["trainable_params"], r["mcc"], color=MODE_C.get(r["mode"], "#52514e"),
                       marker="o" if r["backbone"] == "transformer" else ("^" if r["backbone"] == "s4d" else "s"),
                       s=55, alpha=0.85, edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters (log)")
        ax.set_ylabel("MCC")
        ax.set_title(f"Cost of adaptation at {mid:,} labels")
        for mode in MODE_C:
            ax.scatter([], [], color=MODE_C[mode], label=MODE_L[mode])
        ax.scatter([], [], color="#888", marker="o", label="Transformer")
        ax.scatter([], [], color="#888", marker="^", label="S4D")
        ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(fig_dir / "adaptation_cost.png", dpi=160)
        plt.close(fig)

        lines += ["| Backbone | Fine-tuning | " + " | ".join(f"MCC, {b:,} labels" for b in budgets)
                  + " | Trainable parameters | Attribution on motifs |",
                  "|---|---|" + "---:|" * (len(budgets) + 2)]
        for bb in ("transformer", "s4d", "cnn"):
            for mode in ("full", "lora", "probe", "probe_random", "scratch"):
                rows = [r for r in grid if r["backbone"] == bb and r["mode"] == mode]
                if not rows:
                    continue
                cells = []
                for b in budgets:
                    m = agg.get((bb, mode, b))
                    cells.append(f"{m[0]:.2f} ± {m[1]:.2f}" if m else "")
                tp = rows[0]["trainable_params"]
                att = np.mean([r["attribution_precision"] for r in rows if r["n_labels"] == budgets[-1]])
                lines.append(f"| {BB_L[bb]} | {MODE_L[mode]} | " + " | ".join(cells) + f" | {tp:,} | {att:.0%} |")

    lr = _load(res, "long_range")
    if lr:
        agg = _agg(lr, ("backbone",))
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        names = [b for b in ("transformer", "s4d", "cnn") if (b,) in agg]
        for i, b in enumerate(names):
            m, s, _ = agg[(b,)]
            ax.barh(i, m, xerr=s, color=BB_C[b], height=0.6, capsize=3)
            ax.text(max(m, 0) + 0.03, i, f"{m:.2f}", va="center", fontsize=9, color=INK2)
        ax.set_yticks(range(len(names)), [BB_L[b] for b in names])
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("MCC on 600 test sequences of 1,536 bp")
        ax.set_title("Same motifs, only their order differs: position-aware models win")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "long_range.png", dpi=160)
        plt.close(fig)

    sc = _load(res, "scaling")
    if sc:
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        for b in ("transformer", "s4d", "cnn"):
            pts = [r for r in sc if r["backbone"] == b and "latency_s" in r]
            ax.plot([p["length"] for p in pts], [p["latency_s"] * 1000 for p in pts], marker="o", color=BB_C[b],
                    linewidth=2.2, label=BB_L[b].replace(" (supervised only)", ""))
        ax.set_xscale("log")
        ax.set_yscale("log")
        lens = sorted({r["length"] for r in sc})
        ax.set_xticks(lens, [f"{x / 1000:g}k" if x >= 1000 else str(x) for x in lens])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("sequence length (bp)")
        ax.set_ylabel("latency, batch of 4 (ms)")
        ax.set_title("Inference cost vs sequence length on CPU")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "scaling.png", dpi=160)
        plt.close(fig)

    ddp1, ddp2 = _load(res, "ddp_world1"), _load(res, "ddp_world2")
    if ddp1 and ddp2:
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        ax.plot(ddp1["losses"], color="#9a9893", linewidth=3, label="1 worker, global batch 64")
        ax.plot(ddp2["losses"], color="#2a78d6", linewidth=1.3, linestyle="--", label="2 workers (DDP), 32 each")
        ax.set_xlabel("step")
        ax.set_ylabel("MLM loss")
        ax.set_title("Distributed training reproduces the single-worker run")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "ddp.png", dpi=160)
        plt.close(fig)

    att = _load(res, "attribution_example")
    if att:
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 3.6), sharex=True)
        for ax, bb in zip(axes, ("transformer", "s4d")):
            tr = att[bb]["tracks"][0]
            s = np.array(tr["saliency"])
            m = np.array(tr["mask"])
            x = np.arange(len(s))
            ax.fill_between(x, 0, 1.05, where=m > 0, color="#f6d7a7", step="mid", label="planted motifs (answer key)")
            ax.plot(x, s, color=BB_C[bb], linewidth=0.8, label="model attribution")
            ax.set_ylim(0, 1.08)
            ax.set_yticks([])
            ax.set_ylabel(BB_L[bb].split(" (")[0], rotation=0, ha="right", va="center", fontsize=9)
            ax.grid(False)
        axes[0].legend(fontsize=8, ncol=2, loc="upper center")
        axes[1].set_xlabel("position in the 1,536 bp sequence")
        fig.suptitle("Where the fine-tuned models look: mostly at the insulator near the end, a valid shortcut "
                     "that the answer key reveals", fontsize=10.5, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "attribution.png", dpi=160)
        plt.close(fig)

    return {"lines": lines, "recovery": rec}


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent)
