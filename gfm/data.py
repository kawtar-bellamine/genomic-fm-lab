"""Synthetic regulatory genome with a known answer key.

Sequences are drawn from a human-like background (41 % GC, CpG depleted) into which textbook transcription-factor
motifs are planted following a regulatory grammar. Because every motif position is known, the benchmark can score
not only accuracy but also whether a model looks at the right bases.

Tasks
    promoter   (300 bp)   positive = a TATA box with a GC box (SP1) or CCAAT box (NF-Y) 30-110 bp upstream of it;
                          negatives carry only one of the two, or both in the wrong order.
    long_range (1,536 bp) positive = an enhancer (AP-1 and CREB sites) upstream of a CTCF insulator; negative = the
                          same motifs in the reverse order, about 1,300 bp apart. The class depends on where the
                          motifs sit, which a bag of k-mers or a pooled CNN cannot see.
Real benchmarks (Genomic Benchmarks, Nucleotide Transformer tasks) load through `load_real` when the machine has
internet access; results in the README say which data they come from.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

MOTIFS = {                       # consensus sequences from the literature
    "TATA": ["TATAAAAG", "TATATAAG", "TATAAATG"],         # TBP: TATAWAWR
    "SP1": ["GGGGCGGGGC", "GGGGCGGGAC", "GGGCGGGGCC"],    # GC box
    "NFY": ["AGCCAATCGG", "GGCCAATCAG", "AACCAATCAG"],    # CCAAT box: RRCCAATSRR
    "CTCF": ["CCGCGAGGGGGCAG", "CCACCAGGGGGCGC", "CCGCGTGGTGGCAG"],
    "AP1": ["TGACTCA", "TGAGTCA"],
    "CREB": ["TGACGTCA", "TGACGCAA"],
    "EBOX": ["CACGTG"],
    "GATA": ["AGATAA", "TGATAG"],
}
COMP = str.maketrans("ACGT", "ACGT"[::-1])


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def background(n: int, rng: random.Random) -> list[str]:
    """Human-like background: 41 % GC with CpG dinucleotides depleted about four-fold."""
    out, prev = [], "A"
    for _ in range(n):
        while True:
            b = rng.choices("ACGT", [0.295, 0.205, 0.205, 0.295])[0]
            if not (prev == "C" and b == "G" and rng.random() < 0.75):
                break
        out.append(b)
        prev = b
    return out


def plant(seq: list[str], motif: str, pos: int, mask: np.ndarray):
    seq[pos:pos + len(motif)] = list(motif)
    mask[pos:pos + len(motif)] = 1


@dataclass
class Example:
    seq: str
    label: int
    mask: np.ndarray            # 1 on the bases that decide the label


def promoter_example(rng: random.Random, positive: bool, length: int = 300) -> Example:
    seq, mask = background(length, rng), np.zeros(length, np.int8)
    z = np.zeros(length, np.int8)
    tata_pos = rng.randint(150, 260)
    tata = rng.choice(MOTIFS["TATA"])
    box = rng.choice(MOTIFS[rng.choice(["SP1", "NFY"])])
    gap = rng.randint(30, 110)
    kind = 0 if positive else rng.randrange(3)
    if positive:                     # GC or CCAAT box 30-110 bp upstream of the TATA box
        plant(seq, tata, tata_pos, mask)
        plant(seq, box, tata_pos - gap, mask)
    elif kind == 0:                  # TATA box alone
        plant(seq, tata, tata_pos, z)
    elif kind == 1:                  # GC or CCAAT box alone
        plant(seq, box, tata_pos - gap, z)
    else:                            # both, but in the wrong order: the box sits downstream of the TATA box
        p = rng.randint(30, 110)
        tata_pos = rng.randint(20, length - p - len(box) - 1)
        plant(seq, tata, tata_pos, z)
        plant(seq, box, tata_pos + p, z)
    for _ in range(rng.randint(0, 3)):  # unrelated motifs as distractors
        m = rng.choice(MOTIFS[rng.choice(["EBOX", "GATA", "AP1"])])
        p = rng.randint(0, 15)
        seq[p:p + len(m)] = list(m)
    return Example("".join(seq), int(positive), mask)


def long_range_example(rng: random.Random, positive: bool, length: int = 1536) -> Example:
    """Same motifs in every sequence, only their order over about 1,300 bp differs: a bag of k-mers or a
    translation-invariant model with global pooling cannot tell the classes apart."""
    seq, mask = background(length, rng), np.zeros(length, np.int8)
    z = mask if positive else np.zeros(length, np.int8)
    enh = rng.choice(MOTIFS["AP1"]) + "".join(background(rng.randint(8, 20), rng)) + rng.choice(MOTIFS["CREB"])
    ctcf = rng.choice(MOTIFS["CTCF"])
    start, end = rng.randint(20, 120), length - rng.randint(60, 160)
    first, last = (enh, ctcf) if positive else (ctcf, enh)   # enhancer upstream of the insulator, or the reverse
    plant(seq, first, start, z)
    plant(seq, last, end - len(last), z)
    for _ in range(rng.randint(1, 4)):                        # distractors in the middle
        m = rng.choice(MOTIFS[rng.choice(["EBOX", "GATA", "SP1"])])
        p = rng.randint(300, length - 300)
        seq[p:p + len(m)] = list(m)
    return Example("".join(seq), int(positive), mask)


TASKS = {"promoter": (promoter_example, 300), "long_range": (long_range_example, 1536)}


def make_task(task: str, n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    fn, length = TASKS[task]
    return [fn(rng, i % 2 == 0, length) for i in range(n)]


def pretraining_corpus(n: int, seed: int, length: int = 300) -> list[str]:
    """Unlabelled 'genome': like a real one, it contains regulatory regions that follow the grammar (a box upstream
    of a TATA box, enhancers, insulators) mixed with background carrying stray motifs in any order. No label is
    used: the model can only learn which motifs tend to co-occur, and where."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i % 2 == 0:
            ex = promoter_example(rng, True, length)       # a promoter-like region, label discarded
            seq = list(ex.seq)
        else:
            seq = background(length, rng)
            for _ in range(rng.randint(1, 4)):
                m = rng.choice(MOTIFS[rng.choice(list(MOTIFS))])
                m = m if rng.random() < 0.5 else revcomp(m)
                p = rng.randint(0, length - len(m))
                seq[p:p + len(m)] = list(m)
        out.append("".join(seq))
    return out


def load_real(name: str = "human_nontata_promoters"):
    """Real benchmark from Genomic Benchmarks (Gresova et al., 2023); needs `pip install genomic-benchmarks` and
    internet access. Returns (train, test) lists of Example with an empty motif mask."""
    from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset

    def conv(ds):
        return [Example(s, int(y), np.zeros(len(s), np.int8)) for s, y in ds]
    return conv(get_dataset(name, split="train", version=0)), conv(get_dataset(name, split="test", version=0))
