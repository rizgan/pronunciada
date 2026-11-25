"""Real-learner benchmark of complete pronunciation-assessment systems (results/learner_benchmark.md).

Scope: packaged systems that take a recording and the text that was read and return sound, word and
sentence scores. Items: the four-language learner suite (bench/learners, suite4_test.jsonl: speechocean762
test, Speech Accent Archive, OMPAL test half, Heroico / USMA, LeaP German and English) plus the L2-ARCTIC
and EpaDB test splits (bench/data). Nothing is fitted, tuned or selected on any of it.

    python -m bench.heavy --lane cpu --need 25 --floor 12 -- python -m bench.learnerbench.run_pronunciada --variant fp32
    python -m bench.learnerbench.run_pronunciada --variant int8
    python -m bench.learnerbench.run_openpronounce --mode default        # in OpenPronounce's own venv (driver here)
    python -m bench.learnerbench.report                                  # tables from cached outputs only

Per-item outputs (never committed): $PRONUNCIADA_CACHE/learnerbench/outputs/<system>/<id>.pkl
"""
from __future__ import annotations

import pickle
from pathlib import Path

from bench.data.common import cache_root


def work() -> Path:
    return cache_root() / "learnerbench"


def out_dir(system: str) -> Path:
    d = work() / "outputs" / system
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(system: str, rid: str, obj) -> None:
    import os

    d = out_dir(system)
    tmp = d / f"{rid}.pkl.tmp"
    tmp.write_bytes(pickle.dumps(obj))
    os.replace(tmp, d / f"{rid}.pkl")


def load_outputs(system: str, ids=None) -> dict:
    d = work() / "outputs" / system
    out = {}
    if not d.exists():
        return out
    if ids is None:
        for p in d.glob("*.pkl"):
            out[p.stem] = pickle.loads(p.read_bytes())
    else:
        for i in ids:
            p = d / f"{i}.pkl"
            if p.exists():
                out[i] = pickle.loads(p.read_bytes())
    return out


def items() -> list[dict]:
    """Every benchmark record: suite4 (with labels.corpus) + L2-ARCTIC and EpaDB test splits + the
    LeaP later readings (labels.corpus leap-<lang>-prepost)."""
    from bench.data import load
    from bench.learners import read_jsonl, root

    recs = list(read_jsonl(root() / "suite4_test.jsonl"))
    for ds in ("l2arctic", "epadb"):
        for r in load(ds, split="test"):
            recs.append({**r, "labels": {**r["labels"], "corpus": ds, "native": False}})
    for c in ("leap-de", "leap-en"):          # later readings of LeaP speakers (pre-declared secondary only)
        p = root(c) / "prepost.jsonl"
        if p.exists():
            recs += [{**r, "labels": {**r["labels"], "corpus": f"{c}-prepost"}} for r in read_jsonl(p)]
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids))
    return recs
