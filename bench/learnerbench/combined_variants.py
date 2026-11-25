"""Variants of the combined score, fixed before they were computed (25 November 2026), to show how much the
summary depends on its construction. Nothing here changes the published combined score.

    python -m bench.learnerbench.combined_variants [--boot 2000]

V0  one test per language (as published): en sentence accuracy C, zh sentence accuracy C, es Heroico AUC,
    de LeaP AUC; languages weighted equally.
V1  V0 with each language weighted by its number of speakers.
V2  every level the judges scored: en = mean of sound C, word C, sentence accuracy C (speechocean762),
    L2-ARCTIC AUC and EpaDB AUC; zh = mean of character AUC and sentence accuracy C; es, de as in V0;
    languages weighted equally.
V3  V2 with each language weighted by its number of speakers (the union of its tests' speakers).
Each system is compared with Pronunciada fp32 on the tests both have (a test missing for a system is
left out of that pair, never filled in). Speakers are resampled within language, jointly for the tests of
one corpus; two-sided paired p.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from bench.learnerbench import report as R


def components(rows, srows, sample: bool = False):
    """lang -> list of (corpus, Row) used by V2/V3, and lang -> (corpus, Row) used by V0/V1."""
    v0, v2 = {}, defaultdict(list)
    for r in rows:
        y = getattr(r, "y", None)
        if r.test.startswith("speechocean762") and y is not None and r.metric == "Pearson" and r.unit in ("sound", "word", "sentence accuracy"):
            c = R.Row(r.test, "en", r.label_type, "C", r.unit, r.spk, (lambda yy: lambda v: R.Concordance(v, yy))(np.asarray(y, float)), r.scores)
            v2["en"].append(("so762", c))
            if r.unit == "sentence accuracy":
                v0["en"] = ("so762", c)
        if r.test.startswith(("L2-ARCTIC", "EpaDB")) and r.metric == "AUC":
            v2["en"].append((r.test.split()[0], r))
        if r.test.startswith("OMPAL") and ((r.metric == "AUC" and r.unit == "word (character): any error") or
                                           (r.metric == "Pearson" and r.unit == "sentence accuracy")):
            if r.metric == "Pearson":
                r = R.Row(r.test, "zh", r.label_type, "C", r.unit, r.spk,
                          (lambda yy: lambda v: R.Concordance(v, yy))(np.asarray(r.y, float)), r.scores)
                v0["zh"] = ("ompal", r)
            v2["zh"].append(("ompal", r))
    for r in srows or []:
        if r.metric == "AUC" and r.unit in ("es/heroico", "de/leap"):
            v0[r.lang] = (r.unit, r)
            v2[r.lang].append((r.unit, r))
    return {lg: [x] for lg, x in v0.items()}, dict(v2)


def evaluate(comp: dict, systems, weight: str, n_boot: int, seed: int) -> dict:
    langs = sorted(comp)
    # one speaker index per (language, corpus): tests of one corpus share their resampled speakers
    spk_index, n_spk = {}, {}
    for lg in langs:
        for corpus, r in comp[lg]:
            key = (lg, corpus)
            if key not in spk_index:
                u = sorted(set(map(str, r.spk)))
                spk_index[key] = {s: i for i, s in enumerate(u)}
    lang_speakers = {lg: len({s for c, r in comp[lg] for s in map(str, r.spk)}) for lg in langs}
    maps = {(lg, i): np.array([spk_index[(lg, c)][str(s)] for s in r.spk]) for lg in langs for i, (c, r) in enumerate(comp[lg])}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        d = {k: np.bincount(rng.integers(0, len(v), len(v)), minlength=len(v)).astype(float) for k, v in spk_index.items()}
        draws.append(d)
    fns = {(lg, i): {s: r.build(v) for s, v in r.scores.items() if isinstance(v, np.ndarray)}
           for lg in langs for i, (c, r) in enumerate(comp[lg])}

    def weights(b, lg, i):
        c = comp[lg][i][0]
        return np.ones(len(comp[lg][i][1].spk)) if b is None else draws[b][(lg, c)][maps[(lg, i)]]

    def score(name, shared, b):
        per = []
        for lg, idx in shared.items():
            per.append((np.mean([fns[(lg, i)][name](weights(b, lg, i)) for i in idx]), lang_speakers[lg]))
        w = np.array([1.0 if weight == "equal" else n for _, n in per])
        return float(np.sum(w * np.array([v for v, _ in per])) / w.sum())

    out = {}
    for s in systems:
        shared = {}
        for lg in langs:
            idx = [i for i in range(len(comp[lg])) if s in fns[(lg, i)] and R.BASE in fns[(lg, i)]]
            if idx:
                shared[lg] = idx
        if not shared or s == R.BASE:
            continue
        ps, pb = score(s, shared, None), score(R.BASE, shared, None)
        d = np.array([score(R.BASE, shared, b) - score(s, shared, b) for b in range(n_boot)])
        out[s] = {"languages": sorted(shared), "tests": sum(len(v) for v in shared.values()), "value": ps, "base": pb,
                  "diff": pb - ps, "ci": R.q(d), "p": R.p_two(d)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--vendor", default=None, help="toneperfect or chivox: their sample rows")
    a = ap.parse_args(argv)
    from bench.learners import read_jsonl, root

    os.environ["PRONUNCIADA_LEARNER_SUITE"] = "suite4_test.jsonl"
    recs = R.items()
    recs4 = list(read_jsonl(root() / "suite4_test.jsonl"))
    by = defaultdict(list)
    for r in recs:
        by[r["labels"]["corpus"]].append(r)
    S, _ = R.collect_systems(by)
    if a.vendor:
        A = R.chivox_sample() if a.vendor == "chivox" else R.toneperfect_answers()
        rows = R.so762_rows(S, by["speechocean762"], sample=set(A["speechocean762"]["views"]), chivox=A["speechocean762"], vendor=a.vendor)
        rows += R.ompal_rows(S, sample=set(A["ompal"]["views"]), chivox=A["ompal"], vendor=a.vendor)
        srows = []
        systems = [a.vendor, "azure"]
    else:
        rows = R.so762_rows(S, by["speechocean762"]) + R.errdet_rows(S, by) + R.ompal_rows(S)
        srows, _ = R.suite_rows(S, recs4, by)
        systems = [s for s in S if s != R.BASE]
    v0, v2 = components(rows, srows)
    res = {}
    for name, comp, w in (("V0", v0, "equal"), ("V1", v0, "speakers"), ("V2", v2, "equal"), ("V3", v2, "speakers")):
        res[name] = evaluate(comp, systems, w, a.boot, 31)
        for s, e in res[name].items():
            print(f"{name} {s:30s} {','.join(e['languages']):12s} tests {e['tests']:2d}  ours {e['base']:.3f}  other {e['value']:.3f}  "
                  f"diff {e['diff']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]  p {e['p']:.4f}", flush=True)
    tag = f"_{a.vendor}" if a.vendor else ""
    (R.TABLES / f"learner_benchmark_combined_variants{tag}.json").write_text(json.dumps(R._clean(res), indent=1))


if __name__ == "__main__":
    main()
