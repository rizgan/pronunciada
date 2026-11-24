"""Azure yardstick for the coaching operating point (speechocean762 TEST, cached responses only;
no request is sent and no credential is read).

    python -m bench.coach.azure_yardstick

Flags: per expert-scored phone, ours = coaching error_prob (pronunciada.coach), Azure = 100 -
its phoneme AccuracyScore aligned to the corpus phones exactly as bench/eval_azure_so762.py does
(word by word; an unaligned phone takes the word's accuracy). Target: expert score < 2. Both on
the same phones (the corpus phones our alignment matched). Metrics: precision at our three
recall points, average precision (PR-AUC), speaker-bootstrap 95 % CIs (1000 resamples).
Heard sound: the cached responses were requested with NBestPhonemeCount = 5, so Azure's
candidates exist. On the experts' real substitutions: Azure "names" a sound when its top
candidate differs from the expected phone (exact / within one feature vs the pronounced phone),
and as an upper bound its best candidate other than the expected phone.
Writes results/tables/coaching_azure_yardstick.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from bench.coach.fit_v2 import FOLD, fold, so762_rows, within1
from bench.eval_azure_so762 import ARPA, close
from pronunciada import coach as Co

AZ = {"ɝ": "ɜ˞", "ɚ": "ə˞", "ɡ": "g", "ʌ": "ə"}
N_BOOT = 1000


def az_ipa(p: str) -> str:
    p = AZ.get(p, p)
    if len(p) > 1 and p.endswith("ɹ"):
        p = p[:-1]
    return fold(AZ.get(p, p))


def expand_nb(phonemes):
    out = []
    for p in phonemes:
        ph, sc, nb = p["Phoneme"], p.get("AccuracyScore"), p.get("NBestPhonemes") or []
        if len(ph) > 1 and ph.endswith("ɹ"):
            out += [(ph[:-1], (sc, nb, True)), ("ɹ", (sc, None, False))]
        else:
            out.append((ph, (sc, nb, False)))
    return out


def align_items(ref, hyp):
    """Like bench.eval_azure_so762.align, returning the aligned hyp payload (None if unaligned)."""
    n, m = len(ref), len(hyp)
    if n == m:
        return [h for _, h in hyp]
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (0 if close(ref[i - 1], hyp[j - 1][0]) else 1))
    out = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (0 if close(ref[i - 1], hyp[j - 1][0]) else 1):
            out[i - 1] = hyp[j - 1][1]
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def azure_phones(root):
    """{(utt, word j, phone k): (accuracy, nbest list or None)} from the cached raw responses."""
    from bench.data import load

    out = {}
    for rec in load("speechocean762", split="test"):
        p = os.path.join(root, f"{rec['id']}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p, encoding="utf-8")).get("response") or {}
        nb = (r.get("NBest") or [{}])[0]
        fw = nb.get("Words") or []
        for j, w in enumerate(rec["labels"]["words"]):
            aw = fw[j] if j < len(fw) else None
            acc = aw.get("AccuracyScore", 0.0) if aw else 0.0
            ref = [ARPA["".join(c for c in ph if c.isalpha()).upper()] for ph in w["phones"]]
            items = align_items(ref, expand_nb(aw["Phonemes"])) if aw and aw.get("Phonemes") else [None] * len(ref)
            for k, it in enumerate(items):
                if it is None:
                    out[(rec["id"], j, k)] = (acc, None)
                else:
                    out[(rec["id"], j, k)] = (acc if it[0] is None else it[0], it[1])
    return out


def prec_at_recall(score, y, r):
    o = np.argsort(-score, kind="stable")
    s, yy = score[o], y[o]
    tp = np.cumsum(yy)
    rec = tp / max(yy.sum(), 1)
    k = int(np.searchsorted(rec, r))           # first index reaching recall r
    k = min(k, len(s) - 1)
    thr = s[k]
    f = score >= thr                           # ties at the threshold are all included
    return float(y[f].mean()), float(y[f].sum() / y.sum())


def main():
    from sklearn.metrics import average_precision_score

    root = os.path.expanduser(os.path.join(os.environ.get("PRONUNCIADA_CACHE", "~/.cache/pronunciada"), "azure",
                                           "speechocean762"))
    AZP = azure_phones(root)
    R = so762_rows("so762-test", "test")
    mdl = Co.model()
    ep = Co.error_prob(R, mdl, "en")
    keep = [i for i in range(len(ep)) if np.isfinite(R["acc"][i]) and (R["utt"][i], R["tj"][i], R["tk"][i]) in AZP]
    y = (R["acc"][keep] < 2).astype(float)
    ours = ep[keep]
    az = np.array([100.0 - (AZP[(R["utt"][i], R["tj"][i], R["tk"][i])][0] or 0.0) for i in keep])
    spk = np.asarray(R["spk"])[keep]
    recalls = {s: None for s in ("careful", "balanced", "eager")}
    masks = {s: Co.sound_mask(R, ep, mdl, "en", s)[keep] for s in recalls}
    for s, f in masks.items():
        recalls[s] = float(y[f].sum() / y.sum())
    rng = np.random.default_rng(0)
    us = np.unique(spk)
    idx = {u: np.flatnonzero(spk == u) for u in us}
    boots = []
    for _ in range(N_BOOT):
        b = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        row = {"ap_ours": average_precision_score(y[b], ours[b]), "ap_azure": average_precision_score(y[b], az[b])}
        for s, r in recalls.items():
            f = masks[s][b]
            row[f"p_ours_{s}"] = float(y[b][f].mean())
            row[f"p_azure_{s}"] = prec_at_recall(az[b], y[b], float(y[b][f].sum() / y[b].sum()))[0]
        boots.append(row)

    def ci(k):
        v = np.array([b[k] for b in boots])
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    rep = {"n_phones": len(keep), "error_rate": float(y.mean()),
           "pr_auc": {"ours": float(average_precision_score(y, ours)), "ours_ci": ci("ap_ours"),
                      "azure": float(average_precision_score(y, az)), "azure_ci": ci("ap_azure")},
           "diff_ap_ci": [float(np.quantile([b["ap_ours"] - b["ap_azure"] for b in boots], q)) for q in (0.025, 0.975)],
           "at_recall": {}}
    for s, r in recalls.items():
        f = masks[s]
        pa, ra = prec_at_recall(az, y, r)
        rep["at_recall"][s] = {"recall": r, "precision_ours": float(y[f].mean()), "ours_ci": ci(f"p_ours_{s}"),
                               "precision_azure": pa, "azure_recall_reached": ra, "azure_ci": ci(f"p_azure_{s}"),
                               "diff_ci": [float(np.quantile([b[f"p_ours_{s}"] - b[f"p_azure_{s}"] for b in boots], q))
                                           for q in (0.025, 0.975)]}
    # heard sound on real substitutions (pronounced phone known)
    subs = [i for i in keep if R["pron"][i] not in (None, "<DEL>", "<unk>")]
    named, best = [], []
    for i in subs:
        acc, nb = AZP[(R["utt"][i], R["tj"][i], R["tk"][i])]
        if not nb:
            continue
        exp = fold(R["expected"][i])
        top = az_ipa(nb[0]["Phoneme"])
        if top != exp:
            named.append((R["pron"][i], top, acc if acc is not None else 100.0))
        other = [az_ipa(c["Phoneme"]) for c in nb if az_ipa(c["Phoneme"]) != exp]
        if other:
            best.append((R["pron"][i], other[0]))
    rep["heard"] = {"n_substitutions": len(subs), "nbest_available": True,
                    "azure_top_names_other": {"n": len(named), "coverage": len(named) / max(len(subs), 1),
                                              "exact": float(np.mean([t == h for t, h, _ in named])) if named else None,
                                              "within1": float(np.mean([within1(t, h) for t, h, _ in named])) if named else None},
                    "azure_best_other_candidate": {"n": len(best), "coverage": len(best) / max(len(subs), 1),
                                                   "exact": float(np.mean([t == h for t, h in best])) if best else None,
                                                   "within1": float(np.mean([within1(t, h) for t, h in best])) if best else None}}
    # ours on the same substitutions (balanced), from the shipped decision rule
    utts = np.asarray(R["utt"])
    hits = []
    for u in np.unique(utts[subs]):
        ix = np.flatnonzero(utts == u)
        X = {k: (v[ix] if isinstance(v, np.ndarray) else [v[i] for i in ix]) for k, v in R.items()}
        d = Co.decide(X, "en", "balanced", mdl)
        for q, i in enumerate(ix):
            if i in subs and d["heard"][q]:
                hits.append((R["pron"][i], fold(d["heard"][q])))
    rep["heard"]["ours_balanced"] = {"n": len(hits), "coverage": len(hits) / max(len(subs), 1),
                                     "exact": float(np.mean([t == h for t, h in hits])) if hits else None,
                                     "within1": float(np.mean([within1(t, h) for t, h in hits])) if hits else None}
    # Azure at OUR coverage: its names on the substitutions it scores lowest, as many as we name
    named.sort(key=lambda x: x[2])
    top_n = named[:len(hits)]
    rep["heard"]["azure_top_at_our_coverage"] = {
        "n": len(top_n), "exact": float(np.mean([t == h for t, h, _ in top_n])) if top_n else None,
        "within1": float(np.mean([within1(t, h) for t, h, _ in top_n])) if top_n else None}
    rep["ompal"] = ("not comparable by error type: the cached zh-TW responses carry no phoneme labels (empty "
                    "strings) and zh-CN gives one score per syllable ('ta 1'), not separate initial / final / "
                    "tone scores; Azure has no tone score. Only word-level any-error detection is comparable "
                    "(results/ompal.md).")
    Path("results/tables/coaching_azure_yardstick.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    _ = FOLD
    main()
