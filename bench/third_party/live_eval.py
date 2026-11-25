"""Evaluation of a commercial API's cached answers on the learner sets, next to Pronunciada and Azure.

Each vendor module turns one cached response into a *view* (vendor-neutral, scores 0-100):

    {"sentence": {"accuracy": x, "total": x, "fluency": x, "prosodic": x, "completeness": x,
                  "secondary": x},                       # None where the vendor has no such score
     "words": [{"text": str, "score": x, "phones": [(phoneme, score), ...]}, ...],   # in order
     "zh_units": [(a, b, word_score, tone_score, sounds_score), ...]}   # Mandarin, char spans of the text

The comparisons use exactly the items the vendor answered (a scored response; rejected or failed
requests are counted, not scored), and the same rows, labels and Pronunciada / Azure outputs the
scoreboard uses:

  speechocean762 TEST  bench/eval_english.so762_items (Pronunciada = the api_run outputs of
      `bench.eval_english`, phones = sound model B, words = word_reg; Azure = aligned_test.jsonl).
      Pearson with the experts at sound, word and sentence level (accuracy, total, fluency, prosodic),
      plus the low-word AUC (expert word accuracy <= 6, build_tier2.py's "mispronounced", against the
      word score, low = flagged). A human word no vendor word lines up with gets the vendor's lowest
      score 0; a human phone no vendor phone lines up with gets its word's score (counted).
  OMPAL TEST           bench/eval_ompal unit_table (Pronunciada = the latest `ours` run, Azure = zh-TW
      primary), the rows of results/ompal.md: word AUC (any error / tone-only / consonant-or-vowel;
      all speakers, and learners only), diagnostic channels (tone, sounds), hit rate at 5 % / 10 %
      false alarms (thresholds from native or learner correct units), sentence Pearson and Spearman
      for learners (accuracy, fluency, prosody, secondary).
  Heroico / USMA       bench/learners/evaluate: prompt-matched concordance (= AUC) native vs learner of
      the sentence score (Pronunciada scores["total"], Azure PronScore), USMA only (primary) and with
      the HEROICO natives (secondary).

Uncertainty: N_BOOT (10,000) resamples of speakers with replacement, every system on the same
resamples (numpy default_rng); vendor - Pronunciada with a 95 % interval and two-sided p; the hit
rates use HIT_BOOT (2,000, the count of results/ompal.md) because each one sorts the units.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from bench.third_party.live_core import auc_low_fn, paired, pearson_fn, spearman_fn

N_BOOT = 10_000
HIT_BOOT = 2_000
LOW_WORD_AT = 6           # build_tier2.py: expert word accuracy <= 6 = mispronounced


# ── speechocean762 ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def so762_reference():
    """(P, Wd, S) of bench.eval_english.so762_items, with row keys. Reads the cached api_run outputs
    (calib/api/speechocean762-test-fp32.pkl) and Azure's aligned_test.jsonl; if the api_run cache is
    missing this scores the split through the public API (the scoreboard's own code path)."""
    from bench.eval_english import so762_items
    from bench.calib.api_run import run
    from bench.data import load

    api = run(load("speechocean762", split="test"), "en", precision="fp32", workers=2, tag="speechocean762-test-fp32")
    P, Wd, S, _, n_missing = so762_items(api)
    return P, Wd, S, n_missing


def _arpa_key(p: str) -> str:
    from bench.eval_azure_so762 import ARPA
    b = "".join(ch for ch in p if ch.isalpha()).upper()
    return ARPA.get(b, b.lower())


@lru_cache(maxsize=1)
def so762_human() -> dict:
    from bench.third_party import lingolix as L
    its, _ = L.tier2x_items()
    return {"so762-" + it["id"]: it["human"]["words"] for it in its}


def vendor_so762_rows(views: dict, human: dict) -> tuple[dict, dict, dict]:
    """Per (utt, j) word score, per (utt, j, k) phone score, coverage counts. `views` keyed by the
    scoreboard id ("so762-<utt>")."""
    from bench.eval_azure_so762 import align as palign
    from bench.third_party import lingolix as L

    W, Ph, cnt = {}, {}, {"words": 0, "words_unmatched": 0, "phones": 0, "phones_word_fallback": 0}
    for uid, v in views.items():
        hw = human[uid]
        al = L.align([w["text"] for w in hw], v.get("words") or [])
        for j, (w, e) in enumerate(zip(hw, al)):
            cnt["words"] += 1
            ws = None if e is None else e.get("score")
            if ws is None:
                cnt["words_unmatched"] += 1
                ws = 0.0
            W[(uid, j)] = float(ws)
            ref = [_arpa_key(p) for p in (w.get("phones") or [])]
            hyp = [(str(ph), float(s)) for ph, s in ((e or {}).get("phones") or []) if s is not None]
            sc = palign(ref, hyp) if hyp and ref else [None] * len(ref)
            for k, s in enumerate(sc):
                cnt["phones"] += 1
                if s is None:
                    cnt["phones_word_fallback"] += 1
                Ph[(uid, j, k)] = float(ws if s is None else s)
    return W, Ph, cnt


SO762_SENT = ("accuracy", "total", "fluency", "prosodic")


def eval_so762(views: dict, ref=None, human=None, n_boot: int = N_BOOT, vendor: str = "vendor") -> dict:
    """views: {"so762-<utt>": view} of answered items only."""
    P, Wd, S, n_missing = ref or so762_reference()
    human = human or so762_human()
    W, Ph, cnt = vendor_so762_rows(views, human)
    out = {"coverage": cnt, "n_utts_answered": len(views), "pronunciada_or_azure_missing": n_missing}

    def take(D, keyname, names):
        keep = [i for i, k in enumerate(D[keyname]) if (k if isinstance(k, str) else k[0]) in views]
        return np.asarray(keep, int)

    # sounds
    i = take(P, "key", None)
    if len(i):
        y = np.asarray(P["y"], float)[i]
        spk = np.asarray(P["spk"])[i]
        s = {"pronunciada": np.asarray(P["B"], float)[i], "azure": np.asarray(P["az"], float)[i]}
        # a vendor whose answers carry no phone scores at all has no sound level (its word scores
        # are not passed off as sound scores)
        has = any((w.get("phones") or []) for v in views.values() for w in v.get("words") or [])
        if has:
            s[vendor] = np.array([Ph.get(P["key"][k], np.nan) for k in i])
        out["sounds_pearson"] = {"n": len(i), "vendor_has_score": has,
                                 **paired({k: pearson_fn(y, v) for k, v in s.items()}, spk, n_boot, 11, vendor)}
    # words
    i = take(Wd, "key", None)
    if len(i):
        y = np.asarray(Wd["y"], float)[i]
        spk = np.asarray(Wd["spk"])[i]
        s = {vendor: np.array([W.get(Wd["key"][k], np.nan) for k in i]), "pronunciada": np.asarray(Wd["ours"], float)[i],
             "azure": np.asarray(Wd["az"], float)[i]}
        out["words_pearson"] = {"n": len(i), **paired({k: pearson_fn(y, v) for k, v in s.items()}, spk, n_boot, 12, vendor)}
        pos = y <= LOW_WORD_AT
        out["low_word_auc"] = {"n": len(i), "n_pos": int(pos.sum()),
                               **paired({k: auc_low_fn(pos, v) for k, v in s.items()}, spk, n_boot, 13, vendor)}
    # sentences
    i = take(S, "key", None)
    if len(i):
        spk = np.asarray(S["spk"])[i]
        for n_, k in enumerate(SO762_SENT):
            y = np.asarray(S[k], float)[i]
            vs = [(views[S["key"][m]].get("sentence") or {}).get(k) for m in i]
            fns = {"pronunciada": pearson_fn(y, np.asarray(S["ours"][k], float)[i]),
                   "azure": pearson_fn(y, np.asarray(S[f"az_{k}"], float)[i])}
            if all(v is not None for v in vs):
                fns[vendor] = pearson_fn(y, np.asarray(vs, float))
            out[f"sentence_{k}_pearson"] = {"n": len(i), "vendor_has_score": vendor in fns,
                                            **paired(fns, spk, n_boot, 20 + n_, vendor)}
        vs = [(views[S["key"][m]].get("sentence") or {}).get("secondary") for m in i]
        if all(v is not None for v in vs):
            y = np.asarray(S["accuracy"], float)[i]
            out["sentence_accuracy_secondary_pearson"] = {
                "n": len(i), **paired({vendor: pearson_fn(y, np.asarray(vs, float))}, spk, n_boot, 29, vendor)}
    return out


# ── OMPAL ──────────────────────────────────────────────────────────────────────

def _as_az(views: dict, pos: int) -> dict:
    """One channel of the vendor's Mandarin units in the Azure-output shape unit_table reads (0-1)."""
    return {i: {"status": "Success", "words": [[u[0], u[1], None if u[pos] is None else u[pos] / 100.0]
                                                 for u in v["zh_units"]]}
            for i, v in views.items()}


def eval_ompal(views: dict, recs=None, ours=None, azure=None, n_boot: int = N_BOOT, hit_boot: int = HIT_BOOT,
               vendor: str = "vendor") -> dict:
    """views: {ompal id: view} of answered items; the other arguments default to the scoreboard's data."""
    from bench import eval_ompal as E

    if recs is None:
        recs = [r for r in E.records("test") if r["id"] in views]
    if ours is None:
        ours, meta = E.ours_outputs(recs)
    else:
        meta = {}
    if azure is None:
        azure = E.azure_outputs(recs, "zh-TW")
    out = {"n_utts_answered": len(recs), "pronunciada_cache": meta.get("dir"),
           "azure_answered": sum(1 for r in recs if (azure.get(r["id"]) or {}).get("status") == "Success")}
    U = E.unit_table(recs, ours, {"V": _as_az(views, 2), "VT": _as_az(views, 3), "VS": _as_az(views, 4), "AZ": azure})
    c = U["cols"]
    out["counts"] = U["counts"]
    if not len(c.get("utt", [])):
        return out
    word = {vendor: "az_V", "pronunciada": "ours_word", "azure": "az_AZ"}
    diag = {"tone_only": {vendor: "az_VT", "pronunciada": "ours_tone"},
            "seg_wrong": {vendor: "az_VS", "pronunciada": "ours_sounds"}}
    for scope, keep in (("all", np.ones(len(c["utt"]), bool)), ("learners", ~c["native"].astype(bool))):
        cc = {k: v[keep] for k, v in c.items()}
        neg = cc["correct"].astype(bool)
        for n_, (name, lab, _, _) in enumerate(E.CLASSES):
            pos = cc[lab].astype(bool)
            row = {"n_units": int(len(cc["utt"])), "n_pos": int(pos.sum()), "n_correct": int(neg.sum())}
            row.update(paired({k: E.auc_fn(pos, neg, cc[col]) for k, col in word.items()}, cc["spk"], n_boot, 40 + n_, vendor))
            out[f"word_auc[{scope}] {name}"] = row
            if lab in diag and scope == "all":
                d = paired({k: E.auc_fn(pos, neg, cc[col]) for k, col in diag[lab].items()}, cc["spk"], n_boot, 50 + n_,
                           vendor)
                out[f"word_auc[{scope}] {name}, diagnostic channel ({'tone' if lab == 'tone_only' else 'sounds'})"] = {
                    "n_units": int(len(cc["utt"])), "n_pos": int(pos.sum()), **d}
    nat = c["native"].astype(bool)
    corr = c["correct"].astype(bool)
    for fa_name, negmask in (("native", corr & nat), ("learner", corr & ~nat)):
        if negmask.sum() == 0:
            continue
        for name, lab, _, _ in E.CLASSES:
            pos = c[lab].astype(bool) & ~nat
            for alpha in (0.05, 0.10):
                out[f"hit[{name}] FA on {fa_name} correct units {int(alpha * 100)} %"] = {
                    "n_pos": int(pos.sum()), "n_neg": int(negmask.sum()),
                    **paired({k: E.hit_fn(pos, negmask, c[col], alpha) for k, col in word.items()}, c["spk"], hit_boot,
                             60, vendor)}
    # sentence, learners
    rows = [r for r in recs if not r["labels"]["native"]]
    if rows:
        spk = np.array([r["speaker"] for r in rows])
        pairs = [("accuracy", "accuracy", "accuracy", "AccuracyScore"), ("fluency", "fluency", "fluency", "FluencyScore"),
                 ("prosody", "prosody", "prosodic", "PronScore"), ("accuracy, secondary", "accuracy", "prob", "PronScore")]
        vkey = {"accuracy": "accuracy", "fluency": "fluency", "prosody": "prosodic", "accuracy, secondary": "secondary"}
        for n_, (row, tk, ok, ak) in enumerate(pairs):
            y = np.array([r["labels"]["sentence"][tk] for r in rows], float)

            def get_o(r, k=ok):
                o = ours.get(r["id"]) or {}
                s = o.get("sentence") if o.get("status") == "Success" else None
                return 0.0 if not s or s.get(k) is None else s[k]

            def get_a(r, k=ak):
                a = azure.get(r["id"]) or {}
                s = a.get("sentence") if a.get("status") == "Success" else None
                return 0.0 if not s or s.get(k) is None else s[k]

            vs = [(views[r["id"]].get("sentence") or {}).get(vkey[row]) for r in rows]
            for kind, fn in (("pearson", pearson_fn), ("spearman", spearman_fn)):
                fns = {"pronunciada": fn(y, np.array([get_o(r) for r in rows], float)),
                       "azure": fn(y, np.array([get_a(r) for r in rows], float))}
                if all(v is not None for v in vs):
                    fns[vendor] = fn(y, np.array(vs, float))
                out[f"sentence[learners] {row} {kind}"] = {"n": len(rows), "vendor_has_score": vendor in fns,
                                                          **paired(fns, spk, n_boot, 70 + n_, vendor)}
    return out


# ── Heroico ────────────────────────────────────────────────────────────────────

def eval_heroico(views: dict, recs=None, ours=None, azure=None, n_boot: int = N_BOOT, vendor: str = "vendor") -> dict:
    from bench.learners import evaluate as LE

    if recs is None:
        recs = [r for r in LE.suite() if r["labels"]["corpus"] == "heroico" and r["id"] in views]
    if ours is None:
        ours, _ = LE.ours_scores(recs)
    if azure is None:
        azure = LE.azure_scores(recs)
    out = {"n_utts_answered": len(recs)}
    for n_, (name, sel) in enumerate((("USMA, native vs learner, prompt-matched (primary)",
                                       lambda r: r["labels"]["subcorpus"] == "usma"),
                                      ("USMA learners vs USMA + HEROICO natives, prompt-matched (secondary)",
                                       lambda r: True))):
        rs = [r for r in recs if sel(r)]
        if not rs:
            continue
        T = LE.Test(name, "es", rs, [1.0 if r["labels"]["native"] else 0.0 for r in rs], "native_vs_learner",
                    strata=[r["labels"]["prompt_id"] for r in rs])
        s = {"pronunciada": [(ours.get(r["id"]) or {}).get("score", 0.0) for r in rs],
             "azure": [(azure.get(r["id"]) or {}).get("score", 0.0) for r in rs],
             vendor: [(views[r["id"]].get("sentence") or {}).get("total") for r in rs]}
        s[vendor] = [0.0 if v is None else v for v in s[vendor]]
        fns = {k: (lambda idx, v=np.asarray(v, float): T.metric(v, idx)) for k, v in s.items()}
        out[name] = {"n": len(rs), "n_speakers": len(T.members),
                     "natives": sum(r["labels"]["native"] for r in rs),
                     "azure_answered": sum(1 for r in rs if (azure.get(r["id"]) or {}).get("status") == "Success"),
                     **paired(fns, T.spk, n_boot, 80 + n_, vendor)}
    return out


# ── markdown ───────────────────────────────────────────────────────────────────

LABELS = {"sounds_pearson": "sounds, Pearson", "words_pearson": "words, Pearson",
          "low_word_auc": "low words (expert accuracy <= 6), AUC",
          "sentence_accuracy_pearson": "sentence accuracy, Pearson", "sentence_total_pearson": "sentence total, Pearson",
          "sentence_fluency_pearson": "sentence fluency, Pearson", "sentence_prosodic_pearson": "sentence prosodic, Pearson",
          "sentence_accuracy_secondary_pearson": "sentence accuracy vs the vendor's secondary score, Pearson"}


def md_rows(set_name: str, block: dict, vendor: str) -> list[str]:
    from bench.third_party.live_core import cell, dcell

    out = []
    for metric, r in block.items():
        if not isinstance(r, dict) or "pronunciada" not in r and vendor not in r:
            continue
        metric = LABELS.get(metric, metric)
        n = r.get("n", r.get("n_units", r.get("n_pos", "")))
        out.append(f"| {set_name} | {metric} | {n} | {cell(r.get(vendor))} | {cell(r.get('pronunciada'))} | "
                   f"{dcell(r.get('delta'))} | {cell(r.get('azure'))} |")
    return out


def is_finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
