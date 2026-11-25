"""English head-to-head vs Azure Pronunciation Assessment, through the PUBLIC API, held-out data.

    python -m bench.eval_english [--steps run,so762,calib,errdet,speed,int8,report] [--workers 5]

Every number is computed from pronunciada.Scorer.score(audio_path, text) outputs (bench.calib.api_run)
on speechocean762 TEST, l2arctic TEST and epadb TEST; Azure from its cached responses on the same
clips. Nothing here fits, selects or thresholds on a test split, except that the F1 thresholds are
chosen on the corpora's TRAIN splits (feature tables, same feature code) and applied to test.
Writes results/english_vs_azure.md and results/tables/english_vs_azure.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from bench.calib.common import load_json, save_json

N_BOOT = 1000
SETS = ("A", "B", "C")
MID = {"phone": 1.4, "word": 6.5, "sentence": 7.0}   # midpoint of the lenient / strict thresholds


def az_root(ds):
    return os.path.expanduser(os.path.join(os.environ.get("PRONUNCIADA_CACHE", "~/.cache/pronunciada"), "azure", ds))


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def blend(zl, zs, s):
    from pronunciada.calibrate import blend_logit

    return sig(blend_logit(zl, np.minimum(zs, zl), s))


# ----------------------------------------------------------------------------- statistics

def corr_block(y, ours: dict, az, clusters):
    """Pearson / Spearman with speaker-cluster bootstrap CIs; paired bootstrap ours - Azure."""
    from bench.metrics import bootstrap_ci, paired_bootstrap, pearson, spearman

    y = np.asarray(y, float)
    out = {}
    for name, s in list(ours.items()) + [("azure", az)]:
        if s is None:
            continue
        s = np.asarray(s, float)
        r = {"pearson": pearson(s, y), "spearman": spearman(s, y), "n": int(len(y))}
        ci = bootstrap_ci(pearson, s, y, clusters=clusters, n_boot=N_BOOT)
        r["pearson_ci"] = [ci["lo"], ci["hi"]]
        ci = bootstrap_ci(spearman, s, y, clusters=clusters, n_boot=N_BOOT)
        r["spearman_ci"] = [ci["lo"], ci["hi"]]
        if name != "azure" and az is not None:
            pb = paired_bootstrap(lambda yy, ss: pearson(ss, yy), y, s, np.asarray(az, float),
                                  clusters=clusters, n_boot=N_BOOT)
            r["minus_azure"] = {"delta": pb["delta"], "lo": pb["lo"], "hi": pb["hi"],
                                "p_value_one_sided": pb["p_value"],
                                "p_value_two_sided": min(1.0, 2 * min(pb["p_value"], 1 - pb["p_value"] + 1 / (N_BOOT + 1)))}
        out[name] = r
    return out


def calib_block(y, p, n_bins=15):
    from bench.metrics import brier, reliability_table, roc_auc
    from bench.calib.common import ece

    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    return {"n": int(len(y)), "pos_rate": float(y.mean()), "mean_p": float(p.mean()),
            "ece": ece(y, p, n_bins), "brier": brier(y, p), "auc": roc_auc(y, p),
            "reliability": reliability_table(y, p, n_bins=n_bins, strategy="mass")}


def det_block(y, s, clusters, thr=None):
    """y = 1 error; s = error score (higher = more likely wrong)."""
    from bench.metrics import best_f1, bootstrap_ci, pr_auc, prf_at, roc_auc

    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    ci = bootstrap_ci(roc_auc, y, s, clusters=clusters, n_boot=N_BOOT)
    b = best_f1(y, s)
    r = {"n": int(len(y)), "n_err": int(y.sum()), "auc": roc_auc(y, s), "auc_ci": [ci["lo"], ci["hi"]],
         "pr_auc": pr_auc(y, s), "best_f1": b.f1, "best_f1_thr": b.threshold}
    if thr is not None:
        f = prf_at(y, s, thr)
        r.update(f1_at_train_thr=f.f1, precision_at_train_thr=f.precision, recall_at_train_thr=f.recall,
                 train_thr=thr)
    return r


# ----------------------------------------------------------------------------- run

def run_api(workers, precision="fp32"):
    from bench.calib.api_run import run
    from bench.data import load

    out = {}
    for ds in ("speechocean762", "l2arctic", "epadb"):
        recs = load(ds, split="test")
        out[ds] = run(recs, "en", precision=precision, workers=workers, tag=f"{ds}-test-{precision}")
    return out


# ----------------------------------------------------------------------------- speechocean762

def without_explain(api: dict) -> dict:
    """The same API outputs with the explain-away pass removed (first-pass values)."""
    return {k: ({**o, **o["first"]} if isinstance(o, dict) and "first" in o else o) for k, o in api.items()}


def so762_items(api):
    """Their phones / words / sentences of speechocean762 test with ours (A, B, C, U) and Azure."""
    from bench.data import load
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference, arpa_to_ours

    az = {}
    for line in open(os.path.join(az_root("speechocean762"), "aligned_test.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        az[r["id"]] = r
    # "key": (utt id, word index, phone index) / (utt id, word index) / utt id of every row, so other
    # systems (bench/third_party/*_live.py) can be lined up with these exact rows
    P = {"y": [], "spk": [], "az": [], "fb": 0, "n_match": 0, "key": [], **{k: [] for k in SETS + ("U",)}}
    Wd = {"y": [], "spk": [], "az": [], "fb": 0, "ours": [], "key": []}
    S = {k: [] for k in ("accuracy", "fluency", "prosodic", "total")}
    S.update({f"az_{k}": [] for k in S}, spk=[], ours={k: [] for k in S}, key=[])
    calib = {"phone": {"y": [], "zl": [], "zs": [], "zlU": [], "zsU": [], "spk": []},
             "word": {"y": [], "pmin": [], "zl": [], "zs": [], "spk": []},
             "sentence": {"y": [], "all_len_ok": [], "zl": [], "zs": [], "spk": []}}
    n_missing = 0
    for rec in load("speechocean762", split="test"):
        o = api.get(rec["id"])
        a = az.get(rec["id"])
        if o is None or "error" in o or a is None:
            n_missing += 1
            continue
        ref = build_ref("en", rec["text"])
        assert [p.ipa for p in ref.phones] == o["phones"], rec["id"]
        W = rec["labels"]["words"]
        tp = [[arpa_to_ours(p) for p in w["phones"]] for w in W]
        o2t, wpairs = align_reference(ref, [w["text"] for w in W], tp)
        t2o = {v: k for k, v in o2t.items()}
        tw_ours = {}
        for i, (j, _) in o2t.items():
            tw_ours.setdefault(j, set()).add(ref.phones[i].word)
        for i, j in wpairs:
            tw_ours.setdefault(j, set()).add(i)
        spk = rec["speaker"]
        for j, w in enumerate(W):
            ow = sorted(tw_ours.get(j, set()))
            fb_ph = [k for x in ow for k in range(*ref.words[x].phones)]
            for k, sc in enumerate(w["phones_accuracy"]):
                i = t2o.get((j, k), -1)
                for m in SETS + ("U",):
                    reg = o["sound"][m]["reg"]
                    if i >= 0:
                        v = reg[i]
                    elif fb_ph:
                        v = float(np.mean(reg[fb_ph]))
                    else:
                        v = float(np.mean(reg))
                    P[m].append(float(v))
                P["n_match"] += i >= 0
                P["fb"] += i < 0
                P["y"].append(float(sc))
                P["key"].append((rec["id"], j, k))
                P["spk"].append(spk)
                P["az"].append(float(a["words"][j]["phones"][k]))
                if i >= 0:
                    c = calib["phone"]
                    c["y"].append(float(sc))
                    c["zl"].append(o["sound"]["B"]["z_len"][i])
                    c["zs"].append(o["sound"]["B"]["z_str"][i])
                    c["zlU"].append(o["sound"]["U"]["z_len"][i])
                    c["zsU"].append(o["sound"]["U"]["z_str"][i])
                    c["spk"].append(spk)
            Wd["y"].append(float(w["accuracy"]))
            Wd["key"].append((rec["id"], j))
            Wd["spk"].append(spk)
            Wd["az"].append(float(a["words"][j]["accuracy"]))
            if ow:
                Wd["ours"].append(float(np.mean(o["word_reg"][ow])))
                c = calib["word"]
                c["y"].append(float(w["accuracy"]))
                c["pmin"].append(float(min(w["phones_accuracy"])) if w["phones_accuracy"] else 2.0)
                c["zl"].append(float(np.min(o["word_z"][0][ow])))
                c["zs"].append(float(np.min(o["word_z"][1][ow])))
                c["spk"].append(spk)
            else:
                Wd["ours"].append(float(np.mean(o["word_reg"])))
                Wd["fb"] += 1
        sl = rec["labels"]["sentence"]
        for k in ("accuracy", "fluency", "prosodic", "total"):
            S[k].append(float(sl[k]))
            v = (a["sentence"] or {}).get(k)          # aligned_test.jsonl keys: accuracy / fluency / ...
            S[f"az_{k}"].append(0.0 if v is None else float(v))
            S["ours"][k].append(o["sentence"][k])
        S["spk"].append(spk)
        S["key"].append(rec["id"])
        c = calib["sentence"]
        c["y"].append(float(sl["accuracy"]))
        c["all_len_ok"].append(all(w["accuracy"] >= 5 and (min(w["phones_accuracy"]) if w["phones_accuracy"] else 2) >= 1.0
                                   for w in W))
        c["zl"].append(o["sentence_z"][0])
        c["zs"].append(o["sentence_z"][1])
        c["spk"].append(spk)
    return P, Wd, S, calib, n_missing


def eval_so762(api):
    P, Wd, S, calib, n_missing = so762_items(api)
    out = {"n_utts_missing": n_missing,
           "phone": corr_block(P["y"], {f"ours_{m}": P[m] for m in SETS + ("U",)}, P["az"], np.array(P["spk"])),
           "phone_coverage": {"their_phones": len(P["y"]), "matched_to_ours": P["n_match"],
                              "fallback_word_mean": P["fb"]},
           "word": corr_block(Wd["y"], {"ours": Wd["ours"]}, Wd["az"], np.array(Wd["spk"])),
           "word_coverage": {"their_words": len(Wd["y"]), "no_matching_word_fallback": Wd["fb"]},
           "sentence": {k: corr_block(S[k], {"ours": S["ours"][k]}, S[f"az_{k}"], np.array(S["spk"]))
                        for k in ("accuracy", "fluency", "prosodic", "total")}}
    return out, calib


def eval_calib(calib):
    from bench.calib.train_levels import SENT_LEN, SENT_STR, WORD_LEN, WORD_STR
    from bench.calib.train_sound import LEN_THR, STR_THR

    thr = {"phone": (LEN_THR, STR_THR), "word": (WORD_LEN, WORD_STR), "sentence": (SENT_LEN, SENT_STR)}
    out = {}
    for lev, c in calib.items():
        y = np.asarray(c["y"], float)
        lt, st = thr[lev]
        zl, zs = np.asarray(c["zl"], float), np.minimum(np.asarray(c["zs"], float), np.asarray(c["zl"], float))
        tl, ts, tm = y >= lt, y >= st, y >= MID[lev]
        if lev == "word" and c.get("pmin"):       # round 2: every phone of the word acceptable too
            pm = np.asarray(c["pmin"], float)
            tl, ts, tm = tl & (pm >= 1.0), ts & (pm >= 1.8), tm & (pm >= 1.4)
        if lev == "sentence" and c.get("all_len_ok"):  # round 2: strict also needs every word lenient-ok
            ts = ts & np.asarray(c["all_len_ok"], bool)
        r = {"thresholds": {"lenient": lt, "strict": st, "midpoint": MID[lev]},
             "lenient": calib_block(tl, sig(zl)), "strict": calib_block(ts, sig(zs)),
             "s0.5_vs_midpoint": calib_block(tm, blend(zl, zs, 0.5))}
        if lev == "phone":
            zlU = np.asarray(c["zlU"], float)
            zsU = np.minimum(np.asarray(c["zsU"], float), zlU)
            r["universal_U"] = {"lenient": calib_block(y >= lt, sig(zlU)), "strict": calib_block(y >= st, sig(zsU)),
                                "s0.5_vs_midpoint": calib_block(y >= MID[lev], blend(zlU, zsU, 0.5))}
        out[lev] = r
    return out


# ----------------------------------------------------------------------------- l2arctic / epadb

def _their_entries(rec, ds):
    """Annotation phones of one utterance: list of dicts {word, arpa, ph (ours), label} in order
    (insertions dropped; 'unk' labels kept but excluded later)."""
    from bench.pipeline.labels import arpa_to_ours

    ents = []
    if ds == "l2arctic":
        last_w = 0
        for p in rec["labels"]["phones"]:
            w = p["word"] if p["word"] is not None else last_w
            last_w = w
            if p["error"] == "a":
                continue
            lab = {None: "correct", "s": "sub", "d": "del"}.get(p["error"], "unk")
            ents.append({"word": w, "arpa": p["canonical"], "ph": arpa_to_ours(p["canonical"]), "label": lab})
    else:
        for p in rec["labels"]["phones"]:
            if p["ref"] == "0":
                continue
            lab = "correct" if p["correct"] else ("del" if p["annotated"] == "0" else "sub")
            ents.append({"word": 0, "arpa": p["ref"], "ph": arpa_to_ours(p["ref"]), "label": lab})
    return ents


def _azure_phone_scores(ents, flat, ds, words):
    """Azure accuracy per annotation phone: aligned per word (l2arctic) or globally (epadb) with
    the same alignment as bench/eval_azure_so762.py; unaligned -> word accuracy (l2arctic) or the
    utterance mean (epadb); a failed recognition -> 0."""
    from bench.eval_azure_so762 import ARPA, align, expand

    def key(a):
        b = "".join(ch for ch in a if ch.isalpha()).upper()
        return ARPA.get(b, b.lower())

    n_al = 0
    out = [None] * len(ents)
    if flat is None or flat.get("status") != "Success" or not flat.get("words"):
        return [0.0] * len(ents), 0
    fw = flat["words"]
    if ds == "l2arctic":
        for j in range(len(words)):
            idx = [k for k, e in enumerate(ents) if e["word"] == j]
            if not idx:
                continue
            aw = fw[j] if j < len(fw) else None
            acc = aw["accuracy"] if aw and aw.get("accuracy") is not None else 0.0
            sc = align([key(ents[k]["arpa"]) for k in idx], expand(aw["phones"])) if aw and aw.get("phones") else [None] * len(idx)
            for k, v in zip(idx, sc):
                n_al += v is not None
                out[k] = acc if v is None else v
    else:
        hyp = [x for aw in fw for x in expand(aw.get("phones") or [])]
        sc = align([key(e["arpa"]) for e in ents], hyp) if hyp else [None] * len(ents)
        accs = [aw["accuracy"] for aw in fw if aw.get("accuracy") is not None]
        mean = float(np.mean(accs)) if accs else 0.0
        for k, v in enumerate(sc):
            n_al += v is not None
            out[k] = mean if v is None else v
    return [float(x) for x in out], n_al


def err_items(ds, api):
    from bench.data import load
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference

    flat = {}
    p = os.path.join(az_root(ds), "flat.jsonl")
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        flat[r["id"]] = r
    rows = []
    cov = {"their": 0, "matched_ours": 0, "azure_aligned": 0, "utts": 0, "utts_no_azure": 0}
    for rec in load(ds, split="test"):
        o = api.get(rec["id"])
        if o is None or "error" in o:
            continue
        ref = build_ref("en", rec["text"])
        ents = _their_entries(rec, ds)
        words = [w["text"] for w in rec["labels"]["words"]] if ds == "l2arctic" else None
        if ds == "l2arctic":
            tp = [[e["ph"] for e in ents if e["word"] == j] for j in range(len(words))]
            pos = {}
            for k, e in enumerate(ents):
                pos.setdefault(e["word"], []).append(k)
            o2t, _ = align_reference(ref, words, tp)
            t2o = {pos[j][q]: i for i, (j, q) in o2t.items() if j in pos and q < len(pos[j])}
        else:
            o2t, _ = align_reference(ref, None, [[e["ph"] for e in ents]])
            t2o = {q: i for i, (_, q) in o2t.items()}
        f = flat.get(rec["id"])
        if f is None:
            cov["utts_no_azure"] += 1
        az, n_al = _azure_phone_scores(ents, f, ds, words or rec["labels"].get("words"))
        cov["utts"] += 1
        cov["azure_aligned"] += n_al
        matched = [t2o.get(k, -1) for k in range(len(ents))]
        for k, e in enumerate(ents):
            if e["label"] == "unk":
                continue
            i = matched[k]
            cov["their"] += 1
            cov["matched_ours"] += i >= 0
            if i < 0:   # nearest aligned neighbours' mean (same for every model)
                nb = [matched[q] for q in (k - 1, k + 1) if 0 <= q < len(ents) and matched[q] >= 0]
            sc = {}
            for m, d in o["sound"].items():
                pr = blend(d["z_len"], d["z_str"], 0.5)
                sc[m] = float(1 - pr[i]) if i >= 0 else float(1 - np.mean(pr[nb])) if nb else float(1 - np.mean(pr))
            rows.append({"utt": rec["id"], "spk": rec["speaker"], "label": e["label"], "arpa": e["arpa"],
                         "ph": e["ph"], "matched": i >= 0, "azure": 100.0 - az[k], **{f"ours_{m}": v for m, v in sc.items()}})
    return rows, cov


def train_thresholds(ds, models):
    """Best-F1 threshold of 1 - P_s=0.5 on the corpus' TRAIN split (feature tables; the same
    runtime feature code, so the scores are those the API would produce)."""
    from bench.calib.common import load_table, table_columns
    from bench.metrics import best_f1
    from pronunciada.calibrate import logit

    df = load_table(ds, "train")
    t = table_columns(df)
    y = np.asarray(t["label_err"], float)
    ok = np.isfinite(y)
    out = {}
    for name, m in models.items():
        o = m.predict(t)
        s = 1 - blend(logit(o["len"]), logit(o["str"]), 0.5)
        out[name] = best_f1(y[ok].astype(int), s[ok]).threshold
    return out


def eval_errdet(api_by_ds):
    from bench.calib.common import calib_cache
    from pronunciada.calibrate import LevelModel, data_dir

    models = {"U": LevelModel(data_dir() / "sound_U.npz"), "B": LevelModel(data_dir() / "sound_B.npz"),
              "A": LevelModel(calib_cache() / "sound_A.npz")}
    for tag in ("W", "R", "C"):
        p = calib_cache() / f"sound_{tag}.npz"
        if p.exists():
            models[tag] = LevelModel(p)
    out = {}
    for ds in ("l2arctic", "epadb"):
        rows, cov = err_items(ds, api_by_ds[ds])
        thr = train_thresholds(ds, models)
        y = np.array([r["label"] != "correct" for r in rows], int)
        spk = np.array([r["spk"] for r in rows])
        names = [k[5:] for k in rows[0] if k.startswith("ours_")]
        res = {"coverage": cov, "all": {}, "by_type": {}, "by_phone": {}}
        for m in names:
            res["all"][f"ours_{m}"] = det_block(y, [r[f"ours_{m}"] for r in rows], spk, thr.get(m))
        res["all"]["azure"] = det_block(y, [r["azure"] for r in rows], spk)
        # paired speaker bootstrap of the AUC, shipped model (B) vs Azure, same items and resamples
        from bench.metrics import paired_bootstrap, roc_auc

        pb = paired_bootstrap(roc_auc, y, np.array([r["ours_B"] for r in rows], float),
                              np.array([r["azure"] for r in rows], float), clusters=spk, n_boot=N_BOOT)
        res["all"]["ours_B"]["minus_azure"] = {
            "delta": pb["delta"], "lo": pb["lo"], "hi": pb["hi"], "p_value_one_sided": pb["p_value"],
            "p_value_two_sided": min(1.0, 2 * min(pb["p_value"], 1 - pb["p_value"] + 1 / (N_BOOT + 1)))}
        for et in ("sub", "del"):
            msk = np.array([r["label"] in ("correct", et) for r in rows])
            res["by_type"][et] = {k: det_block(y[msk], np.array([r[k] for r in rows])[msk], spk[msk])
                                  for k in [f"ours_{m}" for m in ("U", "B")] + ["azure"]}
        from collections import Counter

        top = [p for p, _ in Counter(r["arpa"].rstrip("012") for r in rows).most_common(10)]
        for ph in top:
            msk = np.array([r["arpa"].rstrip("012") == ph for r in rows])
            if 0 < y[msk].sum() < msk.sum():
                res["by_phone"][ph] = {k: {"auc": det_block(y[msk], np.array([r[k] for r in rows])[msk], spk[msk])["auc"],
                                           "n": int(msk.sum()), "n_err": int(y[msk].sum())}
                                       for k in ("ours_U", "ours_B", "azure")}
        out[ds] = res
        print(ds, {k: round(v["auc"], 4) for k, v in res["all"].items()}, flush=True)
    return out


# ----------------------------------------------------------------------------- speed / int8

def eval_speed(n=100):
    from bench.data import load
    from pronunciada.scorer import Scorer

    recs = load("speechocean762", split="test")[:n]
    out = {}
    for prec in ("fp32", "int8"):
        for th in (4, 1):
            sc = Scorer("en", precision=prec, threads=th)
            sc.score(recs[0]["audio"], recs[0]["text"])          # warm-up
            tot, stages, dur = [], {}, 0.0
            for r in recs:
                sc._lp_cache.clear()                              # no posterior reuse
                t = time.perf_counter()
                res = sc.score(r["audio"], r["text"])
                tot.append((time.perf_counter() - t) * 1e3)
                dur += r["dur"]
                for k, v in res.timing_ms.items():
                    stages.setdefault(k, []).append(v)
            out[f"{prec}_t{th}"] = {"n": len(recs), "median_ms": float(np.median(tot)),
                                    "p95_ms": float(np.percentile(tot, 95)), "rtf": float(sum(tot) / 1e3 / dur),
                                    "mean_audio_s": dur / len(recs),
                                    "stages_median_ms": {k: float(np.median(v)) for k, v in stages.items()}}
            print(prec, th, out[f"{prec}_t{th}"]["median_ms"], flush=True)
    import os as _os

    from pronunciada import acoustic

    # the backbone file each precision loads (one file per precision in the model cache) and the size of
    # the download it was built from; MB = 10^6 bytes
    name = "zipa-small-crctc-500k"
    out["model_mb"] = {f: round(acoustic.download_size(name, p) / 1e6, 1) for p, f in
                       (("fp32", "model.onnx"), ("int8", "model.int8.onnx"))}
    out["model_file_mb"] = {p: round(_os.path.getsize(acoustic.backbone_model_file(name, p)) / 1e6, 1)
                            for p in ("fp32", "int8")}
    from pronunciada.calibrate import data_dir

    out["calib_kb"] = {p.name: round(p.stat().st_size / 1024, 1) for p in sorted(data_dir().glob("*"))}
    # calibrator-only latency for a ~100-phone utterance
    sc = Scorer("en", threads=4)
    long = max(recs, key=lambda r: len(r["text"]))
    text = " ".join([long["text"]] * 3)
    res = sc.score(long["audio"], text)
    c = res._pronunciada
    ts = []
    for _ in range(30):
        t = time.perf_counter()
        sc._calibrate(c["tab"], c["ref"], c["dur"])
        ts.append((time.perf_counter() - t) * 1e3)
    out["calibrators_ms"] = {"n_phones": len(c["ref"].phones), "median_ms": float(np.median(ts))}
    return out


def eval_int8(workers):
    from bench.calib.api_run import run
    from bench.data import load

    api = run(load("speechocean762", split="test"), "en", precision="int8", workers=workers,
              tag="speechocean762-test-int8")
    r, _ = eval_so762(api)
    return r


# ----------------------------------------------------------------------------- report

def f3(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def _ci(r, k="pearson"):
    return f"{f3(r[k])} [{f3(r[k + '_ci'][0])}, {f3(r[k + '_ci'][1])}]"


def _delta(r):
    d = r.get("minus_azure")
    if not d:
        return "–"
    return f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], p={d['p_value_two_sided']:.3g}"


def report(R):
    cfg = json.loads((__import__("pronunciada.calibrate", fromlist=["data_dir"]).data_dir() / "config.json").read_text())
    shipped = cfg["sound_model"].get("en", cfg["universal_sound_model"])[-1]
    L = ["# English: pronunciada vs Azure Pronunciation Assessment\n",
         "All pronunciada numbers come from the public API (`pronunciada.Scorer(\"en\").score(audio_path, text)`, "
         "fp32, default strictness 0.5) run on the held-out TEST splits; Azure numbers from its cached "
         "responses on the same clips (en-US, `bench/azure_pa.py`). Generated by `python -m bench.eval_english`. "
         "CIs: 95 % speaker-clustered bootstrap (1000 resamples); Δ = ours − Azure with a paired speaker "
         "bootstrap, p = two-sided bootstrap p-value.\n",
         f"Shipped English sound model: **{shipped}** (English-specific, feature set B: evidence + phone identity; it beat "
         "the universal model A on speaker-grouped CV of speechocean762 train). A = universal evidence-only "
         "model (what every other language uses), C = B + utterance duration / speaking rate (benchmark "
         "transparency only, never shipped). Word and sentence calibrators are universal (no word identity) "
         "and consume the universal sound model's outputs.\n"]
    so = R.get("so762")
    if so:
        L.append("## 1. speechocean762 TEST: correlation with the human scores\n")
        cov = so["phone_coverage"]
        L.append(f"Phone level is scored on THEIR {cov['their_phones']} canonical phones: {cov['matched_to_ours']} are aligned to "
                 f"one of our phones, {cov['fallback_word_mean']} unmatched ones take the mean of our phones of the "
                 "aligned word (the same fallback bench/eval_azure_so762.py uses for Azure, whose unaligned phones take "
                 "Azure's word score). Ours = the 0–2 regression head; Azure = phoneme AccuracyScore. "
                 f"Word level: their {so['word_coverage']['their_words']} words, ours = mean word-accuracy regression of the "
                 f"aligned word(s) ({so['word_coverage']['no_matching_word_fallback']} without an aligned word take the utterance mean). "
                 "Sentence: our regression heads vs Azure AccuracyScore / FluencyScore / ProsodyScore / PronScore "
                 "(a failed Azure recognition counts as 0, as in eval_azure_so762).\n")
        L.append("| level | system | Pearson [95 % CI] | Spearman [95 % CI] | Δ Pearson vs Azure [95 % CI], p |")
        L.append("|---|---|---|---|---|")
        for k, v in so["phone"].items():
            name = {"azure": "Azure", "ours_A": "ours A (universal)", "ours_B": "ours B (English)",
                    "ours_C": "ours C (+duration/rate)", "ours_U": "ours U (shipped universal)"}.get(k, k)
            if k == f"ours_{shipped}":
                name = f"**{name} — shipped**"
            L.append(f"| phone | {name} | {_ci(v)} | {_ci(v, 'spearman')} | {_delta(v)} |")
        for k, v in so["word"].items():
            L.append(f"| word accuracy | {'Azure' if k == 'azure' else '**ours — shipped**'} | {_ci(v)} | {_ci(v, 'spearman')} | {_delta(v)} |")
        for lev, blk in so["sentence"].items():
            for k, v in blk.items():
                L.append(f"| sentence {lev} | {'Azure' if k == 'azure' else '**ours — shipped**'} | {_ci(v)} | {_ci(v, 'spearman')} | {_delta(v)} |")
        L.append("")
    ca = R.get("calib")
    if ca:
        L.append("## 2. Calibration on speechocean762 TEST\n")
        L.append("ECE = expected calibration error with 15 equal-mass bins; Brier; AUC of P vs the binary truth. "
                 "Truths: lenient = human score ≥ the lenient threshold, strict = ≥ the strict threshold (phone 1.0 / 1.8 "
                 "on 0–2; word 5 / 8 on 0–10; sentence accuracy 6 / 8 on 0–10). The s = 0.5 blend is compared with the "
                 "**midpoint** truth: human score ≥ (lenient + strict) / 2 = phone ≥ 1.4, word ≥ 6.5 (i.e. ≥ 7, integer "
                 "scores), sentence accuracy ≥ 7. Phone level: our phones aligned to a human-scored phone (shipped English "
                 "model; the universal model U below). Word level: our words aligned to a human-rated word, P = the "
                 "lowest of the aligned words. Sentence: every utterance.\n")
        L.append("| level | probability | truth | n | positive rate | mean P | ECE | Brier | AUC |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for lev, r in ca.items():
            th = r["thresholds"]
            for k, lab in (("lenient", f"≥ {th['lenient']}"), ("strict", f"≥ {th['strict']}"),
                           ("s0.5_vs_midpoint", f"≥ {th['midpoint']}")):
                b = r[k]
                L.append(f"| {lev} | {'P(' + k + ' ok)' if k != 's0.5_vs_midpoint' else 'P at s = 0.5'} | {lab} | {b['n']} | "
                         f"{b['pos_rate']:.3f} | {b['mean_p']:.3f} | {b['ece']:.4f} | {b['brier']:.4f} | {f3(b['auc'])} |")
            if "universal_U" in r:
                for k, lab in (("lenient", f"≥ {th['lenient']}"), ("strict", f"≥ {th['strict']}"),
                               ("s0.5_vs_midpoint", f"≥ {th['midpoint']}")):
                    b = r["universal_U"][k]
                    L.append(f"| phone (U) | {'P(' + k + ' ok)' if k != 's0.5_vs_midpoint' else 'P at s = 0.5'} | {lab} | {b['n']} | "
                             f"{b['pos_rate']:.3f} | {b['mean_p']:.3f} | {b['ece']:.4f} | {b['brier']:.4f} | {f3(b['auc'])} |")
        L.append("")
        L.append("Reliability (equal-mass bins, mean P → observed rate), s = 0.5 vs midpoint truth:\n")
        for lev, r in ca.items():
            rows = r["s0.5_vs_midpoint"]["reliability"]
            L.append(f"- **{lev}**: " + ", ".join(f"{x['mean_p']:.2f}→{x['frac_pos']:.2f}" for x in rows))
        L.append("")
        L.append("Reliability, P(strict ok):\n")
        for lev, r in ca.items():
            rows = r["strict"]["reliability"]
            L.append(f"- **{lev}**: " + ", ".join(f"{x['mean_p']:.2f}→{x['frac_pos']:.2f}" for x in rows))
        L.append("")
    ed = R.get("errdet")
    if ed:
        L.append("## 3. Real learner errors: l2arctic TEST and epadb TEST\n")
        L.append("Items = the annotators' canonical phones (insertions and unparsed labels excluded); error = substituted "
                 "or deleted. Ours: score = 1 − P at strictness 0.5; unaligned annotation phones take the mean of the "
                 "neighbouring aligned phones. Azure: score = 100 − phoneme AccuracyScore, aligned to the annotation's "
                 "canonical phones with the speechocean762 alignment (per word for l2arctic, over the utterance for epadb, "
                 "which has no word grouping); unaligned phones take the word (l2arctic) / utterance-mean (epadb) "
                 "accuracy. \"F1 @ train thr\" uses the best-F1 threshold of the same model on the corpus' TRAIN split "
                 "(no Azure responses exist for the train splits, so Azure has none). Models: U = shipped universal, "
                 "A = universal on speechocean762 only, B = shipped English, C = B + duration/rate, W = universal on "
                 "the wide mix (so762 + l2arctic + epadb + FLEURS synthetic), R = so762 + l2arctic + epadb. W and R "
                 "contain the l2arctic / epadb TRAIN splits; no model saw a test speaker.\n")
        for ds, r in ed.items():
            c = r["coverage"]
            L.append(f"**{ds}** — {c['utts']} utts, {c['their']} annotated phones; aligned to ours "
                     f"{c['matched_ours'] / c['their']:.1%}; aligned to an Azure phone {c['azure_aligned'] / max(c['their'], 1):.1%} "
                     f"(utterances without an Azure response: {c['utts_no_azure']})\n")
            L.append("| system | n | errors | ROC AUC [95 % CI] | PR AUC | best F1 | F1 @ train thr |")
            L.append("|---|---|---|---|---|---|---|")
            for k, v in r["all"].items():
                L.append(f"| {k} | {v['n']} | {v['n_err']} | {f3(v['auc'])} [{f3(v['auc_ci'][0])}, {f3(v['auc_ci'][1])}] | "
                         f"{f3(v['pr_auc'])} | {f3(v['best_f1'])} | {f3(v.get('f1_at_train_thr'))} |")
            L.append("")
            L.append("| error type | " + " | ".join(r["by_type"]["sub"].keys()) + " |")
            L.append("|---|" + "---|" * len(r["by_type"]["sub"]))
            for et, blk in r["by_type"].items():
                L.append(f"| {et} vs correct (n err) | " + " | ".join(f"{f3(v['auc'])} ({v['n_err']})" for v in blk.values()) + " |")
            L.append("")
            L.append("| phone (n / errors) | ours U | ours B | Azure |\n|---|---|---|---|")
            for ph, blk in r["by_phone"].items():
                any_ = next(iter(blk.values()))
                L.append(f"| {ph} ({any_['n']} / {any_['n_err']}) | " + " | ".join(f3(blk[k]["auc"]) for k in ("ours_U", "ours_B", "azure")) + " |")
            L.append("")
    sp = R.get("speed")
    if sp:
        L.append("## 4. Speed (this machine, shared with other jobs)\n")
        L.append(f"First 100 speechocean762 test clips (mean {sp['fp32_t4']['mean_audio_s']:.2f} s of audio), one process, "
                 "end-to-end `Scorer.score(path, text)` incl. audio decoding; posterior cache cleared per call.\n")
        L.append("| precision | threads | median ms | p95 ms | real-time factor | load | g2p | fbank | model | adapter | align | features | calibrate |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in ("fp32_t4", "fp32_t1", "int8_t4", "int8_t1"):
            v = sp[k]
            st = v["stages_median_ms"]
            L.append(f"| {k.split('_')[0]} | {k.split('_t')[1]} | {v['median_ms']:.1f} | {v['p95_ms']:.1f} | {v['rtf']:.4f} | "
                     + " | ".join(f"{st.get(x, 0):.1f}" for x in ("load", "g2p", "fbank", "model", "adapter", "align", "features", "calibrate")) + " |")
        L.append("")
        L.append(f"Calibrators alone (sound, word, sentence; English = shipped B + universal U): "
                 f"{sp['calibrators_ms']['median_ms']:.2f} ms for a {sp['calibrators_ms']['n_phones']}-phone utterance. "
                 f"Model files: {sp['model_mb']}; calibrator files (KB): {sp['calib_kb']}.\n")
    i8 = R.get("int8")
    if i8 and so:
        L.append("## 5. Does int8 change the benchmark? (speechocean762 TEST, same protocol)\n")
        L.append("| level | fp32 Pearson | int8 Pearson |\n|---|---|---|")
        L.append(f"| phone (shipped {shipped}) | {f3(so['phone'][f'ours_{shipped}']['pearson'])} | {f3(i8['phone'][f'ours_{shipped}']['pearson'])} |")
        L.append(f"| phone (U) | {f3(so['phone']['ours_U']['pearson'])} | {f3(i8['phone']['ours_U']['pearson'])} |")
        L.append(f"| word accuracy | {f3(so['word']['ours']['pearson'])} | {f3(i8['word']['ours']['pearson'])} |")
        for k in ("accuracy", "fluency", "prosodic", "total"):
            L.append(f"| sentence {k} | {f3(so['sentence'][k]['ours']['pearson'])} | {f3(i8['sentence'][k]['ours']['pearson'])} |")
        L.append("")
    from bench.pipeline.common import repo_root

    r1p = repo_root() / "results" / "tables" / "round1" / "english_vs_azure.json"
    if r1p.exists() and so:
        R1 = json.loads(r1p.read_text())
        L.append("## 6. Rounds 2 and 3 (word / sentence redesign, recalibration) — before → after on speechocean762 TEST\n")
        L.append("Round 2 was designed AFTER the first held-out evaluation (the test results motivated it); all "
                 "fitting and selection used train splits and FLEURS validation only (results/calibration.md §10). "
                 "Phone-level models are unchanged. Pearson [95 % CI]; Δ vs Azure with paired-bootstrap p.\n")
        r2p = repo_root() / "results" / "tables" / "round2" / "english_vs_azure.json"
        R2 = json.loads(r2p.read_text()) if r2p.exists() else None
        L.append("Columns: round 1, round 2, round 3 (= current: round-2 models + re-fitted word calibration maps, "
                 "explain-away pass, prosody channel; designed after the earlier held-out evaluations).\n")
        L.append("| level | round 1 ours | round 2 ours | round 3 ours | Azure | round 3 Δ vs Azure, p |\n|---|---|---|---|---|---|")
        r2w = _ci(R2['so762']['word']['ours']) if R2 else '–'
        L.append(f"| word accuracy | {_ci(R1['so762']['word']['ours'])} | {r2w} | {_ci(so['word']['ours'])} | "
                 f"{_ci(so['word']['azure'])} | {_delta(so['word']['ours'])} |")
        for k in ("accuracy", "fluency", "prosodic", "total"):
            a1, a3 = R1["so762"]["sentence"][k]["ours"], so["sentence"][k]["ours"]
            a2 = _ci(R2["so762"]["sentence"][k]["ours"]) if R2 else "–"
            L.append(f"| sentence {k} | {_ci(a1)} | {a2} | {_ci(a3)} | {_ci(so['sentence'][k]['azure'])} | {_delta(a3)} |")
        L.append("")
        c1 = R1.get("calib", {})
        if c1 and ca:
            L.append("Calibration (ECE, 15 equal-mass bins). Round-1 word truths: accuracy ≥ 5 / ≥ 8; round-2 word "
                     "truths: accuracy ≥ 5 AND every phone ≥ 1.0 (lenient), accuracy ≥ 8 AND every phone ≥ 1.8 (strict), "
                     "midpoint 6.5 / 1.4; round-2 sentence strict truth: accuracy ≥ 8 AND every word lenient-ok.\n")
            c2 = R2.get("calib", {}) if R2 else {}
            L.append("| level | head | round 1 ECE | round 2 ECE | round 3 ECE | round 3 mean P / observed | round 3 AUC |\n|---|---|---|---|---|---|---|")
            for lev in ("word", "sentence"):
                for h in ("lenient", "strict", "s0.5_vs_midpoint"):
                    e2 = f"{c2[lev][h]['ece']:.4f}" if c2 else "–"
                    L.append(f"| {lev} | {h} | {c1[lev][h]['ece']:.4f} | {e2} | {ca[lev][h]['ece']:.4f} | "
                             f"{ca[lev][h]['mean_p']:.3f} / {ca[lev][h]['pos_rate']:.3f} | {f3(ca[lev][h]['auc'])} |")
            L.append("")
    rxp = repo_root() / "results" / "tables" / "round3_explain" / "english_vs_azure.json"
    RX = json.loads(rxp.read_text()) if rxp.exists() else R
    if RX.get("so762_no_explain") and RX.get("errdet_no_explain") and so:
        sx, ex, en_, so_x = RX["so762_no_explain"], RX["errdet_no_explain"], RX["errdet"], RX["so762"]
        L.append("## 8. Round 3: explain-away second pass — without → with (held-out)\n")
        L.append("Round 3 was designed AFTER the earlier held-out evaluations; its settings were selected on "
                 "FLEURS validation only (results/calibration.md §11). Same API run; \"without\" = the first-pass "
                 "values kept inside each result. The sentence level always uses the first pass. **Result: the pass "
                 "lowers the speechocean762 phone and word correlations, so it is disabled by default** (opt-in "
                 "`Scorer(..., explain_away=True)`); all other tables in this file are from the final run without it.\n")
        L.append("| item | without | with |\n|---|---|---|")
        for m in ("B", "A", "C"):
            L.append(f"| so762 phone Pearson, {m} | {f3(sx['phone'][f'ours_{m}']['pearson'])} | {f3(so_x['phone'][f'ours_{m}']['pearson'])} |")
        L.append(f"| so762 word-accuracy Pearson | {f3(sx['word']['ours']['pearson'])} | {f3(so_x['word']['ours']['pearson'])} |")
        for ds in ("l2arctic", "epadb"):
            for m in ("B", "U"):
                L.append(f"| {ds} ROC AUC, {m} | {f3(ex[ds]['all'][f'ours_{m}']['auc'])} | {f3(en_[ds]['all'][f'ours_{m}']['auc'])} |")
        L.append("")
    lj = repo_root() / "results" / "tables" / "languages.json"
    az_md = repo_root() / "results" / "azure_swap.md"
    if lj.exists() and az_md.exists():
        import re as _re

        langs = json.loads(lj.read_text()).get("by_lang", {})
        azl = {}
        for line in az_md.read_text().splitlines():
            m = _re.match(r"\| ([a-z]{2})-[A-Z]{2} \| (\d+) \| ([\d.]+) \| ([\d.]+) \|", line)
            if m:
                azl[m.group(1)] = (float(m.group(3)), float(m.group(4)))
        L.append("## 7. Latency next to Azure (FLEURS test clips, ~9–14 s of audio)\n")
        L.append("Ours: end-to-end `Scorer.score` on this Mac, 1 ONNX thread per process, 5 processes in parallel, "
                 "median over the FLEURS test clips (languages.md). Azure: wall-clock of the REST request from this Mac "
                 "to westus incl. upload / network, 4 requests in parallel, median over the fleurs-swap clips "
                 "(results/azure_swap.md). Different clip sets of similar length; not a controlled comparison.\n")
        L.append("| lang | ours median ms (median audio s) | ours RTF | Azure median s | Azure p90 s |\n|---|---|---|---|---|")
        for lang, r in langs.items():
            a = azl.get(lang)
            L.append(f"| {lang} | {r['latency']['median_ms']:.0f} ({r['latency']['median_audio_s']:.1f}) | {r['latency']['rtf']:.3f} | "
                     f"{a[0] if a else '–'} | {a[1] if a else '–'} |")
        L.append("")

    (repo_root() / "results" / "english_vs_azure.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote results/english_vs_azure.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="run,so762,calib,errdet,speed,int8,report")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    steps = a.steps.split(",")
    R = load_json("english_vs_azure.json") or {}
    api = run_api(a.workers) if any(s in steps for s in ("run", "so762", "calib", "errdet")) else None
    if "so762" in steps or "calib" in steps:
        so, calib = eval_so762(api["speechocean762"])
        R["so762"] = so
        R["calib"] = eval_calib(calib)
        R["so762_no_explain"], _ = eval_so762(without_explain(api["speechocean762"]))
        save_json("english_vs_azure.json", R)
        print("so762", {k: round(v["pearson"], 4) for k, v in so["phone"].items()}, flush=True)
    if "errdet" in steps:
        R["errdet"] = eval_errdet(api)
        R["errdet_no_explain"] = eval_errdet({k: without_explain(v) for k, v in api.items()})
        save_json("english_vs_azure.json", R)
    if "speed" in steps:
        R["speed"] = eval_speed()
        save_json("english_vs_azure.json", R)
    if "int8" in steps:
        R["int8"] = eval_int8(a.workers)
        save_json("english_vs_azure.json", R)
    if "report" in steps:
        report(R)


if __name__ == "__main__":
    main()
