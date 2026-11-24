"""Candidate minus baseline, row by row, on the same held-out items: the scoreboard's own paired tests
with the baseline's outputs in the place of Azure's.

Inputs are two finished suite runs (bench/opt/suite.py); each run's per-clip outputs are read from
its cache overlay. For every row, the same item set, label definition, resampling unit, number of
resamples and seed as the script that produced the scoreboard row:

  speechocean762 (6)   bench.eval_english.so762_items; paired speaker bootstrap of Pearson,
                       1,000 resamples, eval_english's two-sided p
  learner errors (2)   bench.eval_english.err_items; paired speaker bootstrap of ROC AUC
  coaching (4)         bench.coach.azure_yardstick items; speaker bootstrap, seed 0, 1,000
  OMPAL (6)            bench.eval_ompal unit / sentence tables; speaker bootstrap (seeds 0 / 3)
  swap benchmark, tone swaps (53)
                       bench.eval_swap.paired (utterance bootstrap, 1,000, seed 0), restricted to
                       the records Azure scored, as the scoreboard's rows are
  cv-real (13)         bench.eval_cv.speaker_bootstrap on the clips both runs and Azure scored
  contrast audit (240) clip bootstrap (natives and their edits together; 5 % thresholds recomputed
                       from each resample's natives), 2,000 resamples, seed 0 (as
                       bench/round19_merge/audit_paired.py, extended to sound hit@5 %)
  length (10)          the same clip bootstrap on the per-clip dump of bench.length.heldout_runtime
  coaching tips (26)   clip bootstrap (utterances for the learner rows) on the per-clip dump of
                       bench.coach.tips_r19, 2,000 resamples, seed 0
No paired test (value rules only, bench/opt/gate.py): natives accepted (30) and planted-error AUC
(20) of languages.json, the length native word acceptance, and the five coaching figures quoted in
running text. Every recomputed point value is checked against the run's own table; a row whose
recomputation does not match is reported and treated as having no paired test.
Returns {row key: {"value_base", "value_cand", "delta", "ci", "p", "n", "test"}} with
delta = candidate - baseline as measured (for lower-is-better rows a positive delta is worse).
"""
from __future__ import annotations

import contextlib
import glob
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")


@contextlib.contextmanager
def cache_of(run: Path):
    old = os.environ.get("PRONUNCIADA_CACHE")
    os.environ["PRONUNCIADA_CACHE"] = str(Path(run) / "cache")
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PRONUNCIADA_CACHE", None)
        else:
            os.environ["PRONUNCIADA_CACHE"] = old


def _p2(d) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def _q(d) -> list[float]:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))] if len(d) else [float("nan")] * 2


def _res(vb, vc, delta, ci, p, n, test):
    return {"value_base": float(vb), "value_cand": float(vc), "delta": float(delta), "ci": [float(x) for x in ci],
            "p": float(p), "n": int(n), "test": test}


def _api(run: Path, tag: str) -> dict:
    return pickle.loads((Path(run) / "cache" / "calib" / "api" / f"{tag}.pkl").read_bytes())


# ----------------------------------------------------------------------------- speechocean762 / errdet

def so762(base: Path, cand: Path) -> dict:
    from bench.eval_english import N_BOOT, so762_items
    from bench.metrics import paired_bootstrap, pearson

    with cache_of(base):
        Pb, Wb, Sb, _, _ = so762_items(_api(base, "speechocean762-test-fp32"))
    with cache_of(cand):
        Pc, Wc, Sc, _, _ = so762_items(_api(cand, "speechocean762-test-fp32"))
    if Pb["y"] != Pc["y"] or Wb["y"] != Wc["y"] or Sb["spk"] != Sc["spk"]:
        raise RuntimeError("speechocean762 items differ between the runs")
    out = {}

    def pb(y, c, b, spk, name):
        r = paired_bootstrap(lambda yy, ss: pearson(ss, yy), np.asarray(y, float), np.asarray(c, float),
                             np.asarray(b, float), clusters=np.asarray(spk), n_boot=N_BOOT)
        p2 = min(1.0, 2 * min(r["p_value"], 1 - r["p_value"] + 1 / (N_BOOT + 1)))
        out[f"speechocean762|en|{name}"] = _res(pearson(b, y), pearson(c, y), r["delta"], [r["lo"], r["hi"]], p2,
                                                len(y), "paired speaker bootstrap, Pearson")

    pb(Pb["y"], Pc["B"], Pb["B"], Pb["spk"], "individual sounds")
    pb(Wb["y"], Wc["ours"], Wb["ours"], Wb["spk"], "words")
    for k in ("accuracy", "fluency", "prosodic", "total"):
        pb(Sb[k], Sc["ours"][k], Sb["ours"][k], Sb["spk"], f"sentence {k}" + (" (intonation)" if k == "prosodic" else ""))
    return out


def errdet(base: Path, cand: Path) -> dict:
    from bench.eval_english import N_BOOT, err_items
    from bench.metrics import paired_bootstrap, roc_auc

    out = {}
    for ds, label in (("l2arctic", "L2-ARCTIC test"), ("epadb", "EpaDB test")):
        with cache_of(base):
            rb, _ = err_items(ds, _api(base, f"{ds}-test-fp32"))
        with cache_of(cand):
            rc, _ = err_items(ds, _api(cand, f"{ds}-test-fp32"))
        kb = [(r["utt"], r["ph"], r["label"]) for r in rb]
        if kb != [(r["utt"], r["ph"], r["label"]) for r in rc]:
            raise RuntimeError(f"{ds}: items differ between the runs")
        y = np.array([r["label"] != "correct" for r in rb], int)
        spk = np.array([r["spk"] for r in rb])
        b = np.array([r["ours_B"] for r in rb], float)
        c = np.array([r["ours_B"] for r in rc], float)
        r = paired_bootstrap(roc_auc, y, c, b, clusters=spk, n_boot=N_BOOT)
        p2 = min(1.0, 2 * min(r["p_value"], 1 - r["p_value"] + 1 / (N_BOOT + 1)))
        out[f"learner errors|en|{label}"] = _res(roc_auc(y, b), roc_auc(y, c), r["delta"], [r["lo"], r["hi"]], p2,
                                                 len(y), "paired speaker bootstrap, ROC AUC")
    return out


# ----------------------------------------------------------------------------- coaching yardstick

def coaching(base: Path, cand: Path) -> dict:
    from sklearn.metrics import average_precision_score

    from bench.coach.azure_yardstick import N_BOOT, azure_phones
    from bench.coach.fit_v2 import so762_rows
    from pronunciada import coach as Co

    mdl = Co.model()
    sens = ("careful", "balanced", "eager")
    S = {}
    for tag, run in (("b", base), ("c", cand)):
        with cache_of(run):
            AZP = azure_phones(os.path.join(os.environ["PRONUNCIADA_CACHE"], "azure", "speechocean762"))
            R = so762_rows("so762-test", "test")
        ep = Co.error_prob(R, mdl, "en")
        keep = [i for i in range(len(ep)) if np.isfinite(R["acc"][i]) and (R["utt"][i], R["tj"][i], R["tk"][i]) in AZP]
        S[tag] = {"key": [(R["utt"][i], R["tj"][i], R["tk"][i]) for i in keep], "y": (R["acc"][keep] < 2).astype(float),
                  "s": ep[keep], "spk": np.asarray(R["spk"])[keep],
                  "masks": {s: Co.sound_mask(R, ep, mdl, "en", s)[keep] for s in sens}}
    if S["b"]["key"] != S["c"]["key"]:
        raise RuntimeError("coaching items differ between the runs")
    y, spk = S["b"]["y"], S["b"]["spk"]
    sb, sc = S["b"]["s"], S["c"]["s"]
    mb, mc = S["b"]["masks"], S["c"]["masks"]
    rng = np.random.default_rng(0)
    us = np.unique(spk)
    idx = {u: np.flatnonzero(spk == u) for u in us}
    D = defaultdict(list)
    for _ in range(N_BOOT):
        b = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        D["ap"].append(average_precision_score(y[b], sc[b]) - average_precision_score(y[b], sb[b]))
        for s in sens:
            D[s].append(float(y[b][mc[s][b]].mean()) - float(y[b][mb[s][b]].mean()))
    out = {}
    for s in sens:
        vb, vc = float(y[mb[s]].mean()), float(y[mc[s]].mean())
        out[f"coaching|en|precision at '{s}' recall"] = _res(vb, vc, vc - vb, _q(D[s]), _p2(D[s]), len(y),
                                                             "paired speaker bootstrap, precision at the sensitivity")
    vb, vc = float(average_precision_score(y, sb)), float(average_precision_score(y, sc))
    out["coaching|en|precision-recall AUC"] = _res(vb, vc, vc - vb, _q(D["ap"]), _p2(D["ap"]), len(y),
                                                   "paired speaker bootstrap, average precision")
    return out


# ----------------------------------------------------------------------------- OMPAL

def ompal(base: Path, cand: Path) -> dict:
    from bench import eval_ompal as EO

    recs = EO.records("test")
    U, S = {}, {}
    for tag, run in (("b", base), ("c", cand)):
        with cache_of(run):
            ours, _ = EO.ours_outputs(recs)
        U[tag] = EO.unit_table(recs, ours, {})["cols"]
        S[tag] = EO.sent_table(recs, ours, {})
    if list(U["b"]["utt"]) != list(U["c"]["utt"]) or list(S["b"]["utt"]) != list(S["c"]["utt"]):
        raise RuntimeError("OMPAL items differ between the runs")
    out = {}
    c = U["b"]
    n = len(c["utt"])
    boots = {"speaker": EO.boot_indices(c["spk"], seed=0)}
    neg = c["correct"].astype(bool)
    for name, lab in (("word: any error", "any_wrong"), ("word: tone-only error", "tone_only"),
                      ("word: consonant / vowel error", "seg_wrong")):
        pos = c[lab].astype(bool)
        st = EO.stat_with_ci(EO.auc_fn(pos, neg, U["c"]["ours_word"]), boots, n, EO.auc_fn(pos, neg, U["b"]["ours_word"]))
        out[f"OMPAL|zh|{name}"] = _res(st["azure"], st["value"], st["delta"], st["delta_ci_speaker"], st["p_speaker"],
                                       n, "paired speaker bootstrap, ROC AUC (eval_ompal)")
    keep = ~S["b"]["native"].astype(bool)
    cb = {k: v[keep] for k, v in S["b"].items()}
    cc = {k: v[keep] for k, v in S["c"].items()}
    nb = len(cb["utt"])
    sb = {"speaker": EO.boot_indices(cb["spk"], seed=3)}
    names = {"accuracy": "sentence accuracy", "fluency": "sentence fluency", "prosody": "sentence prosody"}
    for row, tk, ok, _ak, _note in EO.SENT_PAIRS:
        if row not in names:
            continue
        yv = cb[f"t_{tk}"].astype(float)
        st = EO.stat_with_ci(EO.corr_fn(yv, cc[f"ours_{ok}"].astype(float), "pearson"), sb, nb,
                             EO.corr_fn(yv, cb[f"ours_{ok}"].astype(float), "pearson"))
        out[f"OMPAL|zh|{names[row]}"] = _res(st["azure"], st["value"], st["delta"], st["delta_ci_speaker"],
                                             st["p_speaker"], nb, "paired speaker bootstrap, Pearson (eval_ompal)")
    return out


# ----------------------------------------------------------------------------- swap benchmark

SWAP_ROWS = {"swap_far:auc": "far AUC", "swap_near:auc": "near AUC", "swap_near:hit@5%FA": "near hit@5 %",
             "swap_far:right_word": "far right word", "native:fa<60": "native false alarms"}
TONE_ROWS = {"tone:auc": "detection (AUC)", "tone:hit@5%FA": "caught at 5 % native false alarms",
             "tone:right_word": "right word blamed"}


def swap(base: Path, cand: Path) -> dict:
    from bench.data import load
    from bench.eval_swap import azure_outputs, collect, paired

    with cache_of(base):
        recs = load("fleurs-swap", split="test")
        az = azure_outputs(recs)
    O = {}
    for tag, run in (("b", base), ("c", cand)):
        o = {}
        for lang in LANGS:
            p = Path(run) / "cache" / "calib" / "api" / f"fleurs-swap-{lang}.pkl"
            if p.exists():
                o.update({k: v["on"] for k, v in pickle.loads(p.read_bytes()).items()})
        O[tag] = o
    # the scoreboard's set: clusters and variants Azure scored as well (bench.eval_swap.paired keeps,
    # per source clip, the variants every system scored)
    Z = collect(recs, az)
    allowed = {(lang, src, v) for lang, L in Z.items() for src, vs in L["clusters"].items() for v in vs}
    keep_ids = {r["id"] for r in recs
                if (r["lang"], r["labels"]["src_id"], r["labels"]["variant"]) in allowed}
    ob ={k: v for k, v in O["b"].items() if k in keep_ids}
    oc = {k: v for k, v in O["c"].items() if k in keep_ids}
    P = paired(recs, oc, ob)
    out = {}
    for lang, m in P.items():
        for k, x in m.items():
            if not isinstance(x, dict):
                continue
            if k in SWAP_ROWS:
                key = f"swap benchmark|{lang}|{SWAP_ROWS[k]}"
            elif k in TONE_ROWS and lang == "zh":
                key = f"tone swaps|zh|{TONE_ROWS[k]}"
            else:
                continue
            out[key] = _res(x["b"], x["a"], x["delta"], [x["lo"], x["hi"]], x["p_two_sided"], m["n_clusters"],
                            "paired utterance bootstrap (bench.eval_swap.paired)")
    return out


# ----------------------------------------------------------------------------- cv-real

def cvreal(base: Path, cand: Path) -> dict:
    from bench import eval_cv as EC

    recs = EC.records()
    with cache_of(base):
        az = EC.azure_scores(recs)
        ob, _ = EC.ours_scores(recs)
    with cache_of(cand):
        oc, _ = EC.ours_scores(recs)
    groups = defaultdict(list)
    for r in recs:
        groups[(r["locale"], r["test"])].append(r)
    out = {}
    for (loc, test), rs in groups.items():
        if test == "Aweak":
            continue
        have = [r for r in rs if r["id"] in ob and r["id"] in oc and r["id"] in az]
        y = np.array([r["label"] for r in have])
        if not len(have) or y.min() == y.max():
            continue
        spk = [r["client_id"] for r in have]
        a = np.array([oc[r["id"]][EC.OURS_PRIMARY] for r in have])
        b = np.array([ob[r["id"]][EC.OURS_PRIMARY] for r in have])
        pb = EC.speaker_bootstrap(y, a, b, spk)
        k = f"{loc}/{test}"
        lang = loc.split("-")[0]
        out[f"cv-real test {test}|{lang}|{k}"] = _res(pb["azure"], pb["ours"], pb["delta"], pb["delta_ci"],
                                                     pb["p_two_sided"], len(have), "paired speaker bootstrap, ROC AUC")
    return out


# ----------------------------------------------------------------------------- clip bootstraps
# Vectorised: a resample is a vector of per-clip counts m (np.bincount of the drawn clip indices);
# a statistic over the resample is a ratio of count-weighted sums, and a percentile threshold is
# taken over the natives repeated by their clip's count (= np.percentile of the concatenation).

def _counts(n_clips: int, n_boot: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_clips, n_clips), minlength=n_clips) for _ in range(n_boot)])


def _ratio_boot(num: dict, den: dict, M: np.ndarray):
    """num / den: {"b": [n_clips], "c": [n_clips]} per-clip sums. Full-sample values and deltas."""
    v = {t: float(num[t].sum() / den[t].sum()) if den[t].sum() else np.nan for t in ("b", "c")}
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = (M @ num["b"]) / (M @ den["b"])
        rc = (M @ num["c"]) / (M @ den["c"])
    return v["b"], v["c"], rc - rb


def _pct_rep(vals: np.ndarray, clip_of: np.ndarray, m: np.ndarray, q: float, method: str = "linear") -> float:
    w = m[clip_of]
    x = np.repeat(vals, w)
    return float(np.percentile(x, q, method=method)) if len(x) else np.nan


def _audit_load(run: Path, lang: str):
    clips, edits = {}, []
    for f in sorted(glob.glob(str(Path(run) / "cache" / "audit" / "contrast_v1_test" / lang / "chunk_250_*.pkl"))):
        r = pickle.loads(Path(f).read_bytes())
        for c in r["clips"]:
            clips[c["id"]] = c
        edits += [e for e in r["edits"] if "error" not in e]
    return clips, edits


def contrast_audit(base: Path, cand: Path, n_boot: int = 2000) -> dict:
    from bench.audit.contrasts import contrasts_for

    out = {}
    for lang in LANGS:
        cb, eb = _audit_load(base, lang)
        cc, ec = _audit_load(cand, lang)
        if not cb or not cc:
            continue

        def keyed(E):
            occ, d = defaultdict(int), {}
            for e in E:
                k0 = (e["id"], e["contrast"], e["direction"], e["src"], e["dst"])
                d[k0 + (occ[k0],)] = e
                occ[k0] += 1
            return d

        KB, KC = keyed(eb), keyed(ec)
        ids = sorted(set(cb) & set(cc))
        pos = {i: j for j, i in enumerate(ids)}
        names = {}
        for c in contrasts_for(lang):
            a, b = c.members
            names[(c.name, "A>B")] = f"said {a}, expected {b}"
            names[(c.name, "B>A")] = f"said {b}, expected {a}"
        common = [k for k in sorted(set(KB) & set(KC), key=str) if k[0] in pos and (k[1], k[2]) in names
                  and not (lang == "ja" and k[1] in ("vowel_length", "gemination"))]
        rows = sorted({(k[1], k[2]) for k in common})
        rid = {r: j for j, r in enumerate(rows)}
        e_row = np.array([rid[(k[1], k[2])] for k in common])
        e_clip = np.array([pos[k[0]] for k in common])
        e_tone = np.array([KB[k].get("kind") == "tone" for k in common])
        M = _counts(len(ids), n_boot)
        full = np.ones(len(ids), int)
        res = {}
        for t, C, K in (("b", cb, KB), ("c", cc, KC)):
            nat = {f: (np.concatenate([np.asarray(C[i][f], float) for i in ids]),
                       np.concatenate([np.full(len(C[i][f]), j) for j, i in enumerate(ids)]))
                   for f in ("prob", "word_prob")}
            tv = [(v, j) for j, i in enumerate(ids) for v in C[i]["tone_prob"] if v is not None]
            nat["tone"] = (np.array([v for v, _ in tv], float), np.array([j for _, j in tv], int))
            p = np.array([K[k]["prob"] for k in common], float)
            w = np.array([K[k]["word_prob"] for k in common], float)

            def hit(m):
                ts = _pct_rep(*nat["prob"], m, 5)
                tw = _pct_rep(*nat["word_prob"], m, 5)
                tt = _pct_rep(*nat["tone"], m, 5) if len(nat["tone"][0]) else np.nan
                thr = np.where(e_tone, tt, ts)
                wt = m[e_clip].astype(float)
                den = np.bincount(e_row, weights=wt, minlength=len(rows))
                with np.errstate(invalid="ignore", divide="ignore"):
                    hs = np.bincount(e_row, weights=wt * (p <= thr), minlength=len(rows)) / den
                    hw = np.bincount(e_row, weights=wt * (w <= tw), minlength=len(rows)) / den
                return hs, hw

            res[t] = (hit(full), np.array([np.stack(hit(m)) for m in M]))     # boots: [n_boot, 2, rows]
        (hsb, hwb), Bb = res["b"]
        (hsc, hwc), Bc = res["c"]
        D = Bc - Bb
        n_per = np.bincount(e_row, minlength=len(rows))
        for (cn, dr), j in rid.items():
            for lvl, vb, vc, li in (("sound_hit5", hsb[j], hsc[j], 0), ("word_hit5", hwb[j], hwc[j], 1)):
                d = D[:, li, j]
                out[f"contrast audit|{lang}|{cn} | {names[(cn, dr)]} | {lvl}"] = _res(
                    vb, vc, vc - vb, _q(d), _p2(d), int(n_per[j]),
                    "paired clip bootstrap (natives and edits together), 2,000, seed 0")
    return out


def length(base: Path, cand: Path, n_boot: int = 2000) -> dict:
    R = {}
    for tag, run in (("b", base), ("c", cand)):
        p = Path(run) / "cache" / "opt_dumps" / "length_runtime_ja.pkl"
        if not p.exists():
            return {}
        R[tag] = pickle.loads(p.read_bytes())
    ids = sorted(i for i in set(R["b"]) & set(R["c"]) if R["b"][i] is not None and R["c"][i] is not None)
    keys = sorted({(e["contrast"], e["direction"]) for i in ids for e in R["b"][i]["edits"]})
    M = _counts(len(ids), n_boot)
    full = np.ones(len(ids), int)
    vals = {}
    for t in ("b", "c"):
        rows = [R[t][i] for i in ids]
        nl = (np.array([x for r in rows for x in r["nat_len"]], float),
              np.array([j for j, r in enumerate(rows) for _ in r["nat_len"]], int))
        nw = (np.array([x for r in rows for x in r["nat_word"]], float),
              np.array([j for j, r in enumerate(rows) for _ in r["nat_word"]], int))
        E = [(j, e) for j, r in enumerate(rows) for e in r["edits"]]
        ek = np.array([keys.index((e["contrast"], e["direction"])) for _, e in E])
        ec = np.array([j for j, _ in E])
        ln = np.array([np.nan if e["len"] is None else e["len"] for _, e in E], float)
        wd = np.array([e["word"] for _, e in E], float)

        def stat(m):
            t5 = np.quantile(np.repeat(nl[0], m[nl[1]]), 0.05)
            tw5 = np.quantile(np.repeat(nw[0], m[nw[1]]), 0.05)
            wt = m[ec].astype(float)
            okl = np.isfinite(ln)
            out = []
            for k in range(len(keys) + 1):              # last = "all"
                sel = np.ones(len(ek), bool) if k == len(keys) else ek == k
                dl, dw = (wt * (sel & okl)).sum(), (wt * sel).sum()
                out.append((wt * (sel & okl) * (np.nan_to_num(ln, nan=np.inf) <= t5)).sum() / dl if dl else np.nan)
                out.append((wt * sel * (wd <= tw5)).sum() / dw if dw else np.nan)
            natw_w = m[nw[1]].astype(float)
            out.append(float((natw_w * (nw[0] >= 0.5)).sum() / natw_w.sum()))
            return np.array(out)

        vals[t] = (stat(full), np.array([stat(m) for m in M]))
    names = []
    for c, d in keys + [("all", "")]:
        k = c if c == "all" else f"{c}:{d}"
        names += [f"{k} sound caught", f"{k} word caught"]
    names.append("native word acceptance")
    (fb, Bb), (fc, Bc) = vals["b"], vals["c"]
    out = {}
    for j, name in enumerate(names):
        d = Bc[:, j] - Bb[:, j]
        out[f"length|ja|{name}"] = _res(fb[j], fc[j], fc[j] - fb[j], _q(d), _p2(d), len(ids),
                                        "paired clip bootstrap, 2,000, seed 0")
    return out


def tips(base: Path, cand: Path, n_boot: int = 2000) -> dict:
    R = {}
    for tag, run in (("b", base), ("c", cand)):
        p = Path(run) / "cache" / "opt_dumps" / "coaching_tips_r19.pkl"
        if not p.exists():
            return {}
        R[tag] = pickle.loads(p.read_bytes())
    out = {}

    def by_id(t, kind, lang):
        return {c["id"]: c for c in R[t].get(kind, {}).get(lang, [])}

    for lang in LANGS:
        X = {t: by_id(t, "fleurs", lang) for t in ("b", "c")}
        ids = sorted(set(X["b"]) & set(X["c"]))
        if not ids:
            continue
        M = _counts(len(ids), n_boot)
        one = {t: np.ones(len(ids)) for t in ("b", "c")}
        share = {t: np.array([float(bool(X[t][i]["tips"]["after"])) for i in ids]) for t in ("b", "c")}
        per = {t: np.array([float(len(X[t][i]["tips"]["after"])) for i in ids]) for t in ("b", "c")}
        for name, num in (("clips with a pronunciation tip", share), ("tips per clip", per)):
            vb, vc, d = _ratio_boot(num, one, M)
            out[f"coaching tips on natives|{lang}|{name}"] = _res(vb, vc, vc - vb, _q(d), _p2(d), len(ids),
                                                                 "paired clip bootstrap, 2,000, seed 0")
    for kind, key, lang in (("so762", "speechocean762", "en"), ("ompal", "ompal", "zh")):
        X = {t: by_id(t, kind, lang) for t in ("b", "c")}
        ids = sorted(set(X["b"]) & set(X["c"]))
        if not ids:
            continue
        M = _counts(len(ids), n_boot)
        F = {}
        for t in ("b", "c"):
            marked = np.array([float(bool(X[t][i]["marked"])) for i in ids])
            has = np.array([float(bool(X[t][i]["tips"]["after"])) for i in ids])
            right, tot = [], []
            for i in ids:
                u = X[t][i]
                r = sum(int(k in u["wrong_sounds"]) if k is not None else int(wi in u["wrong_words"])
                        for _kind, wi, k in u["tips"]["after"])
                right.append(float(r))
                tot.append(float(len(u["tips"]["after"])))
            F[t] = {"a": (marked * has, marked), "b": (np.array(right), np.array(tot)),
                    "c": ((1 - marked) * has, 1 - marked)}
        for name, f in (("a_marked_utts_with_pron_tip", "a"), ("b_pron_tips_right", "b"),
                        ("c_clean_utts_with_pron_tip", "c")):
            vb, vc, d = _ratio_boot({t: F[t][f][0] for t in F}, {t: F[t][f][1] for t in F}, M)
            out[f"coaching tips on learners ({key})|{lang}|{name}"] = _res(vb, vc, vc - vb, _q(d), _p2(d), len(ids),
                                                                           "paired utterance bootstrap, 2,000, seed 0")
    return out


AREAS = {"speechocean762": so762, "learner errors": errdet, "coaching": coaching, "OMPAL": ompal, "swap": swap,
         "cv-real": cvreal, "contrast audit": contrast_audit, "length": length, "coaching tips": tips}


def all_rows(base: Path, cand: Path, areas=None, log=print) -> tuple[dict, dict]:
    """(row key -> paired result, area -> error message for areas that could not be computed)."""
    out, errors = {}, {}
    for name, f in AREAS.items():
        if areas and name not in areas:
            continue
        try:
            r = f(Path(base), Path(cand))
            out.update(r)
            log(f"paired {name}: {len(r)} rows")
        except Exception as e:  # noqa: BLE001  (an area without outputs falls back to the value rules)
            errors[name] = f"{type(e).__name__}: {e}"
            log(f"paired {name}: not computed ({errors[name]})")
    return out, errors


if __name__ == "__main__":
    import sys

    r, e = all_rows(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else None)
    print(json.dumps({"rows": r, "errors": e}, indent=1, default=float))
