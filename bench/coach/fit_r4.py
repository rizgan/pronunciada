"""Coaching round 4: articulatory attribute heads, the leak check, and their coaching gates.

    python -m bench.coach.fit_r4 heads      # train variants on FLEURS train, leak check on the audit swaps
    python -m bench.coach.fit_r4 gates      # claim policy per sensitivity: speechocean762 train + FLEURS native cap
    python -m bench.coach.fit_r4 eval       # held-out, once: speechocean762 test, OMPAL, L2-ARCTIC / EpaDB

Inputs: bench/coach/dump_r4.py. Rules fixed in advance:
  heads    multinomial logistic regression (sklearn, C = 0.05) on standardised pooled activations,
           FLEURS train, ten languages pooled, at most 2,500 sounds per language and head.
           Variants: layers {enc1, enc1+enc3, enc1+enc3+enc5} x {own sound, + neighbours' mean}.
  leak     FLEURS validation contrast-audit swaps (audio has X, edited reference says Y, vector
           pooled under the EDITED alignment): accuracy w.r.t. X ("honest") and the share that
           predicts Y's class ("follows the reference"), on swaps where X and Y differ in that head.
           Per head the variant with the best honest accuracy is kept, provided it is >= 0.6 and
           above the follow rate; otherwise the head is dropped.
  gates    a head claims "this feature differs" for a flagged sound when 1 - P(expected class) >= q,
           coaching score >= t, and the best competing sound has the claimed class. (q, t) per head
           and sensitivity by a greedy search on speechocean762 train real substitutions (claim
           right = the pronounced phone has the claimed class and the expected one does not):
           claim precision >= 0.8 / 0.75 / 0.75, English FLEURS native wrong claims per clip
           <= 0.15 / 0.3 / none (careful / balanced / eager); per-language coaching-score floor
           for the other languages' native cap.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.coach.gather import out_dir
from pronunciada import attr_heads as AH

LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
VARIANTS = {"e1": slice(0, 256), "e13": slice(0, 768), "e135": slice(0, 1024)}
PER_LANG = 2500
TARGET = 0.75


def load(name):
    return pickle.loads((out_dir() / f"dump-r4-{name}.pkl").read_bytes())


def head_applies(h, lab):
    return h in lab


def train_rows():
    D = load("fleurs-train")
    V, C, L, labs = [], [], [], []
    for rid, o in D.items():
        if "error" in o or not len(o["ipa"]):
            continue
        v = o["vec"].astype(np.float32)
        V.append(v)
        C.append(AH.context_vectors(v))
        L += [o["lang"]] * len(v)
        labs += [AH.labels(p) for p in o["ipa"]]
    return np.concatenate(V), np.concatenate(C), np.array(L), labs


def fit_head(X, y, classes):
    from sklearn.linear_model import LogisticRegression

    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=0.05, max_iter=400).fit((X - mu) / sd, y)
    cls = list(m.classes_)
    W = np.zeros((X.shape[1], len(classes)), np.float32)
    b = np.full(len(classes), -30.0, np.float32)            # classes never seen in training
    if len(cls) == 2:
        k = classes.index(cls[1])
        W[:, k] = m.coef_[0]
        b[k] = m.intercept_[0]
        b[classes.index(cls[0])] = 0.0
    else:
        for j, c in enumerate(cls):
            W[:, classes.index(c)] = m.coef_[j]
            b[classes.index(c)] = m.intercept_[j]
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "W": W, "b": b}


def predict(m, X):
    Z = ((X - m["mu"]) / m["sd"]) @ m["W"] + m["b"]
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def design(V, C, sl, ctx):
    return np.concatenate([V[:, sl], C[:, sl]], 1) if ctx else V[:, sl]


def audit_rows():
    """Native sounds and swap items of the audit clips (vectors + labels)."""
    D = load("audit")
    nat = {"V": [], "C": [], "lang": [], "ipa": []}
    sw = []
    for rid, o in D.items():
        if "error" in o or not len(o["ipa"]):
            continue
        v = o["vec"].astype(np.float32)
        c = AH.context_vectors(v)
        nat["V"].append(v)
        nat["C"].append(c)
        nat["lang"] += [o["lang"]] * len(v)
        nat["ipa"] += o["ipa"]
        for e in o["edits"]:
            ev = e["vec"].astype(np.float32)
            # context: the native neighbours of the edited position (the neighbours' sounds are
            # unchanged; their vectors under the edited alignment were not stored)
            i = min(e["index"], len(v) - 1)
            for q, f in enumerate(e["flag"]):
                sw.append({"lang": o["lang"], "contrast": e["contrast"], "kind": e["kind"],
                           "direction": e["direction"], "src": e["src"], "dst": e["dst"], "ref": e["ref_ipa"][q],
                           "first": q == 0, "V": ev[q], "C": c[i], "alt": e["alt"], "prob": e["prob"][q]})
    for k in ("V", "C"):
        nat[k] = np.concatenate(nat[k])
    nat["lang"] = np.array(nat["lang"])
    return nat, sw


def truth_of(s):
    """(audio symbol, reference symbol) for a swap item, or None when the symbols are not phones."""
    if s["kind"] in ("sub", "final_dev", "ru_unstressed"):
        return s["src"], s["dst"]
    return None


def step_heads():
    V, C, L, labs = train_rows()
    nat, sw = audit_rows()
    rng = np.random.default_rng(0)
    rep = {"n_train": int(len(V)), "n_audit_native": int(len(nat["V"])), "n_swaps": len(sw), "heads": {}}
    shipped = {}
    nat_labs = [AH.labels(p) for p in nat["ipa"]]
    for h, (_, classes, _) in AH.HEADS.items():
        idx = [i for i, lb in enumerate(labs) if h in lb]
        pick = []
        for l in LANGS:
            li = [i for i in idx if L[i] == l]
            if len(li) > PER_LANG:
                li = list(rng.choice(li, PER_LANG, replace=False))
            pick += li
        pick = np.array(sorted(pick))
        y = np.array([labs[i][h] for i in pick])
        if len(set(y)) < 2:
            continue
        ni = np.array([i for i, lb in enumerate(nat_labs) if h in lb])
        ny = np.array([nat_labs[i][h] for i in ni])
        items = []
        for s in sw:
            t = truth_of(s)
            if t is None:
                continue
            a, r = AH.labels(t[0]), AH.labels(t[1])
            if h in a and h in r and a[h] != r[h]:
                items.append((s, a[h], r[h]))
        res = {}
        models = {}
        for vn, sl in VARIANTS.items():
            for ctx in (False, True):
                key = f"{vn}{'+ctx' if ctx else ''}"
                m = fit_head(design(V[pick], C[pick], sl, ctx), y, list(classes))
                models[key] = (m, sl, ctx)
                Pn = predict(m, design(nat["V"][ni], nat["C"][ni], sl, ctx)) if len(ni) else np.zeros((0, len(classes)))
                acc_n = float(np.mean(np.array(classes)[Pn.argmax(1)] == ny)) if len(ni) else None
                if items:
                    Xs = np.stack([np.concatenate([s["V"][sl], s["C"][sl]]) if ctx else s["V"][sl] for s, _, _ in items])
                    Ps = predict(m, Xs)
                    pred = np.array(classes)[Ps.argmax(1)]
                    honest = float(np.mean([p == a for p, (_, a, _) in zip(pred, items)]))
                    follow = float(np.mean([p == r for p, (_, _, r) in zip(pred, items)]))
                    by_lang = defaultdict(list)
                    for p, (s, a, r) in zip(pred, items):
                        by_lang[s["lang"]].append((p == a, p == r))
                    bl = {l: {"n": len(v), "honest": float(np.mean([x for x, _ in v])),
                              "follows_ref": float(np.mean([y_ for _, y_ in v]))} for l, v in sorted(by_lang.items())}
                else:
                    honest = follow = None
                    bl = {}
                nl = {}
                for l in LANGS:
                    mm = nat["lang"][ni] == l
                    if mm.any():
                        nl[l] = float(np.mean(np.array(classes)[Pn[mm].argmax(1)] == ny[mm]))
                res[key] = {"native_acc": acc_n, "native_by_lang": nl, "swap_n": len(items), "swap_honest": honest,
                            "swap_follows_ref": follow, "swap_by_lang": bl}
                print(h, key, "native", acc_n and round(acc_n, 3), "swaps", len(items), "honest",
                      honest and round(honest, 3), "follow", follow and round(follow, 3), flush=True)
        ok = {k: v for k, v in res.items() if v["swap_honest"] is not None and v["swap_n"] >= 30}
        if ok:
            best = max(ok, key=lambda k: (ok[k]["swap_honest"], ok[k]["native_acc"] or 0))
            keep = ok[best]["swap_honest"] >= 0.6 and ok[best]["swap_honest"] > ok[best]["swap_follows_ref"]
        else:                       # no swap evidence: keep the best native variant, flagged untested
            best = max(res, key=lambda k: res[k]["native_acc"] or 0)
            keep = True
        rep["heads"][h] = {"variants": res, "chosen": best, "kept": bool(keep), "swap_tested": bool(ok),
                           "n_train": int(len(pick))}
        if keep:
            m, sl, ctx = models[best]
            shipped[h] = {"m": m, "cols": np.arange(sl.start, sl.stop), "ctx": ctx, "classes": list(classes)}
    z = {"heads": np.array(sorted(shipped))}
    for h, s in shipped.items():
        z[f"{h}/cols"] = s["cols"].astype(np.int16)
        z[f"{h}/ctx"] = np.array([s["ctx"]])
        z[f"{h}/mu"] = s["m"]["mu"].astype(np.float16)
        z[f"{h}/sd"] = s["m"]["sd"].astype(np.float16)
        z[f"{h}/W"] = s["m"]["W"].astype(np.float16)
        z[f"{h}/b"] = s["m"]["b"].astype(np.float32)
        z[f"{h}/classes"] = np.array(s["classes"])
    np.savez_compressed("pronunciada/data/coach/attr_heads.npz", **z)
    side = side_result(nat, sw, AH.AttrHeads.load("pronunciada/data/coach/attr_heads.npz"))
    rep["side"] = side
    Path("results/tables/coaching_r4_heads.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({h: (v["chosen"], v["kept"], v["variants"][v["chosen"]]["native_acc"],
                          v["variants"][v["chosen"]]["swap_honest"], v["variants"][v["chosen"]]["swap_follows_ref"])
                      for h, v in rep["heads"].items()}, indent=0, default=float))
    print(json.dumps(side, indent=0, default=float))


# ----------------------------------------------------------------------------- side result

SIDE = [("ko", "aspiration", "aspirated"), ("zh", "aspiration", "aspirated"), ("ru", "hard_soft", "palatalised"),
        ("pt", "nasal", "nasal"), ("it", "gemination", "long"), ("de", "vowel_length", "long"),
        ("ja", "vowel_length", "long"), ("ja", "gemination", "long")]


def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    # ties: average ranks
    from scipy.stats import rankdata

    r = rankdata(allv)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def side_result(nat, sw, heads):
    """Heads as stand-alone detectors on the audit's blind / weak contrasts: score = 1 - P(the
    reference sound's class); positives = swapped items, negatives = native sounds whose
    reference symbol is one of the contrast's members, same language (threshold for hit@5 % from
    those natives)."""
    out = {}
    audit = json.loads(Path("results/tables/contrast_audit.json").read_text())["languages"]
    for lang, cname, h in SIDE:
        if heads is None or h not in heads.heads:
            out[f"{lang}/{cname}"] = {"head": h, "note": "head not shipped"}
            continue
        items = [s for s in sw if s["lang"] == lang and s["contrast"] == cname and s["first"]]
        if not items:
            continue
        syms = {s["ref"] for s in items} | {s["src"] for s in items if truth_of(s)}
        ni = np.flatnonzero((nat["lang"] == lang) & np.isin(np.array(nat["ipa"], dtype=object), list(syms)))
        P = heads.predict(nat["V"], ni, nat["C"])
        neg = [1 - p[h].get(AH.labels(nat["ipa"][i]).get(h), 0.0) for p, i in zip(P, ni) if h in p and h in AH.labels(nat["ipa"][i])]
        pos, dirs = [], []
        for s in items:
            V = s["V"][None, :]
            Cc = s["C"][None, :]
            p = heads.predict(V, [0], Cc)[0]
            lb = AH.labels(s["ref"])
            if h not in p or h not in lb:
                continue
            pos.append(1 - p[h].get(lb[h], 0.0))
            dirs.append(s["direction"])
        if not pos or not neg:
            continue
        thr = float(np.quantile(neg, 0.95))
        a = audit[lang]["contrasts"].get(cname, {})
        d = {"head": h, "n_swaps": len(pos), "n_native": len(neg), "auc": _auc(pos, neg),
             "hit5": float(np.mean(np.asarray(pos) > thr)),
             "by_direction": {dd: {"n": int(sum(1 for x in dirs if x == dd)),
                                   "auc": _auc([p for p, x in zip(pos, dirs) if x == dd], neg),
                                   "hit5": float(np.mean([p > thr for p, x in zip(pos, dirs) if x == dd]))}
                              for dd in sorted(set(dirs))},
             "audit_auc": a.get("auc"), "audit_hit5": a.get("sound_hit5"), "audit_raw_auc": a.get("raw_auc"),
             "audit_directions": {k: {"auc": v.get("auc"), "hit5": v.get("sound_hit5"), "verdict": v.get("verdict")}
                                  for k, v in (a.get("directions") or {}).items()}}
        out[f"{lang}/{cname}"] = d
    return out




# ----------------------------------------------------------------------------- learners

def so_rows(split):
    """speechocean762 rows (bench.coach.fit_v2.so762_rows: expert scores, pronounced phones,
    out-of-fold coaching logits on train) plus each sound's vector and context vector."""
    from bench.coach.fit_v2 import so762_rows

    R = so762_rows(f"r4-so762-{split}", split)
    D = load(f"so762-{split}")
    ctx = {u: AH.context_vectors(o["vec"].astype(np.float32)) for u, o in D.items() if "error" not in o}
    R["V"] = np.stack([D[u]["vec"][i].astype(np.float32) for u, i in zip(R["utt"], R["i"])])
    R["C"] = np.stack([ctx[u][i] for u, i in zip(R["utt"], R["i"])])
    return R


def decisions(R, sens, mdl, lang="en"):
    """Per row: flagged, named heard sound (coach.decide on each utterance)."""
    from bench.coach.fit_v2 import add_initial
    from pronunciada import coach as Co

    utts = np.asarray(R["utt"])
    flag = np.zeros(len(utts), bool)
    heard = [None] * len(utts)
    for u in dict.fromkeys(R["utt"]):
        ix = np.flatnonzero(utts == u)
        X = {k: (v[ix] if isinstance(v, np.ndarray) else [v[i] for i in ix]) for k, v in R.items() if k not in ("V", "C")}
        add_initial(X)
        d = Co.decide(X, lang, sens, mdl)
        flag[ix] = d["flag"]
        for q, i in enumerate(ix):
            heard[i] = d["heard"][q]
    return flag, heard


def head_probs(R, idx, heads):
    return heads.predict(R["V"], idx, R["C"])


def is_sub(p):
    return p not in (None, "<DEL>", "<unk>")


def claim_right(c, pron, expected):
    lp, le = AH.labels(pron), AH.labels(expected)
    return c["feature"] in lp and lp[c["feature"]] == c["heard"] and le.get(c["feature"]) != c["heard"]


def evaluate(R, flag, heard, P_by_row, bars, mdl=None):
    """Claims on flagged sounds; precision on real substitutions; coverage of real substitutions."""
    from pronunciada.coach_attr import claims_for

    m = np.isfinite(R["acc"])
    subs = [i for i in np.flatnonzero(m) if is_sub(R["pron"][i])]
    n_claims = n_right = 0
    got = got_right = 0
    on_correct = n_correct_flagged = 0
    per_head = defaultdict(lambda: [0, 0])
    named_feat_right = 0
    for i in np.flatnonzero(flag & m):
        P = P_by_row.get(i)
        if P is None:
            continue
        cl = claims_for(P, R["expected"][i], R["alts"][i], heard[i], bars)
        if heard[i]:                                   # a named sound implies its own feature claims
            le, lh = AH.labels(R["expected"][i]), AH.labels(heard[i])
            implied = [{"feature": h, "heard": lh[h]} for h in le if h in lh and lh[h] != le[h]]
            have = {c["feature"] for c in cl}
            cl = cl + [dict(c, implied=True) for c in implied if c["feature"] not in have]
        if R["acc"][i] >= 2:
            n_correct_flagged += 1
            on_correct += len(cl)
        if not is_sub(R["pron"][i]):
            continue
        if cl:
            got += 1
        ok = [claim_right(c, R["pron"][i], R["expected"][i]) for c in cl]
        n_claims += len(cl)
        n_right += sum(ok)
        got_right += any(ok)
        for c, o in zip(cl, ok):
            if not c.get("implied"):
                per_head[c["feature"]][0] += 1
                per_head[c["feature"]][1] += o
    return {"n_substitutions": len(subs), "flagged_substitutions": int(sum(flag[i] for i in subs)),
            "with_claim": got / max(len(subs), 1), "with_right_claim": got_right / max(len(subs), 1),
            "claims": n_claims, "precision": n_right / max(n_claims, 1),
            "claims_on_expert_correct": on_correct, "flagged_expert_correct": n_correct_flagged,
            "per_head": {h: {"n": a, "precision": b / max(a, 1)} for h, (a, b) in sorted(per_head.items())}}


SENS = ("careful", "balanced", "eager")
QS = (0.5, 0.9, 0.99, 0.999)
TS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CAPS = {"careful": 0.15, "balanced": 0.3, "eager": None}           # wrong claims per native clip (English)
PREC = {"careful": 0.8, "balanced": 0.75, "eager": 0.75}


def claim_cfg(M, lang, sens):
    """(bars, min_ep) of the shipped claim policy for a language and sensitivity."""
    A_ = (M.get("attr") or {}).get("by_sens", {}).get(sens) or {}
    bars = {h: (v["q"] if v else None) for h, v in A_.items()}
    floor = ((M.get("attr") or {}).get("by_lang", {}).get(lang) or {}).get(sens, 0.0) or 0.0
    min_ep = {h: max(v["t"], floor) for h, v in A_.items() if v}
    return bars, min_ep


def open_claims(P, expected, alts, heard, cand):
    from pronunciada.coach_attr import claims_for

    return claims_for(P, expected, alts, heard, {h: 0.0 for h in AH.HEADS}, cand=cand, agree=True)


def _passes(c, ep, cfg):
    o = cfg.get(c["feature"])
    return o is not None and c["p"] >= o[0] and ep >= o[1]


def learner_table(R, M, heads):
    """Per flagged (eager) sound: coaching score, flags per sensitivity, its open claims (+ right)."""
    from bench.coach.fit_v2 import add_initial
    from pronunciada import coach as Co

    utts = np.asarray(R["utt"])
    rows = []
    X = {k: v for k, v in R.items() if k not in ("V", "C")}
    for u in dict.fromkeys(R["utt"]):
        ix = np.flatnonzero(utts == u)
        Xu = {k: (v[ix] if isinstance(v, np.ndarray) else [v[i] for i in ix]) for k, v in X.items()}
        add_initial(Xu)
        d = Co.decide(Xu, "en", "eager", M)
        fl = np.flatnonzero(d["flag"])
        if not len(fl):
            continue
        P = heads.predict(R["V"], ix[fl], R["C"])
        masks = {s: Co.sound_mask(Xu, d["error_prob"], M, "en", s) for s in SENS}
        for q, j in enumerate(fl):
            i = ix[j]
            ep = float(d["error_prob"][j])
            fs = {s: bool(mask[j]) for s, mask in masks.items()}
            cl = open_claims(P[q], R["expected"][i], R["alts"][i], d["heard"][j], R["cand"][i])
            sub = is_sub(R["pron"][i])
            for c in cl:
                c["right"] = bool(sub and claim_right(c, R["pron"][i], R["expected"][i]))
            heard_right = None
            if d["heard"][j] and sub:
                le, lh = AH.labels(R["expected"][i]), AH.labels(d["heard"][j])
                imp = [{"feature": h, "heard": lh[h]} for h in le if h in lh and lh[h] != le[h]]
                heard_right = any(claim_right(c, R["pron"][i], R["expected"][i]) for c in imp) if imp else False
            rows.append({"i": int(i), "ep": ep, "flags": fs, "sub": sub, "correct": bool(R["acc"][i] >= 2),
                         "claims": cl, "named": d["heard"][j], "named_right": heard_right})
    nsub = int(sum(1 for i in range(len(R["pron"])) if np.isfinite(R["acc"][i]) and is_sub(R["pron"][i])))
    return rows, nsub


def native_table(M, heads):
    from bench.coach.fit_v2 import add_initial
    from pronunciada import coach as Co

    D = load("fleurs-val")
    out = defaultdict(list)                       # lang -> per clip: [(ep, flags, claims)]
    for rid, o in D.items():
        if "error" in o:
            continue
        lang = rid.split("-")[1]
        X = add_initial(dict(o["sounds"]))
        d = Co.decide(X, lang, "eager", M)
        fl = np.flatnonzero(d["flag"])
        clip = []
        if len(fl):
            V = o["vec"].astype(np.float32)
            P = heads.predict(V, fl, AH.context_vectors(V))
            masks = {s: Co.sound_mask(X, d["error_prob"], M, lang, s) for s in SENS}
            for q, i in enumerate(fl):
                ep = float(d["error_prob"][i])
                fs = {s: bool(mask[i]) for s, mask in masks.items()}
                clip.append((ep, fs, open_claims(P[q], X["expected"][i], X["alts"][i], d["heard"][i], X["cand"][i])))
        out[lang].append(clip)
    return out


def score_cfg(rows, nsub, cfg, sens):
    got = right = n = nr = 0
    for r in rows:
        if not r["flags"][sens] or not r["sub"]:
            continue
        cl = [c for c in r["claims"] if _passes(c, r["ep"], cfg)]
        ok = [c["right"] for c in cl]
        got += bool(cl)
        right += any(ok)
        n += len(cl)
        nr += sum(ok)
    return {"with_claim": got / max(nsub, 1), "with_right_claim": right / max(nsub, 1), "claims": n,
            "precision": nr / max(n, 1)}


def native_rate(nat_clips, cfg, sens, floor=0.0):
    tot = sum(1 for clip in nat_clips for ep, fs, cl in clip if fs[sens] and ep >= floor
              for c in cl if _passes(c, ep, cfg))
    return tot / max(len(nat_clips), 1)


def greedy(rows, nsub, nat_en, sens):
    """Per-head (q, t) options; add / upgrade the head option with the best gain in right
    coverage per added native claim while overall precision >= PREC[sens] and English native
    claims per clip <= CAPS[sens]."""
    opts = []
    for h in AH.HEADS:
        for q in QS:
            for t in TS:
                sel = [c for r in rows if r["flags"][sens] and r["sub"] for c in r["claims"]
                       if c["feature"] == h and c["p"] >= q and r["ep"] >= t]
                if len(sel) >= 15 and np.mean([c["right"] for c in sel]) >= PREC[sens]:
                    opts.append((h, q, t))
    cfg: dict = {}
    cur = score_cfg(rows, nsub, cfg, sens)
    cur_nat = 0.0
    while True:
        best = None
        for h, q, t in opts:
            if cfg.get(h) == (q, t):
                continue
            trial = dict(cfg, **{h: (q, t)})
            sc = score_cfg(rows, nsub, trial, sens)
            nat = native_rate(nat_en, trial, sens)
            if sc["precision"] < PREC[sens] or (CAPS[sens] is not None and nat > CAPS[sens]):
                continue
            gain = sc["with_right_claim"] - cur["with_right_claim"]
            if gain <= 1e-9:
                continue
            ratio = gain / max(nat - cur_nat, 1e-3) if CAPS[sens] is not None else gain
            if best is None or ratio > best[0]:
                best = (ratio, h, q, t, sc, nat)
        if best is None:
            break
        _, h, q, t, cur, cur_nat = best
        cfg[h] = (q, t)
    return cfg, cur, cur_nat


def step_gates():
    """Claim policy per sensitivity on speechocean762 train (flags out-of-fold), English native
    cap on FLEURS validation; then a per-language coaching-score floor for the other languages'
    native cap. Writes model.json attr and results/tables/coaching_r4_gates.json."""
    from pronunciada import coach as Co

    M = Co.model()
    heads = AH.AttrHeads.load("pronunciada/data/coach/attr_heads.npz")
    R = so_rows("train")
    rows, nsub = learner_table(R, M, heads)
    nat = native_table(M, heads)
    rep = {"n_substitutions": nsub, "by_sens": {}, "curve": {}}
    by_sens = {}
    for sens in SENS:
        cfg, sc, nr = greedy(rows, nsub, nat["en"], sens)
        by_sens[sens] = {h: ({"q": cfg[h][0], "t": cfg[h][1]} if h in cfg else None) for h in AH.HEADS}
        rep["by_sens"][sens] = {"policy": by_sens[sens], "train": sc, "native_en": nr}
        print(sens, by_sens[sens], sc, "native en", round(nr, 3), flush=True)
    # coverage / precision curve without the native cap (uniform q, t), balanced flags
    for q in QS:
        for t in TS:
            cfg = {h: (q, t) for h in AH.HEADS}
            rep["curve"].setdefault("balanced", []).append(
                {"q": q, "t": t, **score_cfg(rows, nsub, cfg, "balanced"), "native_en": native_rate(nat["en"], cfg, "balanced")})
    by_lang = {}
    for l in LANGS:
        by_lang[l] = {}
        for sens in SENS:
            cfg = {h: (v["q"], v["t"]) for h, v in by_sens[sens].items() if v}
            cap = CAPS[sens] if CAPS[sens] is not None else float("inf")   # eager: no native cap
            floor = 0.0
            for f in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01):
                floor = f
                if native_rate(nat[l], cfg, sens, f) <= cap:
                    break
            by_lang[l][sens] = floor
    rep["by_lang_floor"] = by_lang
    rep["native"] = {l: {s: native_rate(nat[l], {h: (v["q"], v["t"]) for h, v in by_sens[s].items() if v}, s, by_lang[l][s])
                         for s in SENS} for l in LANGS}
    rep["native_flags"] = {l: {s: sum(1 for clip in nat[l] for ep, fs, cl in clip if fs[s]) / max(len(nat[l]), 1)
                               for s in SENS} for l in LANGS}
    M["attr"] = {"agree_with_candidate": True, "by_sens": by_sens, "by_lang": by_lang,
                 "precision": {s: rep["by_sens"][s]["train"]["precision"] for s in SENS}}
    Path("pronunciada/data/coach/model.json").write_text(json.dumps(M, indent=1))
    Path("results/tables/coaching_r4_gates.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"by_lang": by_lang, "native": rep["native"]}, default=float))


# ----------------------------------------------------------------------------- held-out

def policy(M, lang, sens):
    bars, min_ep = claim_cfg(M, lang, sens)
    return {h: (q, min_ep.get(h, 0.0)) for h, q in bars.items() if q is not None}


def score_rows(rows, nsub, cfg, sens):
    """score_cfg plus: substitutions with any right feature (claims or the named sound's own
    features), claims made on sounds the experts scored 2."""
    sc = score_cfg(rows, nsub, cfg, sens)
    any_right = on_correct = 0
    for r in rows:
        if not r["flags"][sens]:
            continue
        cl = [c for c in r["claims"] if _passes(c, r["ep"], cfg)]
        if r["correct"]:
            on_correct += len(cl)
        if r["sub"] and (any(c["right"] for c in cl) or r["named_right"]):
            any_right += 1
    sc.update(any_right_feature=any_right / max(nsub, 1), claims_on_expert_correct=on_correct)
    return sc


def step_eval():
    """Held-out, once: speechocean762 test, L2-ARCTIC / EpaDB test (report only), OMPAL."""
    from pronunciada import coach as Co

    M = Co.model()
    heads = AH.AttrHeads.load("pronunciada/data/coach/attr_heads.npz")
    R = so_rows("test")
    rows, nsub = learner_table(R, M, heads)
    rep = {"n_substitutions": nsub, "test": {}, "today": {}, "curve": []}
    for s in SENS:
        rep["test"][s] = score_rows(rows, nsub, policy(M, "en", s), s)
        rep["today"][s] = score_rows(rows, nsub, {}, s)
        per = defaultdict(list)
        for r in rows:
            if r["flags"][s] and r["sub"]:
                for c in r["claims"]:
                    if _passes(c, r["ep"], policy(M, "en", s)):
                        per[c["feature"]].append(c["right"])
        rep["test"][s]["per_head"] = {h: {"n": len(v), "precision": float(np.mean(v))} for h, v in sorted(per.items())}
    for q in QS:
        for t in TS:
            cfg = {h: (q, t) for h in AH.HEADS}
            rep["curve"].append({"q": q, "t": t, **score_cfg(rows, nsub, cfg, "balanced")})
    rep["examples"] = examples(R, rows, M)
    rep["learners"] = learners_report(M, heads)
    rep["ompal"] = {sp: ompal_report(M, heads, sp) for sp in ("dev", "test")}
    Path("results/tables/coaching_r4_heldout.json").write_text(json.dumps(rep, indent=1, default=float, ensure_ascii=False))
    for s in SENS:
        print(s, {k: v for k, v in rep["test"][s].items() if k != "per_head"}, "\n  today",
              {k: rep["today"][s][k] for k in ("any_right_feature",)}, "\n  per head", rep["test"][s]["per_head"])
    print(json.dumps({k: rep[k] for k in ("learners", "ompal")}, indent=0, default=float))
    print(json.dumps(rep["examples"], ensure_ascii=False, indent=0))


def examples(R, rows, M, n=2):
    from bench.data import load as dload
    from pronunciada.tips import feature_tip

    recs = {r["id"]: r for r in dload("speechocean762", split="test")}
    cfg = policy(M, "en", "balanced")
    out = []
    for r in rows:
        if not r["flags"]["balanced"] or r["named"] or not r["sub"]:
            continue
        cl = [c for c in r["claims"] if _passes(c, r["ep"], cfg)]
        if not cl or not cl[0]["right"]:
            continue
        i = r["i"]
        rec = recs[R["utt"][i]]
        w = rec["labels"]["words"][int(R["tj"][i])]
        out.append({"id": R["utt"][i], "text": rec["text"], "word": w["text"], "expected": R["expected"][i],
                    "claims": [{k: v for k, v in c.items() if k != "right"} for c in cl],
                    "tip": feature_tip(cl[0], R["expected"][i], w["text"]),
                    "experts": {"phones": w["phones"], "scores": w["phones_accuracy"],
                                "mispronunciations": w["mispronunciations"]}})
        if len(out) >= n:
            break
    return out


def learners_report(M, heads):
    """L2-ARCTIC / EpaDB test, report only: claim precision against the annotators' perceived
    phones (alignment as bench/eval_english.py), balanced."""
    from bench.coach.eval_learners import _fold
    from bench.coach.fit_v2 import add_initial
    from bench.data import load as dload
    from bench.eval_english import _their_entries
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference, arpa_to_ours
    from pronunciada import coach as Co

    cfg = policy(M, "en", "balanced")
    out = {}
    for ds, name in (("l2arctic", "l2arctic-test"), ("epadb", "epadb-test")):
        D = load(name)
        n_sub = got = got_right = n_cl = n_right = named_right = 0
        for rec in dload(ds, split="test"):
            o = D.get(rec["id"])
            if o is None or "error" in o:
                continue
            ref = build_ref("en", rec["text"])
            X = add_initial(dict(o["sounds"]))
            if ref is None or len(ref.phones) != len(X["expected"]):
                continue
            ents = _their_entries(rec, ds)
            raw = [p for p in rec["labels"]["phones"] if (p["error"] != "a" if ds == "l2arctic" else p["ref"] != "0")]
            if ds == "l2arctic":
                words = [w["text"] for w in rec["labels"]["words"]]
                tp = [[e["ph"] for e in ents if e["word"] == j] for j in range(len(words))]
                pos = {}
                for k, e in enumerate(ents):
                    pos.setdefault(e["word"], []).append(k)
                o2t, _ = align_reference(ref, words, tp)
                t2o = {pos[j][q]: i for i, (j, q) in o2t.items() if j in pos and q < len(pos[j])}
            else:
                o2t, _ = align_reference(ref, None, [[e["ph"] for e in ents]])
                t2o = {q: i for i, (_, q) in o2t.items()}
            d = Co.decide(X, "en", "balanced", M)
            ix = np.flatnonzero(d["flag"])
            V = o["vec"].astype(np.float32)
            P = dict(zip(ix.tolist(), heads.predict(V, ix, AH.context_vectors(V))))
            for k, e in enumerate(ents):
                if e["label"] != "sub" or k not in t2o:
                    continue
                per = raw[k]["perceived"] if ds == "l2arctic" else raw[k]["annotated"]
                pron = _fold(arpa_to_ours(per))
                if not pron:
                    continue
                n_sub += 1
                i = t2o[k]
                if i not in P:
                    continue
                cl = [c for c in open_claims(P[i], X["expected"][i], X["alts"][i], d["heard"][i], X["cand"][i])
                      if _passes(c, float(d["error_prob"][i]), cfg)]
                ok = [claim_right(c, pron, X["expected"][i]) for c in cl]
                nr = False
                if d["heard"][i]:
                    le, lh = AH.labels(X["expected"][i]), AH.labels(d["heard"][i])
                    nr = any(claim_right({"feature": h, "heard": lh[h]}, pron, X["expected"][i])
                             for h in le if h in lh and lh[h] != le[h])
                got += bool(cl)
                got_right += any(ok)
                named_right += bool(any(ok) or nr)
                n_cl += len(cl)
                n_right += sum(ok)
        out[ds] = {"n_substitutions": n_sub, "with_claim": got / max(n_sub, 1), "with_right_claim": got_right / max(n_sub, 1),
                   "any_right_feature": named_right / max(n_sub, 1), "claims": n_cl, "precision": n_right / max(n_cl, 1)}
    return out


def ompal_report(M, heads, split):
    """OMPAL (Mandarin learners): characters where a head makes a claim on the syllable's initial
    (or final), against the teachers' consonant (vowel) marks. Teachers mark only the type, so a
    claim is counted right when the matching type is marked wrong."""
    from bench.coach.fit_v2 import add_initial
    from bench.data import load as dload
    from bench.eval_ompal import overlap, syllable_char_spans, word_char_spans
    from pronunciada import coach as Co
    D = load(f"ompal-{split}")
    cfg = policy(M, "zh", "balanced")
    rows = []
    for rec in dload("ompal", split=split):
        lab = rec["labels"]
        if lab["native"] or lab["word_level_dropped"]:
            continue
        o = D.get(rec["id"])
        if o is None or "error" in o:
            continue
        X = add_initial(dict(o["sounds"]))
        d = Co.decide(X, "zh", "balanced", M)
        ix = np.flatnonzero(d["flag"])
        V = o["vec"].astype(np.float32)
        P = dict(zip(ix.tolist(), heads.predict(V, ix, AH.context_vectors(V))))
        spans = word_char_spans(rec["text"], [w["text"] for w in o["words"]])
        syl_span = {}
        for wi, w in enumerate(o["words"]):
            if spans[wi] is not None:
                for k, sp in enumerate(syllable_char_spans(spans[wi], w["n_syl"])):
                    syl_span[(wi, k)] = sp
        claims = defaultdict(set)
        flags = defaultdict(set)
        for i in ix:
            key = (int(X["word"][i]), int(X["syl"][i]))
            if key not in syl_span:
                continue
            t = "initial" if X["initial"][i] else "final"
            flags[syl_span[key]].add(t)
            for c in open_claims(P[i], X["expected"][i], X["alts"][i], d["heard"][i], X["cand"][i]):
                if _passes(c, float(d["error_prob"][i]), cfg):
                    claims[syl_span[key]].add((t, c["feature"]))
        for u in lab["units"]:
            if u["span"] is None:
                continue
            cl, fl = set(), set()
            for sp, v in claims.items():
                if overlap(u["span"][0], u["span"][1], sp[0], sp[1]):
                    cl |= v
            for sp, v in flags.items():
                if overlap(u["span"][0], u["span"][1], sp[0], sp[1]):
                    fl |= v
            rows.append({"cons": u["consonant"] == 0, "vowel": u["vowel"] == 0, "claims": cl, "flags": fl})
    out = {"units": len(rows)}
    for t, lab in (("initial", "cons"), ("final", "vowel")):
        fl = [r for r in rows if t in r["flags"]]
        cl = [r for r in rows if any(x == t for x, _ in r["claims"])]
        pos = sum(r[lab] for r in rows)
        out[t] = {"flagged": len(fl), "flag_precision": float(np.mean([r[lab] for r in fl])) if fl else None,
                  "with_claim": len(cl), "claim_precision": float(np.mean([r[lab] for r in cl])) if cl else None,
                  "share_of_marked_with_claim": sum(r[lab] for r in cl) / max(pos, 1),
                  "by_feature": {}}
        feats = sorted({f for r in rows for x, f in r["claims"] if x == t})
        for f in feats:
            rr = [r for r in rows if (t, f) in r["claims"]]
            out[t]["by_feature"][f] = {"n": len(rr), "precision": float(np.mean([r[lab] for r in rr]))}
    return out


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "heads"
    {"heads": step_heads, "gates": step_gates, "eval": step_eval}[step]()
