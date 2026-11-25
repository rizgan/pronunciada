"""Tables of the real-learner benchmark, from cached per-item outputs only (nothing is scored or sent).

    python -m bench.learnerbench.report [--boot 10000]

Writes results/tables/learner_benchmark_{systems,tests,summary,cost,coverage}.json and
results/tables/learner_benchmark_tables.md.

Systems (category, what is read from the cache):
  A  pronunciada-fp32, pronunciada-int8 (bench.learnerbench.run_pronunciada); a further Pronunciada
     variant scored with --backbone is picked up when its outputs exist (LEARNERBENCH_EXTRA=<name>).
  A  openpronounce-1.1.0 (default mode) and openpronounce-1.1.0-student (distilled English phone model):
     raw outputs of bench.rivals.openpronounce. speechocean762 default mode and every student set from
     the round-19 runs; L2-ARCTIC / EpaDB default mode = the round-19 `--fast` outputs (identical to
     default mode on all 2,500 speechocean762 and 2,700 fleurs-swap records, round 19); the learner
     corpora from this benchmark's runs.
  C  azure: cached responses only (round-19 suite; scoreboard speechocean762, L2-ARCTIC, EpaDB, OMPAL).
  C  lingolix: cached responses only (speechocean762 test, 2,500 clips, 22 Nov 2026).
Scores used at each level (fixed from each system's documentation before any result was looked at):
  Pronunciada  sound: the sound model's regression output (`reg`, 0-2 scale) for Pearson, 1 - P(correct)
               at strictness 0.5 for marked-sound detection; word: `word_reg`; sentence: scores[...]
               (total, accuracy, fluency, prosodic), the suite's sentence score = scores["total"];
               OMPAL words: min WordResult.prob over the words covering the unit (bench.eval_ompal).
  OpenPronounce sound: 100 (1 - p) of its phone model's probability for the expected sound, aligned to
               Pronunciada's reference sounds (bench.rivals.convert.to_reference); word: 100 (1 - p) of its
               word model; sentence: `score` (= scores["total"]) and scores[accuracy, fluency, prosodic].
  Azure        sound / word: AccuracyScore of the phoneme / word; sentence: PronScore (suite),
               Accuracy / Fluency / Prosody / PronScore (speechocean762 aspects), AccuracyScore (OMPAL).
  Lingolix     word: word `accuracy`; sentence: the benchmark adapter's utterance score (`overall`);
               no sound level (it returns syllables), no fluency or prosody score.
A clip a system was given but did not score takes the system's lowest score (counted). A language or
level a system does not support is "not supported"; one it supports but that was not run here is
"not run"; neither is ever filled in.

Statistics: every interval is a 95 % percentile bootstrap over speakers (10,000 replicates); every
difference against Pronunciada fp32 is paired (the same replicate for both systems), two-sided
p = 2 min(P(d <= 0), P(d >= 0)) with the +1 correction. Resampling is done with speaker weights
(multinomial counts), which is the same as concatenating resampled speakers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench.learnerbench import items, load_outputs, work

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "results" / "tables"
N_BOOT = 10000
BASE = "pronunciada-fp32"


# ============================================================================ weighted metric primitives

class AUCPart:
    """One (positives, negatives) comparison on a fixed score vector; value for any item weights."""

    def __init__(self, score, pos, neg):
        idx = np.flatnonzero(pos | neg)
        s = np.asarray(score, float)[idx]
        o = np.argsort(s, kind="stable")
        self.idx = idx[o]
        ss = s[o]
        self.blk = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]) if len(ss) else np.zeros(0, int)
        self.p = pos[self.idx].astype(float)
        self.q = neg[self.idx].astype(float)
        self.nb = int(self.blk[-1]) + 1 if len(ss) else 0

    def numer_pairs(self, w):
        if not self.nb:
            return 0.0, 0.0
        ww = w[self.idx]
        P = np.bincount(self.blk, ww * self.p, self.nb)
        N = np.bincount(self.blk, ww * self.q, self.nb)
        before = np.cumsum(N) - N
        return float(np.sum(P * (before + 0.5 * N))), float(P.sum() * N.sum())


class Concordance:
    """C over pairs with different labels (within strata); = AUC for a binary label."""

    def __init__(self, score, h, strata=None):
        h = np.asarray(h, float)
        st = np.zeros(len(h)) if strata is None else np.asarray(strata)
        self.parts = []
        for g in np.unique(st):
            m = st == g
            levels = np.unique(h[m])
            for b in levels[1:]:
                self.parts.append(AUCPart(score, m & (h == b), m & (h < b)))

    def __call__(self, w):
        c = p = 0.0
        for part in self.parts:
            a, b = part.numer_pairs(w)
            c += a
            p += b
        return c / p if p else float("nan")


class MacroAUC:
    """Unweighted mean over strata of the within-stratum AUC (strata lacking either group skipped)."""

    def __init__(self, score, h, strata):
        h, st = np.asarray(h, float), np.asarray(strata)
        self.parts = [AUCPart(score, (st == g) & (h == 1), (st == g) & (h == 0)) for g in np.unique(st)]

    def __call__(self, w):
        v = []
        for part in self.parts:
            a, b = part.numer_pairs(w)
            if b > 0:
                v.append(a / b)
        return float(np.mean(v)) if v else float("nan")


def wpearson(x, y, w):
    W = w.sum()
    if W <= 0:
        return float("nan")
    mx, my = (w * x).sum() / W, (w * y).sum() / W
    dx, dy = x - mx, y - my
    sxy, sxx, syy = (w * dx * dy).sum(), (w * dx * dx).sum(), (w * dy * dy).sum()
    return float(sxy / math.sqrt(sxx * syy)) if sxx > 0 and syy > 0 else float("nan")


class Pearson:
    def __init__(self, score, y):
        self.x, self.y = np.asarray(score, float), np.asarray(y, float)

    def __call__(self, w):
        return wpearson(self.x, self.y, w)


class _Ranker:
    def __init__(self, v):
        v = np.asarray(v, float)
        self.o = np.argsort(v, kind="stable")
        sv = v[self.o]
        self.blk = np.concatenate([[0], np.cumsum(sv[1:] != sv[:-1])])
        self.nb = int(self.blk[-1]) + 1

    def ranks(self, w):
        W = np.bincount(self.blk, w[self.o], self.nb)
        before = np.cumsum(W) - W
        r = np.empty(len(w))
        r[self.o] = (before + (W + 1) / 2)[self.blk]
        return r


class Spearman:
    def __init__(self, score, y):
        self.rx, self.ry = _Ranker(score), _Ranker(y)

    def __call__(self, w):
        return wpearson(self.rx.ranks(w), self.ry.ranks(w), w)


class SpeakerAUC:
    """AUC over speakers: each speaker's mean item score against its (constant) label."""

    def __init__(self, score, h, spk):
        u, inv = np.unique(spk, return_inverse=True)
        s = np.asarray(score, float)
        mean = np.bincount(inv, s) / np.bincount(inv)
        lab = np.zeros(len(u))
        lab[inv] = np.asarray(h, float)
        self.first = np.array([np.flatnonzero(inv == k)[0] for k in range(len(u))])
        self.part = AUCPart(mean, lab == 1, lab == 0)

    def __call__(self, w):
        a, b = self.part.numer_pairs(w[self.first])
        return a / b if b else float("nan")


class SpeakerSpearman:
    """Spearman over speakers of the speaker's mean score with a per-speaker value."""

    def __init__(self, score, value, spk):
        u, inv = np.unique(spk, return_inverse=True)
        s = np.asarray(score, float)
        self.mean = np.bincount(inv, s) / np.bincount(inv)
        v = np.full(len(u), np.nan)
        v[inv] = np.asarray(value, float)
        self.ok = np.isfinite(v)
        self.first = np.array([np.flatnonzero(inv == k)[0] for k in range(len(u))])[self.ok]
        self.sp = Spearman(self.mean[self.ok], v[self.ok])

    def __call__(self, w):
        return self.sp(w[self.first])


class SpeakerMeanDiff:
    """Mean over speakers of the speaker's mean item value (item value = later - first on one sentence)."""

    def __init__(self, diff, spk):
        u, inv = np.unique(spk, return_inverse=True)
        self.mean = np.bincount(inv, np.asarray(diff, float)) / np.bincount(inv)
        self.first = np.array([np.flatnonzero(inv == k)[0] for k in range(len(u))])

    def __call__(self, w):
        ww = w[self.first]
        return float((ww * self.mean).sum() / ww.sum()) if ww.sum() else float("nan")


METRICS = {"C": Concordance, "AUC": Concordance, "macro_AUC": MacroAUC, "Pearson": Pearson,
           "Spearman": Spearman, "speaker_AUC": SpeakerAUC, "speaker_Spearman": SpeakerSpearman,
           "speaker_mean_diff": SpeakerMeanDiff}


# ============================================================================ rows

class Row:
    """One metric on one test: the label, the speakers, and each system's score vector (or a reason)."""

    def __init__(self, test, lang, label_type, metric, unit, spk, build, scores: dict, note="", n_pos=None):
        self.test, self.lang, self.label_type, self.metric, self.unit = test, lang, label_type, metric, unit
        self.spk = np.asarray(spk)
        self.build = build                 # score vector -> callable(weights) -> value
        self.scores = scores               # system -> np.ndarray | str (reason: not supported / not run / ...)
        self.note, self.n_pos = note, n_pos


def p_two(d) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def q(v) -> list:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [float("nan")] * 2


def evaluate_row(row: Row, n_boot: int, seed: int) -> dict:
    u, inv = np.unique(row.spk, return_inverse=True)
    K = len(u)
    fns = {s: row.build(v) for s, v in row.scores.items() if isinstance(v, np.ndarray)}
    one = np.ones(len(row.spk))
    point = {s: f(one) for s, f in fns.items()}
    rng = np.random.default_rng(seed)
    boots = {s: np.empty(n_boot) for s in fns}
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
        w = cnt[inv]
        for s, f in fns.items():
            boots[s][b] = f(w)
    out = {"test": row.test, "lang": row.lang, "label_type": row.label_type, "metric": row.metric,
           "unit": row.unit, "n_items": int(len(row.spk)), "n_speakers": int(K), "note": row.note,
           "n_pos": row.n_pos, "systems": {}}
    for s, v in row.scores.items():
        if s not in fns:
            out["systems"][s] = {"status": v}
            continue
        e = {"value": point[s], "ci": q(boots[s])}
        if s != BASE and BASE in fns:
            d = boots[BASE] - boots[s]
            e["base_minus"] = {"value": point[BASE] - point[s], "ci": q(d), "p": p_two(d)}
        out["systems"][s] = e
    return out


# ============================================================================ system outputs

def floor_vec(vals, floor=0.0):
    return np.array([floor if v is None or not np.isfinite(v) else v for v in vals], float)


class Sys:
    """Uniform access to a system's per-item outputs."""

    def __init__(self, name, cat, langs, sentence: dict, en_detail=None, zh=None, failed=None, missing_reason="not run"):
        self.name, self.cat, self.langs = name, cat, langs
        self.sentence = sentence           # id -> sentence score (None = failed -> floor)
        self.en = en_detail or {}          # id -> {"phone": [...ref phones], "word": [...ref words], "aspects": {...}}
        self.zh = zh or {}                 # id -> bench.eval_ompal.summarize-shaped output
        self.failed = failed or Counter()
        self.missing_reason = missing_reason


def pronunciada_sys(name: str) -> Sys | None:
    from bench.eval_english import blend

    O = load_outputs(name)
    if not O:
        return None
    sent, en, zh = {}, {}, {}
    for rid, o in O.items():
        if o["status"] != "Success":
            sent[rid] = None
            en[rid] = None
            zh[rid] = {"status": "error"}
            continue
        sent[rid] = o["sentence"]["total"] if o["sentence"]["total"] is not None else o["prob"]
        if "api" in o:
            a = o["api"]
            d = a["sound"][o["sound_name"]]
            en[rid] = {"phones": a["phones"], "phone": np.asarray(d["reg"], float),
                       "phone_err": 1 - blend(d["z_len"], d["z_str"], 0.5), "word": np.asarray(a["word_reg"], float),
                       "aspects": {k: a["sentence"][k] for k in ("accuracy", "fluency", "prosodic", "total")}}
        if "zh" in o:
            zh[rid] = o["zh"]
    return Sys(name, "A", ("en", "zh", "es", "de"), sent, en, zh)


def zipa_sys(recs: list[dict]) -> Sys | None:
    """Reference row: ZIPA alone, bench.eval_zipa_baseline's "gop" scoring (bench.learnerbench.run_zipa)."""
    import math as _m

    from bench import eval_ompal as EO

    d = work() / "outputs" / "zipa-alone"
    if not d.exists():
        return None
    FLOOR = -60.0
    sent, en, zh, failed = {}, {}, {}, Counter()
    for r in recs:
        p = d / f"{r['id'].replace('/', '_')}.json"
        if not p.exists():
            continue
        o = json.loads(p.read_text())
        if "error" in o:
            failed[r["labels"]["corpus"]] += 1
            sent[r["id"]] = FLOOR
            en[r["id"]] = None
            zh[r["id"]] = {"status": "Success", "words": [], "syllables": [], "n_words_unplaced": 0,
                           "sentence": {"prob": FLOOR, "accuracy": FLOOR, "fluency": FLOOR, "prosodic": FLOOR}}
            continue
        sent[r["id"]] = float(o["gop_s"])
        if r["lang"] == "zh":
            spans = EO.word_char_spans(r["text"], o["word_text"])
            zh[r["id"]] = {"status": "Success",
                           "words": [[sp[0], sp[1], _m.exp(g), None, None] for sp, g in zip(spans, o["gop_w"]) if sp],
                           "syllables": [], "n_words_unplaced": sum(sp is None for sp in spans),
                           "sentence": {"prob": o["gop_s"], "accuracy": o["gop_s"], "fluency": o["gop_s"],
                                        "prosodic": o["gop_s"]}}
        else:
            g = np.asarray(o["gop"], float)
            en[r["id"]] = {"phones": o["phones"], "phone": g, "phone_err": -g, "word": np.asarray(o["gop_w"], float),
                           "aspects": {k: float(o["gop_s"]) for k in ("accuracy", "fluency", "prosodic", "total")}}
    return Sys("zipa-alone", "R", ("en", "zh", "es", "de"), sent, en, zh, failed=failed)


def _op_raw(mode: str, ds: str) -> dict:
    from bench.rivals.openpronounce import read_raw

    import bench.rivals.openpronounce as OPM

    out = {}
    roots = [Path.home() / ".cache" / "pronunciada" / "round19" / "rivals", work() / "openpronounce"]
    for r in roots:
        OPM.RAW = r / "raw"
        out.update(read_raw(mode, ds))
    return out


def openpronounce_sys(student: bool, recs_by_corpus: dict) -> Sys:
    from bench.pipeline.common import build_ref
    from bench.rivals.convert import to_reference

    name = "openpronounce-1.1.0-student" if student else "openpronounce-1.1.0"
    mode = "student" if student else "default"
    sent, en, failed = {}, {}, Counter()
    sources = {"speechocean762": mode, "l2arctic": "student" if student else "fast",
               "epadb": "student" if student else "fast"}
    corpora = ["saa", "leap-en", "leap-en-prepost"] + ([] if student else ["heroico", "leap-de", "leap-de-prepost"])
    for c in corpora:
        sources[c] = mode
    for ds, m in sources.items():
        raw = _op_raw(m, ds)
        for rec in recs_by_corpus.get(ds, []):
            r = raw.get(rec["id"])
            if r is None:
                continue
            o = r["out"]
            if "error" in o or o.get("score") is None:
                sent[rec["id"]] = None
                failed[ds] += 1
                continue
            sent[rec["id"]] = float(o["score"])
            if ds in ("speechocean762", "l2arctic", "epadb"):
                ref = build_ref("en", rec["text"])
                w1, _, ph, _, _ = to_reference(ref, r["text"], o)
                en[rec["id"]] = {"phones": [p.ipa for p in ref.phones], "phone": np.asarray(ph, float),
                                 "phone_err": 100.0 - np.asarray(ph, float), "word": np.asarray(w1, float),
                                 "aspects": {k: o["scores"].get(k) for k in ("accuracy", "fluency", "prosodic", "total")}}
    langs = ("en",) if student else ("en", "es", "de")
    return Sys(name, "A", langs, sent, en, failed=failed)


def openpronounce040_sys(recs_by_corpus: dict) -> Sys:
    """OpenPronounce 0.4.0 (bench.rivals.openpronounce_040): one sentence score (`score`), a word score
    100 x (1 - confidence) for the words it flags (100 for the others), no sound scores (each sound takes
    its word's score), as in the 18 November 2026 snapshot. Words are attached to reference words by
    character span (mean when several overlap; a reference word overlapped by none gets the mean)."""
    import re as _re

    from bench.pipeline.common import build_ref
    from bench.rivals import openpronounce_040 as O4

    word_re = _re.compile(r"\b[\w']+\b")
    sent, en, failed = {}, {}, Counter()
    for ds in ("speechocean762", "l2arctic", "epadb", "heroico", "saa", "leap-en", "leap-de",
               "leap-en-prepost", "leap-de-prepost"):
        raw = O4.read_raw(ds)
        for rec in recs_by_corpus.get(ds, []):
            r = raw.get(rec["id"])
            if r is None:
                continue
            o = r["out"]
            if "error" in o or o.get("score") is None:
                sent[rec["id"]] = None
                failed[ds] += 1
                continue
            sent[rec["id"]] = float(o["score"])
            if ds in ("speechocean762", "l2arctic", "epadb"):
                ref = build_ref("en", rec["text"])
                spans = [(m.start(), m.end()) for m in word_re.finditer(r["text"])]
                flag = {int(e["position"]): float(e["confidence"]) for e in o.get("errors", [])}
                v = [100.0 * (1 - flag.get(i, 0.0)) for i in range(len(spans))]
                mean = sum(v) / len(v) if v else float(o["score"])
                words = []
                for w in ref.words:
                    a, b = w.char_span
                    hit = [v[i] for i, (s0, e0) in enumerate(spans) if s0 < b and e0 > a]
                    words.append(sum(hit) / len(hit) if hit else mean)
                ph = np.asarray([words[p.word] for p in ref.phones], float)
                sc = float(o["score"])
                en[rec["id"]] = {"phones": [p.ipa for p in ref.phones], "phone": ph, "phone_err": 100.0 - ph,
                                 "word": np.asarray(words, float),
                                 "aspects": {"total": sc, "accuracy": sc, "fluency": None, "prosodic": None}}
    return Sys("openpronounce-0.4.0", "A", ("en", "es", "de"), sent, en, failed=failed)


def azure_sys(recs_by_corpus: dict) -> Sys:
    from bench.learners.evaluate import azure_scores

    sent = {}
    for c in ("heroico", "saa", "speechocean762", "ompal"):
        for rid, o in azure_scores(recs_by_corpus.get(c, [])).items():
            sent[rid] = o["score"] if o["status"] == "Success" else None
    return Sys("azure", "C", ("en", "zh", "es", "de"), sent)


def lingolix_sys(recs_by_corpus: dict) -> tuple[Sys, dict]:
    from bench.third_party import lingolix as L
    from bench.third_party import lingolix_live as LL

    root = LL.live_root(False)
    jobs = LL.tier2x_jobs()
    R, prov = LL.lingolix_records(jobs, root)
    R = R.get(LL.TIER2X, {})
    sent, words = {}, {}
    for rec in recs_by_corpus.get("speechocean762", []):
        uid = rec["id"].removeprefix("so762-")
        r = R.get(uid)
        if r is None:
            continue
        if r.get("error"):
            sent[rec["id"]] = None
            continue
        target = [w["text"] for w in rec["labels"]["words"]]
        al = L.align(target, r["words"])
        sc = [e["score"] for e in al if e is not None]
        mean = float(np.mean(sc)) if sc else 0.0
        words[rec["id"]] = np.array([e["score"] if e is not None else mean for e in al], float)
        u = L.utterance_score(r, "benchmark")
        sent[rec["id"]] = None if u is None else float(u)
        words[rec["id"] + "#acc"] = r["raw"].get("accuracy")
    # Heroico / USMA test (600 items), measured 2026-11-23 by the coordinator's run; reduced per-item scores
    hp = work().parent / "third_party" / "lingolix_learner_live" / "per_item.jsonl"
    if hp.exists():
        n = 0
        for line in hp.read_text().splitlines():
            o = json.loads(line)
            sent[o["id"]] = None if o.get("sentence_score") is None else float(o["sentence_score"])
            n += 1
        prov["heroico"] = {"items": n, "file": "third_party/lingolix_learner_live/per_item.jsonl",
                           "date": "2026-11-23", "model_version": "2026-11"}
    s = Sys("lingolix", "C", ("en", "es", "de"), sent)
    s.words = words
    return s, prov


# ============================================================================ tests

def so762_rows(S: dict, recs: list[dict], sample: set | None = None, chivox: dict | None = None,
               vendor: str = "chivox") -> list[Row]:
    """speechocean762 test: the scoreboard's units (bench.eval_zipa_baseline.eval_so762), every system."""
    from bench.eval_english import az_root, so762_items
    import bench.eval_english as EE
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference, arpa_to_ours

    base = S[BASE]
    P = {"y": [], "spk": [], "plan": [], "key": []}
    Wd = {"y": [], "spk": [], "plan": [], "utt": [], "j": []}
    St = {"spk": [], "utt": [], **{f"y_{k}": [] for k in ("accuracy", "fluency", "prosodic", "total")}}
    for rec in recs:
        e = base.en.get(rec["id"])
        if e is None:
            continue                               # scoreboard rule: an utterance ours failed on is excluded
        ref = build_ref("en", rec["text"])
        assert [p.ipa for p in ref.phones] == e["phones"], rec["id"]
        W = rec["labels"]["words"]
        tp = [[arpa_to_ours(p) for p in w["phones"]] for w in W]
        o2t, wpairs = align_reference(ref, [w["text"] for w in W], tp)
        t2o = {v: k for k, v in o2t.items()}
        tw = {}
        for i, (j, _) in o2t.items():
            tw.setdefault(j, set()).add(ref.phones[i].word)
        for i, j in wpairs:
            tw.setdefault(j, set()).add(i)
        n_ph, n_w = len(ref.phones), len(ref.words)
        for j, w in enumerate(W):
            ow = sorted(tw.get(j, set()))
            fb = [k for x in ow for k in range(*ref.words[x].phones)]
            for k, s in enumerate(w["phones_accuracy"]):
                i = t2o.get((j, k), -1)
                P["plan"].append((rec["id"], [i] if i >= 0 else (fb or list(range(n_ph)))))
                P["key"].append((rec["id"], j, k))
                P["y"].append(float(s))
                P["spk"].append(rec["speaker"])
            Wd["plan"].append((rec["id"], ow or list(range(n_w))))
            Wd["y"].append(float(w["accuracy"]))
            Wd["spk"].append(rec["speaker"])
            Wd["utt"].append(rec["id"])
            Wd["j"].append(j)
        for k in ("accuracy", "fluency", "prosodic", "total"):
            St[f"y_{k}"].append(float(rec["labels"]["sentence"][k]))
        St["spk"].append(rec["speaker"])
        St["utt"].append(rec["id"])

    def vec(sys_: Sys, plan, key, floor):
        if "en" not in sys_.langs:
            return "not supported"
        out = []
        n_ok = 0
        for rid, ix in plan:
            e = sys_.en.get(rid)
            if e is None:
                out.append(floor)
                continue
            v = e[key]
            out.append(float(np.mean(v[ix])))
            n_ok += 1
        if n_ok == 0:
            return "not run"
        return np.array(out, float)

    rows = []
    sound, word, aspects = {}, {}, {k: {} for k in ("accuracy", "fluency", "prosodic", "total")}
    for name, s in S.items():
        if s.cat in ("A", "R"):
            fl = -60.0 if s.cat == "R" else 0.0
            sound[name] = vec(s, P["plan"], "phone", fl)
            word[name] = vec(s, Wd["plan"], "word", fl)
            for k in aspects:
                if "en" not in s.langs:
                    aspects[k][name] = "not supported"
                    continue
                vals = [None if s.en.get(u) is None else s.en[u]["aspects"].get(k) for u in St["utt"]]
                aspects[k][name] = floor_vec(vals, fl) if any(v is not None for v in vals) else "not run"
    # Azure, from the scoreboard's own extraction (same units, same order; checked below)
    EE.SETS = ("B",)
    api = {}
    for rec in recs:
        e = base.en.get(rec["id"])
        if e is not None:
            api[rec["id"]] = {"phones": e["phones"], "sound": {"B": {"reg": e["phone"], "z_len": e["phone"], "z_str": e["phone"]},
                                                               "U": {"reg": e["phone"], "z_len": e["phone"], "z_str": e["phone"]}},
                              "word_reg": e["word"], "word_z": (e["word"], e["word"]),
                              "sentence": e["aspects"], "sentence_z": (0.0, 0.0)}
    AP, AW, AS, _, n_missing = so762_items(api)
    assert n_missing == 0 and AP["y"] == P["y"] and AW["y"] == Wd["y"], "Azure units differ from the scoreboard's"
    sound["azure"] = np.asarray(AP["az"], float)
    word["azure"] = np.asarray(AW["az"], float)
    for k in aspects:
        aspects[k]["azure"] = np.asarray(AS[f"az_{k}"], float)
    # Lingolix: word and sentence only
    lx = S.get("lingolix")
    if lx is not None:
        sound["lingolix"] = "not given (syllable level only)"
        lw = []
        for u, j in zip(Wd["utt"], Wd["j"]):
            v = lx.words.get(u)
            lw.append(0.0 if v is None else float(v[j]))
        word["lingolix"] = np.array(lw)
        aspects["total"]["lingolix"] = floor_vec([lx.sentence.get(u) for u in St["utt"]])
        aspects["accuracy"]["lingolix"] = floor_vec([lx.words.get(u + "#acc") for u in St["utt"]])
        aspects["fluency"]["lingolix"] = aspects["prosodic"]["lingolix"] = "not given"
    T = "speechocean762 test"
    if sample is not None:
        T = f"speechocean762 test, {VENDOR_NAME[vendor]} sample ({len(sample)} recordings)"
        mP = np.array([k[0] in sample for k in P["key"]])
        mW = np.array([u in sample for u in Wd["utt"]])
        mS = np.array([u in sample for u in St["utt"]])

        def sub(d, m):
            return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in d.items()}

        sound, word = sub(sound, mP), sub(word, mW)
        aspects = {k: sub(v, mS) for k, v in aspects.items()}
        P_keys = [k for k, m in zip(P["key"], mP) if m]
        P = {"y": list(np.asarray(P["y"])[mP]), "spk": list(np.asarray(P["spk"])[mP])}
        Wd = {"y": list(np.asarray(Wd["y"])[mW]), "spk": list(np.asarray(Wd["spk"])[mW]),
              "key": [(u, j) for u, j, m in zip(Wd["utt"], Wd["j"], mW) if m]}
        St = {"spk": list(np.asarray(St["spk"])[mS]), "utt": [u for u, m in zip(St["utt"], mS) if m],
              **{f"y_{k}": list(np.asarray(St[f"y_{k}"])[mS]) for k in ("accuracy", "fluency", "prosodic", "total")}}
        if chivox is not None:
            from bench.third_party import live_eval as LE

            views = chivox["views"]
            W, Ph, _ = LE.vendor_so762_rows(views, LE.so762_human())
            if vendor == "chivox":
                sound[vendor] = "— (the English sentence engine returns no sound scores)"
            else:
                sound[vendor] = np.array([Ph.get(k, 0.0) for k in P_keys], float)
            word[vendor] = np.array([W.get(k, 0.0) for k in Wd["key"]], float)
            for k in aspects:
                vals = [(views[u].get("sentence") or {}).get(k) for u in St["utt"]]
                aspects[k][vendor] = floor_vec(vals) if all(v is not None for v in vals) else "not given"
    rows.append(Row(T, "en", "expert scores", "Pearson", "sound", P["spk"], lambda v: Pearson(v, P["y"]), sound,
                    note="expert phone accuracy 0-2"))
    rows[-1].y = P["y"]
    rows.append(Row(T, "en", "expert scores", "Spearman", "sound", P["spk"], lambda v: Spearman(v, P["y"]), sound))
    rows.append(Row(T, "en", "expert scores", "Pearson", "word", Wd["spk"], lambda v: Pearson(v, Wd["y"]), word,
                    note="expert word accuracy 0-10"))
    rows[-1].y = Wd["y"]
    rows.append(Row(T, "en", "expert scores", "Spearman", "word", Wd["spk"], lambda v: Spearman(v, Wd["y"]), word))
    for k in ("total", "accuracy", "fluency", "prosodic"):
        y = St[f"y_{k}"]
        rows.append(Row(T, "en", "expert scores", "Pearson", f"sentence {k}", St["spk"],
                        (lambda yy: lambda v: Pearson(v, yy))(y), aspects[k]))
        rows[-1].y = y
        rows.append(Row(T, "en", "expert scores", "Spearman", f"sentence {k}", St["spk"],
                        (lambda yy: lambda v: Spearman(v, yy))(y), aspects[k]))
    return rows


def errdet_rows(S: dict, recs_by: dict) -> list[Row]:
    """L2-ARCTIC / EpaDB test: AUC of each system's error signal for the sounds annotators marked."""
    from bench.eval_english import _their_entries, err_items
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference

    rows = []
    base = S[BASE]
    for ds, label in (("l2arctic", "L2-ARCTIC test"), ("epadb", "EpaDB test")):
        plan, y, spk = [], [], []
        for rec in recs_by[ds]:
            e = base.en.get(rec["id"])
            if e is None:
                continue
            ref = build_ref("en", rec["text"])
            ents = _their_entries(rec, ds)
            if ds == "l2arctic":
                words = [w["text"] for w in rec["labels"]["words"]]
                tp = [[x["ph"] for x in ents if x["word"] == j] for j in range(len(words))]
                pos = {}
                for k, x in enumerate(ents):
                    pos.setdefault(x["word"], []).append(k)
                o2t, _ = align_reference(ref, words, tp)
                t2o = {pos[j][qq]: i for i, (j, qq) in o2t.items() if j in pos and qq < len(pos[j])}
            else:
                o2t, _ = align_reference(ref, None, [[x["ph"] for x in ents]])
                t2o = {qq: i for i, (_, qq) in o2t.items()}
            matched = [t2o.get(k, -1) for k in range(len(ents))]
            for k, x in enumerate(ents):
                if x["label"] == "unk":
                    continue
                i = matched[k]
                if i >= 0:
                    ix = [i]
                else:
                    nb = [matched[qq] for qq in (k - 1, k + 1) if 0 <= qq < len(ents) and matched[qq] >= 0]
                    ix = nb or list(range(len(ref.phones)))
                plan.append((rec["id"], ix))
                y.append(int(x["label"] != "correct"))
                spk.append(rec["speaker"])
        y = np.asarray(y)
        scores = {}
        for name, s in S.items():
            if s.cat not in ("A", "R"):
                continue
            if "en" not in s.langs:
                scores[name] = "not supported"
                continue
            vals, n_ok = [], 0
            for rid, ix in plan:
                e = s.en.get(rid)
                if e is None:     # failed: the lowest score = the strongest error signal
                    vals.append(1.0 if name.startswith("pronunciada") else 60.0 if s.cat == "R" else 100.0)
                    continue
                vals.append(float(np.mean(e["phone_err"][ix])))
                n_ok += 1
            scores[name] = np.array(vals) if n_ok else "not run"
        # Azure: the scoreboard's extraction (same iteration and units)
        api = {r["id"]: {"sound": {"B": {"z_len": np.zeros(len(base.en[r["id"]]["phones"])),
                                         "z_str": np.zeros(len(base.en[r["id"]]["phones"]))}}}
               for r in recs_by[ds] if base.en.get(r["id"]) is not None}
        try:
            arows, _ = err_items(ds, api)
            assert len(arows) == len(y) and [int(r["label"] != "correct") for r in arows] == y.tolist()
            scores["azure"] = np.array([r["azure"] for r in arows], float)
        except FileNotFoundError:
            scores["azure"] = "not run"
        scores["lingolix"] = "not run (no cached responses)"
        pos, neg = y == 1, y == 0
        f1 = {k: (v if k.startswith("pronunciada") or isinstance(v, str) else
                  "not defined (no decision rule in the recipe)" if k == "zipa-alone" else
                  "no sound-level decision in its output") for k, v in scores.items()}
        rows.append(Row(label, "en", "marked errors", "F1", "sound, own operating point", spk,
                        (lambda p_: lambda v: F1At(v, p_, 0.5))(pos), f1,
                        note="Pronunciada: a sound is flagged when its calibrated probability of being acceptable "
                             "(PhoneResult.prob, default strictness) is below 0.5", n_pos=int(pos.sum())))
        rows.append(Row(label, "en", "marked errors", "AUC", "sound", spk,
                        (lambda p_, n_: lambda v: _AUCWrap(v, p_, n_))(pos, neg), scores,
                        note="annotator-marked mispronounced sounds; error signal, higher = more likely wrong",
                        n_pos=int(pos.sum())))
    return rows


class F1At:
    """F1 of the decision `error signal > thr` against the marked label (weighted counts)."""

    def __init__(self, v, pos, thr):
        self.flag, self.pos = np.asarray(v, float) > thr, np.asarray(pos, bool)

    def __call__(self, w):
        tp = (w * (self.flag & self.pos)).sum()
        fp = (w * (self.flag & ~self.pos)).sum()
        fn = (w * (~self.flag & self.pos)).sum()
        return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan")


class _AUCWrap:
    def __init__(self, v, pos, neg):
        self.part = AUCPart(v, pos, neg)

    def __call__(self, w):
        a, b = self.part.numer_pairs(w)
        return a / b if b else float("nan")


def ompal_rows(S: dict, sample: set | None = None, chivox: dict | None = None, vendor: str = "chivox") -> list[Row]:
    from bench import eval_ompal as EO

    recs = EO.records("test")
    if sample is not None:
        recs = [r for r in recs if r["id"] in sample]
    az = EO.azure_outputs(recs, "zh-TW")
    azs = {"zh-TW": az}
    if chivox is not None:
        from bench.third_party import live_eval as LE

        azs["CV"] = LE._as_az(chivox["views"], 2)
        cvs = {i: {"status": "Success", "sentence": {"AccuracyScore": v["sentence"].get("accuracy"),
                                                     "FluencyScore": v["sentence"].get("fluency"),
                                                     "PronScore": v["sentence"].get("total")}}
               for i, v in chivox["views"].items()}
    seg, flu = {}, {}
    rows = []
    word, sent = {}, {}
    U0 = S0 = None
    for name, s in S.items():
        if "zh" not in s.langs and s.cat == "A":
            word[name] = sent[name] = "not supported"
            continue
        if not (s.name.startswith("pronunciada") or s.name == "zipa-alone"):
            continue
        ours = {r["id"]: (s.zh.get(r["id"]) or {"status": "missing"}) for r in recs}
        U = EO.unit_table(recs, ours, azs)["cols"]
        St = EO.sent_table(recs, ours, {"zh-TW": az, **({"CV": cvs} if chivox is not None else {})})
        if U0 is None:
            U0, S0 = U, St
        word[name] = np.asarray(U["ours_word"], float)
        keep = ~St["native"].astype(bool)
        sent[name] = np.asarray(St["ours_accuracy"], float)[keep]
        flu[name] = np.asarray(St["ours_fluency"], float)[keep]
    word["azure"] = np.asarray(U0["az_zh-TW"], float)
    keep = ~S0["native"].astype(bool)
    sent["azure"] = np.asarray(S0["az_zh-TW_AccuracyScore"], float)[keep]
    flu["azure"] = np.asarray(S0["az_zh-TW_FluencyScore"], float)[keep]
    for k in word:
        if isinstance(word[k], str):
            flu.setdefault(k, word[k])
    if chivox is not None:
        word[vendor] = np.asarray(U0["az_CV"], float)
        sent[vendor] = np.asarray(S0["az_CV_AccuracyScore"], float)[keep]
        given = any((v.get("sentence") or {}).get("fluency") is not None for v in chivox["views"].values())
        flu[vendor] = np.asarray(S0["az_CV_FluencyScore"], float)[keep] if given else "not given"
    for nm in ("lingolix",):
        word[nm] = sent[nm] = "not supported (no Mandarin)"
    pos, neg = U0["any_wrong"].astype(bool), U0["correct"].astype(bool)
    T = "OMPAL test (Mandarin, French L1)"
    if sample is not None:
        T = f"OMPAL test, {VENDOR_NAME[vendor]} sample ({len(recs)} recordings)"
    rows.append(Row(T, "zh", "teacher marks", "AUC", "word (character): any error", U0["spk"],
                    lambda v: _AUCWrap(-v, pos, neg), word, note="all 21 speakers, natives included (as results/ompal.md)",
                    n_pos=int(pos.sum())))
    if sample is not None:
        ps = U0["seg_wrong"].astype(bool)
        rows.append(Row(T, "zh", "teacher marks", "AUC", "word (character): consonant or vowel error", U0["spk"],
                        lambda v: _AUCWrap(-v, ps, neg), word, n_pos=int(ps.sum())))
        pt = U0["tone_only"].astype(bool)
        rows.append(Row(T, "zh", "teacher marks", "AUC", "word (character): tone-only error", U0["spk"],
                        lambda v: _AUCWrap(-v, pt, neg), word, n_pos=int(pt.sum())))
    y = S0["t_accuracy"].astype(float)[keep]
    spk = S0["spk"][keep]
    rows.append(Row(T, "zh", "teacher scores", "Pearson", "sentence accuracy", spk, lambda v: Pearson(v, y), sent,
                    note="learners only; teachers' mean sentence accuracy"))
    rows[-1].y = y
    rows.append(Row(T, "zh", "teacher scores", "Spearman", "sentence accuracy", spk, lambda v: Spearman(v, y), sent))
    if sample is not None:
        yf = S0["t_fluency"].astype(float)[keep]
        rows.append(Row(T, "zh", "teacher scores", "Pearson", "sentence fluency", spk, lambda v: Pearson(v, yf), flu))
    return rows


VENDOR_NAME = {"chivox": "Chivox", "toneperfect": "TonePerfect"}


def toneperfect_answers() -> dict:
    """TonePerfect's cached answers (speechocean762 sample and OMPAL test), read only."""
    from bench.third_party import live_core as LC
    from bench.third_party import toneperfect_live as TP

    items = LC.select_items(TP, ["speechocean762", "ompal"], pilot=False, limit=None, seed=0)
    A = LC.collect_answers(TP, items, LC.live_root(TP, False))

    def key(ds, k):
        return f"so762-{k}" if ds == "speechocean762" and not k.startswith("so762-") else k

    return {ds: {"views": {key(ds, k): v for k, v in A[ds]["views"].items()}, "versions": A[ds].get("versions"),
                 "first_utc": A[ds].get("first_utc"), "last_utc": A[ds].get("last_utc")} for ds in A}


def chivox_sample() -> dict:
    """Chivox's cached pilot answers (299 speechocean762 + 299 OMPAL test items), read only."""
    from bench.third_party import chivox_live as CV
    from bench.third_party import live_core as LC

    items = LC.select_items(CV, ["speechocean762", "ompal"], pilot=False, limit=299, seed=0)
    A = LC.collect_answers(CV, items, LC.live_root(CV, False))
    def key(ds, k):
        return f"so762-{k}" if ds == "speechocean762" and not k.startswith("so762-") else k

    return {ds: {"views": {key(ds, k): v for k, v in A[ds]["views"].items()}, "versions": A[ds].get("versions"), "first_utc": A[ds].get("first_utc"),
                 "last_utc": A[ds].get("last_utc")} for ds in A}


def suite_rows(S: dict, recs4: list[dict], recs_by: dict) -> tuple[list[Row], dict]:
    """The learner-suite tests of bench.learners.evaluate (definitions fixed there), every system."""
    from bench.learners import evaluate as E

    T = E.build_tests(recs4)
    rows = []
    vecs = {}

    def sysvec(s: Sys, t):
        if t.lang not in s.langs:
            return "not supported"
        vals = [s.sentence.get(r["id"], "missing") for r in t.recs]
        if all(v == "missing" for v in vals):
            return "not run"
        if any(v == "missing" for v in vals):
            return f"incomplete ({sum(v == 'missing' for v in vals)} of {len(vals)} items missing)"
        return floor_vec(vals, -60.0 if s.cat == "R" else 0.0)

    for key, t in T.items():
        sc = {name: sysvec(s, t) for name, s in S.items()}
        if key == "en/saa_all":
            sc["azure"] = "not run (47 clips over Azure's 30 s limit)"
        vecs[key] = sc
        strata = t.strata
        h = t.h
        spk = t.spk
        lt = {"expert": "expert scores", "native_vs_learner": "native vs learner", "proxy": "self-report",
              "proficiency": "corpus group", "proxy_onset": "corpus metadata"}[t.kind]
        if t.kind in ("expert", "native_vs_learner", "proficiency"):
            rows.append(Row(t.name, t.lang, lt, "C" if t.kind == "expert" else "AUC", key, spk,
                            (lambda hh, ss: lambda v: Concordance(v, hh, ss))(h, strata), sc))
        if t.kind == "native_vs_learner":
            rows.append(Row(t.name, t.lang, lt, "speaker_AUC", key, spk,
                            (lambda hh, sp: lambda v: SpeakerAUC(v, hh, sp))(h, spk), sc))
            if strata is not None and key.startswith(("de/leap", "en/leap")):
                rows.append(Row(t.name, t.lang, lt, "macro_AUC", key, spk,
                                (lambda hh, ss: lambda v: MacroAUC(v, hh, ss))(h, strata), sc))
            dur = -np.array([r["dur"] for r in t.recs])
            rows.append(Row(t.name, t.lang, "baseline", "AUC", key + " (-duration)", spk,
                            (lambda hh, ss: lambda v: Concordance(v, hh, ss))(h, strata), {"-duration": dur}))
        if t.kind == "proxy_onset":
            onset = [r["labels"].get("age_of_onset") for r in t.recs]
            onset = np.array([np.nan if v is None else v for v in onset], float)
            rows.append(Row(t.name, t.lang, lt, "speaker_Spearman", key, spk,
                            (lambda val, sp: lambda v: SpeakerSpearman(v, val, sp))(onset, spk), sc,
                            note=f"expected negative; {int(np.isfinite(onset).sum())} items with an age of onset"))
    # LeaP later vs first reading (pre-declared secondary)
    for lang in ("de", "en"):
        pp = recs_by.get(f"leap-{lang}-prepost", [])
        first = {(r["speaker"], r["labels"]["prompt_id"]): r for r in recs_by.get(f"leap-{lang}", [])}
        pairs = [(first[(r["speaker"], r["labels"]["prompt_id"])], r) for r in pp
                 if (r["speaker"], r["labels"]["prompt_id"]) in first]
        if not pairs:
            continue
        sc = {}
        for name, s in S.items():
            if lang not in s.langs:
                sc[name] = "not supported"
                continue
            a = [s.sentence.get(p[0]["id"], "missing") for p in pairs]
            b = [s.sentence.get(p[1]["id"], "missing") for p in pairs]
            if any(x == "missing" for x in a + b):
                sc[name] = "not run"
                continue
            sc[name] = floor_vec(b) - floor_vec(a)
        spk = [p[0]["speaker"] for p in pairs]
        rows.append(Row(f"LeaP {lang}: later reading minus first reading, same sentences", lang, "corpus design",
                        "speaker_mean_diff", f"{lang}/leap_prepost", spk, lambda v, sp=spk: SpeakerMeanDiff(v, sp), sc,
                        note="no expected sign (the courses taught prosody)"))
    return rows, {"tests": T, "vecs": vecs}


# ============================================================================ summaries

def summary(T: dict, vecs: dict, picks: dict, systems: list[str], n_boot: int, seed: int) -> dict:
    """Unweighted mean over languages of each pick's C; speakers resampled within language."""
    names = [picks[k] for k in sorted(picks)]
    ok = [s for s in systems if all(isinstance(vecs[n].get(s), np.ndarray) for n in names)]
    fns = {s: [Concordance(vecs[n][s], T[n].h, T[n].strata) for n in names] for s in ok}
    invs = [np.unique(T[n].spk, return_inverse=True)[1] for n in names]
    Ks = [inv.max() + 1 for inv in invs]
    point = {s: float(np.mean([f(np.ones(len(T[n].h))) for f, n in zip(fns[s], names)])) for s in ok}
    rng = np.random.default_rng(seed)
    B = {s: np.empty(n_boot) for s in ok}
    for b in range(n_boot):
        ws = [np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[inv] for K, inv in zip(Ks, invs)]
        for s in ok:
            B[s][b] = np.mean([f(w) for f, w in zip(fns[s], ws)])
    out = {"languages": sorted(picks), "tests": names, "speakers": int(sum(Ks)),
           "items": int(sum(len(T[n].h) for n in names)), "systems": {}}
    for s in systems:
        if s not in ok:
            miss = [n for n in names if not isinstance(vecs[n].get(s), np.ndarray)]
            out["systems"][s] = {"status": "; ".join(f"{n}: {vecs[n].get(s)}" for n in miss)}
            continue
        e = {"value": point[s], "ci": q(B[s])}
        if s != BASE and BASE in ok:
            d = B[BASE] - B[s]
            e["base_minus"] = {"value": point[BASE] - point[s], "ci": q(d), "p": p_two(d)}
        out["systems"][s] = e
    return out


# ============================================================================ combined score

COMBINED_NOTE = ("Combined score: the unweighted mean over languages of one concordance C per language (0.5 = "
                 "chance, 1 = perfect agreement): en speechocean762 test, the system's sentence accuracy score "
                 "against the experts' sentence accuracy; zh OMPAL test learners, the same against the teachers' "
                 "mean accuracy; es Heroico / USMA and de LeaP, native against learner recordings of the same "
                 "prompt (the system's sentence score, the learner suite's primary tests). Speakers are resampled "
                 "within each language; a system is compared with Pronunciada fp32 on the languages both cover.")


def combined_components(rows: list, srows: list | None = None) -> dict:
    """lang -> Row of the concordance used by the combined score (built from the tagged rows)."""
    comp = {}
    for r in rows:
        if getattr(r, "y", None) is not None and r.unit == "sentence accuracy" and r.metric == "Pearson":
            y = np.asarray(r.y, float)
            comp[r.lang] = Row(r.test, r.lang, r.label_type, "C", "sentence accuracy", r.spk,
                               (lambda yy: lambda v: Concordance(v, yy))(y), r.scores,
                               note="combined-score component")
    for r in srows or []:
        if r.metric == "AUC" and r.unit in ("es/heroico", "de/leap"):
            comp[r.lang] = r
    return comp


def combined(comp: dict, systems: list[str], n_boot: int, seed: int, langs: tuple | None = None) -> dict:
    """Mean over languages of each component's C, every system against BASE on the languages both have
    (or on `langs` for every system that has all of them); speakers resampled within language, paired."""
    L = sorted(comp) if langs is None else list(langs)
    fns = {lg: {s: comp[lg].build(v) for s, v in comp[lg].scores.items() if isinstance(v, np.ndarray)} for lg in L}
    invs = {lg: np.unique(comp[lg].spk, return_inverse=True)[1] for lg in L}
    rng = np.random.default_rng(seed)
    W = {lg: [] for lg in L}
    for _ in range(n_boot):
        for lg in L:
            K = invs[lg].max() + 1
            W[lg].append(np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[invs[lg]])
    one = {lg: np.ones(len(comp[lg].spk)) for lg in L}
    out = {"languages": L, "note": COMBINED_NOTE, "per_language": {}, "systems": {}}
    for lg in L:
        out["per_language"][lg] = {"test": comp[lg].test, "items": int(len(comp[lg].spk)),
                                   "speakers": int(invs[lg].max() + 1),
                                   "values": {s: f(one[lg]) for s, f in fns[lg].items()}}
    for s in systems:
        shared = [lg for lg in L if s in fns[lg] and BASE in fns[lg]]
        if langs is not None and len(shared) < len(L):
            out["systems"][s] = {"status": "not every language: " + ", ".join(lg for lg in L if lg not in shared)}
            continue
        if not shared:
            out["systems"][s] = {"status": "no shared language"}
            continue
        val = lambda name, ws: float(np.mean([fns[lg][name](w) for lg, w in zip(shared, ws)]))
        ps, pb = val(s, [one[lg] for lg in shared]), val(BASE, [one[lg] for lg in shared])
        bs = np.array([val(s, [W[lg][b] for lg in shared]) for b in range(n_boot)])
        bb = bs if s == BASE else np.array([val(BASE, [W[lg][b] for lg in shared]) for b in range(n_boot)])
        e = {"languages": shared, "value": ps, "ci": q(bs), "base_value": pb}
        if s != BASE:
            d = bb - bs
            e["base_minus"] = {"value": pb - ps, "ci": q(d), "p": p_two(d)}
        out["systems"][s] = e
    return out


# ============================================================================ main

def collect_systems(recs_by: dict) -> tuple[dict, dict]:
    S, prov = {}, {}
    for name in ["pronunciada-fp32", "pronunciada-int8"] + [x for x in os.environ.get("LEARNERBENCH_EXTRA", "").split(",") if x]:
        s = pronunciada_sys(name)
        if s is not None:
            S[name] = s
    S["openpronounce-1.1.0"] = openpronounce_sys(False, recs_by)
    S["openpronounce-1.1.0-student"] = openpronounce_sys(True, recs_by)
    S["openpronounce-0.4.0"] = openpronounce040_sys(recs_by)
    S["azure"] = azure_sys(recs_by)
    S["lingolix"], prov["lingolix"] = lingolix_sys(recs_by)
    z = zipa_sys([r for v in recs_by.values() for r in v])
    if z is not None:
        S["zipa-alone"] = z
    return S, prov


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--only", default=None, help="comma list of row groups: so762,errdet,ompal,suite,summary")
    a = ap.parse_args(argv)
    from bench.learners import read_jsonl, root

    os.environ["PRONUNCIADA_LEARNER_SUITE"] = "suite4_test.jsonl"
    recs = items()
    recs4 = list(read_jsonl(root() / "suite4_test.jsonl"))
    recs_by = defaultdict(list)
    for r in recs:
        recs_by[r["labels"]["corpus"]].append(r)
    S, prov = collect_systems(recs_by)
    order = list(S)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "learner_benchmark_systems.json").write_text((Path(__file__).parent / "systems.json").read_text())
    groups = set((a.only or "so762,errdet,ompal,suite,summary").split(","))
    rows = []
    vend = "chivox" if "chivox" in groups else "toneperfect" if "toneperfect" in groups else None
    if vend:
        CVA = chivox_sample() if vend == "chivox" else toneperfect_answers()
        so = CVA["speechocean762"]["views"]
        om = CVA["ompal"]["views"]
        rows = so762_rows(S, recs_by["speechocean762"], sample=set(so), chivox=CVA["speechocean762"], vendor=vend)
        rows += ompal_rows(S, sample=set(om), chivox=CVA["ompal"], vendor=vend)
        res = {"n_boot": a.boot, "systems": order + [vend], "rows": [],
               vend: {k: {kk: vv for kk, vv in v.items() if kk != "views"} | {"answered": len(v["views"])}
                      for k, v in CVA.items()}}
        for k, row in enumerate(rows):
            res["rows"].append(evaluate_row(row, a.boot, seed=5000 + k))
            print(f"{row.test} | {row.metric} | {row.unit}: " + ", ".join(
                f"{s_}={v['value']:.3f}" for s_, v in res["rows"][-1]["systems"].items() if "value" in v), flush=True)
        comp = combined_components(rows)
        for k, lg in enumerate(sorted(comp)):
            res["rows"].append(evaluate_row(comp[lg], a.boot, seed=9500 + k))
        res["combined"] = combined(comp, order + [vend], a.boot, 23)
        tag = "" if vend == "chivox" else "_toneperfect"
        (TABLES / f"learner_benchmark_sample{tag}.json").write_text(json.dumps(_clean(res), indent=1, ensure_ascii=False))
        from bench.learnerbench.tables import write_markdown

        write_markdown(res, TABLES / f"learner_benchmark_sample{tag}_tables.md")
        return res
    if "so762" in groups:
        rows += so762_rows(S, recs_by["speechocean762"])
    if "errdet" in groups:
        rows += errdet_rows(S, recs_by)
    if "ompal" in groups:
        rows += ompal_rows(S)
    srows, SU = suite_rows(S, recs4, recs_by)
    if "suite" in groups:
        rows += srows
    res = {"n_boot": a.boot, "systems": order, "rows": []}
    for k, row in enumerate(rows):
        res["rows"].append(evaluate_row(row, a.boot, seed=1000 + k))
        print(f"{row.test} | {row.metric} | {row.unit}: " + ", ".join(
            f"{s}={v['value']:.3f}" for s, v in res["rows"][-1]["systems"].items() if "value" in v), flush=True)
    if "summary" in groups:
        from bench.learners.evaluate import NL, NL4, PRIMARY, PRIMARY4

        T, V = SU["tests"], SU["vecs"]
        res["summaries"] = {
            "LS4": summary(T, V, PRIMARY4, order, a.boot, 7),
            "LS-NL4": summary(T, V, NL4, order, a.boot, 8),
            "LS (en, es, zh; round 19)": summary(T, V, PRIMARY, order, a.boot, 9),
            "LS-NL (en, es, zh; round 19)": summary(T, V, NL, order, a.boot, 10),
            "LS restricted to en, es, de": summary(T, V, {k: PRIMARY4[k] for k in ("en", "es", "de")}, order, a.boot, 11),
            "LS-NL restricted to en, es, de": summary(T, V, {k: NL4[k] for k in ("en", "es", "de")}, order, a.boot, 12),
        }
    if "summary" in groups and {"so762", "ompal", "suite"} <= groups:
        comp = combined_components(rows, srows)
        for k, lg in enumerate(sorted(comp)):
            if comp[lg].metric == "C":            # the suite's own rows (es, de) are already in the table
                res["rows"].append(evaluate_row(comp[lg], a.boot, seed=9000 + k))
        res["combined"] = combined(comp, order, a.boot, 21)
        res["combined_en_es_de"] = combined(comp, order, a.boot, 22, langs=("de", "en", "es"))
    res["failures"] = {s: dict(S[s].failed) for s in order}
    res["provenance"] = {"lingolix": prov.get("lingolix")}
    TABLES.mkdir(parents=True, exist_ok=True)
    tag = "" if not a.only else "_partial"
    (TABLES / f"learner_benchmark_results{tag}.json").write_text(json.dumps(_clean(res), indent=1, ensure_ascii=False))
    from bench.learnerbench.tables import write_markdown

    write_markdown(res, TABLES / f"learner_benchmark_tables{tag}.md")
    return res


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


if __name__ == "__main__":
    main()
