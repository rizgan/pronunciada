"""Score the learner suite with Pronunciada and Azure and report it.

    python -m bench.learners.evaluate --steps ours [--workers 3]
    python -m bench.learners.evaluate --steps azure --dry-run      # hours that would be sent
    python -m bench.learners.evaluate --steps azure
    python -m bench.learners.evaluate --steps report               # results/tables/learners_suite.{json,md}

Input: $PRONUNCIADA_CACHE/data/learners/suite_test.jsonl (python -m bench.learners.suite).

PRE-SPECIFIED (written down before any system was run on the suite):
  sentence score   ours  = SentenceResult.scores["total"] (fallback SentenceResult.prob when absent);
                   Azure = NBest[0] PronScore. (pronunciada.compat.azure maps PronScore to the same
                   field.) Default settings, Pronunciada 1.0.2 as on this branch.
  failures         no assessment (ours: ValueError; Azure: RecognitionStatus != Success, HTTP 400, or
                   clip longer than Azure's 30 s limit for corpora where the clip was eligible) ->
                   the lowest score 0; counted, never dropped. Exception: Speech Accent Archive clips
                   > 30 s cannot be sent to Azure at all, so both systems are compared on the <= 30 s
                   clips (primary) and ours is also reported on all clips.
  concordance C    P(system orders a pair of recordings the same way as the human label), over all
                   pairs whose human labels differ (system ties count 1/2). With a binary label
                   (native = better than learner) C is the ROC AUC of bench.metrics.roc_auc; with an
                   ordinal score it is Harrell's C. "Prompt-matched": only pairs that read the same
                   prompt are counted.
  per language primary learner metric (one per language, the strongest evidence available):
     en  speechocean762 test: C of the sentence score vs the expert sentence `total` (2,500 utts,
         125 learners; expert-scored)
     zh  OMPAL test, learners only: C vs the teachers' mean sentence `accuracy` (663 utts, 18
         learners; expert-scored)
     es  Heroico USMA: prompt-matched C (= AUC) of native (8 speakers) vs learner (10 speakers)
         recordings (native-vs-learner only: weaker evidence than expert scores)
  multilingual summary  LS = unweighted mean over those languages of the primary C; 95 % interval:
                   2,000 bootstrap replicates, each resampling speakers with replacement independently
                   within each language (both systems on the same replicates); paired difference
                   ours - Azure with two-sided p = 2 min(P(d <= 0), P(d >= 0)). Reported with the
                   number of languages, speakers and items it rests on.
  second summary   LS-NL = mean over languages of the native-vs-learner AUC: en Speech Accent Archive
                   (<= 30 s), es Heroico USMA (prompt-matched), zh OMPAL test (prompt-matched, only 3
                   native speakers).
  secondary        Pearson r (bench.metrics.pearson) for the expert-scored sets; Heroico with the 30
                   HEROICO natives added (other site and equipment); speaker-level AUC (mean score per
                   speaker); a duration-only baseline (-duration as the score) for every native-vs-
                   learner test; Speech Accent Archive learners: Spearman of the score with English
                   onset age (expected negative) and with log(1 + years of residence) (expected positive).
  intervals        95 % percentile bootstrap, 2,000 replicates, speakers resampled.

ADDED 2026-11-23, PRE-SPECIFIED before any system was run on LeaP (bench/learners/leap.py). Nothing above
is changed; the round-19 tests, LS and LS-NL keep their definitions and their items.
  suite file       $PRONUNCIADA_CACHE/data/learners/suite4_test.jsonl = every line of suite_test.jsonl plus
                   the LeaP test selections (leap-de, leap-en); python -m bench.learners.suite --four.
  de  LeaP German read stories (B, C): prompt-matched C (= AUC) of the sentence score, native (7
         speakers) vs learner (46 speakers) recordings; pairs counted only within the same story
         sentence; speakers resampled. Primary German metric (native-vs-learner only: weaker evidence
         than expert scores, as for Spanish).
  per-prompt paired version (de, and en LeaP): the unweighted mean over prompts of the AUC computed
         within each prompt (prompts with no native or no learner reading skipped), same bootstrap.
  en  LeaP English read story (A): the same two metrics, native (4) vs learner (46). A secondary English
         native-vs-learner test; it does not replace Speech Accent Archive in LS-NL.
  LeaP secondary (both languages): speaker-level AUC; -duration baseline; per story (de: B and C);
         only items read exactly as printed (labels.read_deviations == 0, the corpus' word tier).
  LeaP proficiency (pre-declared secondary; the corpus has no proficiency rating, only group codes and
         age of onset): among learners, prompt-matched C of "superlearner" (group sl, written s1; the
         corpus' "very advanced learner") vs all other learners, expected > 0.5; Spearman of each
         learner's mean score with labels.age_of_onset, expected negative; later vs first reading of
         the same speaker on the same sentences (prepost.jsonl: e2 vs e1, c2 / c3 vs c1): mean over
         speakers of the mean per-sentence difference later - first, reported with a speaker bootstrap,
         no expected sign (the courses taught prosody).
  LS4              = unweighted mean over en, zh, es, de of the primary C (en speechocean762, zh OMPAL,
                   es Heroico USMA, de LeaP German); speakers resampled independently within each
                   language, both systems on the same replicates, two-sided p as for LS.
  LS-NL4           = mean of the native-vs-learner AUC over en (Speech Accent Archive <= 30 s), es (Heroico
                   USMA), zh (OMPAL test) and de (LeaP German).
  partial coverage A system that does not support one of the four languages gets no LS4. It is compared
                   with Pronunciada on the same formula restricted to the languages it supports (for
                   example LS-en,es,de), labelled with that language set; a language is never filled in
                   from a workaround. Failures score the system's lowest value, as above.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

from bench.data.common import cache_root
from bench.learners import read_jsonl, root
from bench.metrics import pearson

REPO = Path(__file__).resolve().parents[2]
N_BOOT = 2000
AZ_MAX_SECONDS = 30.0
AZURE_HOURS_CAP = 6.0
# where each corpus' Azure responses live (bench.azure_pa layout). speechocean762 and OMPAL were sent
# in earlier rounds with the same request settings; they are only read here, never re-sent.
AZURE_CACHE = {"heroico": "learners-heroico", "saa": "learners-saa", "speechocean762": "speechocean762",
               "ompal": "ompal-zh-TW"}
AZURE_SEND = ("heroico", "saa")
LOCALE = {"en": "en-US", "es": "es-MX", "zh": "zh-TW"}


def suite_file() -> str:
    """suite_test.jsonl (round 19) unless PRONUNCIADA_LEARNER_SUITE names another file in data/learners/."""
    return os.environ.get("PRONUNCIADA_LEARNER_SUITE", "suite_test.jsonl")


def suite() -> list[dict]:
    return list(read_jsonl(root() / suite_file()))


def work_dir() -> Path:
    return cache_root() / "round19" / "evalsuite"


# ============================================================================ ours

def variant() -> str:
    """'' for the default (fp32, shipped backbone); else e.g. 'int8' or 'backbone-<sha256[:12]>'.
    PRONUNCIADA_LEARNER_PRECISION selects the shipped precision; PRONUNCIADA_BACKBONE a backbone file."""
    v = os.environ.get("PRONUNCIADA_LEARNER_PRECISION", "fp32")
    tag = "" if v == "fp32" else v
    b = os.environ.get("PRONUNCIADA_BACKBONE")
    if b:
        from bench.learners import sha256 as _sha

        tag = (tag + "-" if tag else "") + "backbone-" + _sha(Path(b))[:12]
    return tag


def fingerprint() -> str:
    h = hashlib.sha256()
    pkg = REPO / "pronunciada"
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            h.update(str(p.relative_to(pkg)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


_SC = None


def _init(lang: str) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _SC
    import pronunciada

    assert Path(pronunciada.__file__).resolve().is_relative_to(REPO), pronunciada.__file__
    _SC = pronunciada.Scorer(lang, threads=1, precision=os.environ.get("PRONUNCIADA_LEARNER_PRECISION", "fp32"))


def _score_one(rec: dict) -> tuple[str, dict]:
    t = time.perf_counter()
    try:
        res = _SC.score(rec["audio"], rec["text"])
        sc = dict(res.scores or {})
        o = {"status": "Success", "prob": float(res.prob),
             "total": None if sc.get("total") is None else float(sc["total"]),
             "accuracy": None if sc.get("accuracy") is None else float(sc["accuracy"]),
             "fluency": None if res.fluency is None else float(res.fluency),
             "completeness": None if res.completeness is None else float(res.completeness),
             "warnings": list(res.warnings), "audio_issues": list((res.audio or {}).get("issues") or [])}
    except ValueError as e:
        o = {"status": "error", "error": str(e)[:200]}
    o["seconds"] = round(time.perf_counter() - t, 3)
    return rec["id"], o


def run_ours(workers: int) -> None:
    import pronunciada

    assert Path(pronunciada.__file__).resolve().is_relative_to(REPO), \
        f"pronunciada imported from {pronunciada.__file__}; run from the repo root with python -m"
    fp = fingerprint() + (f"-{variant()}" if variant() else "")
    d = work_dir() / "ours" / fp
    d.mkdir(parents=True, exist_ok=True)
    recs = suite()
    todo = defaultdict(list)
    for r in recs:
        if not (d / f"{r['id']}.json").exists():
            todo[r["lang"]].append(r)
    n = sum(len(v) for v in todo.values())
    print(f"ours: pronunciada {pronunciada.__version__} fingerprint {fp}: {len(recs)} clips, {n} to score", flush=True)
    t0 = time.time()
    for lang, rs in sorted(todo.items()):
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(lang,)) as ex:
            for k, (uid, o) in enumerate(ex.map(_score_one, rs, chunksize=4)):
                tmp = d / f"{uid}.json.tmp"
                tmp.write_text(json.dumps(o, ensure_ascii=False))
                os.replace(tmp, d / f"{uid}.json")
                if (k + 1) % 250 == 0:
                    print(f"  {lang}: {k + 1}/{len(rs)} ({time.time() - t0:.0f} s)", flush=True)
        print(f"  {lang}: {len(rs)} scored ({time.time() - t0:.0f} s)", flush=True)
    meta = {"fingerprint": fp, "version": pronunciada.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "variant": variant(), "suite": suite_file()}
    (d / "_meta.json").write_text(json.dumps(meta))
    if not variant() and suite_file() == "suite_test.jsonl":
        (work_dir() / "ours" / "latest.json").write_text(json.dumps(meta))


def ours_scores(recs: list[dict]) -> tuple[dict, dict]:
    meta = json.loads((work_dir() / "ours" / "latest.json").read_text())
    if meta["fingerprint"] != fingerprint():
        print(f"note: pronunciada/ changed since the ours run ({meta['fingerprint']} -> {fingerprint()})")
    d = work_dir() / "ours" / meta["fingerprint"]
    out = {}
    for r in recs:
        p = d / f"{r['id']}.json"
        if p.exists():
            o = json.loads(p.read_text())
            if o["status"] == "Success":
                o["score"] = o["total"] if o.get("total") is not None else o["prob"]
            else:
                o["score"] = 0.0
            out[r["id"]] = o
    return out, meta


# ============================================================================ Azure

def azure_dir(corpus: str) -> Path:
    from bench.azure_pa import cache_dir

    return cache_dir(AZURE_CACHE[corpus])


def azure_hours_sent() -> float:
    s = 0.0
    for c in AZURE_SEND:
        for p in azure_dir(c).glob("*.json"):
            s += json.loads(p.read_text()).get("audio_dur", 0.0)
    return s / 3600


def run_azure(dry_run: bool, workers: int = 4, max_tries: int = 8) -> None:
    from bench import azure_pa as AZ

    recs = [r for r in suite() if r["labels"]["corpus"] in AZURE_SEND]
    todo, cached, too_long = [], 0, 0
    for r in recs:
        if (azure_dir(r["labels"]["corpus"]) / f"{r['id']}.json").exists():
            cached += 1
        elif r["dur"] > AZ_MAX_SECONDS:
            too_long += 1
        else:
            todo.append(r)
    secs = defaultdict(float)
    for r in todo:
        secs[(r["labels"]["corpus"], LOCALE[r["lang"]])] += r["dur"]
    sent = azure_hours_sent()
    new = sum(secs.values()) / 3600
    print(f"azure: {len(recs)} clips, {cached} cached, {too_long} longer than {AZ_MAX_SECONDS:.0f} s (never sent), "
          f"{len(todo)} to send = {new:.3f} audio-h; already sent by this job {sent:.3f} h; cap {AZURE_HOURS_CAP} h")
    for (c, loc), s in sorted(secs.items()):
        print(f"  {c} {loc}: {sum(1 for r in todo if r['labels']['corpus'] == c)} requests, {s / 3600:.3f} h")
    if dry_run or not todo:
        return
    if sent + new > AZURE_HOURS_CAP:
        raise SystemExit("refusing: the Azure hour cap for this job would be exceeded")
    key, region = AZ.credentials()
    import requests

    local = threading.local()
    abort = threading.Event()

    def one(r: dict) -> str:
        if abort.is_set():
            return "skipped"
        out = azure_dir(r["labels"]["corpus"])
        out.mkdir(parents=True, exist_ok=True)
        if (out / f"{r['id']}.json").exists():
            return "ok"
        if not hasattr(local, "s"):
            local.s = requests.Session()
        locale = LOCALE[r["lang"]]
        params = AZ.pa_params(r["text"], locale)
        audio = AZ.wav_bytes(r["audio"])
        resp, attempts, t = None, 0, time.perf_counter()
        for attempt in range(max_tries):
            attempts = attempt + 1
            t = time.perf_counter()
            try:
                resp = AZ.send(local.s, key, region, locale, params, audio, max_tries=1)
                break
            except RuntimeError as e:
                msg = str(e)
                if msg.startswith(("Azure returned HTTP 401", "Azure returned HTTP 403")):
                    abort.set()
                    return f"auth failure ({msg[:40]}); stopping"
                if msg.startswith("Azure returned HTTP 400"):
                    resp = {"RecognitionStatus": "HTTP400", "error": msg[:300]}
                    break
                time.sleep(min(120.0, 4.0 * 2 ** attempt))
        if resp is None:
            return f"fail {r['id']}: giving up after {max_tries} tries (not cached)"
        rec = {"id": r["id"], "locale": locale, "params": params,
               "audio_sha1": hashlib.sha1(Path(r["audio"]).read_bytes()).hexdigest(),
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "latency_s": time.perf_counter() - t,
               "attempts": attempts, "audio_dur": r["dur"], "response": resp}
        tmp = out / f"{r['id']}.json.tmp{threading.get_ident()}"
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        os.replace(tmp, out / f"{r['id']}.json")
        return "ok"

    n_ok = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for i, f in enumerate(as_completed(futs)):
            res = f.result()
            if res == "ok":
                n_ok += 1
            elif res != "skipped":
                n_fail += 1
                print(res, flush=True)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)} ({n_fail} failed, {time.time() - t0:.0f} s)", flush=True)
    print(f"azure done: {n_ok} ok, {n_fail} failed, {time.time() - t0:.0f} s; job total sent "
          f"{azure_hours_sent():.3f} h", flush=True)


def azure_scores(recs: list[dict]) -> dict:
    from bench.azure_pa import parse_response

    out = {}
    for r in recs:
        p = azure_dir(r["labels"]["corpus"]) / f"{r['id']}.json"
        if not p.exists():
            continue
        resp = json.loads(p.read_text())["response"]
        parsed = parse_response(resp)
        s = parsed.get("sentence") if parsed["status"] == "Success" else None
        v = None if s is None else s.get("PronScore")
        out[r["id"]] = {"status": "Success" if v is not None else str(parsed["status"]),
                        "score": 0.0 if v is None else float(v),
                        "accuracy": None if s is None or s.get("AccuracyScore") is None else float(s["AccuracyScore"])}
    return out


# ============================================================================ metrics

def concordance(h, s, strata=None) -> tuple[float, float]:
    """(concordant pairs incl. 1/2 ties, comparable pairs) over pairs with h_i != h_j (within strata)."""
    h, s = np.asarray(h, float), np.asarray(s, float)
    if strata is None:
        groups = [np.arange(len(h))]
    else:
        st = np.asarray(strata)
        _, inv = np.unique(st, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        cuts = np.flatnonzero(np.diff(inv[order])) + 1
        groups = np.split(order, cuts)
    conc = pairs = 0.0
    for g in groups:
        hg, sg = h[g], s[g]
        levels = np.unique(hg)
        if len(levels) < 2:
            continue
        for b in levels[1:]:
            lo, hi = sg[hg < b], sg[hg == b]
            if not len(lo) or not len(hi):
                continue
            r = stats.rankdata(np.r_[hi, lo])
            u = r[: len(hi)].sum() - len(hi) * (len(hi) + 1) / 2
            conc += u
            pairs += len(hi) * len(lo)
    return conc, pairs


def cindex(h, s, strata=None) -> float:
    c, p = concordance(h, s, strata)
    return c / p if p else float("nan")


class Test:
    """One evaluation: items, human label h (higher = better), speakers, optional strata."""

    def __init__(self, name, lang, recs, h, kind, strata=None, evidence=""):
        self.name, self.lang, self.recs, self.kind, self.evidence = name, lang, recs, kind, evidence
        self.h = np.asarray(h, float)
        self.spk = np.array([r["speaker"] for r in recs])
        self.strata = None if strata is None else np.asarray(strata)
        _, inv = np.unique(self.spk, return_inverse=True)
        self.members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]

    def metric(self, s, idx=None):
        idx = np.arange(len(self.h)) if idx is None else idx
        st = None if self.strata is None else self.strata[idx]
        return cindex(self.h[idx], np.asarray(s)[idx], st)

    def resample(self, rng):
        pick = rng.integers(0, len(self.members), len(self.members))
        idx = np.concatenate([self.members[k] for k in pick])
        # copy id of every resampled item (a speaker drawn twice counts as two speakers)
        self.last_copy = np.repeat(np.arange(len(pick)), [len(self.members[k]) for k in pick])
        return idx

    def info(self) -> dict:
        lab = Counter("native" if r["labels"]["native"] else "learner" for r in self.recs)
        spk = defaultdict(set)
        for r in self.recs:
            spk["native" if r["labels"]["native"] else "learner"].add(r["speaker"])
        return {"name": self.name, "lang": self.lang, "kind": self.kind, "evidence": self.evidence,
                "items": len(self.recs), "speakers": len(self.members),
                "items_by_group": dict(lab), "speakers_by_group": {k: len(v) for k, v in spk.items()},
                "hours": round(sum(r["dur"] for r in self.recs) / 3600, 3)}


def q(v) -> list[float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [float("nan")] * 2


def p_two(d) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def evaluate_pair(t: Test, a, b, fn=None, seed=0) -> dict:
    """fn(test, scores, idx) -> value; defaults to the test's concordance."""
    fn = fn or (lambda tt, s, idx: tt.metric(s, idx))
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    va, vb = [], []
    for _ in range(N_BOOT):
        idx = t.resample(rng)
        va.append(fn(t, a, idx))
        vb.append(fn(t, b, idx) if b is not None else np.nan)
    va, vb = np.array(va), np.array(vb)
    A, B = fn(t, a, None), fn(t, b, None)
    return {"ours": A, "ours_ci": q(va), "azure": B, "azure_ci": q(vb), "delta": A - B, "delta_ci": q(va - vb),
            "p_two_sided": p_two(va - vb)}


def evaluate_single(t: Test, a, fn=None, seed=1) -> dict:
    fn = fn or (lambda tt, s, idx: tt.metric(s, idx))
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    v = [fn(t, a, t.resample(rng)) for _ in range(N_BOOT)]
    return {"value": fn(t, a, None), "ci": q(v)}


def _pearson(t, s, idx):
    idx = np.arange(len(t.h)) if idx is None else idx
    return pearson(t.h[idx], np.asarray(s)[idx])


def _speaker_auc(t, s, idx):
    """C over speakers: each speaker's mean score vs its label (resampled copies are separate speakers)."""
    if idx is None:
        idx = np.arange(len(t.h))
        _, copy = np.unique(t.spk, return_inverse=True)
    else:
        copy = t.last_copy
    s = np.asarray(s, float)[idx]
    h = t.h[idx]
    n = copy.max() + 1
    mean = np.bincount(copy, weights=s, minlength=n) / np.bincount(copy, minlength=n)
    lab = np.zeros(n)
    lab[copy] = h
    return cindex(lab, mean)


def _spearman_with(values):
    vals = np.asarray(values, float)

    def f(t, s, idx):
        idx = np.arange(len(t.h)) if idx is None else idx
        ok = np.isfinite(vals[idx])
        return float(stats.spearmanr(np.asarray(s)[idx][ok], vals[idx][ok]).statistic)

    return f


# ============================================================================ the tests

def build_tests(recs: list[dict]) -> dict[str, Test]:
    by = defaultdict(list)
    for r in recs:
        by[r["labels"]["corpus"]].append(r)
    T = {}
    so = by["speechocean762"]
    T["en/speechocean762"] = Test("speechocean762 test: sentence total (experts)", "en", so,
                                  [r["labels"]["human_score"] for r in so], "expert", evidence="expert scores")
    om = [r for r in by["ompal"] if not r["labels"]["native"]]
    T["zh/ompal"] = Test("OMPAL test learners: teacher sentence accuracy", "zh", om,
                         [r["labels"]["human_score"] for r in om], "expert", evidence="expert scores")
    us = [r for r in by["heroico"] if r["labels"]["subcorpus"] == "usma"]
    T["es/heroico"] = Test("Heroico USMA: native vs learner, prompt-matched", "es", us,
                           [1.0 if r["labels"]["native"] else 0.0 for r in us], "native_vs_learner",
                           strata=[r["labels"]["prompt_id"] for r in us], evidence="native vs learner")
    # secondary / second-summary tests
    he = by["heroico"]
    T["es/heroico+heroico_natives"] = Test("Heroico: learners vs USMA + HEROICO natives, prompt-matched", "es", he,
                                           [1.0 if r["labels"]["native"] else 0.0 for r in he], "native_vs_learner",
                                           strata=[r["labels"]["prompt_id"] for r in he], evidence="native vs learner")
    sa = [r for r in by["saa"] if r["dur"] <= AZ_MAX_SECONDS]
    T["en/saa"] = Test("Speech Accent Archive (<= 30 s): native vs learner, same paragraph", "en", sa,
                       [1.0 if r["labels"]["native"] else 0.0 for r in sa], "native_vs_learner",
                       evidence="native vs learner")
    T["en/saa_all"] = Test("Speech Accent Archive, all clips (ours only)", "en", by["saa"],
                           [1.0 if r["labels"]["native"] else 0.0 for r in by["saa"]], "native_vs_learner",
                           evidence="native vs learner")
    sl = [r for r in sa if not r["labels"]["native"]]
    T["en/saa_learners"] = Test("Speech Accent Archive learners (<= 30 s): proficiency proxies", "en", sl,
                                np.zeros(len(sl)), "proxy", evidence="self-reported background")
    oa = by["ompal"]
    T["zh/ompal_nl"] = Test("OMPAL test: native vs learner, prompt-matched", "zh", oa,
                            [1.0 if r["labels"]["native"] else 0.0 for r in oa], "native_vs_learner",
                            strata=[r["text"] for r in oa], evidence="native vs learner")
    # LeaP (added 2026-11-23; see the docstring)
    for lang, corpus in (("de", "leap-de"), ("en", "leap-en")):
        lp = by.get(corpus) or []
        if not lp:
            continue
        nl = lambda rs: [1.0 if r["labels"]["native"] else 0.0 for r in rs]  # noqa: E731
        pid = lambda rs: [r["labels"]["prompt_id"] for r in rs]  # noqa: E731
        T[f"{lang}/leap"] = Test(f"LeaP {lang} read stories: native vs learner, prompt-matched", lang, lp, nl(lp),
                                 "native_vs_learner", strata=pid(lp), evidence="native vs learner")
        ex = [r for r in lp if r["labels"].get("read_deviations") == 0]
        T[f"{lang}/leap_exact"] = Test(f"LeaP {lang}: items read exactly as printed", lang, ex, nl(ex),
                                       "native_vs_learner", strata=pid(ex), evidence="native vs learner")
        for st in sorted({r["labels"]["story"] for r in lp}):
            ss = [r for r in lp if r["labels"]["story"] == st]
            if lang == "de":
                T[f"de/leap_story{st}"] = Test(f"LeaP de story {st} only", lang, ss, nl(ss), "native_vs_learner",
                                               strata=pid(ss), evidence="native vs learner")
        le = [r for r in lp if not r["labels"]["native"]]
        T[f"{lang}/leap_superlearner"] = Test(f"LeaP {lang} learners: superlearner vs other learners, prompt-matched",
                                              lang, le, [1.0 if r["labels"].get("superlearner") else 0.0 for r in le],
                                              "proficiency", strata=pid(le), evidence="corpus group code")
        T[f"{lang}/leap_onset"] = Test(f"LeaP {lang} learners: speaker mean score vs age of onset", lang, le,
                                       np.zeros(len(le)), "proxy_onset", evidence="corpus metadata")
    return T


def prompt_macro_auc(t, s, idx):
    """Unweighted mean over prompts of the within-prompt AUC (prompts lacking either group skipped)."""
    idx = np.arange(len(t.h)) if idx is None else idx
    s = np.asarray(s, float)[idx]
    h, st = t.h[idx], t.strata[idx]
    vals = []
    for p in np.unique(st):
        m = st == p
        if len(np.unique(h[m])) == 2:
            vals.append(cindex(h[m], s[m]))
    return float(np.mean(vals)) if vals else float("nan")


def speaker_spearman(values_by_item):
    """Spearman over speakers of the speaker's mean score with a per-speaker value (resampled copies kept)."""
    vals = np.asarray(values_by_item, float)

    def f(t, s, idx):
        if idx is None:
            idx = np.arange(len(t.h))
            _, copy = np.unique(t.spk, return_inverse=True)
        else:
            copy = t.last_copy
        n = copy.max() + 1
        ss = np.asarray(s, float)[idx]
        mean = np.bincount(copy, weights=ss, minlength=n) / np.bincount(copy, minlength=n)
        v = np.full(n, np.nan)
        v[copy] = vals[idx]
        ok = np.isfinite(v)
        return float(stats.spearmanr(mean[ok], v[ok]).statistic) if ok.sum() > 2 else float("nan")

    return f


PRIMARY = {"en": "en/speechocean762", "zh": "zh/ompal", "es": "es/heroico"}
NL = {"en": "en/saa", "zh": "zh/ompal_nl", "es": "es/heroico"}
# added 2026-11-23
PRIMARY4 = {**PRIMARY, "de": "de/leap"}
NL4 = {**NL, "de": "de/leap"}


def summary(T: dict, picks: dict, a: dict, b: dict, seed: int = 7) -> dict:
    """Macro mean over languages of each pick's concordance; speakers resampled within language."""
    rng = np.random.default_rng(seed)
    names = [picks[k] for k in sorted(picks)]
    A = np.mean([T[n].metric(a[n]) for n in names])
    B = np.mean([T[n].metric(b[n]) for n in names])
    va, vb = [], []
    for _ in range(N_BOOT):
        xa, xb = [], []
        for n in names:
            idx = T[n].resample(rng)
            xa.append(T[n].metric(a[n], idx))
            xb.append(T[n].metric(b[n], idx))
        va.append(np.mean(xa))
        vb.append(np.mean(xb))
    va, vb = np.array(va), np.array(vb)
    return {"languages": sorted(picks), "tests": names, "ours": float(A), "ours_ci": q(va), "azure": float(B),
            "azure_ci": q(vb), "delta": float(A - B), "delta_ci": q(va - vb), "p_two_sided": p_two(va - vb),
            "speakers": int(sum(len(T[n].members) for n in names)), "items": int(sum(len(T[n].recs) for n in names))}


def report() -> dict:
    recs = suite()
    ours, meta = ours_scores(recs)
    az = azure_scores(recs)
    T = build_tests(recs)
    res = {"pronunciada": meta, "tests": {}, "n_boot": N_BOOT}
    a_s, b_s = {}, {}
    for key, t in T.items():
        if t.kind == "proxy_onset":       # LeaP secondary, reported by bench.learnerbench
            continue
        ids = [r["id"] for r in t.recs]
        miss_o = [i for i in ids if i not in ours]
        miss_a = [i for i in ids if i not in az]
        e = {**t.info(), "missing_ours": len(miss_o), "missing_azure": len(miss_a),
             "ours_failures": dict(Counter(ours[i]["status"] for i in ids if i in ours and ours[i]["status"] != "Success")),
             "azure_failures": dict(Counter(az[i]["status"] for i in ids if i in az and az[i]["status"] != "Success"))}
        if miss_o:
            res["tests"][key] = e
            continue
        a = np.array([ours[i]["score"] for i in ids])
        a_s[key] = a
        have_az = not miss_a
        if have_az:
            b = np.array([az[i]["score"] for i in ids])
            b_s[key] = b
        if t.kind == "proxy":
            for lab, sign in (("onset_age", "expected negative"), ("log_residence", "expected positive")):
                vals = [np.nan if r["labels"].get(lab) is None else float(r["labels"][lab]) for r in t.recs]
                f = _spearman_with(vals)
                e[f"spearman_{lab}"] = (evaluate_pair(t, a, b, f) if have_az else evaluate_single(t, a, f)) | {"note": sign}
            res["tests"][key] = e
            continue
        e["C"] = evaluate_pair(t, a, b) if have_az else evaluate_single(t, a)
        if t.kind == "expert":
            e["pearson"] = evaluate_pair(t, a, b, _pearson) if have_az else evaluate_single(t, a, _pearson)
        else:
            e["speaker_level_auc"] = evaluate_pair(t, a, b, _speaker_auc) if have_az else evaluate_single(t, a, _speaker_auc)
            e["baseline_minus_duration"] = evaluate_single(t, -np.array([r["dur"] for r in t.recs]))
        res["tests"][key] = e
    for nm, picks in (("LS", PRIMARY), ("LS_NL", NL)):
        if all(p in a_s and p in b_s for p in picks.values()):
            res[nm] = summary(T, picks, a_s, b_s)
    out = REPO / "results" / "tables"
    (out / "learners_suite.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / "learners_suite.md").write_text(markdown(res))
    print(markdown(res))
    return res


def _f(v, ci=None, d=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    s = f"{v:.{d}f}"
    return s if ci is None else f"{s} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def _pv(p) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def markdown(res: dict) -> str:
    L = ["# Learner suite: tables", "",
         f"Generated by `python -m bench.learners.evaluate --steps report` (Pronunciada {res['pronunciada'].get('version')}, "
         f"package fingerprint {res['pronunciada']['fingerprint']}). C = concordance (AUC for native vs learner). "
         "95 % intervals: speaker bootstrap, 2,000 replicates; p: paired, two-sided.", ""]
    for nm, title in (("LS", "Multilingual learner summary LS (primary metric per language, unweighted mean)"),
                      ("LS_NL", "Second summary LS-NL (native vs learner AUC per language)")):
        if nm in res:
            s = res[nm]
            L += [f"## {title}", "", "| languages | tests | speakers | items | Pronunciada | Azure | ours - Azure | p |",
                  "|---|---|---|---|---|---|---|---|",
                  f"| {', '.join(s['languages'])} | {', '.join(s['tests'])} | {s['speakers']} | {s['items']} | "
                  f"{_f(s['ours'], s['ours_ci'])} | {_f(s['azure'], s['azure_ci'])} | {_f(s['delta'], s['delta_ci'])} | {_pv(s['p_two_sided'])} |", ""]
    L += ["## Per test", "", "| test | lang | evidence | speakers | items | metric | Pronunciada | Azure | ours - Azure | p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, e in res["tests"].items():
        rows = []
        for m in ("C", "pearson", "speaker_level_auc", "spearman_onset_age", "spearman_log_residence"):
            if m in e:
                rows.append((m, e[m]))
        for m, v in rows:
            if "ours" in v:
                L.append(f"| {e['name']} | {e['lang']} | {e['evidence']} | {e['speakers']} | {e['items']} | {m} | "
                         f"{_f(v['ours'], v['ours_ci'])} | {_f(v['azure'], v['azure_ci'])} | {_f(v['delta'], v['delta_ci'])} | {_pv(v['p_two_sided'])} |")
            else:
                L.append(f"| {e['name']} | {e['lang']} | {e['evidence']} | {e['speakers']} | {e['items']} | {m} | "
                         f"{_f(v['value'], v['ci'])} | - | - | - |")
        if "baseline_minus_duration" in e:
            v = e["baseline_minus_duration"]
            L.append(f"| {e['name']} | {e['lang']} | baseline | {e['speakers']} | {e['items']} | C of -duration | "
                     f"{_f(v['value'], v['ci'])} | | | |")
    L += ["", "## Coverage and failures", "", "| test | items by group | speakers by group | hours | missing ours / Azure | ours failures | Azure failures |",
          "|---|---|---|---|---|---|---|"]
    for key, e in res["tests"].items():
        L.append(f"| {key} | {e['items_by_group']} | {e['speakers_by_group']} | {e['hours']} | {e['missing_ours']} / {e['missing_azure']} | "
                 f"{e['ours_failures'] or '-'} | {e['azure_failures'] or '-'} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="report")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    steps = a.steps.split(",")
    if "ours" in steps:
        run_ours(a.workers)
    if "azure" in steps:
        run_azure(a.dry_run)
    if "report" in steps:
        report()


if __name__ == "__main__":
    main()
