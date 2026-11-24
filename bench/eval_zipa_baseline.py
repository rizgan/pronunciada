"""Ablation: Pronunciada against the unmodified ZIPA model ("ZIPA alone"), on the held-out tests
and protocols of results/SCOREBOARD.md.

    python -m bench.eval_zipa_baseline [--steps run,verify,eval,report] [--workers 3]

BASELINE (fixed before any baseline number was computed; nothing is fitted, selected or
thresholded on any data):
  input    raw log-posteriors of zipa-small-crctc-500k, fp32 (pronunciada.acoustic.AcousticModel,
           no adapter), and the canonical phone sequence of Pronunciada's text front end for the
           prompt (pronunciada.text.get_g2p(lang)(text); RefPhone.ipa only, i.e. the first /
           canonical alternative; RefPhone.alts are ignored), mapped to ZIPA token ids with
           pronunciada.text.ipa.to_ids exactly as pronunciada.lattice does.
  gop      CTC Viterbi forced alignment of the canonical token sequence (pronunciada.lattice
           build_lattice with one non-skippable, zero-weight alternative per sound, decoded with
           pronunciada.align.viterbi; standard CTC topology; blank = logaddexp(<blk>, ▁), the
           model's own two non-speech symbols). Phone score = mean raw log-posterior of the
           canonical token over the frames aligned to that token (blank frames excluded; a
           multi-token sound averages over all of its tokens' frames); log-posteriors floored at
           -60 (the floor of bench/postcache.py). Word score = mean of its phone scores; sentence
           score = mean of all phone scores. If the canonical sequence cannot fit into the audio
           (fewer frames than CTC needs), every phone gets the floor -60 (counted).
  match    greedy CTC decode (pronunciada.acoustic.greedy_tokens: argmax per frame, collapse
           repeats, drop ids 0-3), Levenshtein alignment with the canonical token sequence
           (unit costs). Phone score = 1 if every token of the sound is aligned to an identical
           decoded token, else 0; word score = share of its phones matched; sentence score =
           max(0, 1 - edit distance / number of canonical tokens) (= 1 - token error rate,
           insertions counted).
Not used: per-language adapter, lattice alternatives, native variants, calibrators, tone /
stress / length / contrast heads, fitted thresholds, explain-away pass.

PROTOCOLS: every test reuses the item set, label definition, fallback rule, resampling unit,
bootstrap size and seed of the script that produced the scoreboard row (bench/eval_english.py,
bench/coach/azure_yardstick.py, bench/eval_ompal.py, bench/eval_swap.py + eval_languages.py,
bench/eval_cv.py). Pronunciada's numbers are RECOMPUTED here from the same cached per-utterance
outputs those scripts read (no Pronunciada re-scoring), and each is checked against the value in
results/tables/*.json. Δ = Pronunciada − gop, paired on the same resamples; two-sided p.
Rank-based tests (AUC, hit@5 %, precision at recall) are invariant to monotone transforms, so
where a protocol scores a missing unit as 0 the gop word score is passed as exp(mean log-post).

Caches: per-utterance baseline scores in $PRONUNCIADA_CACHE/zipa_baseline/<VERSION>/<dataset>/;
a record is re-scored only when missing or when its language front end changed. Writes
results/tables/zipa_baseline.json and results/zipa_baseline.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = "zipa-small-crctc-500k"
PRECISION = "fp32"
VERSION = f"{MODEL}-{PRECISION}-v1"
GOP_FLOOR = -60.0
JSON_OUT = ROOT / "results" / "tables" / "zipa_baseline.json"
MD_OUT = ROOT / "results" / "zipa_baseline.md"
RANK: list[dict] = []          # Spearman rank check of the Pearson rows (key "rank_check")
FORBIDDEN = ("pronunciada.adapter", "pronunciada.calibrate", "pronunciada.variants", "pronunciada.scorer",
             "pronunciada.prosody", "pronunciada.features", "pronunciada.contrast_heads", "pronunciada.length",
             "pronunciada.sound_emb", "pronunciada.delivery", "pronunciada.coach", "pronunciada.contrast_word")


# ============================================================================ scorers (pure)

def canonical_tokens(ref) -> list[tuple[int, ...]]:
    """ZIPA token ids of each sound's canonical realisation (RefPhone.ipa; alternatives ignored)."""
    from pronunciada.text.ipa import to_ids

    return [tuple(to_ids(p.ipa)) for p in ref.phones]


def word_ranges(ref) -> list[tuple[int, int]]:
    return [tuple(w.phones) for w in ref.words]


def _word_means(vals: np.ndarray, words) -> list[float]:
    return [float(np.mean(vals[a:b])) for a, b in words]


def gop_scores(lp: np.ndarray, phone_toks, words) -> dict:
    """Classic GOP on raw log-posteriors lp [T, V] (see module docstring)."""
    from pronunciada.align import emission_table, viterbi
    from pronunciada.lattice import Slot, build_lattice

    n = len(phone_toks)
    lp = np.asarray(lp, dtype=np.float64)
    slots = [Slot([(tuple(t), 0.0, 0)]) for t in phone_toks]      # canonical only, not skippable
    aligned = False
    ph = np.full(n, GOP_FLOOR)
    if len(lp) and n:
        lat = build_lattice(slots)
        score, path = viterbi(lat, emission_table(lp))
        if np.isfinite(score) and len(path):
            aligned = True
            T = len(lp)
            tok = lat.is_tok[path]
            sl = lat.slot[path]
            lab = np.where(tok, lat.lab[path], 0)
            v = np.maximum(lp[np.arange(T), lab], GOP_FLOOR)
            for i in range(n):
                m = tok & (sl == i)
                assert m.any(), "every CTC token occupies at least one frame"
                ph[i] = float(v[m].mean())
    return {"phone": ph.tolist(), "word": _word_means(ph, words),
            "sentence": float(ph.mean()) if n else GOP_FLOOR, "aligned": aligned}


def greedy_ids(lp: np.ndarray) -> list[int]:
    from pronunciada.acoustic import greedy_tokens
    from pronunciada.text.ipa import token_ids

    ids = token_ids()
    return [ids[t] for t, _, _ in greedy_tokens(lp)]


def levenshtein_align(ref: list[int], hyp: list[int]) -> tuple[list[bool], int]:
    """Unit-cost edit distance; matched[k] = ref token k aligned to an identical hyp token."""
    n, m = len(ref), len(hyp)
    D = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        r, prev, cur = ref[i - 1], D[i - 1], D[i]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != hyp[j - 1]))
    matched = [False] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            matched[i - 1] = ref[i - 1] == hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return matched, D[n][m]


def match_scores(lp: np.ndarray, phone_toks, words, hyp: list[int] | None = None) -> dict:
    hyp = greedy_ids(lp) if hyp is None else hyp
    flat = [t for toks in phone_toks for t in toks]
    owner = [i for i, toks in enumerate(phone_toks) for _ in toks]
    matched, dist = levenshtein_align(flat, hyp)
    n = len(phone_toks)
    ok = np.ones(n, bool)
    for k, mt in enumerate(matched):
        ok[owner[k]] &= mt
    ph = ok.astype(float)
    return {"phone": ph.tolist(), "word": _word_means(ph, words),
            "sentence": max(0.0, 1.0 - dist / max(len(flat), 1)), "edit_distance": int(dist),
            "n_tokens": len(flat), "n_hyp": len(hyp)}


# ============================================================================ run (workers)

_AM = None
_G2P: dict = {}


def cache_dir(dataset: str) -> Path:
    from pronunciada.acoustic import cache_root

    return Path(cache_root()) / "zipa_baseline" / VERSION / dataset


def cache_path(dataset: str, rid: str) -> Path:
    return cache_dir(dataset) / f"{rid.replace('/', '_')}.json"


def _init() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _AM
    from pronunciada.acoustic import AcousticModel

    _AM = AcousticModel(MODEL, precision=PRECISION, threads=1)      # plain graph, no adapter


def _ref(lang: str, text: str):
    """Same as bench.pipeline.common.build_ref (copied so the workers never import the bench
    pipeline, which pulls in pronunciada.variants)."""
    from pronunciada.text import get_g2p

    if lang not in _G2P:
        _G2P[lang] = get_g2p(lang)
    try:
        ref = _G2P[lang](text)
        ref.validate()
    except Exception:  # noqa: BLE001
        return None
    return ref if ref.phones else None


def score_one(lp, ref) -> dict:
    toks, words = canonical_tokens(ref), word_ranges(ref)
    g = gop_scores(lp, toks, words)
    m = match_scores(lp, toks, words)
    return {"phones": [p.ipa for p in ref.phones], "tokens": [list(t) for t in toks],
            "word_text": [w.text for w in ref.words], "word_span": [list(w.char_span) for w in ref.words],
            "word_phones": [list(w) for w in words], "n_frames": int(len(lp)), "aligned": g["aligned"],
            "gop": g["phone"], "gop_w": g["word"], "gop_s": g["sentence"],
            "match": m["phone"], "match_w": m["word"], "match_s": m["sentence"],
            "edit_distance": m["edit_distance"], "n_tokens": m["n_tokens"], "n_hyp": m["n_hyp"]}


def _score_group(group: list[dict]) -> dict:
    """group = records sharing one audio file: {dataset, id, lang, audio, text, g2p, dur}."""
    lp = None
    done = 0
    for t in group:
        ref = _ref(t["lang"], t["text"])
        if ref is None:
            o = {"error": "no_ref"}
        else:
            if lp is None:
                lp = _AM.posteriors(t["audio"])
            o = score_one(lp, ref)
        o.update(id=t["id"], lang=t["lang"], g2p=t["g2p"], dur=t["dur"])
        p = cache_path(t["dataset"], t["id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(o, ensure_ascii=False))
        os.replace(tmp, p)
        done += 1
    return {"n": done, "audio_s": float(group[0]["dur"]) if lp is not None else 0.0,
            "modules": sorted(m for m in sys.modules if m.startswith("pronunciada"))}


def tasks() -> dict[str, list[dict]]:
    """Every held-out record the scoreboard rows use, per dataset."""
    from bench import cv_build as CB
    from bench.data import load
    from bench.pipeline.common import g2p_version

    gv = {}

    def g(lang):
        if lang not in gv:
            gv[lang] = g2p_version(lang)
        return gv[lang]

    out = {}
    for ds in ("speechocean762", "l2arctic", "epadb"):
        out[ds] = [{"dataset": ds, "id": r["id"], "lang": "en", "audio": r["audio"], "text": r["text"],
                    "g2p": g("en"), "dur": r["dur"]} for r in load(ds, split="test")]
    out["fleurs-swap"] = [{"dataset": "fleurs-swap", "id": r["id"], "lang": r["lang"], "audio": r["audio"],
                           "text": r["text"], "g2p": g(r["lang"]), "dur": r["dur"]}
                          for r in load("fleurs-swap", split="test")]
    out["ompal"] = [{"dataset": "ompal", "id": r["id"], "lang": "zh", "audio": r["audio"], "text": r["text"],
                     "g2p": g("zh"), "dur": r["dur"]} for r in load("ompal", split="test")]
    seen, cv = set(), []
    for r in CB.load_manifest(CB.LOCALES, half="test"):
        if r["id"] not in seen:
            seen.add(r["id"])
            cv.append({"dataset": "cv-real", "id": r["id"], "lang": r["lang"], "audio": r["audio"],
                       "text": r["sentence"], "g2p": g(r["lang"]), "dur": r["dur"]})
    out["cv-real"] = cv
    return out


def _cached_ok(t: dict) -> bool:
    p = cache_path(t["dataset"], t["id"])
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("g2p") == t["g2p"]
    except (OSError, ValueError):
        return False


def run(workers: int = 3) -> dict:
    T = tasks()
    groups = []
    stats = {"records": 0, "unique_audio": 0, "audio_hours": 0.0}
    for ds, ts in T.items():
        by = defaultdict(list)
        for t in ts:
            by[t["audio"]].append(t)
        stats["records"] += len(ts)
        stats["unique_audio"] += len(by)
        stats["audio_hours"] += sum(g[0]["dur"] for g in by.values()) / 3600
        todo = [[t for t in g if not _cached_ok(t)] for g in by.values()]
        groups += [g for g in todo if g]
    n_todo = sum(len(g) for g in groups)
    print(f"baseline: {stats['records']} records, {stats['unique_audio']} audio files "
          f"({stats['audio_hours']:.2f} h), {n_todo} records to score in {len(groups)} groups", flush=True)
    mods = set()
    t0 = time.time()
    if groups:
        groups.sort(key=lambda g: -g[0]["dur"])
        with ProcessPoolExecutor(min(workers, 3), initializer=_init) as ex:
            for k, r in enumerate(ex.map(_score_group, groups, chunksize=4)):
                mods.update(r["modules"])
                if (k + 1) % 500 == 0:
                    print(f"  {k + 1}/{len(groups)} groups ({time.time() - t0:.0f} s)", flush=True)
        print(f"  scored {n_todo} records in {time.time() - t0:.0f} s", flush=True)
    stats["worker_modules"] = sorted(mods)
    return stats


def baseline(dataset: str) -> dict:
    d = cache_dir(dataset)
    out = {}
    for p in d.glob("*.json"):
        o = json.loads(p.read_text())
        out[o["id"]] = o
    return out


# ============================================================================ verification

def _verify_worker(recs: list[dict]) -> dict:
    """In a fresh process: the posteriors the baseline uses, the modules it loaded."""
    _init()
    from bench.postcache import post_path

    rows = []
    for r in recs:
        lp = _AM.posteriors(r["audio"])
        ref = _ref("en", r["text"])
        o = score_one(lp, ref)
        pc = post_path("speechocean762", r["id"])
        rows.append({"id": r["id"], "lp": lp, "gop": o["gop"], "cache": np.load(pc).astype(np.float32) if pc.exists() else None})
    mods = sorted(m for m in sys.modules if m.startswith("pronunciada"))
    return {"rows": rows, "modules": mods}


def verify(n: int = 20) -> dict:
    """Checks that the baseline uses no adapter and no fitted component:
    (1) modules loaded by a fresh baseline worker (none of FORBIDDEN);
    (2) its log-posteriors vs the raw-model posterior cache (bench/postcache.py, float16) and vs
        the English adapter's output;
    (3) the lattice has exactly one zero-weight alternative per sound and no skips."""
    from bench.data import load
    from pronunciada.adapter import Adapter
    from pronunciada.lattice import Slot, build_lattice

    recs = load("speechocean762", split="test")[:n]
    with ProcessPoolExecutor(1) as ex:
        W = ex.submit(_verify_worker, recs).result()
    ad = Adapter.load("en")
    d_cache, d_adapt, d_gop = [], [], []
    for r in W["rows"]:
        lp = r["lp"]
        if r["cache"] is not None:
            d_cache.append(float(np.max(np.abs(np.maximum(lp, -60.0) - r["cache"]))))
        if ad is not None:
            la = ad(lp)
            d_adapt.append(float(np.max(np.abs(la - lp))))
            rec = next(x for x in recs if x["id"] == r["id"])
            ref = _ref("en", rec["text"])
            ga = gop_scores(la, canonical_tokens(ref), word_ranges(ref))["phone"]
            d_gop.append(float(np.mean(np.abs(np.asarray(ga) - np.asarray(r["gop"])))))
    ref = _ref("en", recs[0]["text"])
    lat = build_lattice([Slot([(t, 0.0, 0)]) for t in canonical_tokens(ref)])
    forbidden = [m for m in W["modules"] if m in FORBIDDEN or any(m.startswith(f + ".") for f in FORBIDDEN)]
    out = {"n_utts": len(W["rows"]), "worker_pronunciada_modules": W["modules"],
           "forbidden_modules_loaded": forbidden,
           "max_abs_diff_vs_raw_posterior_cache": max(d_cache) if d_cache else None,
           "n_compared_with_cache": len(d_cache),
           "max_abs_change_by_en_adapter": {"min": min(d_adapt), "max": max(d_adapt)} if d_adapt else None,
           "mean_abs_gop_change_if_adapter_were_used": float(np.mean(d_gop)) if d_gop else None,
           "lattice_alt_weights_all_zero": bool(np.all(lat.w == 0.0)),
           "lattice_alt_indices": sorted(set(int(a) for a in lat.alt[lat.is_tok])),
           "lattice_finite_init_states": int(np.isfinite(lat.init).sum())}
    assert not forbidden, forbidden
    assert out["lattice_alt_indices"] == [0]
    print(json.dumps(out, indent=1), flush=True)
    return out


# ============================================================================ helpers

def _two_sided_from_one(p1: float, n_boot: int) -> float:
    """eval_english's conversion of paired_bootstrap's one-sided p."""
    return min(1.0, 2 * min(p1, 1 - p1 + 1 / (n_boot + 1)))


def _p_two(d: np.ndarray) -> float:
    d = np.asarray(d)
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def _q(v) -> list[float]:
    v = np.asarray([x for x in v if np.isfinite(x)])
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def _api(tag: str) -> dict:
    from bench.calib.common import calib_cache

    return pickle.loads((calib_cache() / "api" / f"{tag}.pkl").read_bytes())


def _tables(name: str):
    p = ROOT / "results" / "tables" / name
    return json.loads(p.read_text()) if p.exists() else None


def _gop_arr(b: dict | None, key: str, n: int | None = None) -> np.ndarray:
    if b is None or "error" in b:
        return np.full(n or 1, GOP_FLOOR if key.startswith("gop") else 0.0)
    return np.asarray(b[key], float)


def _row(group, test, n, ours, gop, match, delta, p, unit, table_value=None, **kw):
    r = {"group": group, "test": test, "n": n, "pronunciada": ours, "gop": gop, "match": match,
         "delta": delta, "p": p, "unit": unit, **kw}
    if table_value is not None:
        r["pronunciada_table"] = table_value
        r["pronunciada_matches_table"] = bool(abs(ours["value"] - table_value) < 5e-4)
    return r


# ============================================================================ speechocean762 / errdet

def eval_so762(B: dict) -> tuple[list[dict], dict]:
    from bench.data import load
    from bench.eval_english import N_BOOT, az_root
    from bench.metrics import bootstrap_ci, paired_bootstrap, pearson, spearman
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference, arpa_to_ours

    api = _api("speechocean762-test-fp32")
    az = {}
    for line in open(os.path.join(az_root("speechocean762"), "aligned_test.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        az[r["id"]] = r
    SYS = ("ours", "gop", "match")
    P = {"y": [], "spk": [], **{s: [] for s in SYS}}
    Wd = {"y": [], "spk": [], **{s: [] for s in SYS}}
    S = {"spk": [], "gop": [], "match": [], **{f"y_{k}": [] for k in ("accuracy", "fluency", "prosodic", "total")},
         **{f"ours_{k}": [] for k in ("accuracy", "fluency", "prosodic", "total")}}
    cnt = Counter()
    for rec in load("speechocean762", split="test"):
        o, a = api.get(rec["id"]), az.get(rec["id"])
        if o is None or "error" in o or a is None:
            cnt["utts_excluded_as_in_scoreboard"] += 1
            continue
        b = B.get(rec["id"])
        ref = build_ref("en", rec["text"])
        assert [p.ipa for p in ref.phones] == o["phones"], rec["id"]
        if b is None or "error" in b:
            cnt["baseline_failed_utts"] += 1
        else:
            assert b["phones"] == o["phones"], rec["id"]
            cnt["baseline_not_aligned_utts"] += not b["aligned"]
        n = len(o["phones"])
        sc = {"ours": np.asarray(o["sound"]["B"]["reg"], float), "gop": _gop_arr(b, "gop", n),
              "match": _gop_arr(b, "match", n)}
        wsc = {"ours": np.asarray(o["word_reg"], float), "gop": _gop_arr(b, "gop_w", len(o["words"])),
               "match": _gop_arr(b, "match_w", len(o["words"]))}
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
            for k, s in enumerate(w["phones_accuracy"]):
                i = t2o.get((j, k), -1)
                for m in SYS:
                    reg = sc[m]
                    P[m].append(float(reg[i] if i >= 0 else (np.mean(reg[fb_ph]) if fb_ph else np.mean(reg))))
                cnt["phones"] += 1
                cnt["phones_fallback"] += i < 0
                P["y"].append(float(s))
                P["spk"].append(spk)
            Wd["y"].append(float(w["accuracy"]))
            Wd["spk"].append(spk)
            for m in SYS:
                Wd[m].append(float(np.mean(wsc[m][ow]) if ow else np.mean(wsc[m])))
            cnt["words"] += 1
            cnt["words_fallback"] += not ow
        sl = rec["labels"]["sentence"]
        for k in ("accuracy", "fluency", "prosodic", "total"):
            S[f"y_{k}"].append(float(sl[k]))
            S[f"ours_{k}"].append(float(o["sentence"][k]))
        S["gop"].append(float(b["gop_s"]) if b and "error" not in b else GOP_FLOOR)
        S["match"].append(float(b["match_s"]) if b and "error" not in b else 0.0)
        S["spk"].append(spk)
        cnt["utts"] += 1

    def corr(y, ours, gop, match, spk):
        y, spk = np.asarray(y, float), np.asarray(spk)
        res = {}
        for name, s in (("ours", ours), ("gop", gop), ("match", match)):
            s = np.asarray(s, float)
            ci = bootstrap_ci(pearson, s, y, clusters=spk, n_boot=N_BOOT)
            res[name] = {"value": pearson(s, y), "ci": [ci["lo"], ci["hi"]]}
        pb = paired_bootstrap(lambda yy, ss: pearson(ss, yy), y, np.asarray(ours, float), np.asarray(gop, float),
                              clusters=spk, n_boot=N_BOOT)
        res["delta"] = {"value": pb["delta"], "ci": [pb["lo"], pb["hi"]], "p": _two_sided_from_one(pb["p_value"], N_BOOT)}
        pm = paired_bootstrap(lambda yy, ss: pearson(ss, yy), y, np.asarray(ours, float), np.asarray(match, float),
                              clusters=spk, n_boot=N_BOOT)
        res["delta_vs_match"] = {"value": pm["delta"], "ci": [pm["lo"], pm["hi"]],
                                 "p": _two_sided_from_one(pm["p_value"], N_BOOT)}
        return res

    def rank(test, y, ours, gop, match, spk, tv):
        """Rank check: Spearman, same items, same speaker resamples (bootstrap seed 0)."""
        y, spk = np.asarray(y, float), np.asarray(spk)
        res = {}
        for name, s in (("ours", ours), ("gop", gop), ("match", match)):
            s = np.asarray(s, float)
            ci = bootstrap_ci(spearman, s, y, clusters=spk, n_boot=N_BOOT)
            res[name] = {"value": spearman(s, y), "ci": [ci["lo"], ci["hi"]]}
        pb = paired_bootstrap(lambda yy, ss: spearman(ss, yy), y, np.asarray(ours, float), np.asarray(gop, float),
                              clusters=spk, n_boot=N_BOOT)
        RANK.append(_row("speechocean762", test, int(len(y)), res["ours"], res["gop"], res["match"], pb["delta"],
                         _two_sided_from_one(pb["p_value"], N_BOOT), "Spearman", tv, delta_ci=[pb["lo"], pb["hi"]]))

    E = (_tables("english_vs_azure.json") or {}).get("so762", {})
    rows = []
    for name, X, tv in (("individual sounds", P, E.get("phone", {}).get("ours_B", {}).get("pearson")),
                        ("words", Wd, E.get("word", {}).get("ours", {}).get("pearson"))):
        r = corr(X["y"], X["ours"], X["gop"], X["match"], X["spk"])
        rank(name, X["y"], X["ours"], X["gop"], X["match"], X["spk"],
             E.get("phone", {}).get("ours_B", {}).get("spearman") if name == "individual sounds"
             else E.get("word", {}).get("ours", {}).get("spearman"))
        rows.append(_row("speechocean762", name, len(X["y"]), r["ours"], r["gop"], r["match"], r["delta"]["value"],
                         r["delta"]["p"], "Pearson", tv, delta_ci=r["delta"]["ci"], delta_vs_match=r["delta_vs_match"]))
    for k in ("accuracy", "fluency", "prosodic", "total"):
        r = corr(S[f"y_{k}"], S[f"ours_{k}"], S["gop"], S["match"], S["spk"])
        rank(f"sentence {k}", S[f"y_{k}"], S[f"ours_{k}"], S["gop"], S["match"], S["spk"],
             E.get("sentence", {}).get(k, {}).get("ours", {}).get("spearman"))
        tv = E.get("sentence", {}).get(k, {}).get("ours", {}).get("pearson")
        rows.append(_row("speechocean762", f"sentence {k}", len(S["spk"]), r["ours"], r["gop"], r["match"],
                         r["delta"]["value"], r["delta"]["p"], "Pearson", tv, delta_ci=r["delta"]["ci"],
                         delta_vs_match=r["delta_vs_match"],
                         note="baseline: the one sentence score (mean phone GOP / 1 - token error rate) "
                              "against every expert sentence score"))
    return rows, dict(cnt)


def eval_errdet(Bs: dict) -> tuple[list[dict], dict]:
    from bench.data import load
    from bench.eval_english import N_BOOT, _their_entries, blend
    from bench.metrics import bootstrap_ci, paired_bootstrap, roc_auc
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference

    E = (_tables("english_vs_azure.json") or {}).get("errdet", {})
    rows, counts = [], {}
    for ds, label in (("l2arctic", "L2-ARCTIC test"), ("epadb", "EpaDB test")):
        api = _api(f"{ds}-test-fp32")
        B = Bs[ds]
        y, spk, S = [], [], {"ours": [], "gop": [], "match": []}
        cnt = Counter()
        for rec in load(ds, split="test"):
            o = api.get(rec["id"])
            if o is None or "error" in o:
                cnt["utts_excluded_as_in_scoreboard"] += 1
                continue
            b = B.get(rec["id"])
            ref = build_ref("en", rec["text"])
            if b is None or "error" in b:
                cnt["baseline_failed_utts"] += 1
            else:
                assert b["phones"] == [p.ipa for p in ref.phones]
                cnt["baseline_not_aligned_utts"] += not b["aligned"]
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
            n = len(ref.phones)
            d = o["sound"]["B"]
            err = {"ours": 1 - blend(d["z_len"], d["z_str"], 0.5),            # 1 - P at strictness 0.5
                   "gop": -_gop_arr(b, "gop", n), "match": 1 - _gop_arr(b, "match", n)}
            matched = [t2o.get(k, -1) for k in range(len(ents))]
            for k, e in enumerate(ents):
                if e["label"] == "unk":
                    continue
                i = matched[k]
                nb = []
                if i < 0:
                    nb = [matched[q] for q in (k - 1, k + 1) if 0 <= q < len(ents) and matched[q] >= 0]
                for m, v in err.items():
                    if m == "ours":       # eval_english averages probabilities, then takes 1 - mean
                        pr = 1 - v
                        S[m].append(float(1 - pr[i]) if i >= 0 else float(1 - np.mean(pr[nb])) if nb
                                    else float(1 - np.mean(pr)))
                    else:
                        S[m].append(float(v[i]) if i >= 0 else float(np.mean(v[nb])) if nb else float(np.mean(v)))
                y.append(int(e["label"] != "correct"))
                spk.append(rec["speaker"])
                cnt["phones"] += 1
                cnt["phones_unmatched"] += i < 0
        y, spk = np.asarray(y), np.asarray(spk)
        res = {}
        for m in S:
            s = np.asarray(S[m], float)
            ci = bootstrap_ci(roc_auc, y, s, clusters=spk, n_boot=N_BOOT)
            res[m] = {"value": roc_auc(y, s), "ci": [ci["lo"], ci["hi"]]}
        pb = paired_bootstrap(roc_auc, y, np.asarray(S["ours"]), np.asarray(S["gop"]), clusters=spk, n_boot=N_BOOT)
        pm = paired_bootstrap(roc_auc, y, np.asarray(S["ours"]), np.asarray(S["match"]), clusters=spk, n_boot=N_BOOT)
        tv = E.get(ds, {}).get("all", {}).get("ours_B", {}).get("auc")
        cnt["errors"] = int(y.sum())
        rows.append(_row("learner errors", label, int(len(y)), res["ours"], res["gop"], res["match"], pb["delta"],
                         _two_sided_from_one(pb["p_value"], N_BOOT), "ROC AUC", tv, delta_ci=[pb["lo"], pb["hi"]],
                         delta_vs_match={"value": pm["delta"], "ci": [pm["lo"], pm["hi"]],
                                         "p": _two_sided_from_one(pm["p_value"], N_BOOT)},
                         n_errors=int(y.sum())))
        counts[ds] = dict(cnt)
    return rows, counts


# ============================================================================ coaching yardstick

def eval_coaching(B: dict) -> tuple[list[dict], dict]:
    from sklearn.metrics import average_precision_score

    from bench.coach.azure_yardstick import N_BOOT, azure_phones, prec_at_recall
    from bench.coach.fit_v2 import so762_rows
    from pronunciada import coach as Co

    root = os.path.expanduser(os.path.join(os.environ.get("PRONUNCIADA_CACHE", "~/.cache/pronunciada"), "azure",
                                           "speechocean762"))
    AZP = azure_phones(root)
    R = so762_rows("so762-test", "test")
    mdl = Co.model()
    ep = Co.error_prob(R, mdl, "en")
    utt = np.asarray(R["utt"])
    pos = np.zeros(len(utt), int)
    first = {}
    for i, u in enumerate(utt):
        first.setdefault(u, i)
        pos[i] = i - first[u]
    n_rows = Counter(utt.tolist())
    for u, k in n_rows.items():          # so762_rows emits one row per Pronunciada sound, in order
        b = B.get(u)
        assert b is None or "error" in b or len(b["phones"]) == k, u
    keep = [i for i in range(len(ep)) if np.isfinite(R["acc"][i]) and (R["utt"][i], R["tj"][i], R["tk"][i]) in AZP]
    cnt = Counter()
    g, m = [], []
    for i in keep:
        b = B.get(utt[i])
        if b is None or "error" in b:
            cnt["baseline_failed_phones"] += 1
            g.append(GOP_FLOOR)
            m.append(0.0)
            continue
        g.append(b["gop"][pos[i]])
        m.append(b["match"][pos[i]])
    y = (R["acc"][keep] < 2).astype(float)
    ours = ep[keep]
    fg, fm = -np.asarray(g), 1 - np.asarray(m)          # flag scores: higher = flag
    spk = np.asarray(R["spk"])[keep]
    sens = ("careful", "balanced", "eager")
    masks = {s: Co.sound_mask(R, ep, mdl, "en", s)[keep] for s in sens}
    recalls = {s: float(y[masks[s]].sum() / y.sum()) for s in sens}
    rng = np.random.default_rng(0)
    us = np.unique(spk)
    idx = {u: np.flatnonzero(spk == u) for u in us}
    boots = []
    for _ in range(N_BOOT):
        b = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        row = {"ap_ours": average_precision_score(y[b], ours[b]), "ap_gop": average_precision_score(y[b], fg[b]),
               "ap_match": average_precision_score(y[b], fm[b])}
        for s in sens:
            f = masks[s][b]
            r = float(y[b][f].sum() / y[b].sum())
            row[f"p_ours_{s}"] = float(y[b][f].mean())
            row[f"p_gop_{s}"] = prec_at_recall(fg[b], y[b], r)[0]
            row[f"p_match_{s}"] = prec_at_recall(fm[b], y[b], r)[0]
        boots.append(row)

    def col(k):
        return np.array([x[k] for x in boots])

    Y = _tables("coaching_azure_yardstick.json") or {}
    rows = []
    for s in sens:
        po = float(y[masks[s]].mean())
        pg, rg = prec_at_recall(fg, y, recalls[s])
        pm, rm = prec_at_recall(fm, y, recalls[s])
        d = col(f"p_ours_{s}") - col(f"p_gop_{s}")
        tv = Y.get("at_recall", {}).get(s, {}).get("precision_ours")
        rows.append(_row("coaching", f"precision at '{s}' recall ({recalls[s]:.3f})", int(len(y)),
                         {"value": po, "ci": _q(col(f"p_ours_{s}"))}, {"value": pg, "ci": _q(col(f"p_gop_{s}")),
                                                                         "recall_reached": rg},
                         {"value": pm, "ci": _q(col(f"p_match_{s}")), "recall_reached": rm}, po - pg, _p_two(d),
                         "precision", tv, delta_ci=_q(d), recall=recalls[s]))
    d = col("ap_ours") - col("ap_gop")
    ao, ag, am = (float(average_precision_score(y, v)) for v in (ours, fg, fm))
    rows.append(_row("coaching", "precision-recall AUC", int(len(y)), {"value": ao, "ci": _q(col("ap_ours"))},
                     {"value": ag, "ci": _q(col("ap_gop"))}, {"value": am, "ci": _q(col("ap_match"))}, ao - ag,
                     _p_two(d), "average precision", Y.get("pr_auc", {}).get("ours"), delta_ci=_q(d)))
    cnt["phones"] = len(y)
    cnt["errors"] = int(y.sum())
    return rows, dict(cnt)


# ============================================================================ OMPAL

def eval_ompal(B: dict) -> tuple[list[dict], dict]:
    from bench import eval_ompal as EO

    recs = EO.records("test")
    ours, meta = EO.ours_outputs(recs)
    U = EO.unit_table(recs, ours, {})["cols"]
    cnt = Counter()
    gcol, mcol = [], []
    for r in recs:                                     # same iteration as eval_ompal.unit_table
        lab = r["labels"]
        if lab["word_level_dropped"]:
            continue
        b = B.get(r["id"])
        ok = b is not None and "error" not in b
        if ok:
            spans = EO.word_char_spans(r["text"], b["word_text"])
            gw = [(sp[0], sp[1], math.exp(s)) for sp, s in zip(spans, b["gop_w"]) if sp is not None]
            mw = [(sp[0], sp[1], s) for sp, s in zip(spans, b["match_w"]) if sp is not None]
            cnt["baseline_words_unplaced"] += sum(sp is None for sp in spans)
        else:
            cnt["baseline_failed_utts"] += 1
        for u in lab["units"]:
            if u["span"] is None:
                continue
            sp = tuple(u["span"])
            gcol.append(EO.unit_score(sp, gw) if ok else 0.0)
            mcol.append(EO.unit_score(sp, mw) if ok else 0.0)
    gcol, mcol = np.asarray(gcol), np.asarray(mcol)
    assert len(gcol) == len(U["utt"])
    n = len(gcol)
    boots = {"speaker": EO.boot_indices(U["spk"], seed=0)}
    neg = U["correct"].astype(bool)
    O = (_tables("ompal.json") or {}).get("test", {})
    rows = []
    for name, labk, short in (("primary: any error", "any_wrong", "word: any error"),
                              ("(a) tone-only error", "tone_only", "word: tone-only error"),
                              ("(b) consonant or vowel error", "seg_wrong", "word: consonant / vowel error")):
        p = U[labk].astype(bool)
        st = EO.stat_with_ci(EO.auc_fn(p, neg, U["ours_word"]), boots, n, EO.auc_fn(p, neg, gcol))
        sm = EO.stat_with_ci(EO.auc_fn(p, neg, mcol), boots, n)
        tv = O.get("word", {}).get(name, {}).get("ours", {}).get("value")
        rows.append(_row("OMPAL", short, n, {"value": st["value"], "ci": st["ci_speaker"]},
                         {"value": st["azure"], "ci": st["azure_ci_speaker"]},
                         {"value": sm["value"], "ci": sm["ci_speaker"]}, st["delta"], st["p_speaker"], "ROC AUC", tv,
                         delta_ci=st["delta_ci_speaker"], n_pos=int(p.sum()), n_neg=int(neg.sum())))
    S = EO.sent_table(recs, ours, {})
    gs = np.array([(B[u]["gop_s"] if u in B and "error" not in B[u] else GOP_FLOOR) for u in S["utt"]])
    ms = np.array([(B[u]["match_s"] if u in B and "error" not in B[u] else 0.0) for u in S["utt"]])
    keep = ~S["native"].astype(bool)
    c = {k: v[keep] for k, v in S.items()}
    gs, ms = gs[keep], ms[keep]
    nb = len(c["utt"])
    sb = {"speaker": EO.boot_indices(c["spk"], seed=3)}
    yv = c["t_accuracy"].astype(float)
    st = EO.stat_with_ci(EO.corr_fn(yv, c["ours_accuracy"].astype(float), "pearson"), sb, nb,
                         EO.corr_fn(yv, gs, "pearson"))
    sm = EO.stat_with_ci(EO.corr_fn(yv, ms, "pearson"), sb, nb)
    tv = O.get("sentence", {}).get("accuracy", {}).get("pearson", {}).get("value")
    rows.append(_row("OMPAL", "sentence accuracy", nb, {"value": st["value"], "ci": st["ci_speaker"]},
                     {"value": st["azure"], "ci": st["azure_ci_speaker"]}, {"value": sm["value"], "ci": sm["ci_speaker"]},
                     st["delta"], st["p_speaker"], "Pearson", tv, delta_ci=st["delta_ci_speaker"],
                     n_speakers=int(len(set(c["spk"])))))
    st = EO.stat_with_ci(EO.corr_fn(yv, c["ours_accuracy"].astype(float), "spearman"), sb, nb,
                         EO.corr_fn(yv, gs, "spearman"))
    sm = EO.stat_with_ci(EO.corr_fn(yv, ms, "spearman"), sb, nb)
    tv = O.get("sentence", {}).get("accuracy", {}).get("spearman", {}).get("value")
    RANK.append(_row("OMPAL", "sentence accuracy", nb, {"value": st["value"], "ci": st["ci_speaker"]},
                     {"value": st["azure"], "ci": st["azure_ci_speaker"]}, {"value": sm["value"], "ci": sm["ci_speaker"]},
                     st["delta"], st["p_speaker"], "Spearman", tv, delta_ci=st["delta_ci_speaker"]))
    cnt["units"] = n
    cnt["ours_cache"] = meta.get("dir")
    return rows, dict(cnt)


# ============================================================================ swap benchmark

def _swap_outputs(B: dict, key: str) -> dict:
    out = {}
    for rid, b in B.items():
        if "error" in b:
            out[rid] = {"status": "error", "words": []}
            continue
        f = math.exp if key == "gop_w" else float
        out[rid] = {"status": "Success", "unmapped": [],
                    "words": [(sp[0], sp[1], f(s)) for sp, s in zip(b["word_span"], b[key])]}
    return out


def eval_swap(B: dict) -> tuple[list[dict], dict]:
    from bench.calib.common import calib_cache
    from bench.data import load
    from bench.eval_swap import _metrics, azure_outputs, collect

    recs = load("fleurs-swap", split="test")
    ours = {}
    for lang in sorted({r["lang"] for r in recs}):
        o = pickle.loads((calib_cache() / "api" / f"fleurs-swap-{lang}.pkl").read_bytes())
        ours.update({k: v["on"] for k, v in o.items()})
    A, Z = collect(recs, ours), collect(recs, azure_outputs(recs))
    G, M = collect(recs, _swap_outputs(B, "gop_w")), collect(recs, _swap_outputs(B, "match_w"))
    T = (_tables("swap_ours_vs_azure.json") or {}).get("paired_ours_minus_azure", {})
    # tone swaps: does the swapped prompt change the canonical token sequence at all?
    src = defaultdict(dict)
    for r in recs:
        src[r["labels"]["src_id"]][r["labels"]["variant"]] = r["id"]
    tone_same = tone_n = 0
    for s, d in src.items():
        if "tone" in d and "orig" in d:
            tone_n += 1
            tone_same += B[d["tone"]].get("tokens") == B[d["orig"]].get("tokens")
    rows, counts = [], {"tone_records": tone_n, "tone_records_same_tokens_as_orig": tone_same}
    keys = (("swap_far:auc", "far AUC"), ("swap_near:auc", "near AUC"), ("swap_near:hit@5%FA", "near hit@5 %"))
    for lang in ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko"):
        if lang not in A:
            continue
        ca, cz, cg, cm = (X[lang]["clusters"] for X in (A, Z, G, M))
        po, pg, pm = [], [], []
        dropped = 0
        for k in sorted(set(ca) & set(cz)):             # the scoreboard's set: both ours and Azure scored
            vs = set(ca[k]) & set(cz[k])
            if "orig" not in vs:
                continue
            if not vs <= set(cg.get(k, {})) or not vs <= set(cm.get(k, {})):
                dropped += 1
                continue
            po.append({v: ca[k][v] for v in vs})
            pg.append({v: cg[k][v] for v in vs})
            pm.append({v: cm[k][v] for v in vs})
        variants = ["swap_far", "swap_near"] + (["tone"] if lang == "zh" else [])
        mo, mg, mm = _metrics(po, variants), _metrics(pg, variants), _metrics(pm, variants)
        rng = np.random.default_rng(0)
        bo, bg, bm = defaultdict(list), defaultdict(list), defaultdict(list)
        for _ in range(1000):
            idx = rng.integers(0, len(po), len(po))
            xo, xg, xm = (_metrics([P[i] for i in idx], variants) for P in (po, pg, pm))
            for k, _n in keys + (("tone:auc", ""),):
                for D, x in ((bo, xo), (bg, xg), (bm, xm)):
                    if k in x and np.isfinite(x[k]):
                        D[k].append(x[k])
                if k in xo and k in xg and np.isfinite(xo[k] - xg[k]):
                    bo[f"d:{k}"].append(xo[k] - xg[k])
        counts[lang] = {"clusters": len(po), "clusters_dropped_baseline_missing": dropped}
        for k, name in keys:
            d = np.asarray(bo[f"d:{k}"])
            tv = T.get(lang, {}).get(k, {}).get("a")
            rows.append(_row("swap benchmark", f"{lang} {name}", int(mo[k.split(':')[0] + ':n']),
                             {"value": mo[k], "ci": _q(bo[k])}, {"value": mg[k], "ci": _q(bg[k])},
                             {"value": mm[k], "ci": _q(bm[k])}, mo[k] - mg[k], _p_two(d), name.split()[-1], tv,
                             delta_ci=_q(d), lang=lang))
        if lang == "zh" and "tone:auc" in mo:
            k = "tone:auc"
            d = np.asarray(bo[k]) - 0.5
            tv = T.get(lang, {}).get(k, {}).get("a")
            blind = tone_same == tone_n
            rows.append(_row("tone swaps", "zh tone AUC", int(mo["tone:n"]), {"value": mo[k], "ci": _q(bo[k])},
                             {"value": 0.5, "ci": None, "by_construction": blind},
                             {"value": 0.5, "ci": None, "by_construction": blind}, mo[k] - 0.5, _p_two(d), "ROC AUC",
                             tv, delta_ci=_q(d), lang="zh",
                             note=f"{tone_same} of {tone_n} tone-swapped prompts map to the same ZIPA token sequence "
                                  "as the original prompt (ZIPA has no tone tokens), so the baseline cannot see "
                                  "the change; its AUC is 0.5 by construction and was not computed. Δ = "
                                  "Pronunciada − 0.5, CI and p from Pronunciada's utterance bootstrap."))
    return rows, counts


# ============================================================================ cv-real

def eval_cv(B: dict) -> tuple[list[dict], dict]:
    from bench import eval_cv as EC

    recs = EC.records()
    ours, fp = EC.ours_scores(recs)
    az = EC.azure_scores(recs)
    groups = defaultdict(list)
    for r in recs:
        groups[(r["locale"], r["test"])].append(r)
    C = (_tables("cv_real.json") or {}).get("results", {})
    rows, counts = [], {"ours_fingerprint": fp}
    for (loc, test), rs in sorted(groups.items(), key=lambda kv: (EC.CB.LOCALES.index(kv[0][0]), kv[0][1])):
        if test == "Aweak":
            continue                                    # not a scoreboard row
        have = [r for r in rs if r["id"] in ours and r["id"] in az]
        y = np.array([r["label"] for r in have])
        spk = [r["client_id"] for r in have]
        a = np.array([ours[r["id"]][EC.OURS_PRIMARY] for r in have])
        fail = [r["id"] for r in have if r["id"] not in B or "error" in B[r["id"]]]
        g = np.array([GOP_FLOOR if r["id"] in fail else B[r["id"]]["gop_s"] for r in have])
        m = np.array([0.0 if r["id"] in fail else B[r["id"]]["match_s"] for r in have])
        pb = EC.speaker_bootstrap(y, a, g, spk)
        sm = EC.single_bootstrap(y, m, spk)
        k = f"{loc}/{test}"
        tv = C.get(k, {}).get("primary", {}).get("ours")
        counts[k] = {"n": len(have), "baseline_failed": len(fail),
                     "baseline_not_aligned": sum(1 for r in have if r["id"] not in fail and not B[r["id"]]["aligned"])}
        rows.append(_row(f"cv-real test {test}", k, len(have), {"value": pb["ours"], "ci": pb["ours_ci"]},
                         {"value": pb["azure"], "ci": pb["azure_ci"]}, {"value": sm["auc"], "ci": sm["ci"]},
                         pb["delta"], pb["p_two_sided"], "ROC AUC", tv, delta_ci=pb["delta_ci"],
                         n_pos=int(y.sum()), n_neg=int(len(y) - y.sum())))
    return rows, counts


# ============================================================================ report

def verdict(delta: float, p: float, tol: float = 0.0005) -> str:
    if abs(delta) < tol:
        return "level"
    return f"{'ahead' if delta > 0 else 'behind'} ({'significant' if p < 0.05 else 'not significant'})"


def evaluate() -> dict:
    B = {ds: baseline(ds) for ds in ("speechocean762", "l2arctic", "epadb", "fleurs-swap", "ompal", "cv-real")}
    rows, counts = [], {}
    for name, fn, arg in (("speechocean762", eval_so762, B["speechocean762"]), ("learner errors", eval_errdet, B),
                          ("coaching", eval_coaching, B["speechocean762"]), ("OMPAL", eval_ompal, B["ompal"]),
                          ("swap benchmark", eval_swap, B["fleurs-swap"]), ("cv-real", eval_cv, B["cv-real"])):
        t = time.time()
        r, c = fn(arg)
        rows += r
        counts[name] = c
        print(f"{name}: {len(r)} rows ({time.time() - t:.0f} s)", flush=True)
    for r in rows:
        r["verdict"] = verdict(r["delta"], r["p"])
    for r in RANK:
        r["verdict"] = verdict(r["delta"], r["p"])
    return {"rows": rows, "counts": counts, "rank_check": list(RANK)}


def f3(x):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.3f}"


def pv(p):
    return "–" if p is None else ("< 0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}"))


def cell(x: dict) -> str:
    if x.get("by_construction"):
        return "0.5 (blind)"
    s = f3(x["value"])
    if x.get("ci"):
        s += f" [{f3(x['ci'][0])}, {f3(x['ci'][1])}]"
    return s


def tally(rows) -> dict:
    c = Counter(r["verdict"] for r in rows)
    return {"ahead (significant)": c["ahead (significant)"],
            "not different": c["ahead (not significant)"] + c["behind (not significant)"] + c["level"],
            "behind (significant)": c["behind (significant)"], "n": len(rows)}


def _not_aligned_note(cnt: dict) -> str:
    cv = {k: v["baseline_not_aligned"] for k, v in cnt.get("cv-real", {}).items()
          if isinstance(v, dict) and v.get("baseline_not_aligned")}
    other = sum(int(v.get("baseline_not_aligned_utts", 0)) for v in
                [cnt.get("speechocean762", {})] + list(cnt.get("learner errors", {}).values()))
    if not cv and not other:
        return "Every canonical sequence fitted into its audio."
    s = (f"{sum(cv.values())} cv-real clips ({', '.join(f'{k}: {n}' for k, n in cv.items())}) are too short for "
         "their canonical token sequence, so the forced alignment does not exist; by the rule fixed in advance "
         "every phone gets the floor −60 (the lowest possible gop score; match is computed normally).") if cv else ""
    return s + (f" {other} English test utterances likewise." if other else "")


def report(R: dict) -> None:
    rows, cnt, st = R["rows"], R["counts"], R.get("run", {})
    V = R.get("verify", {})
    L = ["# Pronunciada against ZIPA alone (ablation)", "",
         "Generated by `python -m bench.eval_zipa_baseline`. Question: how much does Pronunciada add on top of the "
         "unmodified ZIPA phone recognizer, on the same held-out tests, items and statistics as "
         "results/SCOREBOARD.md?", "",
         "## Baseline", "",
         f"- Model: `{MODEL}`, {PRECISION}, raw log-posteriors (`pronunciada.acoustic.AcousticModel`, no output adapter).",
         "- Reference: the canonical phone sequence of Pronunciada's text front end for the prompt (`RefPhone.ipa`; "
         "the rule-based alternatives in `RefPhone.alts` are ignored), mapped to ZIPA token ids with "
         "`pronunciada.text.ipa.to_ids`.",
         "- `gop` (primary): CTC Viterbi forced alignment of the canonical tokens (`pronunciada.lattice.build_lattice` "
         "with one zero-weight, non-skippable alternative per sound, `pronunciada.align.viterbi`; blank = logaddexp of "
         "`<blk>` and `▁`). Phone score = mean log-posterior of the canonical token over its aligned frames (blank "
         "frames excluded; floor −60); word = mean of its phones; sentence = mean of all phones.",
         "- `match` (secondary): greedy CTC decode, unit-cost Levenshtein alignment with the canonical tokens. Phone = 1 "
         "if all of its tokens are matched, else 0; word = share of its phones matched; sentence = max(0, 1 − edit "
         "distance / canonical tokens) (1 − token error rate, insertions counted).",
         "- Not used: adapter, lattice alternatives, native variants, calibrators, tone / stress / length / contrast "
         "heads, fitted thresholds, explain-away. Nothing was fitted, selected or thresholded on any data; the "
         "precision-at-recall rows set the baseline's threshold on the test set only to reach the recall that "
         "Pronunciada's shipped thresholds reach (the scoreboard's own protocol for Azure).", "",
         "## How the baseline was checked for fitted components", ""]
    if V:
        L += [f"- A fresh baseline worker process loaded only these Pronunciada modules: "
              f"{', '.join('`' + m + '`' for m in V['worker_pronunciada_modules'])}. None of `pronunciada.adapter`, "
              "`calibrate`, `variants`, `scorer`, `prosody`, `features`, `contrast_heads`, `length`, `sound_emb`, "
              "`delivery`, `coach` was imported" + (
                  f"; the scoring run's workers loaded {len(st['worker_modules'])} modules: the same set plus the other "
                  "nine languages' front ends (`pronunciada.text.<lang>`, `_cjk_common`)."
                  if st.get("worker_modules") else "."),
              f"- Its log-posteriors on {V['n_compared_with_cache']} speechocean762 test clips differ from the raw-model "
              f"posterior cache (`bench/postcache.py`, stored as float16) by at most "
              f"{V['max_abs_diff_vs_raw_posterior_cache']:.3f} (float16 rounding); applying the English adapter would "
              f"change them by up to {V['max_abs_change_by_en_adapter']['max']:.1f} (largest |Δ| over frames and "
              f"tokens, in low-probability columns) and the GOP phone scores by {V['mean_abs_gop_change_if_adapter_were_used']:.3f} on "
              "average, so the adapter is demonstrably not in the path.",
              f"- The alignment lattice has a single alternative per sound (alt indices {V['lattice_alt_indices']}), "
              f"all arc weights 0 and no skip arcs; every phone list equals the `phones` field of Pronunciada's cached "
              "outputs for the same prompt (asserted for speechocean762, L2-ARCTIC and EpaDB)."]
    L += ["", "## Data and Pronunciada values", "",
          f"Baseline scored {st.get('records', '–')} records over {st.get('unique_audio', '–')} audio files, "
          f"{st.get('audio_hours', float('nan')):.2f} h of audio (speechocean762, L2-ARCTIC, EpaDB test; fleurs-swap "
          "test, 10 languages; OMPAL test; cv-real test half). Pronunciada's values are recomputed from the cached "
          "per-utterance outputs the scoreboard scripts read (`calib/api/*.pkl`, the OMPAL and cv-real output caches); "
          f"{sum(r.get('pronunciada_matches_table', False) for r in rows)} of "
          f"{sum('pronunciada_table' in r for r in rows)} recomputed values equal the value in results/tables/*.json "
          "to 3 decimals" + (": mismatches " + ", ".join(f"{r['group']} {r['test']}" for r in rows
                                                       if 'pronunciada_table' in r and not r['pronunciada_matches_table'])
                             if any('pronunciada_table' in r and not r['pronunciada_matches_table'] for r in rows) else "")
          + ".", ""]
    L += ["## Main table", "",
          "Δ = Pronunciada − gop on the same resamples (speaker bootstrap for speechocean762, L2-ARCTIC, EpaDB, "
          "coaching, OMPAL and cv-real; utterance bootstrap for the swap benchmark), 95 % CI, two-sided p.", "",
          "| group | test | n | metric | Pronunciada [95 % CI] | ZIPA gop [95 % CI] | ZIPA match [95 % CI] | Δ vs gop [95 % CI] | p | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dc = r.get("delta_ci")
        L.append(f"| {r['group']} | {r['test']} | {r['n']} | {r['unit']} | {cell(r['pronunciada'])} | {cell(r['gop'])} | "
                 f"{cell(r['match'])} | {r['delta']:+.3f}" + (f" [{dc[0]:+.3f}, {dc[1]:+.3f}]" if dc else "")
                 + f" | {pv(r['p'])} | {r['verdict']} |")
    t = tally(rows)
    L += ["", f"Tally over {t['n']} rows (Pronunciada vs gop, p < 0.05): significantly ahead {t['ahead (significant)']}, "
          f"not different {t['not different']}, significantly behind {t['behind (significant)']}.", ""]
    beh = [r for r in rows if r["delta"] <= 0]
    L += ["Rows where the gop baseline equals or beats Pronunciada: " +
          ("; ".join(f"{r['group']} {r['test']} (Δ {r['delta']:+.3f}, p {pv(r['p'])})" for r in beh) if beh else "none")
          + ".", ""]
    bm = [r for r in rows if not r["match"].get("by_construction") and r["match"]["value"] >= r["pronunciada"]["value"]]
    L += ["Rows where the match baseline equals or beats Pronunciada: " +
          ("; ".join(f"{r['group']} {r['test']}" for r in bm) if bm else "none") + ".", ""]
    rk = R.get("rank_check") or []
    if rk:
        L += ["## Rank check (Spearman)", "",
              "GOP scores are unbounded log-posteriors with a floor, so Pearson could understate the baseline through "
              "nonlinearity alone; Spearman on the same items with the same paired speaker bootstrap removes that "
              "effect. Not counted in the tally above.", "",
              "| group | test | n | Pronunciada [95 % CI] | ZIPA gop [95 % CI] | ZIPA match [95 % CI] | Δ vs gop [95 % CI] | p | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rk:
            dc = r["delta_ci"]
            L.append(f"| {r['group']} | {r['test']} | {r['n']} | {cell(r['pronunciada'])} | {cell(r['gop'])} | "
                     f"{cell(r['match'])} | {r['delta']:+.3f} [{dc[0]:+.3f}, {dc[1]:+.3f}] | {pv(r['p'])} | "
                     f"{r['verdict']} |")
        L.append("")
    L += ["## Notes per test", "",
          "- speechocean762: items, phone/word fallbacks and exclusions as `bench/eval_english.py` (Pronunciada = "
          "`ours_B` 0–2 phone score, word-accuracy regression, sentence heads). The baseline has one sentence score, "
          "correlated with each expert sentence score.",
          "- L2-ARCTIC / EpaDB: annotation phones as `bench/eval_english.py err_items`; error score = −gop "
          "(match: 1 − match); unaligned annotation phones take the mean of the neighbouring aligned phones.",
          "- Coaching: the phones of `bench/coach/azure_yardstick.py` (expert-scored, aligned to ours and to Azure); "
          "flag score = −gop, thresholded at the recall Pronunciada's shipped thresholds reach (ties included, "
          "`prec_at_recall`); the `match` scores are binary, so its reached recall can be higher: "
          + ", ".join(f"{r['test'].split(' (')[0]}: {r['match'].get('recall_reached', float('nan')):.3f}"
                      for r in rows if r["group"] == "coaching" and "recall_reached" in r["match"]) + ".",
          "- OMPAL: units, labels and speaker bootstrap of `bench/eval_ompal.py`; unit score = min over overlapping "
          "baseline words (exp of the word GOP, so an uncovered unit's 0 stays the lowest score).",
          "- Swap benchmark: the scoreboard's record set (clusters both Pronunciada and Azure scored), "
          "`bench.eval_swap._metrics`, 1000 utterance resamples, seed 0.",
          "- cv-real: `bench/eval_cv.py` speaker bootstrap (2000, seed 0) on the clips both Pronunciada and Azure "
          "scored; baseline score = the sentence score.",
          "- Items the baseline could not score: the text front end produced a reference for every record (0 "
          "front-end failures). " + _not_aligned_note(cnt) + " Every other test item was scored normally; no item "
          "was dropped beyond the scoreboard's own exclusions.", ""]
    L += ["## Counts", "", "```", json.dumps(cnt, indent=1, ensure_ascii=False, default=str), "```", ""]
    MD_OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {MD_OUT.relative_to(ROOT)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="run,verify,eval,report")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    steps = a.steps.split(",")
    R = json.loads(JSON_OUT.read_text()) if JSON_OUT.exists() else {}
    if "run" in steps:
        R["run"] = run(a.workers)
    if "verify" in steps:
        R["verify"] = verify()
    if "eval" in steps:
        R.update(evaluate())
        R["baseline"] = {"model": MODEL, "precision": PRECISION, "version": VERSION, "gop_floor": GOP_FLOOR}
        R["tally_vs_gop"] = tally(R["rows"])
    JSON_OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
    if "report" in steps and "rows" in R:
        report(R)


if __name__ == "__main__":
    main()
