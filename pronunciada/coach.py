"""Coaching feedback: structured, versioned diagnoses of what went wrong in one utterance.

    items = result.coaching()                         # sensitivity "balanced"
    items = result.coaching(sensitivity="careful")    # fewer, surer items ("eager": more)
    tips = pronunciada.tips.tips(items)               # optional default English tip sentences

Every item is a plain JSON-serialisable dict with the same keys (SCHEMA_VERSION 2):

  schema_version  2
  kind            "substitution" | "missing" | "extra" | "tone" | "stress" | "missing_word"
                  | word-level miscue and delivery kinds (2.1) | "length" (2.4, Japanese)
  word, word_index        the prompt word and its index in result.words
  syllable_index          index into that word's syllables (None for missing_word)
  phone_index             index into that word's phones (sounds only, else None)
  position                "initial" | "medial" | "final" | "only" (sound in its word)
  expected, heard         sounds: IPA strings; tones: 1-5; stress: syllable index in the word;
                          heard None = "this sound was off", without naming a replacement
  expected_ipa, heard_ipa sounds only (same as expected / heard), else None
  diff_keys               articulatory features that differ (pronunciada.articulatory.diff);
                          ["tone"] / ["stress_position"]; [] when heard is None
  error_prob              coaching score: P(an expert would mark this sound / tone as an error),
                          the model of pronunciada/data/coach/model.json (fitted on learner data)
  prob                    the calibrated "acceptable?" probability (PhoneResult.prob,
                          SyllableResult.tone_prob / stress_prob, WordResult.present_prob)
  confidence              estimated precision of `heard` (sounds: exact or within one feature;
                          tones / stress: exact) on learner / planted data; None without heard
  start, end              seconds
  gate                    gate inputs, for logging

Round 2: sounds and Mandarin tones are flagged by a separate coaching operating point (the
coaching score above a per-sensitivity threshold set for precision 0.8 / 0.7 / 0.6 on
speechocean762 train and OMPAL dev), not by prob < 0.5. Nothing here changes any probability:
it reads the evidence cached on the result.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources

import numpy as np

from . import articulatory as A
from .text.ipa import MODIFIERS, token_ids

SCHEMA_VERSION = 2
SCHEMA_MINOR = 6      # 2.1 (round 3): word-level miscue and delivery kinds, key "schema_minor";
#                       2.6 (round 17): contrast-evidence items / claims ("source": "contrast");
#                       2.5 (coaching round 4): key "features" (attribute-head claims)
#                       2.2 (round 13): an experimental Japanese pitch-accent kind
#                       2.3 (round 15): that kind removed again (Japanese pitch accent is not scored)
#                       2.4 (round 15): kind "length" (long / short sounds, pronunciada.length)
SENSITIVITIES = ("careful", "balanced", "eager")
KINDS = ("substitution", "missing", "extra", "tone", "stress", "length", "missing_word", "mispronounced_word",
         "extra_speech", "long_pause", "pause_mid_phrase", "rushed_punctuation", "flat_pitch", "slow", "fast")
WORD_KINDS = ("mispronounced_word", "extra_speech")
DELIVERY_KINDS = ("long_pause", "pause_mid_phrase", "rushed_punctuation", "flat_pitch", "slow", "fast")
FIXED_RANK = {"extra_speech": 0.6, "long_pause": 0.35, "pause_mid_phrase": 0.3, "rushed_punctuation": 0.25,
              "flat_pitch": 0.2, "slow": 0.2, "fast": 0.2}
DEFAULT_GATE = {
    "flag_prob": 0.5,          # a sound / tone / stress item is reported when prob < this
    "min_log_ratio": 0.0,      # log(P heard / P expected) over the sound's span, at least
    "min_heard_post": 0.05,    # geometric-mean span posterior of the heard sound, at least
    "min_frames": 0,           # aligned frames of the sound, at least
    "missing_max_frames": 0,   # "missing" when the sound got at most this many frames ...
    "missing_min_llr_del": 0.0,  # ... and the "skipped" evidence llr_del is at most this
    "tone_min_post": 0.5,      # heard tone: its posterior at least this ...
    "tone_min_log_ratio": 0.0,  # ... and log(P heard tone / P expected tone) at least this
    "stress_min_post": 0.5,    # heard stressed syllable: its share at least this
    "missing_word_prob": 0.5,  # missing_word when present_prob < this
    "confidence": {"edges": [], "values": [0.85]},   # log_ratio bins -> precision
    "tone_confidence": 0.8, "stress_confidence": 0.8, "missing_confidence": 0.8,
    "by_lang": {},             # per-language overrides of the above (fitted where needed)
}


@lru_cache(maxsize=1)
def _shipped_gate() -> dict:
    p = resources.files("pronunciada").joinpath("data/coach/gate.json")
    try:
        g = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        g = {}
    return {**DEFAULT_GATE, **g.get("thresholds", {})}


def gate_params(**overrides) -> dict:
    """Shipped gate thresholds with keyword overrides."""
    return {**_shipped_gate(), **overrides}


@lru_cache(maxsize=None)
def inventory(lang: str) -> tuple[str, ...]:
    p = resources.files("pronunciada").joinpath("data/coach/inventory.json")
    return tuple(json.loads(p.read_text(encoding="utf-8"))["phones"].get(lang, ()))


@lru_cache(maxsize=None)
def _cand_ids(lang: str, vowel: bool):
    """Candidate sounds of the same type (vowel / consonant) as the expected one: planted and
    annotated learner substitutions are within a type, and a cross-type "heard" is nearly
    always a boundary effect of the alignment. Returns (names, padded token ids [n, L],
    1 / length per row, mask [n, L])."""
    tid = token_ids()
    cands = [c for c in inventory(lang) if A.covered(c) and (A.features(c)["type"] == "vowel") == vowel]
    L = max((len(c) for c in cands), default=1)
    ids = np.zeros((len(cands), L), dtype=np.int64)
    mask = np.zeros((len(cands), L))
    for r, c in enumerate(cands):
        ids[r, :len(c)] = [tid[ch] for ch in c]
        mask[r, :len(c)] = 1.0
    return cands, ids, 1.0 / np.maximum(mask.sum(1), 1), mask


def span_mean(lpa: np.ndarray, s: int, e: int) -> np.ndarray:
    """Mean posterior over the sound's span padded by one frame (pronunciada.features rule)."""
    T = len(lpa)
    a, b = (max(s - 1, 0), min(s + 1, T)) if e <= s else (max(s - 1, 0), min(e + 1, T))
    if b <= a:
        a, b = max(T - 1, 0), T
    return np.exp(lpa[a:b].astype(np.float64)).mean(0)


def heard_candidate(m: np.ndarray, expected: str, alts, lang: str):
    """Best other sound of the language (same type) by the geometric-mean span posterior of its
    tokens. Returns (heard, log P heard, log P expected)."""
    tid = token_ids()
    lm = np.log(m + 1e-6)
    cands, ids, inv, mask = _cand_ids(lang, A.features(expected)["type"] == "vowel")
    le = float(lm[[tid[ch] for ch in expected]].mean())
    if not cands:
        return None, -math.inf, le
    sc = (lm[ids] * mask).sum(1) * inv
    skip = {expected, *alts}
    for r in np.argsort(-sc):
        if cands[r] not in skip:
            return cands[r], float(sc[r]), le
    return None, -math.inf, le


def _position(k: int, n: int) -> str:
    return "only" if n == 1 else "initial" if k == 0 else "final" if k == n - 1 else "medial"


@lru_cache(maxsize=1)
def model() -> dict:
    """The fitted coaching model (bench/coach/fit_v2.py); {} when not shipped."""
    p = resources.files("pronunciada").joinpath("data/coach/model.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return {}


def sound_class(ipa: str) -> str:
    """'vowel' or the consonant manner (stop, affricate, fricative, nasal, approximant, ...)."""
    f = A.features(ipa)
    return "vowel" if f["type"] == "vowel" else f["manner"]


GLIDES = frozenset(("j", "w", "ɥ"))
SOUND_FEATURES = ("zl", "zs", "llr_sub", "z_llr_sub", "log_ratio_comp", "log_frames", "nucleus",
                  "llr_del", "peak_post", "blank_mean")


def sound_inputs(result, heard: bool | list = False) -> dict:
    """Per reference sound (arrays, one entry per sound): everything the coaching model and the
    gates read. heard=True computes the heard candidate for every sound, a list of indices for
    those only (others None)."""
    c = result._pronunciada  # noqa: SLF001
    ref, ev, tab = c["ref"], c["ev"], c["tab"]
    n = len(ref.phones)
    f = lambda k: np.asarray(tab[k], dtype=np.float64)  # noqa: E731
    first = np.zeros(n, bool)
    for sy in ref.syllables:
        first[sy.phones[0]] = True
    X = {"i": np.arange(n), "word": np.array([p.word for p in ref.phones]),
         "syl": np.array([p.syllable - ref.words[p.word].syllables[0] for p in ref.phones]),
         "k": np.array([i - ref.words[p.word].phones[0] for i, p in enumerate(ref.phones)]),
         "expected": [p.ipa for p in ref.phones], "alts": [[a for a, _ in p.alts] for p in ref.phones],
         "nucleus": np.array([float(p.is_nucleus) for p in ref.phones]), "first_in_syl": first,
         "zl": np.asarray(c.get("coach_zl", c["s_zl"]), np.float64),   # round 12b: detail model
         "zs": np.asarray(c.get("coach_zs", c["s_zs"]), np.float64),
         "llr_sub": f("llr_sub"), "llr_del": f("llr_del"), "z_llr_sub": f("z_llr_sub"),
         "log_ratio_comp": f("log_ratio_comp"), "comp_mean": f("comp_mean"), "can_mean": f("can_mean"),
         "peak_post": f("peak_post"), "blank_mean": f("blank_mean"), "n_frames": f("n_frames"),
         "slot_heard": [s.heard for s in ev.slots],
         "start_frame": np.array([s.start_frame for s in ev.slots]),
         "end_frame": np.array([s.end_frame for s in ev.slots]), "phone_vec": c.get("phone_vec")}
    X["log_frames"] = np.log1p(X["n_frames"])
    # syllable-initial consonant (Mandarin "initial"; medial glides belong to the final)
    X["initial"] = first & (X["nucleus"] == 0) & np.array([e not in GLIDES for e in X["expected"]])
    X["cand"], X["lh"], X["le"] = [None] * n, np.full(n, np.nan), np.full(n, np.nan)
    _add_heard(X, result, range(n) if heard is True else (heard or []))
    return X


def _add_heard(X: dict, result, idx) -> None:
    c = result._pronunciada  # noqa: SLF001
    lpa = c.get("lpa")
    if lpa is None or not len(lpa):
        return
    for i in idx:
        m = span_mean(lpa, int(X["start_frame"][i]), int(X["end_frame"][i]))
        X["cand"][i], X["lh"][i], X["le"][i] = heard_candidate(m, X["expected"][i], X["alts"][i], c["ref"].lang)


_CLIP = {"zl": (-12, 12), "zs": (-12, 12), "llr_sub": (-30, 0), "z_llr_sub": (-6, 6),
         "log_ratio_comp": (-10, 10), "log_frames": (0, 4), "nucleus": (0, 1), "llr_del": (-30, 30),
         "peak_post": (0, 1), "blank_mean": (0, 1)}


def feature_matrix(X: dict, names=SOUND_FEATURES) -> np.ndarray:
    cols = []
    for k in names:
        v = np.nan_to_num(np.asarray(X[k], np.float64), nan=0.0, posinf=_CLIP[k][1], neginf=_CLIP[k][0])
        cols.append(np.clip(v, *_CLIP[k]))
    return np.column_stack(cols) if cols else np.zeros((0, 0))


def _logistic(M: dict, F: np.ndarray) -> np.ndarray:
    z = ((F - np.asarray(M["mean"])) / np.asarray(M["scale"])) @ np.asarray(M["coef"]) + M["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def error_prob(X: dict, mdl: dict | None = None, lang: str | None = None) -> np.ndarray:
    """Coaching score per sound: P(an expert marks it as an error). A language may carry its own
    logistic (by_lang[lang]["sound_model"]; round 12b: English, fitted on sound_E inputs)."""
    mdl = mdl or model()
    M = mdl.get("by_lang", {}).get(lang, {}).get("sound_model", mdl["sound"])
    ep = _logistic(M, feature_matrix(X, M["features"]))
    if lang == "zh" and mdl.get("by_lang", {}).get("zh", {}).get("emb"):
        from .coach_zh import error_prob as _zh   # round 12b: per-sound activations (OMPAL dev)

        ep = _zh(X, ep)
    return ep


def missing_prob(X: dict, mdl: dict | None = None) -> np.ndarray | None:
    mdl = mdl or model()
    M = mdl.get("missing")
    if not M or M.get("threshold") is None:
        return None
    return _logistic(M, feature_matrix(X, M["features"]))


def sound_threshold(mdl: dict, lang: str, sensitivity: str, initial: bool) -> float:
    bl = mdl.get("by_lang", {}).get(lang, {})
    if "initial" in bl and "final" in bl:
        return bl["initial" if initial else "final"][sensitivity]
    if "sound" in bl:
        value = bl["sound"][sensitivity]
        return value["fallback"] if isinstance(value, dict) else value
    return mdl["thresholds"][sensitivity]


def sound_mask(X: dict, ep: np.ndarray, mdl: dict, lang: str, sensitivity: str,
               threshold: float | None = None) -> np.ndarray:
    if threshold is not None:
        return ep >= threshold
    if lang == "en" and sensitivity == "balanced":
        rule = mdl.get("by_lang", {}).get(lang, {}).get("sound", {}).get(sensitivity)
        if isinstance(rule, dict):
            cuts = [rule["thresholds"].get(sound_class(phone), rule["fallback"])
                    for phone in X["expected"]]
            return ep >= cuts
    ti = sound_threshold(mdl, lang, sensitivity, True)
    tf = sound_threshold(mdl, lang, sensitivity, False)
    return np.where(X["initial"], ep >= ti, ep >= tf)


def heard_ok(mdl: dict, lang: str, expected: str, cand, lh: float, le: float) -> bool:
    """The fitted heard-sound gate (learner data): class switched on, log-ratio and posterior bars."""
    H = mdl.get("heard", {})
    if cand is None or not np.isfinite(lh):
        return False
    cls = sound_class(expected)
    if cls in H.get("off_classes", []):
        return False
    g = {**H, **H.get("by_lang", {}).get(lang, {})}
    return (lh - le) >= g.get("min_log_ratio", -1e9) and math.exp(lh) >= g.get("min_heard_post", 0.0)


def decide(X: dict, lang: str, sensitivity: str = "balanced", mdl: dict | None = None,
           threshold: float | None = None, skip_words=()) -> dict:
    """Per-sound decisions shared by coaching() and the benchmarks: error_prob, flagged,
    kind ("missing" / "substitution" / None) and heard (None = not named). X must carry the
    heard candidate for the flagged sounds (sound_inputs(..., heard=...))."""
    mdl = mdl or model()
    ep = error_prob(X, mdl, lang)
    flag = sound_mask(X, ep, mdl, lang, sensitivity, threshold)
    flag &= ~np.isin(X["word"], list(skip_words))
    mp = missing_prob(X, mdl)
    n = len(ep)
    kind, heard = [None] * n, [None] * n
    for i in np.flatnonzero(flag):
        if mp is not None and mp[i] >= mdl["missing"]["threshold"]:
            kind[i], heard[i] = "missing", ""
            continue
        kind[i] = "substitution"
        if heard_ok(mdl, lang, X["expected"][i], X["cand"][i], X["lh"][i], X["le"][i]):
            heard[i] = X["cand"][i]
    return {"error_prob": ep, "flag": flag, "kind": kind, "heard": heard, "missing_prob": mp}


def flagged_indices(X: dict, lang: str, sensitivity: str = "balanced", mdl: dict | None = None,
                    threshold: float | None = None) -> np.ndarray:
    mdl = mdl or model()
    ep = error_prob(X, mdl, lang)
    return np.flatnonzero(sound_mask(X, ep, mdl, lang, sensitivity, threshold))


def _item(kind, w, wi, si, pi, pos, expected, heard, diff, eprob, prob, conf, start, end, gate=None, ipa=True):
    return {"schema_version": SCHEMA_VERSION, "schema_minor": SCHEMA_MINOR, "kind": kind, "word": w.text,
            "word_index": wi,
            "syllable_index": si, "phone_index": pi, "position": pos, "expected": expected,
            "heard": heard, "expected_ipa": expected if ipa else None,
            "heard_ipa": heard if ipa else None, "diff_keys": list(diff),
            "error_prob": None if eprob is None else round(float(eprob), 4),
            "prob": None if prob is None else round(float(prob), 4),
            "confidence": None if conf is None else round(float(conf), 3),
            "start": start, "end": end, "gate": gate}


def _rank(it) -> float:
    if it["kind"] in FIXED_RANK:
        return FIXED_RANK[it["kind"]]
    e = it["error_prob"] if it["error_prob"] is not None else 1.0 - (it["prob"] or 0.0)
    c = it["confidence"] if it["confidence"] is not None else 0.5
    return e * (0.5 + 0.5 * c)


def coaching(result, max_items: int | None = None, sensitivity: str = "balanced", miscues: bool = True,
             delivery: bool | str = True, **gate) -> list[dict]:
    """Coaching items for a SentenceResult from Scorer.score / rescore, most important first
    (by error_prob x (0.5 + 0.5 confidence)). sensitivity: "careful" | "balanced" | "eager".
    Keyword overrides: `threshold` (coaching-score bar for sounds), `tone_threshold`, and the
    stress / missing-word bars of DEFAULT_GATE. miscues / delivery add the word-level reading
    miscues (mispronounced_word, extra_speech) and delivery flags (pronunciada.delivery_coach);
    delivery="all" also reports the unvalidated slow / flat_pitch flags."""
    if sensitivity not in SENSITIVITIES:
        raise ValueError(f"sensitivity must be one of {SENSITIVITIES}")
    c = getattr(result, "_pronunciada", None)
    if c is None:
        raise ValueError("result has no cached evidence (was it produced by Scorer.score?)")
    mdl = model()
    if not mdl:
        raise RuntimeError("pronunciada/data/coach/model.json is missing")
    ref = c["ref"]
    lang = ref.lang
    g = gate_params(**{k: v for k, v in gate.items() if k in DEFAULT_GATE})
    pr = c.get("prosody") or []
    omitted = {wi for wi, w in enumerate(result.words)
               if w.present_prob is not None and w.present_prob < g["missing_word_prob"]}
    X = sound_inputs(result)
    sel = [i for i in flagged_indices(X, lang, sensitivity, mdl, gate.get("threshold"))
           if X["word"][i] not in omitted]
    if sel:
        _add_heard(X, result, sel)
    D = decide(X, lang, sensitivity, mdl, gate.get("threshold"), omitted)
    hconf = mdl.get("heard", {}).get("confidence", {})
    items: list[dict] = []
    attr_rows: list[tuple[int, int]] = []          # (item index, sound index) for the attribute heads
    for wi in sorted(omitted):
        w = result.words[wi]
        items.append(_item("missing_word", w, wi, None, None, None, w.text, None, [], 1.0 - w.present_prob,
                           w.present_prob, None, w.start, w.end, ipa=False))
    for i in np.flatnonzero(D["flag"]):
        wi, k = int(X["word"][i]), int(X["k"][i])
        w = result.words[wi]
        p = w.phones[k]
        e = X["expected"][i]
        gin = {"error_prob": round(float(D["error_prob"][i]), 4)}
        if X["cand"][i] is not None:
            gin.update(candidate=X["cand"][i], heard_post=round(math.exp(X["lh"][i]), 4),
                       log_ratio=round(float(X["lh"][i] - X["le"][i]), 3))
        h = D["heard"][i]
        if D["kind"][i] == "missing":
            items.append(_item("missing", w, wi, int(X["syl"][i]), k, _position(k, len(w.phones)), e, "", [],
                               D["error_prob"][i], p.prob, mdl["missing"].get("confidence"), p.start, p.end, gin))
            continue
        d = A.diff(e, h) if h else []
        conf = hconf.get(sound_class(e), hconf.get("all")) if h else None
        items.append(_item("substitution", w, wi, int(X["syl"][i]), k, _position(k, len(w.phones)), e, h, d,
                           D["error_prob"][i], p.prob, conf, p.start, p.end, gin))
        attr_rows.append((len(items) - 1, i))
    _attach_features(result, items, attr_rows, X, lang, mdl, sensitivity, D["error_prob"], gate.get("features", True))
    _contrast_items(result, items, X, lang, sensitivity, omitted, D["error_prob"], gate.get("features", True))
    T = mdl.get("tone", {})
    for wi, w in enumerate(result.words):
        if wi in omitted:
            continue
        s0 = ref.words[wi].syllables[0]
        for k, sy in enumerate(w.syllables):
            si = s0 + k
            d = pr[si] if si < len(pr) else {}
            if sy.tone_prob is not None and d.get("tone_probs") and T:
                tthr = gate.get("tone_threshold", T["thresholds"][sensitivity])
                if sy.tone_prob <= tthr:
                    P = np.asarray(d["tone_probs"], dtype=np.float64)
                    acc = {sy.tone_expected, *ref.syllables[si].tone_alts}
                    ht = int(P.argmax()) + 1
                    lr = math.log(P[ht - 1] + 1e-6) - math.log(P[sy.tone_expected - 1] + 1e-6)
                    ok = ht not in acc and P[ht - 1] >= g["tone_min_post"] and lr >= g["tone_min_log_ratio"]
                    items.append(_item("tone", w, wi, k, None, None, sy.tone_expected, ht if ok else None,
                                       ["tone"] if ok else [], tone_error_prob(T, sy.tone_prob),
                                       sy.tone_prob, g["tone_confidence"] if ok else None, sy.start, sy.end,
                                       {"heard_post": round(float(P[ht - 1]), 4), "log_ratio": round(lr, 3)},
                                       ipa=False))
            ws = w.stress_prob if w.stress_prob is not None else 0.0
            if (sy.stress_prob is not None and sy.stress_prob < g["flag_prob"] and ws < g["flag_prob"]
                    and d.get("stress_dist")):
                P = np.asarray(d["stress_dist"], dtype=np.float64)
                hs = int(P.argmax())
                ok = hs != k and P[hs] >= g["stress_min_post"]
                items.append(_item("stress", w, wi, k, None, None, k, hs if ok else None,
                                   ["stress_position"] if ok else [], None, sy.stress_prob,
                                   g["stress_confidence"] if ok else None, w.start, w.end,
                                   {"heard_post": round(float(P[hs]), 4), "n_syllables": len(P)},
                                   ipa=False))
    items += length_items(result, sensitivity, omitted)
    if miscues or delivery:
        from . import delivery_coach as DC

        if miscues:
            items += DC.miscue_items(result, items, mdl)
        if delivery:
            kinds = DELIVERY_KINDS if delivery == "all" else DC.DEFAULT_DELIVERY
            items += DC.delivery_items(result, mdl, kinds)
    for it in items:
        it.setdefault("features", [])
    items.sort(key=_rank, reverse=True)
    return items[:max_items] if max_items else items


def length_items(result, sensitivity: str, omitted=()) -> list[dict]:
    """Round 15: kind "length" for sounds the length channel judges (PhoneResult.length_prob).
    A sound is reported when length_prob <= the language's coaching threshold for the
    sensitivity (pronunciada/data/length/<lang>.json "coach": the loosest native quantile whose
    estimated precision on FLEURS dev reaches 0.8 / 0.7 / 0.6; threshold 0 = off). expected /
    heard are "long" / "short" (the heard length is the other one); expected_ipa is the sound."""
    from . import length as L

    par = L.params(result.lang)
    co = (par or {}).get("coach")
    if not co:
        return []
    thr = co["thresholds"][sensitivity]
    if thr <= 0:                    # no operating point reaches this sensitivity's precision
        return []
    out = []
    for wi, w in enumerate(result.words):
        if wi in omitted:
            continue
        for k, p in enumerate(w.phones):
            if p.length_prob is None or p.length_prob > thr:
                continue
            heard = "short" if p.length_expected == "long" else "long"
            it = _item("length", w, wi, None, k, _position(k, len(w.phones)), p.length_expected, heard,
                       ["length"], 1.0 - p.length_prob, p.length_prob, co["precision"][sensitivity],
                       p.start, p.end, {"length_prob": round(float(p.length_prob), 4), "threshold": thr},
                       ipa=False)
            it["expected_ipa"] = p.ipa
            out.append(it)
    return out


def _contrast_items(result, items, X, lang, sensitivity, omitted, eprob, enabled=True) -> None:
    """Round 17: the head-based contrast evidence (pronunciada.contrast_heads; Korean / Mandarin
    aspiration, Russian soft / hard) below the sensitivity's bar adds that feature's claim to the
    sound's item, or a new item when the sound was not flagged."""
    c = result._pronunciada  # noqa: SLF001
    ev = c.get("contrast_ev")
    if not enabled or not ev:
        return
    from . import attr_heads as AH
    from .contrast_heads import entries

    by_sound = {(it["word_index"], it["phone_index"]): it for it in items if it["kind"] == "substitution"}
    for e in entries(lang):
        z = ev.get(e["contrast"])
        bar = (e.get("coach") or {}).get(sensitivity)
        if z is None or bar is None:
            continue
        h = e["head"]
        for i in np.flatnonzero(np.isfinite(z)):
            p = 1.0 / (1.0 + math.exp(-float(z[i])))
            if p >= bar:
                continue
            wi, k = int(X["word"][i]), int(X["k"][i])
            if wi in omitted:
                continue
            exp = X["expected"][i]
            if h == "tense":                          # round 18: Korean lax / tense (RefPhone.tense)
                t = getattr(c["ref"].phones[i], "tense", None)
                if t is None:
                    continue
                ec = "yes" if t else "no"
            else:
                ec = AH.labels(exp).get(h)
            other = "no" if ec == "yes" else "yes"
            claim = {"feature": h, "key": AH.HEADS[h][2] if h in AH.HEADS else h, "expected": ec, "heard": other,
                     "p": round(1.0 - p, 4), "source": "contrast"}
            it = by_sound.get((wi, k))
            if it is not None:
                if it["heard"] and AH.labels(it["heard"]).get(h) != other:
                    continue                          # would contradict the named sound
                if not any(f["feature"] == h for f in it.get("features") or []):
                    it["features"] = [claim] + list(it.get("features") or [])
                    if it["heard"] is None:
                        it["diff_keys"] = list(dict.fromkeys([claim["key"]] + it["diff_keys"]))
                continue
            w = result.words[wi]
            ph = w.phones[k]
            new = _item("substitution", w, wi, int(X["syl"][i]), k, _position(k, len(w.phones)), exp, None,
                        [claim["key"]], 1.0 - p, ph.prob, None, ph.start, ph.end,
                        {"contrast": e["contrast"], "evidence": round(p, 4), "bar": bar})
            new["features"] = [claim]
            items.append(new)
            by_sound[(wi, k)] = new


def attr_policy(mdl: dict, lang: str, sensitivity: str) -> tuple[dict, dict]:
    """(bars, min_ep) of the feature-claim policy (model.json "attr", bench/coach/fit_r4.py)."""
    A_ = mdl.get("attr") or {}
    pol = (A_.get("by_sens") or {}).get(sensitivity) or {}
    floor = ((A_.get("by_lang") or {}).get(lang) or {}).get(sensitivity, 0.0) or 0.0
    bars = {h: (v["q"] if v else None) for h, v in pol.items()}
    return bars, {h: max(v["t"], floor) for h, v in pol.items() if v}


def _attach_features(result, items, rows, X, lang, mdl, sensitivity, eprob, enabled=True) -> None:
    """Round 4: feature-level claims of the attribute heads on flagged substitutions; with no
    named sound they also fill diff_keys (pronunciada.coach_attr)."""
    A_ = mdl.get("attr")
    if not enabled or not rows or not A_:
        return
    from . import attr_heads as AH
    from .coach_attr import claims_for, sound_vectors

    heads = AH.shipped()
    vc = sound_vectors(result) if heads is not None else None
    if vc is None:
        return
    bars, min_ep = attr_policy(mdl, lang, sensitivity)
    agree = bool(A_.get("agree_with_candidate"))
    P = heads.predict(vc[0], [i for _, i in rows], vc[1])
    prec = (A_.get("precision") or {}).get(sensitivity)
    for (j, i), p in zip(rows, P):
        it = items[j]
        cl = claims_for(p, X["expected"][i], X["alts"][i], it["heard"], bars, cand=X["cand"][i],
                        eprob=float(eprob[i]), min_ep=min_ep, agree=agree)
        it["features"] = cl
        if it["heard"] is None and cl:
            it["diff_keys"] = list(dict.fromkeys(c["key"] for c in cl))
            it["confidence"] = None if prec is None else round(float(prec), 3)


def tone_error_prob(T: dict, tone_prob: float) -> float:
    """P(teachers mark the tone wrong) from the syllable tone probability (OMPAL dev fit)."""
    q = min(max(float(tone_prob), 1e-4), 1 - 1e-4)
    return float(1.0 / (1.0 + math.exp(-(T["a"] + T["b"] * math.log(q / (1 - q))))))


def to_json(items: list[dict], **kw) -> str:
    return json.dumps({"schema_version": SCHEMA_VERSION, "schema_minor": SCHEMA_MINOR, "items": items}, ensure_ascii=False, **kw)


__all__ = ["SCHEMA_VERSION", "SENSITIVITIES", "KINDS", "DEFAULT_GATE", "coaching", "gate_params",
           "heard_candidate", "sound_inputs", "error_prob", "to_json", "MODIFIERS"]
