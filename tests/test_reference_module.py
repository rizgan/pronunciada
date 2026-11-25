"""The dependency-free reference (pronunciada/reference): standard library only, reproduces its
golden vectors exactly, and agrees with production.

Measured agreement with production (fp32 posteriors; tests below assert it):
  L_can, llr_sub, llr_del, llr_ins, spans, alternative index, heard, free tokens: identical
  (the reference adds in production's summation order, numeric.py);
  peak_post: within one float32 unit in the last place (<= 5.97e-8; production evaluates exp in
  float32, which equals C expf here; the reference rounds the double exp to float32);
  sound / word tree outputs, calibration maps, log-odds, omission probability, strictness blend:
  identical.
"""
from __future__ import annotations

import base64
import gzip
import json
import math
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "tests" / "data" / "reference_golden"
LANGS = ("en", "de", "es", "fr", "it", "pt", "ru", "ja", "ko", "zh")
PEAK_BOUND = 5.97e-8          # one float32 ulp just below 1.0 (peak_post <= 1)
# The golden vectors were written on macOS arm64, where the reference reproduces them to the last
# bit. Other platforms have a different C math library and NumPy kernels (exp, log), so results
# there differ in the last bits; the same checks then allow a small tolerance.
EXACT = sys.platform == "darwin" and platform.machine() == "arm64"
TOL = 1e-5
if not EXACT:
    PEAK_BOUND = TOL


def _near(a, b):
    """Deep equality. Floats are compared exactly on the platform the vectors were written on, within TOL elsewhere."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_near(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_near(a[k], b[k]) for k in a)
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if not (number(a) and number(b)) or (isinstance(a, int) and isinstance(b, int)):
        return a == b
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if EXACT or math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def _enc(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    if isinstance(x, dict):
        return {k: _enc(v) for k, v in x.items()}
    return x


def _dec(x):
    if x == "NaN":
        return float("nan")
    if x == "Infinity":
        return float("inf")
    if x == "-Infinity":
        return float("-inf")
    if isinstance(x, list):
        return [_dec(v) for v in x]
    return x


def _load(lang):
    with gzip.open(GOLD / f"{lang}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def _lp(clip):
    raw = base64.b64decode(clip["lp_q"])
    T = clip["T"]
    vals = struct.unpack("<%dh" % (T * 127), raw)
    s = float(clip["lp_scale"])
    return [[vals[t * 127 + k] / s for k in range(127)] for t in range(T)]


def test_reference_imports_no_numpy():
    code = ("import sys\n"
            "import pronunciada.reference as R\n"
            "from pronunciada.reference import align, calibrate, features, lattice, npz, numeric, pipeline, tokens\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('numpy', 'scipy', 'onnxruntime'))\n"
            "print(','.join(bad))\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT), check=True)
    assert out.stdout.strip() == ""


def test_reference_sources_are_stdlib_only():
    allowed = {"__future__", "ast", "fractions", "json", "math", "os", "struct", "zipfile"}
    for p in (ROOT / "pronunciada" / "reference").glob("*.py"):
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("import ") or (s.startswith("from ") and " import " in s):
                mod = s.split()[1]
                if mod.startswith("."):
                    continue
                assert mod.split(".")[0] in allowed, (p.name, s)


@pytest.mark.parametrize("lang", LANGS)
def test_golden_vectors_reproduce_exactly(lang):
    from pronunciada.reference import align as RA
    from pronunciada.reference import calibrate as RC

    doc = _load(lang)
    toks = doc["tokens"]
    assert len(doc["clips"]) >= 2
    for clip in doc["clips"]:
        lp = _lp(clip)
        phones = clip["phones"]
        for cs in clip["cases"]:
            t0, t1 = cs["frames"]
            p0, p1 = cs["phones"]
            got = RA.score_phones(lp[t0:t1], phones[p0:p1], toks, wild_slots=cs["wild_slots"],
                                  insertion=cs["insertion"], exact=cs["exact"])
            assert _near(json.loads(json.dumps(_enc(got))), cs["expected"]), (clip["id"], cs["name"])
        for key in ("sound_model", "word_model"):
            m = RC.LevelModel(str(ROOT / clip[key]["file"]))
            got = [(m.predict(_dec(r)) | m.logits(_dec(r))) if key == "sound_model" else m.logits(_dec(r))
                   for r in clip[key]["rows"]]
            assert _near(_enc(got), clip[key]["expected"]), (clip["id"], key)
        om = json.loads((ROOT / clip["omission"]["file"]).read_text())
        got = [RC.present_prob(om, _dec(x)) for x in clip["omission"]["llr_del"]]
        assert _near(got, clip["omission"]["expected"])
        from pronunciada.reference import pipeline as RP

        ev = RA.score_phones(lp, phones, toks)
        us = RP.universal_scores(lp, clip["target"], ev, lang, toks, clip["dur"])
        assert _near(_enc({"sound": us["sound"], "word": us["word"]}), clip["universal"]["expected"]), clip["id"]


def _prod_evidence(lp, phones, wild_slots, insertion, exact):
    from pronunciada.align import score_slots
    from pronunciada.lattice import slots_from_reference, wild_slot

    ref = SimpleNamespace(phones=[SimpleNamespace(ipa=p["ipa"], alts=[tuple(a) for a in p["alts"]]) for p in phones])
    slots = slots_from_reference(ref)
    fixed = sorted(set(wild_slots))
    for i in fixed:
        slots[i] = wild_slot(slots[i])
    return score_slots(np.asarray(lp, np.float32).reshape(-1, 127), slots, exact=exact, fixed=fixed,
                       insertion=insertion)


def _reference(target):
    from pronunciada.types import Reference, RefPhone, RefSyllable, RefWord

    ref = Reference(lang="", text="")
    ref.phones = [RefPhone(p["ipa"], p["word"], p["syllable"], p["is_nucleus"], [tuple(a) for a in p["alts"]])
                  for p in target["phones"]]
    ref.syllables = [RefSyllable(word=-1, phones=(0, 0), stress=s["stress"]) for s in target["syllables"]]
    ref.words = [RefWord(w["text"], (0, 0), tuple(w["phones"])) for w in target["words"]]
    return ref


def _same(a, b):
    return _near(float(a), float(b))


def _check_evidence(e, r, insertion):
    assert _same(e.L_can, r["L_can"])
    assert e.free_tokens == [tuple(x) for x in r["free_tokens"]]
    assert len(e.slots) == len(r["slots"])
    for s, q in zip(e.slots, r["slots"]):
        assert _same(s.llr_sub, q["llr_sub"]) and _same(s.llr_del, q["llr_del"])
        assert (s.start_frame, s.end_frame, s.alt_index, s.heard, s.n_frames) == \
            (q["start_frame"], q["end_frame"], q["alt_index"], q["heard"], q["n_frames"])
        assert abs(s.peak_post - q["peak_post"]) <= PEAK_BOUND
    if insertion:
        assert all(_same(a, b) for a, b in zip(e.llr_ins, r["llr_ins"]))


@pytest.mark.parametrize("lang", LANGS)
def test_golden_vectors_match_production(lang):
    from pronunciada import calibrate as C
    from pronunciada.scorer import _present_prob

    doc = _load(lang)
    for clip in doc["clips"]:
        lp = _lp(clip)
        for cs in clip["cases"]:
            t0, t1 = cs["frames"]
            p0, p1 = cs["phones"]
            e = _prod_evidence(lp[t0:t1], clip["phones"][p0:p1], cs["wild_slots"], cs["insertion"], cs["exact"])
            r = json.loads(json.dumps(cs["expected"]), parse_constant=float)
            r = {k: _dec(v) if k in ("L_can", "llr_ins") else v for k, v in r.items()}
            for q in r["slots"]:
                for k in ("llr_sub", "llr_del"):
                    q[k] = _dec(q[k])
            _check_evidence(e, r, cs["insertion"])
        for key in ("sound_model", "word_model"):
            m = C.load_level(Path(clip[key]["file"]).stem)
            X = np.array([_dec(r) for r in clip[key]["rows"]], dtype=np.float64)
            out = m.predict(X=X) | m.logits(X=X) if key == "sound_model" else m.logits(X=X)
            for i, exp_row in enumerate(clip[key]["expected"]):
                for h, v in exp_row.items():
                    assert _same(out[h][i], _dec(v)), (clip["id"], key, h, i)
        om = json.loads((ROOT / clip["omission"]["file"]).read_text())
        prod = [_present_prob(om, np.array(_dec(x))) for x in clip["omission"]["llr_del"]]
        assert _near([float(x) for x in prod], clip["omission"]["expected"])
        # universal sound model + word model on production's own evidence and feature table
        from pronunciada import features as F
        from pronunciada.variants import NativeStats

        ref = _reference(clip["target"])
        lp32 = np.asarray(lp, np.float32)
        from pronunciada.align import score_reference

        tab = F.phone_table(lp32, ref, score_reference(lp32, ref), NativeStats.load(lang), clip["dur"])
        cfg = C.config()
        uni, word = C.load_level(cfg["universal_sound_model"]), C.load_level("word")
        u = uni.logits(X=uni.matrix(tab))
        bias = float(cfg.get("bias", {}).get(lang, 0.0))
        zl, zs = u["z_len"] + bias, u["z_str"] + bias
        W = F.word_table(tab, zl, zs, u["reg"])
        w = word.logits(X=np.column_stack([np.asarray(W[k], dtype=np.float64) for k in word.num]))
        exp_u = clip["universal"]["expected"]
        for got, key in ((zl, ("sound", "z_len")), (zs, ("sound", "z_str")), (u["reg"], ("sound", "reg")),
                         (w["z_len"], ("word", "z_len")), (w["z_str"], ("word", "z_str"))):
            want = _dec(exp_u[key[0]][key[1]])
            assert all(_same(a, b) for a, b in zip(got.tolist(), want)), (clip["id"], key)


def test_scorer_formulas_match_production():
    from pronunciada import calibrate as C
    from pronunciada.reference import calibrate as RC
    from pronunciada.scorer import _pros_prob

    rng = np.random.default_rng(7)
    cal = {"coef": [0.3, 0.9, 0.2, -0.05], "c_len": 0.7, "c_str": -0.4}
    for _ in range(200):
        probs = rng.uniform(0, 1, rng.integers(1, 5)).tolist()
        s = float(rng.uniform(-0.6, 1.6))
        assert _near(RC.prosody_prob(cal, probs, s), float(_pros_prob(cal, probs, s)))
        p1, p2 = float(rng.uniform(0, 1)), float(rng.uniform(0, 1))
        assert _near(RC.blend(p1, p2, s), float(C.blend(p1, p2, s)))
    xs = np.sort(rng.normal(size=30)).astype(np.float32).astype(np.float64)
    ys = np.sort(rng.uniform(size=30)).astype(np.float32).astype(np.float64)
    for x in np.r_[rng.normal(scale=2, size=300), xs[:5], [-np.inf, np.inf]]:
        assert _near(RC.interp(float(x), xs.tolist(), ys.tolist()), float(np.interp(x, xs, ys)))


def test_live_practice_clips_match_production():
    """One practice clip per language not in the golden set, straight from the fp32 backbone."""
    from pronunciada.acoustic import model_cached

    if not model_cached():
        pytest.skip("fp32 backbone not in the model cache")
    try:
        from bench.opt.common import practice_clips

        clips = {lang: sorted(practice_clips([lang]), key=lambda c: (c["dur"], c["id"]))[2] for lang in LANGS}
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"practice clips not available: {e}")
    from pronunciada.acoustic import AcousticModel
    from pronunciada.adapter import Adapter
    from pronunciada.align import score_reference
    from pronunciada.audio import load_audio
    from pronunciada.reference import align as RA
    from pronunciada.reference.tokens import load_tokens
    from pronunciada.text import get_g2p

    toks = load_tokens()
    am = AcousticModel(precision="fp32", threads=2)
    for lang, c in clips.items():
        lpa = Adapter.load(lang)(am.posteriors(load_audio(c["audio"])))
        ref = get_g2p(lang)(c["text"])
        phones = [{"ipa": p.ipa, "alts": [list(a) for a in p.alts]} for p in ref.phones]
        for ins in (False, True):
            e = score_reference(lpa, ref, insertion=ins)
            r = RA.score_phones(lpa.tolist(), phones, toks, insertion=ins)
            _check_evidence(e, r, ins)
