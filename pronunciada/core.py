"""Scoring from target phones, without a text front-end (the on-device core).

    from pronunciada.core import phones_for, score_phones
    target = phones_for("Hola, ¿cómo estás?", "es")       # text side: today's front-end (a dict, JSON-able)
    res = score_phones(samples, 16000, target, "es")       # core side: float32 samples + the target

`score_phones(samples, sample_rate, target, lang)` returns the same `SentenceResult` as
`Scorer(lang).score(audio, text)` and, given `phones_for(text, lang)`, the same values (checked field
by field in tests/test_core.py and by `python -m pronunciada.tools.mobile_parity`).

What `score_phones` loads: NumPy, ONNX Runtime, kaldi-native-fbank, the backbone, and the language's
scoring data (adapter, calibration, native statistics, the per-language heads and the small stress /
tone / sentence ONNX models). It never imports a text front-end (`pronunciada.text.<lang>`), num2words,
soundfile, pyopenjtalk, sudachipy, jieba, pypinyin or opencc, and never opens a lexicon or other
front-end data file (`pronunciada/data/lex/`, `pronunciada/data/cjk/`); tests/test_core.py checks
this in a fresh process for every language. Audio decoding and resampling are the caller's job:
`samples` must already be mono 16 kHz.

Target format (version 1). A JSON object; the canonical description with the phone inventory and
the normalisation rules is docs/PHONES.md:

    {"format": "pronunciada-target", "version": 1, "lang": "es",
     "text": "Hola",                       # optional: the prompt as shown (SentenceResult.text)
     "warnings": [],                       # optional: front-end notes passed to the result; for German
                                           # "oov:<word>" marks a word that was not in the lexicon
                                           # (the German word recalibration reads it)
     "words": [
       {"text": "Hola",                    # optional: the word as shown (WordResult.text)
        "char_span": [0, 4],               # optional: offsets into "text"
        "syllables": [
          {"text": "ho",                   # optional; read for zh (一 / 不 tone rule) and ko (spelled vowel)
           "stress": 1,                    # 1 primary, 2 secondary, 0 unstressed, null = no lexical stress
           "tone": null, "tone_lexical": null, "tone_alts": [],      # zh: surface tone 1-5, before sandhi, others ok
           "phones": [
             {"ipa": "o", "nucleus": true, # ipa: one sound in ZIPA tokens; exactly one nucleus per syllable
              "alts": [["u", -1.5]],       # optional: [ipa, log prior] acceptable variants; "" = may be dropped
              "tense": null,               # ko: true / false for fortis / lax obstruents, else null
              "length": null}]}],          # ja it de: "long" / "short" when the prompt fixes the length
        "alternatives": []}]}              # optional: other readings of the whole word (round 20, below)

`words[i].alternatives` is a list of readings, each a list of syllables in the format above. The
round-20 word-level pass (pronunciada.r20; on by default for Spanish number tokens) re-scores a word
whose probability is below 0.9 against each reading and keeps the best word-level values. The
readings come from the front-end (pronunciada.text.numbers spells the numbers), so the core takes them as
data: `phones_for` fills them exactly as the text path would compute them, and a target without them
gives that word no second pass (the first-pass values are returned, a documented difference, not an
approximation of it).

Every field the scorer reads is in the target; there are no hidden inputs. Optional fields that are
absent take the values the front-ends write when they have nothing to say (alts [], tense / length /
stress / tone null, text ""), and the scorer then behaves as for such a front-end output: no stress
or tone channel for syllables without a target, the length channel derives "long" / "short" from
the symbol, Korean tense evidence off for sounds without a flag, the German word recalibration
treats every word as in the lexicon. docs/PHONES.md lists per language which marks change scores.
"""
from __future__ import annotations

import json
import threading

from .types import RefPhone, RefSyllable, RefWord, Reference, SentenceResult

FORMAT = "pronunciada-target"
VERSION = 1
SAMPLE_RATE = 16000

__all__ = ["score_phones", "phones_for", "phones_from_kana", "phones_from_pinyin", "target_from_reference",
           "reference_from_target", "get_scorer", "FORMAT", "VERSION", "SAMPLE_RATE"]


def phones_from_kana(reading, text=None, *, style="pron", devoicing="auto") -> dict:
    """Japanese target from a kana reading, by rules (layer 1, no dictionary): pronunciada.reading.ja."""
    from .reading import phones_from_kana as f

    return f(reading, text, style=style, devoicing=devoicing)


def phones_from_pinyin(reading, text=None, *, sandhi=True) -> dict:
    """Mandarin target from pinyin, by rules (layer 1, no dictionary): pronunciada.reading.zh."""
    from .reading import phones_from_pinyin as f

    return f(reading, text, sandhi=sandhi)


# ----------------------------------------------------------------------------- target <-> Reference

def _phone_dict(p: RefPhone) -> dict:
    return {"ipa": p.ipa, "nucleus": bool(p.is_nucleus), "alts": [[a, float(w)] for a, w in p.alts],
            "tense": p.tense, "length": p.length}


def _syllables_of(ref: Reference, s0: int, s1: int) -> list[dict]:
    out = []
    for si in range(s0, s1):
        sy = ref.syllables[si]
        out.append({"text": sy.text, "stress": sy.stress, "tone": sy.tone, "tone_lexical": sy.tone_lexical,
                    "tone_alts": list(sy.tone_alts),
                    "phones": [_phone_dict(ref.phones[i]) for i in range(*sy.phones)]})
    return out


def target_from_reference(ref: Reference, alternatives: dict | None = None) -> dict:
    """The target (JSON-able dict) of a Reference; `alternatives` = {word index: [one-word References]}."""
    words = []
    for wi, w in enumerate(ref.words):
        alts = []
        for frag in (alternatives or {}).get(wi, ()):
            alts.append(_syllables_of(frag, 0, len(frag.syllables)))
        words.append({"text": w.text, "char_span": [int(w.char_span[0]), int(w.char_span[1])],
                      "syllables": _syllables_of(ref, *w.syllables), "alternatives": alts})
    return {"format": FORMAT, "version": VERSION, "lang": ref.lang, "text": ref.text,
            "warnings": list(ref.warnings), "words": words}


def _build(lang: str, text: str, warnings, words) -> Reference:
    """Reference from [(word text, char span, [syllable dicts])] (indices assigned in order)."""
    ref = Reference(lang=lang, text=text, warnings=list(warnings))
    for wi, (wtext, span, syls) in enumerate(words):
        s0, p0 = len(ref.syllables), len(ref.phones)
        for sy in syls:
            si = len(ref.syllables)
            a = len(ref.phones)
            for ph in sy["phones"]:
                tense = ph.get("tense")
                ref.phones.append(RefPhone(
                    ipa=str(ph["ipa"]), word=wi, syllable=si, is_nucleus=bool(ph.get("nucleus", False)),
                    alts=[(str(x[0]), float(x[1])) for x in ph.get("alts") or ()],
                    tense=None if tense is None else bool(tense), length=ph.get("length")))
            ref.syllables.append(RefSyllable(
                word=wi, phones=(a, len(ref.phones)), text=str(sy.get("text") or ""),
                tone=_opt_int(sy.get("tone")), tone_lexical=_opt_int(sy.get("tone_lexical")),
                tone_alts=[int(t) for t in sy.get("tone_alts") or ()], stress=_opt_int(sy.get("stress"))))
        ref.words.append(RefWord(text=str(wtext), syllables=(s0, len(ref.syllables)),
                                 phones=(p0, len(ref.phones)), char_span=(int(span[0]), int(span[1]))))
    return ref


def _opt_int(x):
    return None if x is None else int(x)


def reference_from_target(target, lang: str | None = None) -> Reference:
    """Reference for the scorer from a target (dict or JSON text). Raises ValueError on a malformed
    target or a symbol that is not ZIPA IPA (Reference.validate). The word alternatives are attached
    as `Reference.word_alternatives` ({word index: [one-word References]}), read by pronunciada.r20."""
    if isinstance(target, (str, bytes)):
        target = json.loads(target)
    if not isinstance(target, dict) or "words" not in target:
        raise ValueError("target must be a dict with a 'words' list (see pronunciada.core)")
    if target.get("format", FORMAT) != FORMAT or int(target.get("version", VERSION)) != VERSION:
        raise ValueError(f"unsupported target format {target.get('format')!r} version {target.get('version')!r}")
    tl = target.get("lang")
    if lang is not None and tl is not None and tl != lang:
        raise ValueError(f"target is for {tl!r}, scorer language is {lang!r}")
    lang = lang or tl
    if not lang:
        raise ValueError("language missing (target 'lang' or the lang argument)")
    words = target["words"]
    if not isinstance(words, list) or not words:
        raise ValueError("target has no words")
    items = []
    for w in words:
        syls = w.get("syllables")
        if not syls:
            raise ValueError(f"word {w.get('text')!r} has no syllables")
        items.append((w.get("text", ""), w.get("char_span") or (0, 0), syls))
    ref = _build(lang, str(target.get("text", "")), target.get("warnings") or (), items)
    ref.validate()
    alts = {}
    for wi, w in enumerate(words):
        frs = []
        for reading in w.get("alternatives") or ():
            fr = _build(lang, str(w.get("text", "")), (), [(w.get("text", ""), (0, len(str(w.get("text", "")))),
                                                             reading)])
            fr.validate()
            frs.append(fr)
        if frs:
            alts[wi] = frs
    ref.word_alternatives = alts
    return ref


# ----------------------------------------------------------------------------- text side

def phones_for(text: str, lang: str) -> dict:
    """The target of `text` from today's text front-end (pronunciada.text), including the round-20
    candidate readings of each word that the text path would compute (pronunciada.r20.text_fragments,
    for the languages and fixes enabled now). Runs the layer-2 text front-end (lexicons, dictionaries) and the
    number speller; not part of the on-device core. See docs/TEXT_LAYERS.md."""
    from . import r20 as R20
    from .text import get_g2p, normalize_lang

    lang = normalize_lang(lang)
    if text is None or not str(text).strip():
        raise ValueError("text is empty")
    g2p = get_g2p(lang)
    ref = g2p(str(text))
    ref.validate()
    alts = {}
    if R20.any_on(lang):
        for wi, w in enumerate(ref.words):
            fr = R20.text_fragments(g2p, lang, ref, wi, w.text)
            if fr:
                alts[wi] = fr
    return target_from_reference(ref, alts)


# ----------------------------------------------------------------------------- core side

_SCORERS: dict = {}
_LOCK = threading.Lock()


def get_scorer(lang: str, **kw):
    """The cached front-end-free Scorer for (lang, options); `kw` goes to pronunciada.scorer.Scorer."""
    from .scorer import Scorer
    from .text.base import normalize_lang

    lang = normalize_lang(lang)
    key = (lang, tuple(sorted(kw.items())))
    with _LOCK:
        sc = _SCORERS.get(key)
        if sc is None:
            sc = _SCORERS[key] = Scorer(lang, frontend=False, **kw)
    return sc


def score_phones(samples, sample_rate: int, target, lang: str, strictness: float = 0.5, *,
                 scorer=None, **kw) -> SentenceResult:
    """Score mono float samples (16 kHz, values in [-1, 1]) against a target (see the module doc).

    samples: 1-D float32 / float64 array-like (values outside [-1, 1] are clipped, as for decoded
    audio); sample_rate: must be 16000, any other rate raises ValueError (resample first, e.g. with
    pronunciada._resample.resample_poly as pronunciada.audio.resample does);
    target: dict or JSON text (pronunciada.core format) for the same language;
    scorer: an existing Scorer(lang, frontend=False) to use, else one is built and cached per
    (lang, kw) with `kw` passed to Scorer (precision="fp32" default, threads, ...).
    Returns the SentenceResult of Scorer.score (rescore / details / coaching work on it as usual).
    A Scorer is not safe to call from several threads at once; use one per thread."""
    import numpy as np

    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"sample_rate must be {SAMPLE_RATE} (got {sample_rate}); resample before scoring")
    wav = np.asarray(samples)
    if wav.ndim != 1:
        raise ValueError(f"samples must be 1-D mono audio, got shape {wav.shape}")
    if not np.issubdtype(wav.dtype, np.floating):
        raise ValueError(f"samples must be float in [-1, 1], got dtype {wav.dtype}")
    wav = np.clip(wav.astype(np.float32, copy=False), -1.0, 1.0)
    sc = scorer if scorer is not None else get_scorer(lang, **kw)
    ref = reference_from_target(target, sc.lang)
    return sc.score(wav, ref, strictness=strictness)
