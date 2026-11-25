"""Public scoring API.

    scorer = pronunciada.Scorer("en")                    # loads ZIPA, adapter, calibrators once
    res = scorer.score("clip.wav", "The quick brown fox")      # -> SentenceResult
    res2 = scorer.rescore(res, strictness=1.0)       # no model re-run
    pronunciada.score(audio, text, "en")                 # convenience, caches Scorer objects

Pipeline: audio -> fbank -> ZIPA (ONNX) -> per-language output adapter -> text front-end ->
lattice evidence (pronunciada.align.score_reference) -> features (pronunciada.features) -> calibrators
(pronunciada.calibrate: sound -> word -> sentence) -> optional tone / stress (pronunciada.prosody) ->
SentenceResult. The sound probabilities shown come from the language's shipped sound model
(English: the English-specific model if it won on CV, else the universal one; every other
language: the universal model + its logit bias); word and sentence calibrators always consume
the universal sound model's log-odds, which is what they were trained on.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict

import numpy as np

from . import calibrate as C
from . import features as F
from .types import PhoneResult, Reference, SentenceResult, SyllableResult, WordResult

MIN_AUDIO_S = 0.25
SILENCE_PEAK = 1e-4


class Scorer:
    def __init__(self, lang: str, model: str = "zipa-small-crctc-500k", precision: str = "fp32",
                 strictness: float = 0.5, threads: int | None = None, *,
                 calib_dir: str | None = None, sound_model: str | None = None,
                 prosody: bool = True, cache_size: int = 4, explain_away: bool | None = None,
                 delivery: bool | None = None,
                 audio_check: bool = True, length: bool = True, frontend: bool = True,
                 device: str = "cpu"):
        from .acoustic import AcousticModel
        from .adapter import Adapter
        from .text.base import normalize_lang
        from .variants import NativeStats

        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        self.lang = normalize_lang(lang)
        self.strictness = float(strictness)
        # frontend=False (pronunciada.core.score_phones): no text front-end is loaded; score() then
        # takes a ready Reference only
        self._g2p_error: ImportError | None = None
        if frontend:
            from .text import get_g2p

            try:
                self.g2p = get_g2p(self.lang)
            except ImportError as e:
                # layer 2 (the language's dictionary extra) is not installed: scoring a Reference
                # still works; text raises this error, which names the extra (docs/TEXT_LAYERS.md)
                self.g2p = None
                self._g2p_error = e
        else:
            self.g2p = None
        self.adapter = Adapter.load(self.lang)
        self.native = NativeStats.load(self.lang)
        cfg = C.config(calib_dir)
        # sentence calibrator (round 5): per-language choice in config "sentence_model"
        # ({lang: packaged name}, default "sentence"); round-5 models also read the delivery
        # features of pronunciada.delivery (+ a projection of pooled encoder activations)
        self.sentence_name = cfg.get("sentence_model", {}).get(self.lang, "sentence")
        # round 19: a newer sentence model per language (config "sentence_head"; English:
        # sentence_r19, delivery + F0 inputs + std-pooled activations of every encoder stack);
        # PRONUNCIADA_SENTENCE_HEAD=0 keeps the "sentence_model" choice
        self.seq_head, self.seq_weight = None, 0.0
        self.seq_sound_weight = self.seq_word_weight = 0.0
        if os.environ.get("PRONUNCIADA_SENTENCE_HEAD", "1") != "0":
            sh = cfg.get("sentence_head", {}).get(self.lang)
            if sh:
                self.sentence_name = sh["trees"]
                if sh.get("seq"):              # sequence head blended into the four 0-10 scores
                    from .sentence_head import SentenceHead

                    self.seq_head = SentenceHead(sh["seq"], calib_dir, threads=min(2, threads or 1))
                    self.seq_weight = float(sh["seq_weight"])
                    self.seq_sound_weight = float(sh.get("sound_weight", 0.0))
                    self.seq_word_weight = float(sh.get("word_weight", 0.0))
        self.sentence = C.load_level(self.sentence_name, calib_dir)
        self.sentence_proj = C.load_projection(self.sentence_name, calib_dir)
        self.sentence_lin = C.load_linear(self.sentence_name, calib_dir)
        from . import delivery as D
        from . import sentence_head as SH

        dcols = set(D.ALL) | set((self.sentence_proj or {}).get("names", [])) | set(SH.F0_KEYS) | set(SH.VOICE_KEYS)
        uses = any(c in dcols for c in self.sentence.num)
        # reported 0-10 word score (round 6): per-language choice in config "word_score_model";
        # reads the word-pooled ZIPA stack-1 activations. The sentence level keeps reading the
        # round-2 word regression (word_reg.npz), so sentence inputs do not change.
        ws_name = cfg.get("word_score_model", {}).get(self.lang)
        from pathlib import Path as _Pw

        _dw = _Pw(calib_dir) if calib_dir else C.data_dir()
        self.word_score = C.load_level(ws_name, calib_dir) if ws_name and (_dw / f"{ws_name}.npz").exists() else None
        self.word_score_proj = C.load_projection(ws_name, calib_dir) if self.word_score is not None else None
        uses = uses or self.word_score is not None
        self.delivery = uses if delivery is None else bool(delivery)
        self._delivery_forced = delivery is True        # training extraction: every feature
        self._dcols = [c for c in self.sentence.num if c in dcols]
        want_emb = self.delivery and (self.sentence_proj is not None or self.word_score is not None
                                      or delivery is True)
        if prosody:                              # a selected prosody model may read ZIPA activations
            from .prosody import needs_embeddings

            want_emb = want_emb or needs_embeddings(self.lang)
        extra_layers = {k.split(".")[0] for k in (self.sentence_proj or {}).get("layers", [])}
        if self.seq_head is not None:
            extra_layers |= set(self.seq_head.layers)
        if want_emb and extra_layers - {"enc1", "enc3", "enc5", "final"}:
            want_emb = sorted(extra_layers)      # round 19: stacks beyond the round-5 outputs
        if want_emb and delivery is True and os.environ.get("PRONUNCIADA_ALL_LAYERS") == "1":
            want_emb = "all"                     # training extraction: every encoder stack
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else None
        self.am = AcousticModel(model, precision, threads, providers=providers, embeddings=want_emb)
        self._word_layers = tuple(sorted(set(
            (["enc1"] if delivery is True else []) + list((self.word_score_proj or {}).get("layers", [])))))
        self.universal_name = cfg["universal_sound_model"]
        name, bias = C.sound_model_for(self.lang, calib_dir)
        if sound_model is not None:
            name = sound_model
            bias = float(cfg.get("bias", {}).get(self.lang, 0.0)) if name == self.universal_name else 0.0
        self.sound_name, self.sound_bias = name, bias
        self.universal_bias = float(cfg.get("bias", {}).get(self.lang, 0.0))
        self.sound = C.load_level(name, calib_dir)
        # round 12b: a "detail" sound model (config "sound_detail_model", English: sound_E, which
        # also reads per-sound ZIPA-layer inputs, pronunciada/sound_emb.py) gives the reported 0-2
        # phone score and the coaching inputs; the probabilities that feed words / sentences and
        # PhoneResult.prob stay on self.sound.
        from . import sound_emb as _SE

        dname = cfg.get("sound_detail_model", {}).get(self.lang) if sound_model is None else None
        self.sound_detail = C.load_level(dname, calib_dir) if dname else None
        self.sound_aux = _SE.load(dname or name, calib_dir)
        self.universal = C.load_level(self.universal_name, calib_dir)
        self.word = C.load_level("word", calib_dir)
        from pathlib import Path as _P

        _d = _P(calib_dir) if calib_dir else C.data_dir()
        # round 2: the word-accuracy regression is a separate model that also reads the word log-odds
        self.word_reg = C.load_level("word_reg", calib_dir) if (_d / "word_reg.npz").exists() else None
        # prosody channel (round 3): per-language word-level calibrator of the tone / stress
        # evidence, pronunciada/data/calib/prosody/<lang>.json (bench/calib/fit_prosody.py)
        pp = _d / "prosody" / f"{self.lang}.json"
        from .prosody import default_model, selected_model

        if selected_model(self.lang) != default_model(self.lang):   # its own channel calibrator
            pm = _d / "prosody" / f"{self.lang}.{selected_model(self.lang)}.json"
            pp = pm if pm.exists() else pp
        self.pros_cal = json.loads(pp.read_text()) if pp.exists() else None
        # omission channel (round 4): P(word was said at all), pooled over languages
        # (bench/calib/fit_omission.py); replaces the hard llr_del rule for WordResult.omitted
        # round 7: Mandarin sentence accuracy / prosodic scores recalibrated on OMPAL DEV learners
        # (bench/calib/fit_sentence_zh.py); other languages and other scores unchanged
        pr7 = _d / f"sentence_{self.lang}_recal.json"
        self.sent_recal = json.loads(pr7.read_text()) if pr7.exists() else None
        if self.sent_recal and self.sent_recal.get("tone_model"):
            from .prosody import selected_tone_model

            if self.sent_recal["tone_model"] != selected_tone_model():
                self.sent_recal = None        # fitted on another tone model's outputs
        po = _d / "omission.json"
        self.omit_cal = json.loads(po.read_text()) if po.exists() else None
        # explain-away second pass (round 3): config "explain_away" {threshold, max_passes, enabled}
        ea = {"threshold": 0.3, "max_passes": 2, **cfg.get("explain_away", {})}
        # round 19: on by default for the languages listed in config "explain_away.langs" (de)
        if self.lang in ea.get("langs", ()) and os.environ.get("PRONUNCIADA_EXPLAIN_LANGS", "1") != "0" \
                and not (self.lang == "de" and os.environ.get("PRONUNCIADA_DE_R19", "1") == "0"):
            ea["enabled"] = True
            # bounded form: the pass runs only when the clip's lowest word (sounds-based P, s = 0.5)
            # lies in an uncertain band [lo, hi), with at most max_passes passes
            ea.update(ea.get("lang", {}).get(self.lang, {}))
        if os.environ.get("PRONUNCIADA_EA_BAND"):               # selection runs only
            ea["band"] = [float(x) for x in os.environ["PRONUNCIADA_EA_BAND"].split(",")]
        if os.environ.get("PRONUNCIADA_EA_PASSES"):
            ea["max_passes"] = int(os.environ["PRONUNCIADA_EA_PASSES"])
        if explain_away is not None:
            ea["enabled"] = bool(explain_away)
        self.ea = ea if ea.get("enabled") else None
        self._calib_dir = calib_dir
        # length channel (round 15): long / short sounds judged from alignment timing
        # (pronunciada/length.py, pronunciada/data/length/<lang>.json); None = not shipped / off
        from . import length as _L

        self.length_cal = _L.params(self.lang) if length else None
        from . import contrast_word as _CW                 # round 18: word-level contrast factor

        self.contrast_word = _CW.params(self.lang)
        from . import word_lang as _WL                     # round 19: per-language word recalibration

        self.word_lang = _WL.params(self.lang)
        from . import stress_rel as _SR                   # round 19: reliability-weighted stress factor

        self.stress_rel = _SR.params(self.lang)
        self.audio_check = bool(audio_check)     # pronunciada.quality gate -> SentenceResult.audio
        self.prosody = None
        self.prosody_error = None
        if prosody:
            try:
                from .prosody import ProsodyScorer

                ps = ProsodyScorer(self.lang, threads=min(2, threads or 2))
                self.prosody = ps if getattr(ps, "sess", None) is not None else None
            except Exception as e:  # noqa: BLE001  (models are built concurrently; optional)
                self.prosody_error = f"{type(e).__name__}: {e}"
        self._lp_cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ acoustic stage
    def _posteriors(self, wav: np.ndarray, timing: dict):
        """(adapted log-posteriors, fbank, encoder activations or None), cached per waveform."""
        key = hashlib.sha1(wav.tobytes()).hexdigest()
        if key in self._lp_cache:
            self._lp_cache.move_to_end(key)
            timing.update(fbank=0.0, model=0.0, adapter=0.0)
            return self._lp_cache[key]
        t = time.perf_counter()
        x = self.am._features(wav)
        vt = os.environ.get("PRONUNCIADA_VTLN")      # round 19 experiment: "alpha:min_median_f0_hz"
        if vt:
            from .audio import vtln_warp
            from .prosody.f0 import track as _trk

            alpha, thr = (float(v) for v in vt.split(":"))
            tr = _trk(np.asarray(wav, np.float32))
            v = tr.f0[tr.voiced & (tr.f0 > 0)]
            if len(v) and float(np.median(v)) > thr:
                x = vtln_warp(x, alpha)
        timing["fbank"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        emb = None
        if self.am.embeddings:
            lp, emb = self.am.posteriors_from_features(x, return_embeddings=True)
        else:
            lp = self.am.posteriors_from_features(x)
        timing["model"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        lpa = self.adapter(lp) if self.adapter is not None else lp
        timing["adapter"] = (time.perf_counter() - t) * 1e3
        self._lp_cache[key] = (lpa, x, emb)
        while len(self._lp_cache) > self._cache_size:
            self._lp_cache.popitem(last=False)
        return self._lp_cache[key]

    # ------------------------------------------------------------------ public
    def score(self, audio, text, strictness: float | None = None, _second: bool = False) -> SentenceResult:
        """audio: path | bytes | (np.ndarray, sample_rate) | float32 16 kHz np.ndarray;
        text: the prompt (str) or a ready pronunciada.types.Reference.
        _second (internal, round 20): a word-level re-scoring pass of pronunciada.r20 on cached posteriors;
        skips the sentence-only delivery features and sequence head and any further round-20 pass."""
        from .align import score_reference
        from .audio import SAMPLE_RATE, load_audio

        t_all = time.perf_counter()
        timing: dict = {}
        warnings: list[str] = []
        if isinstance(text, Reference):
            ref = text
            t = time.perf_counter()
        else:
            if text is None or not str(text).strip():
                raise ValueError("text is empty")
            if self.g2p is None:
                if self._g2p_error is not None:
                    raise ImportError(str(self._g2p_error)) from self._g2p_error
                raise ValueError("this Scorer has no text front-end (frontend=False): pass a Reference")
            t = time.perf_counter()
            ref = self.g2p(str(text))
        ref.validate()
        if not ref.phones:
            raise ValueError(f"no scorable sounds in {getattr(ref, 'text', text)!r}")
        timing["g2p"] = (time.perf_counter() - t) * 1e3
        warnings += list(ref.warnings)
        t = time.perf_counter()
        wav = load_audio(audio)
        timing["load"] = (time.perf_counter() - t) * 1e3
        dur = len(wav) / SAMPLE_RATE
        if dur < MIN_AUDIO_S:
            warnings.append(f"audio is very short ({dur:.2f} s)")
        if len(wav) == 0 or float(np.max(np.abs(wav))) < SILENCE_PEAK:
            warnings.append("audio is silent")
        aq = None
        if self.audio_check:
            from . import quality as Q

            t = time.perf_counter()
            aq = Q.check(wav, len(ref.syllables), self.lang)
            timing["audio_check"] = (time.perf_counter() - t) * 1e3
        lpa, fb, emb = self._posteriors(wav, timing)
        t = time.perf_counter()
        # round 19: the extra-sound (insertion) evidence between sounds (pronunciada.align, ~1 ms;
        # one implementation): Korean always (its word step), Japanese when its word step is on
        from .ja_round19 import on as _r19

        from . import identity as _IDr

        ev = score_reference(lpa, ref, insertion=self.lang == "ko" or (self.lang == "ja" and _r19("step"))
                             or (self.lang != "de" and _IDr.needs_insertion(self.lang)))
        timing["align"] = (time.perf_counter() - t) * 1e3
        if not np.isfinite(ev.L_can) or ev.L_can <= -1e5:
            warnings.append("the text is too long for the audio (the expected sounds cannot all fit)")
        t = time.perf_counter()
        tab = F.phone_table(lpa, ref, ev, self.native, dur)
        if self.sound_aux is not None:
            self.sound_aux.augment(tab, ref, ev, emb)
        _pvs: list = []

        def _pv(emb_, ev_, n_):
            """Per-sound encoder vectors of this call's (emb, ev): every caller passes the same
            inputs and only reads the result, so it is computed once and shared."""
            if not _pvs:
                from .sound_emb import phone_vectors

                _pvs.append(phone_vectors(emb_, ev_, n_))
            return _pvs[0]

        phone_vec = None
        if self.lang == "zh" and emb:            # round 12b: Mandarin consonant / vowel coaching
            from . import coach_zh as _CZ

            if _CZ.params() is not None:
                phone_vec = _pv(emb, ev, len(ref.phones))
        contrast_ev = None
        V17 = None
        if emb:                                  # round 17: head-based contrast evidence (ko / ru / zh)
            from . import contrast_heads as _CH

            if _CH.entries(self.lang):
                V17 = phone_vec if phone_vec is not None else _pv(emb, ev, len(ref.phones))
                contrast_ev = _CH.evidence(self.lang, [p.ipa for p in ref.phones], V17,
                                           tense=[getattr(p, "tense", None) for p in ref.phones])
                capz = _CH.cap_column(self.lang, contrast_ev)
                if capz is not None:
                    tab["head_cap_z"] = capz
        ko_id = None
        if emb and self.lang == "ko":            # round 19: Korean sound-identity evidence (word factor)
            from . import ko_identity as _KI

            if _KI.enabled():
                # the same phone vectors as the contrast heads' (same inputs), computed once
                Vk = phone_vec if phone_vec is not None else V17 if V17 is not None \
                    else _pv(emb, ev, len(ref.phones))
                kcls = _KI.classes(ref)
                ko_id = _KI.evidence(ref, Vk, cls=kcls)
                if _KI._word_params().get("form") == "delta":     # round 19 part 2: word log-odds change
                    ko_id = {"delta": _KI.word_deltas(ref, ev, ko_id, _KI._word_params(), cls=kcls), "s": ko_id}
        timing["features"] = (time.perf_counter() - t) * 1e3
        trk = dv = pooled = wpooled = spans = None
        if self.delivery and not _second:
            from . import delivery as D
            from .prosody.f0 import track

            t = time.perf_counter()
            # the pitch track is shared with the prosody channel; skipped when neither needs it
            need_pitch = self.prosody is not None or any(k in D.PITCH or k.startswith("f0_") for k in self._dcols) \
                or self._delivery_forced
            trk = track(wav) if need_pitch else None
            dv = D.features(wav, fb, lpa, ref, ev, trk, pitch=need_pitch)
            from . import sentence_head as SH

            if self._delivery_forced or self.seq_head is not None or any(k in SH.F0_KEYS for k in self._dcols):
                dv.update(SH.f0_summary(trk))          # round 19: speaker F0 summary
            if self._delivery_forced or any(k in SH.VOICE_KEYS for k in self._dcols):
                dv.update(SH.voice_cues(wav))          # round 19: spectral voice cues
            pooled = D.pool_embeddings(emb, ev)
            dv.update(D.project(pooled, self.sentence_proj))
            if emb is not None:      # word-level pooled activations (round 6 word head / training)
                spans = D.word_frame_spans(ref, ev, len(lpa))
                wpooled = {k: D.pool_words(emb[k], spans) for k in self._word_layers}
            else:
                spans = wpooled = None
            timing["delivery"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        cache = self._calibrate(tab, ref, dur, dv, wpooled)
        cache.update(delivery=dv, pooled=pooled, word_pooled=wpooled, word_spans=spans)
        timing["calibrate"] = (time.perf_counter() - t) * 1e3
        if self.seq_head is not None and not _second:   # round 19: sequence head blended into the 0-10 scores
            t = time.perf_counter()
            feeds = self.seq_head.inputs(tab, ref, ev, emb, cache["S"], dv, trk)
            if feeds is not None:
                o = self.seq_head.run(feeds, [p.word for p in ref.phones])
                s = o["sentence"]
                sent = dict(cache["sent"])
                w = self.seq_weight
                for j, k in enumerate(("accuracy", "fluency", "prosodic", "total")):
                    sent[f"reg_{k}"] = np.clip((1 - w) * np.asarray(sent[f"reg_{k}"], np.float64) + w * s[j], 0.0, 10.0)
                cache["sent"], cache["sent_seq"] = sent, s
                # reported 0-2 sound and 0-10 word scores (blend weights from config; 0 = unchanged);
                # probabilities, coaching inputs and the sentence inputs do not read them
                ws, ww = self.seq_sound_weight, self.seq_word_weight
                if ws and "phone" in o and cache.get("s_reg") is not None:
                    cache["s_reg"] = np.clip((1 - ws) * np.asarray(cache["s_reg"], np.float64) + ws * o["phone"], 0.0, 2.0)
                if ww and "word" in o and len(o["word"]) == len(ref.words):
                    wv = np.asarray(cache["w_reg"], np.float64)
                    cache["w_reg"] = np.where(np.isfinite(o["word"]),
                                              np.clip((1 - ww) * wv + ww * np.nan_to_num(o["word"]), 0.0, 10.0), wv)
                cache["seq_level"] = o
            timing["sentence_head"] = (time.perf_counter() - t) * 1e3
        cache["phone_vec"] = phone_vec
        cache["emb"] = emb              # round 4 coaching: attribute heads read the activations (no score uses it)
        cache["contrast_ev"] = contrast_ev  # round 17: per-contrast head evidence (coaching)
        cache["ko_identity_s"] = ko_id      # round 19: per-sound Korean identity evidence (word factor)
        if emb and self.lang == "de":            # round 19b: German sound-identity evidence (word factor)
            from . import de_identity as _DI

            if _DI.enabled():
                cache["de_identity_s"] = _DI.evidence(ref, _pv(emb, ev, len(ref.phones)))
        if emb and self.lang != "de":            # round 19 rollout: generic sound-identity evidence
            from . import identity as _ID

            if _ID.enabled(self.lang):
                pv = phone_vec if phone_vec is not None else _pv(emb, ev, len(ref.phones))
                cache["identity_s"] = _ID.evidence(self.lang, ref, pv)
                cache["identity_cls"] = _ID.classes(ref, _ID._cal(self.lang), _ID._head(self.lang))[0]
        cache.update(ref=ref, ev=ev, tab=tab)
        cache.update(lpa=lpa, audio=aq)          # read by pronunciada.coach; no score uses them
        if not np.isfinite(ev.L_can) or ev.L_can <= -1e5:
            # the expected sounds cannot all fit into the audio: nothing was said in a scorable
            # way, so every level gets the floor probability (the calibrators never saw the
            # llr = -1000 sentinel values and would extrapolate arbitrarily)
            lo = float(C.logit(C.P_CLIP))
            for k in ("s_zl", "s_zs", "w_zl", "w_zs"):
                cache[k] = np.full_like(np.asarray(cache[k], dtype=np.float64), lo)
            cache["sent_zl"] = cache["sent_zs"] = lo
            cache["S"] = dict(cache["S"], completeness=np.zeros(1))
        elif self.ea is not None and self._ea_in_band(cache):
            t = time.perf_counter()
            cache = self.explain_away(lpa, cache, dur, self.ea["threshold"], self.ea["max_passes"])
            timing["explain_away"] = (time.perf_counter() - t) * 1e3
        elif self.lang == "en" and self.omit_cal and not _second:
            t = time.perf_counter()
            cache = self._recheck_omission(cache, dur)
            timing["omission_recheck"] = (time.perf_counter() - t) * 1e3
        pros = None
        if self.prosody is not None:
            t = time.perf_counter()
            try:
                if trk is None:     # the same track ProsodyScorer.score would compute; kept for coaching
                    from .prosody.f0 import track as _f0_track

                    trk = _f0_track(np.asarray(wav, dtype=np.float32))
                pros = self.prosody.score(wav, ref, ev, track=trk, embeddings=emb)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"prosody failed: {type(e).__name__}: {e}")
            timing["prosody"] = (time.perf_counter() - t) * 1e3
        lres = None
        V19 = None
        if self.length_cal is not None and np.isfinite(ev.L_can) and ev.L_can > -1e5:
            from . import length as _L

            t = time.perf_counter()
            hs = None
            if emb and self.length_cal.get("head"):        # round 17: attribute-head length evidence
                from . import contrast_heads as _CH
                V19 = _pv(emb, ev, len(ref.phones))
                hs = _CH.length_evidence(self.lang, self.length_cal, [p.ipa for p in ref.phones], V19)
            lres = _L.score(ref, _L.frame_inputs(ref, ev, lpa, fb), self.length_cal, head_s=hs)
            timing["length"] = (time.perf_counter() - t) * 1e3
        from . import word_step as _WS                   # round 19 (ja): word step inputs

        wsp = _WS.params(self.lang)
        if wsp is not None and emb and np.isfinite(ev.L_can) and ev.L_can > -1e5:
            from . import ja_identity as _JI

            t = time.perf_counter()
            if V19 is None:
                V19 = _pv(emb, ev, len(ref.phones))
            mm = None
            if wsp["cfg"].get("a_mm"):
                from .word_match import extra_counts

                mm = extra_counts(ref, ev)
            cache["word_step_in"] = {"s": _JI.evidence(ref, V19), "cls": _JI.classes(ref), "extra": mm}
            timing["word_step"] = (time.perf_counter() - t) * 1e3
        cache.update(prosody=pros, dur=dur, warnings=warnings, f0=trk, wav=wav,   # f0 / wav: coaching only
                     length=lres)
        t = time.perf_counter()
        res = self._assemble(cache, self.strictness if strictness is None else strictness)
        timing["assemble"] = (time.perf_counter() - t) * 1e3
        if not _second:                          # round 20 word-level passes (pronunciada.r20.DEFAULTS)
            from . import r20 as _R20

            if _R20.any_on(self.lang):
                t = time.perf_counter()
                res = _R20.apply(self, res, wav, self.strictness if strictness is None else strictness)
                timing["r20"] = (time.perf_counter() - t) * 1e3
        timing["total"] = (time.perf_counter() - t_all) * 1e3
        res.timing_ms = {k: round(v, 3) for k, v in timing.items()}
        return res

    def rescore(self, result: SentenceResult, strictness: float) -> SentenceResult:
        """Same audio and text at another strictness, from the evidence cached on `result`
        (no acoustic model, alignment or calibrator run)."""
        cache = getattr(result, "_pronunciada", None)
        if cache is None:
            raise ValueError("result has no cached evidence (was it produced by Scorer.score?)")
        t = time.perf_counter()
        res = self._assemble(cache, strictness)
        res.timing_ms = {"rescore": round((time.perf_counter() - t) * 1e3, 3)}
        return res

    def details(self, result: SentenceResult) -> dict:
        """Human-scale scores behind a result: per phone the 0-2 regression and both logits,
        per word the 0-10 accuracy regression, per sentence accuracy / fluency / prosodic /
        total on 0-10 (the speechocean762 scales)."""
        c = result._pronunciada  # noqa: SLF001
        if c["s_reg"] is None:
            c["s_reg"] = self.sound.predict(c["tab"], heads=("reg",))["reg"]
        return {"phone_score": c["s_reg"], "phone_z_len": c["s_zl"], "phone_z_str": c["s_zs"],
                "word_accuracy": c["w_reg"], "word_z_len": c["w_zl"], "word_z_str": c["w_zs"],
                "sentence": {k: float(c["sent"][f"reg_{k}"][0]) for k in ("accuracy", "fluency", "prosodic", "total")},
                "sentence_z_len": float(c["sent_zl"]), "sentence_z_str": float(c["sent_zs"]),
                "omitted": c["W"]["omitted_f"].astype(bool)}

    def sound_outputs(self, result: SentenceResult, model) -> dict:
        """Outputs of another sound calibrator (LevelModel or packaged name) on the cached
        features of `result` (benchmarks: feature sets A / B / C side by side)."""
        m = model if isinstance(model, C.LevelModel) else C.load_level(model, self._calib_dir)
        return m.predict(result._pronunciada["tab"])  # noqa: SLF001

    # ------------------------------------------------------------------ explain-away pass
    def _ea_in_band(self, c: dict) -> bool:
        band = self.ea.get("band")
        if not band:
            return True
        pw = C.sigmoid(C.blend_logit(c["w_zl"], np.minimum(c["w_zs"], c["w_zl"]), 0.5))
        lo = float(np.min(pw)) if len(pw) else 1.0
        return band[0] <= lo < band[1]

    def explain_away(self, lpa, c: dict, dur: float, threshold: float, max_passes: int) -> dict:
        """Second pass(es): while the lowest not-yet-wildcarded word has sounds-based P < threshold
        (s = 0.5), wildcard its sounds in the base lattice (pronunciada.align wild_slots), re-run the
        evidence (no acoustic model), features and calibrators, at most `max_passes` times. Every
        other word / sound takes the last pass; a wildcarded word keeps the values of the pass in
        which it was selected (the first pass for the first one). The sentence level keeps the
        first pass. Returns the merged cache (c["explained"] = wildcarded word indices)."""
        from .align import score_reference

        ref = c["ref"]
        passes = [c]
        keep: dict[int, int] = {}                  # word -> index of the pass it is taken from
        for _ in range(int(max_passes)):
            cur = passes[-1]
            pw = C.sigmoid(C.blend_logit(cur["w_zl"], np.minimum(cur["w_zs"], cur["w_zl"]), 0.5))
            cand = [k for k in range(len(ref.words)) if k not in keep]
            if not cand:
                break
            wi = min(cand, key=lambda k: pw[k])
            if pw[wi] >= threshold:
                break
            keep[wi] = len(passes) - 1
            slots = [i for k in keep for i in range(*ref.words[k].phones)]
            ev2 = score_reference(lpa, ref, wild_slots=slots)
            tab2 = F.phone_table(lpa, ref, ev2, self.native, dur)
            for k in ("emb_proj", "proto_cos", "proto_other", "proto_margin", "proto_has", "head_cap_z"):
                if k in c["tab"]:                 # round 12b inputs: kept from the first pass
                    tab2[k] = c["tab"][k]
            c2 = self._calibrate(tab2, ref, dur, c.get("delivery"), c.get("word_pooled"))
            c2.update(ref=ref, ev=ev2, tab=tab2)
            passes.append(c2)
        if not keep:
            c["explained"] = []
            return c
        last = len(passes) - 1
        wsrc = np.array([keep.get(k, last) for k in range(len(ref.words))])
        psrc = np.array([wsrc[p.word] for p in ref.phones])
        out = dict(c)                               # sentence level stays first-pass
        for key in ("s_zl", "s_zs"):
            out[key] = np.array([passes[psrc[i]][key][i] for i in range(len(psrc))])
        out["s_reg"] = None if any(p["s_reg"] is None for p in passes) else \
            np.array([passes[psrc[i]]["s_reg"][i] for i in range(len(psrc))])
        for key in ("w_zl", "w_zs", "w_reg"):
            out[key] = np.array([passes[wsrc[k]][key][k] for k in range(len(wsrc))])
        out["W"] = {k: (np.array([passes[wsrc[j]]["W"][k][j] for j in range(len(wsrc))])
                        if isinstance(v, np.ndarray) and len(v) == len(wsrc) else v)
                    for k, v in c["W"].items()}
        out["tab"] = {k: ([passes[psrc[i]]["tab"][k][i] for i in range(len(psrc))] if isinstance(v, list)
                          else np.array([passes[psrc[i]]["tab"][k][i] for i in range(len(psrc))]))
                      if (isinstance(v, (list, np.ndarray)) and len(v) == len(psrc)) else v
                      for k, v in c["tab"].items()}
        ev = c["ev"]
        # spans too: a wildcarded word's from the pass it was selected in (not the wildcard)
        out["ev"] = type(ev)([passes[psrc[i]]["ev"].slots[i] for i in range(len(psrc))],
                             ev.L_can, ev.n_frames, ev.frame_shift, ev.free_tokens)
        out["explained"] = sorted(keep)
        out["n_passes"] = len(passes)
        out["first_pass"] = c                      # for benchmarks: the result without the pass
        return out

    def _recheck_omission(self, c: dict, dur: float) -> dict:
        """Try freeing the frames of one suspected omission for its immediate neighbours."""
        from dataclasses import replace

        from .align import score_slots, slots_from_reference

        ref, ev, lpa = c["ref"], c["ev"], c["lpa"]
        if len(ref.words) < 2:
            return c
        present = np.array([_present_prob(self.omit_cal, c["tab"]["llr_del"][slice(*w.phones)])
                            for w in ref.words])
        wi = int(present.argmin())
        if present[wi] >= 0.5:
            return c

        left, right = max(wi - 1, 0), min(wi + 1, len(ref.words) - 1)
        a, b = ref.words[left].phones[0], ref.words[right].phones[1]
        p, q = ref.words[wi].phones
        start = ev.slots[a - 1].end_frame if a else 0
        end = ev.slots[b].start_frame if b < len(ev.slots) else len(lpa)
        if end <= start or end - start > 200 or b - a > 32:
            return c
        slots = slots_from_reference(ref)
        frames = lpa[start:end]
        reduced = score_slots(frames, slots[a:p] + slots[q:b])
        original = score_slots(frames, slots[a:b])
        if reduced.L_can <= original.L_can:
            return c

        lifted = [replace(slot, start_frame=slot.start_frame + start, end_frame=slot.end_frame + start)
                  for slot in reduced.slots]
        adjusted = replace(ev, slots=list(ev.slots))
        adjusted.slots[a:p] = lifted[:p - a]
        adjusted.slots[q:b] = lifted[p - a:]
        tab = F.phone_table(lpa, ref, adjusted, self.native, dur)
        for key in c["tab"].keys() - tab.keys():
            tab[key] = c["tab"][key]
        revised = self._calibrate(tab, ref, dur, c.get("delivery"), c.get("word_pooled"))
        candidate_words = [j for j in range(left, right + 1) if j != wi]
        accepted = []
        for j in candidate_words:
            span = slice(*ref.words[j].phones)
            support = _present_prob(self.omit_cal, tab["llr_del"][span])
            moved = any((old.start_frame, old.end_frame) != (new.start_frame, new.end_frame)
                        for old, new in zip(ev.slots[span], adjusted.slots[span]))
            if (moved and present[j] >= 0.5 and support >= present[j]
                    and revised["w_zl"][j] >= c["w_zl"][j] and revised["w_zs"][j] >= c["w_zs"][j]):
                accepted.append(j)
        if not accepted:
            return c

        # The omitted word has no audio span in the revised alignment. The surviving
        # neighbouring spans must be ordered even if only one came from the local pass.
        left_slots = adjusted.slots if wi - 1 in accepted else ev.slots
        right_slots = adjusted.slots if wi + 1 in accepted else ev.slots
        before = left_slots[p - 1].end_frame if p else 0
        after = right_slots[q].start_frame if q < len(ev.slots) else len(lpa)
        if before > after:
            return c
        anchor = min(after, max(before, ev.slots[p].start_frame))

        out = dict(c)
        out["ev"] = replace(ev, slots=list(ev.slots))
        out["ev"].slots[p:q] = [replace(slot, start_frame=anchor, end_frame=anchor,
                                        n_frames=0, alt_index=-1, heard="")
                                 for slot in ev.slots[p:q]]
        out["tab"] = {k: v.copy() for k, v in c["tab"].items()}
        out["W"] = {k: v.copy() for k, v in c["W"].items()}
        for key, value in (("start_frame", anchor), ("end_frame", anchor), ("n_frames", 0)):
            out["tab"][key][p:q] = value
        word_keys = ("w_zl", "w_zs", "w_reg", "w_reg_r2")
        phone_keys = ("s_zl", "s_zs", "s_reg", "coach_zl", "coach_zs", "contrast_z")
        for key in word_keys + phone_keys:
            if c.get(key) is not None:
                out[key] = c[key].copy()
        for j in accepted:
            span = slice(*ref.words[j].phones)
            out["ev"].slots[span] = adjusted.slots[span]
            for key, v in out["tab"].items():
                v[span] = tab[key][span]
            for key, v in out["W"].items():
                v[j] = revised["W"][key][j]
            for key in word_keys:
                out[key][j] = revised[key][j]
            for key in phone_keys:
                if c.get(key) is not None:
                    out[key][span] = revised[key][span]
        return out

    # ------------------------------------------------------------------ calibration
    def _calibrate(self, tab: dict, ref, dur: float, dv: dict | None = None,
                   wpooled: dict | None = None) -> dict:
        Xu = self.universal.matrix(tab)
        u = self.universal.logits(X=Xu)
        uzl, uzs = u["z_len"] + self.universal_bias, u["z_str"] + self.universal_bias
        ureg = u["reg"]
        if self.sound is self.universal:
            zl, zs, sreg = uzl, uzs, ureg
        else:   # probabilities shown from the language's own model; its 0-2 head on demand
            s = self.sound.logits(tab, heads=("len", "str"))
            zl, zs = s["z_len"] + self.sound_bias, s["z_str"] + self.sound_bias
            sreg = None
        detail = {}
        if self.sound_detail is not None:
            d = self.sound_detail.logits(tab)
            sreg = d["reg"]
            detail = {"coach_zl": d["z_len"], "coach_zs": np.minimum(d["z_str"], d["z_len"])}
        caps = C.contrast_caps(self.lang, self._calib_dir) if hasattr(self, "_calib_dir") else []
        if caps:                       # round 16: contrast caps on the SHOWN sound log-odds only;
            detail.setdefault("coach_zl", zl)   # coaching and word / sentence inputs unchanged
            detail.setdefault("coach_zs", np.minimum(zs, zl))
            zl, zs = C.apply_contrast_caps(caps, tab, zl, zs)
        cz = C.contrast_cap_z(caps, tab) if caps else np.full(len(zl), np.nan)   # round 18: word factor input
        if "head_cap_z" in tab:
            hz0 = np.asarray(tab["head_cap_z"], np.float64)
            cz = np.where(np.isfinite(cz) & np.isfinite(hz0), np.minimum(cz, hz0), np.where(np.isfinite(cz), cz, hz0))
        if "head_cap_z" in tab:         # round 17: head-based contrast caps, composed by the minimum
            detail.setdefault("coach_zl", zl)
            detail.setdefault("coach_zs", np.minimum(zs, zl))
            hz = np.asarray(tab["head_cap_z"], np.float64)
            fire = np.isfinite(hz)
            zl, zs = np.array(zl, np.float64), np.array(zs, np.float64)
            zl[fire] = np.minimum(zl[fire], hz[fire])
            zs[fire] = np.minimum(zs[fire], hz[fire])
        W = F.word_table(tab, uzl, uzs, ureg)
        XW = np.column_stack([np.asarray(W[c], dtype=np.float64) for c in self.word.num])
        w = self.word.logits(X=XW)
        if self.word_reg is not None:
            W["word_z_len"], W["word_z_str"] = w["z_len"], w["z_str"]
            XR = np.column_stack([np.asarray(W[c], dtype=np.float64) for c in self.word_reg.num])
            w["reg"] = self.word_reg.predict(X=XR)["reg"]
        S = F.sentence_table(tab, uzl, uzs, ureg, W, w["z_len"], w["z_str"], w["reg"], dur)
        for k in self._dcols:            # round-5 delivery inputs (NaN when not computed)
            S[k] = np.array([np.nan if dv is None else float(dv.get(k, np.nan))])
        XS = np.column_stack([np.asarray(S[c], dtype=np.float64) for c in self.sentence.num])
        so = C.apply_linear_blend(self.sentence_lin, XS, self.sentence.logits(X=XS))
        w_rep = w["reg"]
        pj = self.word_score_proj
        if self.word_score is not None and wpooled and all(k in wpooled for k in pj["layers"]):
            P = np.concatenate([wpooled[k] for k in pj["layers"]], axis=1).astype(np.float64)
            z = ((P - pj["mu"]) / pj["sd"]) @ pj["W"] + pj["b"]
            for j, nm in enumerate(pj["names"]):
                W[nm] = z[:, j]
            XR2 = np.column_stack([np.asarray(W[c], dtype=np.float64) for c in self.word_score.num])
            w_rep = self.word_score.predict(X=XR2)["reg"]
        return {**detail, "s_zl": zl, "s_zs": np.minimum(zs, zl), "s_reg": sreg, "W": W, "w_zl": w["z_len"],
                "w_zs": w["z_str"], "w_reg": w_rep, "w_reg_r2": w["reg"], "S": S, "sent": so,
                "sent_zl": so["z_len"][0], "sent_zs": so["z_str"][0], "contrast_z": cz}

    # ------------------------------------------------------------------ result objects
    def _assemble(self, c: dict, strictness: float) -> SentenceResult:
        s = C.clip_strictness(strictness)
        ref, ev, tab = c["ref"], c["ev"], c["tab"]
        ps = C.sigmoid(C.blend_logit(c["s_zl"], c["s_zs"], s))
        if getattr(self, "word_lang", None) is not None:      # round 19 (de): shown word P only
            from . import word_lang as _WL

            wzl, wzs = _WL.transform(self.word_lang, c["W"], c["w_zl"], c["w_zs"], _WL.oov_words(ref))
            pw = C.sigmoid(C.blend_logit(wzl, np.minimum(wzs, wzl), s))
        else:
            pw = C.sigmoid(C.blend_logit(c["w_zl"], np.minimum(c["w_zs"], c["w_zl"]), s))
        pr = c.get("prosody") or [dict() for _ in ref.syllables]
        lres = c.get("length")
        f2t = self.am.frame_to_time
        phones = []
        for i, p in enumerate(ref.phones):
            sl = ev.slots[i]
            st, en = float(f2t(sl.start_frame)), float(f2t(sl.end_frame))
            # what was heard instead: greedy tokens over the sound's span, "" = nothing (missing)
            heard = (sl.heard if sl.n_frames > 0 else "") if ps[i] < 0.5 else None
            feats = {k: _num(tab[k][i]) for k in ("llr_sub", "llr_del", "z_llr_sub", "n_frames", "peak_post")}
            feats.update(p_lenient=float(C.sigmoid(c["s_zl"][i])), p_strict=float(C.sigmoid(c["s_zs"][i])))
            phones.append(PhoneResult(p.ipa, float(ps[i]), round(st, 3), round(en, 3), heard, feats))
            if lres is not None and lres["prob"][i] is not None:
                phones[-1].length_expected = lres["expected"][i]
                phones[-1].length_prob = float(lres["prob"][i])
        words = []
        omitted = c["W"]["omitted_f"].astype(bool)
        srel = getattr(self, "stress_rel", None)
        if srel is not None and self.pros_cal and self.pros_cal["kind"] == "stress":
            from . import stress_rel as _SR

            wsp = []
            for w in ref.words:
                sp_ = [(x.start, x.end) for x in phones[w.phones[0]:w.phones[1]] if x.end > x.start]
                wsp.append((min(a for a, _ in sp_), max(b for _, b in sp_)) if sp_ else (0.0, 0.0))
            sr_ctx = _SR.utterance_context(wsp, len(ref.syllables), c.get("f0"), c.get("audio"))
        else:
            srel = None
        for wi, w in enumerate(ref.words):
            ph = phones[w.phones[0]:w.phones[1]]
            spans = [(x.start, x.end) for x in ph if x.end > x.start]
            st = min(a for a, _ in spans) if spans else (ph[0].start if ph else 0.0)
            en = max(b for _, b in spans) if spans else st
            syls = []
            tone_p, stress_p = [], []
            for si in range(*w.syllables):
                sy = ref.syllables[si]
                d = pr[si] if si < len(pr) else {}
                a, b = sy.phones
                sp = [(x.start, x.end) for x in phones[a:b] if x.end > x.start]
                syls.append(SyllableResult(
                    sy.text, min(x for x, _ in sp) if sp else st, max(y for _, y in sp) if sp else st,
                    tone_expected=sy.tone, tone_heard=d.get("tone_heard"), tone_prob=d.get("tone_prob"),
                    stress_expected=sy.stress, stress_prob=d.get("stress_prob")))
                for lst, k in ((tone_p, "tone_prob"), (stress_p, "stress_prob")):
                    if d.get(k) is not None:
                        lst.append(float(d[k]))
            sounds = float(pw[wi])
            # Prosody channel (round 3): P_pros(word) from a word-level calibrator of the lowest /
            # second-lowest tone / stress logit over the word's scored syllables
            # (bench/calib/fit_prosody.py, FLEURS validation); WordResult.prob = sounds x P_pros.
            # Without a fitted channel the raw syllable minimum is reported and prob = sounds.
            raw = {"tone": tone_p, "stress": stress_p}
            p_pros = None
            if self.pros_cal and raw.get(self.pros_cal["kind"]):
                p_pros = _pros_prob(self.pros_cal, raw[self.pros_cal["kind"]], s)
            tone = min(tone_p) if tone_p else None
            stress = min(stress_p) if stress_p else None
            if p_pros is not None:
                if self.pros_cal["kind"] == "stress":
                    stress = p_pros
                else:
                    tone = p_pros
            f_pros = p_pros
            if srel is not None and p_pros is not None and stress_p:
                # round 19 (de): reliability-weighted stress factor in the word probability only;
                # stress_prob (word and syllable) keeps the shipped channel
                f_pros = _SR.factor(srel, self.pros_cal, stress_p,
                                    _SR.reliability(sr_ctx, st, en, len(range(*w.syllables))), s)
            prob = sounds * (f_pros if f_pros is not None else 1.0)
            # length channel (round 15): a factor that stays 1 unless the word's lowest length
            # logit is below a low native quantile (pronunciada.length.word_factor,
            # bench/length/fit_length.py)
            p_len = None
            if lres is not None:
                lv = [x.length_prob for x in ph if x.length_prob is not None]
                if lv:
                    from . import length as _L

                    p_len = _L.word_factor(self.length_cal, lv, s)
                    if c.get("word_step_in") is None:
                        prob *= p_len
            p_con = None                 # round 18: word-level contrast factor (pronunciada.contrast_word)
            if self.contrast_word is not None and c.get("contrast_z") is not None:
                from . import contrast_word as _CW

                p_con = _CW.word_factor(self.contrast_word, c["contrast_z"][w.phones[0]:w.phones[1]], s)
                if p_con is not None:
                    prob *= p_con
            if c.get("de_identity_s") is not None:      # round 19b: German sound-identity factor
                from . import de_identity as _DI

                p_id = _DI.word_factor(_DI._word_params(), c["de_identity_s"][w.phones[0]:w.phones[1]], s)
                if p_id is not None:
                    prob *= p_id
            if c.get("identity_s") is not None:         # round 19 rollout: sound-identity evidence
                from . import identity as _ID

                # round 20: the rule may also read the word's tone factor, contrast factor and sound classes
                # (pronunciada.identity.rule; PRONUNCIADA_IDENTITY_<LANG>_R20=0 restores the round-19 rule)
                is_tone = bool(self.pros_cal) and self.pros_cal["kind"] == "tone"
                icls = c.get("identity_cls")
                prob = _ID.apply_word(_ID.rule(self.lang), c["identity_s"][w.phones[0]:w.phones[1]], prob, s,
                                      sounds=sounds, tone=f_pros if is_tone else None, contrast=p_con,
                                      cls=None if icls is None else icls[w.phones[0]:w.phones[1]],
                                      ins=_ID.word_ins(getattr(ev, "llr_ins", None), w.phones[0], w.phones[1]))
            if self.omit_cal:
                present = _present_prob(self.omit_cal, tab["llr_del"][w.phones[0]:w.phones[1]])
                prob, om = prob * present, present < 0.5
            else:
                present, om = None, bool(omitted[wi])
            if c.get("ko_identity_s") is not None:      # round 19: Korean identity factor (pronunciada.ko_identity)
                from . import ko_identity as _KI

                kid = c["ko_identity_s"]
                if isinstance(kid, dict):                # "delta" form: shift the final word log-odds
                    pq = min(max(prob, 1e-6), 1 - 1e-6)
                    pn = 1.0 / (1.0 + np.exp(-(np.log(pq / (1 - pq)) + kid["delta"][wi])))
                    p_id = float(pn / pq)
                else:                                    # first form: knee factor on sounds x present
                    p_id = _KI.word_factor(_KI._word_params(), kid[w.phones[0]:w.phones[1]], s,
                                           sounds_prob=sounds)
                if p_id is not None:
                    prob *= p_id
                    p_con = p_id if p_con is None else p_con * p_id
            wsi = c.get("word_step_in")
            if wsi is not None:
                # round 19 (ja): capped two-sided step on the word log-odds (pronunciada.word_step);
                # the length evidence enters here instead of through the round-15 factor
                from . import length as _L
                from . import word_step as _WS
                from .text.ipa import is_vowel

                P19 = _WS.params(self.lang)
                lg = lambda x: math.log(max(x, 1e-6) / max(1 - x, 1e-6))  # noqa: E731
                lv = [x.length_prob for x in ph if x.length_prob is not None]
                zlen = min(lg(x) for x in lv) if lv else None
                ll = [x.length_prob for x in ph if x.length_prob is not None and x.length_expected == "long"
                      and is_vowel(x.ipa[0])]
                ins_v = [] if ev.llr_ins is None else [float(x) for x in ev.llr_ins[max(w.phones[0] - 1, 0):w.phones[1] - 1]
                                                         if np.isfinite(x)]
                st19 = _WS.step(P19, _WS.word_s(P19, ref, w, wsi["s"], wsi["cls"]), zlen,
                                wsi["extra"][wi] if wsi["extra"] is not None else 0,
                                min(lg(x) for x in ll) if ll else None, min(ins_v) if ins_v else 0.0)
                prob = _WS.apply(prob, st19)
            words.append(WordResult(w.text, prob, st, en, ph, syls, sounds_prob=sounds,
                                    stress_prob=stress, tone_prob=tone,
                                    present_prob=present, omitted=om, length_prob=p_len, contrast_prob=p_con))
        zl, zs = c["sent_zl"], min(c["sent_zs"], c["sent_zl"])
        sp = float(C.sigmoid(C.blend_logit(zl, zs, s)))
        flu = float(np.clip(c["sent"]["reg_fluency"][0] / 10.0, 0.0, 1.0))
        comp = (1.0 - float(np.mean([w.omitted for w in words]))) if (self.omit_cal and words) \
            else float(c["S"]["completeness"][0])
        scores = {k: float(np.clip(c["sent"][f"reg_{k}"][0] / 10.0, 0.0, 1.0))
                  for k in ("accuracy", "fluency", "prosodic", "total")}
        if self.sent_recal:
            scores = _sentence_recal(self.sent_recal, scores, [sy.tone_prob for w in words for sy in w.syllables])
        res = SentenceResult(self.lang, ref.text, sp, s, words, fluency=flu, completeness=comp,
                             warnings=list(c["warnings"]), scores=scores)
        res.audio = c.get("audio")
        res._pronunciada = c  # noqa: SLF001  (cached evidence for rescore / details)
        return res


def _pros_prob(cal: dict, probs: list[float], s: float) -> float:
    """P_pros(word) at strictness s: z = b0 + b1 * min logit + b2 * second-min logit + b3 * n;
    lenient / strict operating points = z + c_len / z + c_str, blended in log-odds."""
    q = np.sort(C.logit(np.asarray(probs, dtype=np.float64)))
    f = [q[0], q[1] if len(q) > 1 else q[0], float(len(q))]
    b = cal["coef"]
    z = b[0] + b[1] * f[0] + b[2] * f[1] + b[3] * f[2]
    zl, zs = z + cal["c_len"], z + cal["c_str"]
    return float(C.sigmoid(C.blend_logit(zl, min(zs, zl), s)))


def _sentence_recal(cal: dict, scores: dict, tone_probs) -> dict:
    """Round 7 (zh): score_k = clip((ridge_k(features) - 1) / 4, 0, 1) for the recalibrated k;
    features = the four sentence heads + mean / mean-log / share-below-0.5 of syllable tone_prob
    (bench/calib/fit_sentence_zh.features)."""
    t = np.asarray([x for x in tone_probs if x is not None] or [1.0], np.float64)
    f = np.array([scores[k] for k in ("accuracy", "fluency", "prosodic", "total")]
                 + [t.mean(), np.log(np.clip(t, 1e-3, 1)).mean(), (t < 0.5).mean()], np.float64)
    out = dict(scores)
    for k, m in cal["scores"].items():
        c = np.asarray(m["coef"], np.float64)
        out[k] = float(np.clip((c[0] + c[1:] @ f - 1) / 4, 0.0, 1.0))
    return out


def _present_prob(cal: dict, llr_del) -> float:
    """P(word was said): logistic on [max llr_del, mean llr_del, one-sound word, two-sound word]."""
    d = np.clip(np.nan_to_num(np.asarray(llr_del, dtype=np.float64), nan=0.0), -cal["clip"], cal["clip"])
    b = cal["coef"]
    z = b[0] + b[1] * d.max() + b[2] * d.mean() + b[3] * (len(d) == 1) + b[4] * (len(d) == 2)
    return float(C.sigmoid(z))


def _num(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else x


# ----------------------------------------------------------------------------- convenience

_SCORERS: dict = {}


def score(audio, text, lang: str, strictness: float = 0.5, **kw) -> SentenceResult:
    """One-call scoring; Scorer objects are cached per (lang, options)."""
    key = (lang, tuple(sorted(kw.items())))
    if key not in _SCORERS:
        _SCORERS[key] = Scorer(lang, **kw)
    return _SCORERS[key].score(audio, text, strictness=strictness)
