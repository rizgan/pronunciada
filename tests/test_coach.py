"""Coaching feedback: articulatory table coverage, diff / tip logic, schema, audio gate."""
import json

import numpy as np
import pytest

from pronunciada import articulatory as A
from pronunciada import coach, quality, tips
from pronunciada.text.ipa import SPECIAL, load_tokens

SAMPLES = {
    "en": "The quick brown fox jumps over the lazy dog while the children watch.",
    "zh": "我们明天去北京看朋友，你想一起去吗？",
    "ja": "今日はとても暑いので、冷たい水を飲みたいです。",
    "es": "Se recomienda a los viajeros que se informen sobre el clima.",
    "fr": "Les enfants jouent dans le jardin pendant que leurs parents mangent.",
    "de": "Die Kinder spielen im Garten, während die Eltern Kaffee trinken.",
    "it": "I bambini giocano nel giardino mentre i genitori bevono il caffè.",
    "pt": "As crianças brincam no jardim enquanto os pais tomam café.",
    "ru": "Дети играют в саду, пока родители пьют чай на веранде.",
    "ko": "아이들은 정원에서 놀고 부모님은 차를 마십니다.",
}


def test_every_zipa_token_has_features():
    toks = [t for t in load_tokens() if t not in SPECIAL]
    missing = [t for t in toks if t not in A.symbols()]
    assert not missing
    for t in toks:
        if t not in A.MODIFIERS:
            f = A.features(t)
            assert f["type"] in ("consonant", "vowel")
            keys = A.CONSONANT_KEYS if f["type"] == "consonant" else A.VOWEL_KEYS
            assert set(keys) <= set(f)


def test_every_reference_phone_of_every_language_has_features():
    inv = coach.inventory
    for lang in SAMPLES:
        phones = inv(lang)
        assert len(phones) >= 20, lang
        bad = [p for p in phones if not A.covered(p)]
        assert not bad, (lang, bad)


@pytest.mark.parametrize("lang", sorted(SAMPLES))
def test_front_end_output_and_alternatives_covered(lang):
    from pronunciada.text import get_g2p

    try:
        ref = get_g2p(lang)(SAMPLES[lang])
    except Exception as e:  # noqa: BLE001  (optional text dependencies)
        pytest.skip(f"front-end unavailable: {e}")
    for p in ref.phones:
        assert A.covered(p.ipa), (lang, p.ipa)
        for a, _ in p.alts:
            assert not a or A.covered(a), (lang, a)


def test_composites_and_diff():
    assert A.features("tʃ")["manner"] == "affricate" and A.features("tʃ")["place"] == "postalveolar"
    assert A.features("tt")["long"] and A.features("tts")["manner"] == "affricate"
    assert A.features("aɪ")["glide"] == "ɪ" and A.features("aj")["glide"] == "i"
    assert A.features("ɑɻ")["rhotic"] and A.features("kʰ")["aspirated"]
    assert A.diff("t", "t") == []
    assert A.diff("s", "z") == ["voicing"]
    assert A.diff("tʰ", "t") == ["aspirated"]
    assert A.diff("i", "ɪ") == ["height"]
    assert A.diff("θ", "s") == ["place"]
    assert A.diff("l", "ɹ") == ["lateral"]
    assert A.diff("a", "t") == ["type"]
    assert A.diff("tʃ", "ʃ") == ["manner"]


def test_tip_keys_are_consistent():
    cases = {("z", "s"): "voicing.add", ("s", "z"): "voicing.remove", ("pʰ", "p"): "aspiration.add",
             ("θ", "s"): "place", ("s", "t"): "manner.continuant", ("t", "s"): "manner.stop",
             ("i", "ɪ"): "vowel.close", ("ɪ", "i"): "vowel.open", ("u", "i"): "vowel.back",
             ("y", "i"): "vowel.round", ("aɪ", "a"): "vowel.glide.add", ("ɔ̃", "ɔ"): "nasal.add",
             ("tʲ", "t"): "palatalisation.add", ("aː", "a"): "length.longer", ("n", "d"): "manner.nasal"}
    for (e, h), key in cases.items():
        k, _ = tips.sound_tip_key(e, h, A.diff(e, h))
        assert k == key, (e, h, k)
        assert k in tips.TEMPLATES


def _item(**kw):
    base = {"schema_version": 1, "kind": "substitution", "word": "think", "word_index": 0,
            "syllable_index": 0, "phone_index": 0, "position": "initial", "expected": "θ",
            "heard": "s", "expected_ipa": "θ", "heard_ipa": "s", "diff_keys": ["place"],
            "prob": 0.1, "confidence": 0.9, "start": 0.1, "end": 0.2, "gate": None}
    base.update(kw)
    return base


def test_tips_cap_and_one_per_sound():
    items = [_item(), _item(heard="f", diff_keys=["place"], prob=0.2),
             _item(word_index=1, word="tone", kind="tone", expected=2, heard=4, expected_ipa=None,
                   heard_ipa=None, diff_keys=["tone"], phone_index=None),
             _item(word_index=2, word="x", heard=None, heard_ipa=None, diff_keys=[], confidence=None),
             _item(word_index=3, word="gone", kind="missing_word", expected="gone", heard=None)]
    out = tips.tips(items, max_tips=3)
    assert len(out) == 3
    assert [t["item"]["word_index"] for t in out] == [0, 1, 3]     # diagnosed before flag-only
    assert out[0]["tip_key"] == "place" and "/θ/" in out[0]["text"]
    assert out[1]["tip_key"] == "tone.wrong" and "rising" in out[1]["text"] and "falling" in out[1]["text"]
    for key, tmpl in tips.TEMPLATES.items():
        assert tmpl.count(".") <= 2 and tmpl.endswith("."), key


def test_gate_file_and_overrides():
    g = coach.gate_params()
    assert set(coach.DEFAULT_GATE) <= set(g)
    assert coach.gate_params(min_log_ratio=9.0)["min_log_ratio"] == 9.0
    edges, vals = g["confidence"]["edges"], g["confidence"]["values"]
    assert len(vals) == len(edges) + 1 and all(0 <= v <= 1 for v in vals)


def test_model_file():
    m = coach.model()
    assert m["version"] == 2 and len(m["sound"]["coef"]) == len(m["sound"]["features"])
    t = m["thresholds"]
    assert t["careful"] > t["balanced"] > t["eager"]
    for lang in SAMPLES:
        for s in coach.SENSITIVITIES:
            assert 0 < coach.sound_threshold(m, lang, s, True) <= 1.01
    assert m["tone"]["thresholds"]["careful"] <= m["tone"]["thresholds"]["eager"]
    with pytest.raises(ValueError):
        coach.coaching(object(), sensitivity="loud")


def test_heard_candidate_picks_dominant_sound():
    from pronunciada.text.ipa import token_ids

    m = np.full(127, 1e-4)
    m[token_ids()["s"]] = 0.6
    m[token_ids()["θ"]] = 0.01
    h, lh, le = coach.heard_candidate(m, "θ", [], "en")
    assert h == "s" and lh > le


# ----------------------------------------------------------------------------- audio gate

def _speechy(sec=3.0, level=0.1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * 16000)) / 16000
    env = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)            # 3 "syllables" per second
    x = level * env * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * rng.standard_normal(len(t)))
    x = np.r_[np.zeros(8000), x, np.zeros(8000)] + 1e-4 * rng.standard_normal(len(t) + 16000)
    return x.astype(np.float32)


def test_audio_gate_codes():
    clean = quality.check(_speechy(), n_syllables=9, lang="en")
    assert clean["issues"] == [] and clean["retry_recommended"] is False
    json.dumps(clean)
    clip = quality.check(np.clip(_speechy() * 40, -1, 1), 9, "en")
    assert "clipped" in clip["issues"]
    rng = np.random.default_rng(1)
    noisy = quality.check(_speechy() + 0.08 * rng.standard_normal(len(_speechy())).astype(np.float32), 9, "en")
    assert "too_noisy" in noisy["issues"]
    quiet = quality.check(_speechy() * 1e-3, 9, "en")
    assert "too_quiet" in quiet["issues"]
    short = quality.check(_speechy(sec=1.0), 30, "en")
    assert "too_short" in short["issues"]
    sil = quality.check(np.r_[np.zeros(160000, np.float32), _speechy(sec=0.5)], None, "en")
    assert "mostly_silence" in sil["issues"]
    assert quality.check(np.zeros(16000, np.float32))["retry_recommended"]


@pytest.mark.model
def test_coaching_on_scorer_result(tts):
    import pronunciada

    sc = pronunciada.Scorer("en", threads=2)
    wav = tts("The quick brown fox jumps over the lazy dog")
    res = sc.score((wav, 16000), "The quick brown cat jumps over the lazy dog")
    assert res.audio is not None and "issues" in res.audio
    items = res.coaching()
    json.dumps(items)
    omitted = {wi for wi, w in enumerate(res.words) if w.present_prob is not None and w.present_prob < 0.5}
    X = coach.sound_inputs(res)
    flagged = {(int(X["word"][i]), int(X["k"][i]))
               for i in coach.flagged_indices(X, "en") if X["word"][i] not in omitted}
    assert {(it["word_index"], it["phone_index"]) for it in items if it["kind"] in ("substitution", "missing")} == flagged
    for it in items:
        assert it["schema_version"] == coach.SCHEMA_VERSION and it["kind"] in coach.KINDS
        if it["heard_ipa"]:
            assert it["diff_keys"] == A.diff(it["expected_ipa"], it["heard_ipa"])
            assert coach.sound_class(it["expected_ipa"]) not in coach.model()["heard"]["off_classes"]
    ranks = [coach._rank(it) for it in items]  # noqa: SLF001
    assert ranks == sorted(ranks, reverse=True)
    # sensitivities are nested: careful <= balanced <= eager
    key = lambda L: {(i["word_index"], i["phone_index"]) for i in L if i["kind"] == "substitution"}  # noqa: E731
    assert key(res.coaching(sensitivity="careful")) <= key(items) <= key(res.coaching(sensitivity="eager"))
    assert len(tips.tips(items)) <= 3
    # rescore keeps the audio report and coaching still works
    r2 = sc.rescore(res, 0.9)
    assert r2.audio == res.audio and isinstance(r2.coaching(), list)
    for it in res.coaching(delivery="all"):
        assert it["kind"] in coach.KINDS and it["schema_minor"] == coach.SCHEMA_MINOR
        json.dumps(it)
    assert all(t["tip_key"] in tips.TEMPLATES for t in tips.tips(res.coaching(delivery="all"), 10, res))


# ----------------------------------------------------------------------------- round 3

def test_delivery_flags_rules():
    from pronunciada.delivery_coach import delivery_flags

    T = {"long_pause_s": 1.0, "mid_pause_s": 0.6, "rushed": {"stop": 0.1, "comma": None},
         "flat_range_st": 5.0, "slow_rate": 10.0, "fast_rate": 16.0}
    D = {"pauses": [{"start": 1.0, "end": 2.2, "duration": 1.2, "after_word": 1, "before_word": 2, "in_word": False,
                     "punctuation": "stop"},
                    {"start": 3.0, "end": 3.7, "duration": 0.7, "after_word": 3, "before_word": 4, "in_word": False,
                     "punctuation": None},
                    {"start": 4.0, "end": 4.7, "duration": 0.7, "after_word": 5, "before_word": 6, "in_word": False,
                     "punctuation": "comma"}],
         "marks": [{"after_word": 7, "kind": "stop", "gap": 0.02, "time": 5.0},
                   {"after_word": 8, "kind": "comma", "gap": 0.0, "time": 6.0}],
         "articulation_rate": 18.0, "n_sounds": 40, "f0_range_st": 3.0}
    kinds = [k for k, _ in delivery_flags(D, T, 10)]
    assert kinds == ["long_pause", "pause_mid_phrase", "rushed_punctuation", "flat_pitch", "fast"]
    assert [k for k, _ in delivery_flags(dict(D, n_sounds=5), T, 2)] == ["long_pause", "pause_mid_phrase",
                                                                       "rushed_punctuation"]


def test_model_has_round3_thresholds():
    m = coach.model()
    for lang in SAMPLES:
        T = m["delivery"][lang]
        assert T["slow_rate"] < T["fast_rate"] and T["mid_pause_s"] <= T["long_pause_s"]
        assert m["extra_speech"]["min_sounds"][lang] >= 1


def test_tips_for_word_and_delivery_kinds():
    base = {"schema_version": 2, "schema_minor": 1, "syllable_index": None, "phone_index": None,
            "position": None, "expected": None, "heard": None, "expected_ipa": None, "heard_ipa": None,
            "diff_keys": [], "error_prob": None, "prob": None, "confidence": None, "start": 0.0, "end": 1.0,
            "gate": None}
    words = ["the", "cat", "sat", "down"]
    items = [dict(base, kind="extra_speech", word=None, word_index=None, heard="ə", heard_ipa="ə", after_word=1,
                  before_word=2, where="between", restart=False, n_sounds=3),
             dict(base, kind="long_pause", word="cat", word_index=1, duration=1.4, after_word=1, before_word=2,
                  at_punctuation=False),
             dict(base, kind="rushed_punctuation", word="sat", word_index=2, punctuation="stop", gap=0.0, after_word=2),
             dict(base, kind="fast", word=None, word_index=None, sounds_per_s=19.0),
             dict(base, kind="flat_pitch", word=None, word_index=None, pitch_range_st=2.0)]
    out = [tips.tip_for(it, None, words) for it in items]
    assert [t["tip_key"] for t in out] == ["speech.extra", "pause.long", "punctuation.rushed", "rate.fast", "pitch.flat"]
    assert "between “cat” and “sat”" in out[0]["text"] and "1.4 s" in out[1]["text"] and "full stop" in out[2]["text"]
    assert len(tips.tips(items, max_tips=5)) == 5


# ----------------------------------------------------------------------------- round 4 (attribute heads)

def test_attr_labels():
    from pronunciada.attr_heads import HEADS, labels

    assert labels("b")["voicing"] == "voiced" and labels("pʰ")["aspirated"] == "yes" and "aspirated" not in labels("s")
    assert labels("tʃ")["manner"] == "affricate" and labels("ʂ")["place"] == "post" and labels("k")["place"] == "dorsal"
    assert labels("i")["height"] == "close" and labels("u")["rounded"] == "yes" and labels("ɔ̃")["nasal"] == "yes"
    assert labels("aː")["long"] == "yes" and labels("tʲ")["palatalised"] == "yes" and labels("l")["lateral"] == "yes"
    for p in coach.inventory("en") + coach.inventory("ko") + coach.inventory("ru"):
        lb = labels(p)
        assert lb and all(lb[h] in HEADS[h][1] for h in lb), p


def test_feature_claims_agree_with_named_sound():
    from pronunciada.coach_attr import claims_for

    P = {"voicing": {"voiced": 0.1, "voiceless": 0.9}, "place": {"labial": 0.05, "front": 0.9, "post": 0.03,
                                                                  "dorsal": 0.01, "laryngeal": 0.01}}
    bars = {"voicing": 0.6, "place": 0.6}
    cl = claims_for(P, "z", [], None, bars)
    assert [c["feature"] for c in cl] == ["voicing"] and cl[0]["heard"] == "voiceless"     # z is already "front"
    assert claims_for(P, "z", [], "s", bars)[0]["heard"] == "voiceless"                    # agrees with /s/
    assert claims_for(P, "z", [], "ʒ", bars) == []                                          # /ʒ/ is voiced: dropped
    assert claims_for(P, "z", ["s"], None, bars) == []                                      # /s/ is accepted here
    assert claims_for(P, "z", [], None, {"voicing": 0.95, "place": None}) == []


def test_feature_tips():
    from pronunciada.tips import FEATURE_TEMPLATES, feature_tip, tip_for

    t = feature_tip({"feature": "voicing", "expected": "voiced", "heard": "voiceless"}, "z", "zoo")
    assert t["tip_key"] == "feature.voicing.add" and "/z/" in t["text"] and "voiceless" in t["text"]
    t = feature_tip({"feature": "height", "expected": "close", "heard": "mid"}, "i", "see")
    assert t["tip_key"] == "feature.height.close"
    t = feature_tip({"feature": "place", "expected": "front", "heard": "post"}, "s", "see")
    assert "tongue tip" in t["text"]
    item = {"kind": "substitution", "word": "zoo", "expected": "z", "heard": None, "diff_keys": ["voicing"],
            "features": [{"feature": "voicing", "key": "voicing", "expected": "voiced", "heard": "voiceless", "p": 0.9}]}
    assert tip_for(item)["tip_key"] == "feature.voicing.add"
    for k, v in FEATURE_TEMPLATES.items():
        assert v.endswith(".") and "{expected}" in v, k


def test_attr_heads_shipped_and_small():
    import os

    from pronunciada.attr_heads import HEADS, shipped

    h = shipped()
    assert h is not None and set(h.heads) <= set(HEADS) and {"voicing", "place", "rounded"} <= set(h.heads)
    p = os.path.join(os.path.dirname(coach.__file__), "data", "coach", "attr_heads.npz")
    assert os.path.getsize(p) < 400_000
    V = np.random.default_rng(0).standard_normal((5, 1024)).astype(np.float32)
    out = h.predict(V, [0, 3])
    assert len(out) == 2 and all(abs(sum(d.values()) - 1) < 1e-5 for d in out[0].values())
    bars, min_ep = coach.attr_policy(coach.model(), "en", "balanced")
    assert any(v is not None for v in bars.values())


# ----------------------------------------------------------------------------- round 19 (tips gate)

def _fake_result(word_probs, phone_probs):
    from types import SimpleNamespace as NS

    words = [NS(text=f"w{i}", prob=wp, phones=[NS(prob=p) for p in pp], syllables=[NS(text=f"w{i}")])
             for i, (wp, pp) in enumerate(zip(word_probs, phone_probs))]
    return NS(words=words)


def test_tips_gate_on_off_and_result_none():
    res = _fake_result([0.93, 0.93, 0.4], [[0.9, 0.9], [0.3, 0.9], [0.9, 0.9]])
    acc = _item(word_index=0, word="w0", prob=0.9)                    # word and sound accepted
    bad_sound = _item(word_index=1, word="w1", prob=0.3)              # sound not accepted
    bad_word = _item(word_index=2, word="w2", prob=0.9)               # word not accepted
    feat = _item(word_index=0, word="w0", phone_index=1, heard=None, heard_ipa=None, diff_keys=["voicing"], prob=0.9,
                 features=[{"feature": "voicing", "key": "voicing", "expected": "voiced", "heard": "voiceless", "p": 0.9}])
    items = [acc, bad_sound, bad_word, feat]
    got = {t["item"]["word_index"] for t in tips.tips(items, 10, res)}
    assert got == {1, 2}                                                # accepted candidates give no tip
    assert len(tips.tips(items, 10, res, include_accepted=True)) == 3  # gate off: one tip per word, as before
    assert tips.tips(items, 10) == tips.tips(items, 10, None, include_accepted=True)  # no result: unchanged
    assert tips.tips([acc], 3, res) == []
    # max_tips fills from the items that pass
    assert [t["item"]["word_index"] for t in tips.tips([acc, bad_sound, bad_word], 1, res)] == [1]
    # no item prob: the sound's prob is read from the result
    assert len(tips.tips([_item(word_index=1, word="w1", phone_index=0, prob=None)], 3, res)) == 1
    assert tips.tips([_item(word_index=1, word="w1", phone_index=1, prob=None)], 3, res) == []


def test_tips_gate_mispronounced_word():
    res = _fake_result([0.93, 0.93], [[0.9, 0.3], [0.9, 0.9]])
    mw = lambda wi: _item(kind="mispronounced_word", word=f"w{wi}", word_index=wi, phone_index=None,  # noqa: E731
                          syllable_index=None, heard=None, heard_ipa=None, diff_keys=[], prob=None)
    sub = lambda wi, k, p: _item(word=f"w{wi}", word_index=wi, phone_index=k, heard=None, heard_ipa=None,  # noqa: E731
                                 diff_keys=[], prob=p)
    # word 0 has a flagged sound that is not accepted, word 1 only accepted ones
    out = tips.tips([mw(0), mw(1), sub(0, 1, 0.3), sub(1, 0, 0.9)], 10, res)
    assert [t["item"]["word_index"] for t in out] == [0] and out[0]["tip_key"] == "word.mispronounced"
    assert tips.tips([mw(1), sub(1, 0, 0.9)], 10, res) == []
    assert len(tips.tips([mw(1), sub(1, 0, 0.9)], 10, res, include_accepted=True)) == 1


def test_tips_gate_leaves_other_kinds_alone():
    res = _fake_result([0.99] * 6, [[0.99]] * 6)
    base = {"schema_version": 2, "schema_minor": 6, "syllable_index": 0, "phone_index": None, "position": None,
            "heard": None, "expected_ipa": None, "heard_ipa": None, "diff_keys": [], "error_prob": None,
            "prob": 0.99, "confidence": None, "start": 0.0, "end": 1.0, "gate": None}
    items = [dict(base, kind="missing_word", word="w0", word_index=0, expected="w0", syllable_index=None),
             dict(base, kind="tone", word="w1", word_index=1, expected=2, heard=4, diff_keys=["tone"]),
             dict(base, kind="stress", word="w2", word_index=2, expected=0, heard=1),
             dict(base, kind="length", word="w3", word_index=3, phone_index=0, expected="long", heard="short",
                  expected_ipa="aː"),
             dict(base, kind="long_pause", word="w4", word_index=4, expected=None, duration=1.4, after_word=4, before_word=5,
                  at_punctuation=False),
             dict(base, kind="extra_speech", word=None, word_index=None, expected=None, heard="ə", after_word=1, before_word=2,
                  where="between", restart=False, n_sounds=3)]
    on, off = tips.tips(items, 10, res), tips.tips(items, 10, res, include_accepted=True)
    assert len(on) == len(items) and [t["tip_key"] for t in on] == [t["tip_key"] for t in off]


@pytest.mark.model
def test_tips_gate_follows_strictness(tts):
    import pronunciada

    sc = pronunciada.Scorer("en", threads=2)
    wav = tts("The quick brown fox jumps over the lazy dog")
    res = sc.score((wav, 16000), "The quick brown cat jumps over the lazy dog")
    for r in (sc.rescore(res, 0.0), sc.rescore(res, 1.0)):
        items = r.coaching(miscues=False, delivery=False)
        cand = [it for it in items if it["kind"] == "substitution"]
        out = tips.tips(cand, 50, r)
        for t in out:
            it = t["item"]
            w = r.words[it["word_index"]]
            assert w.prob < 0.5 or it["prob"] < 0.5
        want = {it["word_index"] for it in cand if r.words[it["word_index"]].prob < 0.5 or it["prob"] < 0.5}
        assert {t["item"]["word_index"] for t in out} == want
    strict = tips.tips(sc.rescore(res, 1.0).coaching(), 50, sc.rescore(res, 1.0))
    lenient = tips.tips(sc.rescore(res, 0.0).coaching(), 50, sc.rescore(res, 0.0))
    pk = lambda T: sum(t["item"]["kind"] in tips.PRONUNCIATION_KINDS for t in T)  # noqa: E731
    assert pk(strict) >= pk(lenient)


@pytest.mark.parametrize("lang,sensitivity,override", [
    (lang, sensitivity, override)
    for lang in ("en", "zh", "ja", "de", "es", "fr", "it", "ko", "pt", "ru")
    for sensitivity in coach.SENSITIVITIES for override in (None, 0.5)
    if (lang, sensitivity, override) != ("en", "balanced", None)
])
def test_sound_mask_preserves_legacy_thresholds(lang, sensitivity, override):
    model = coach.model()
    rules = model["by_lang"].get(lang, {})
    X = {"expected": ["t"] * 6, "initial": np.array([True, False] * 3)}
    if override is not None:
        cuts = np.full(6, override)
    elif "initial" in rules and "final" in rules:
        cuts = np.array([rules["initial" if i else "final"][sensitivity] for i in X["initial"]])
    else:
        cuts = np.full(6, rules.get("sound", model["thresholds"])[sensitivity])
    scores = cuts + np.repeat([-0.001, 0, 0.001], 2)
    assert coach.sound_mask(X, scores, model, lang, sensitivity, override).tolist() == [
        False, False, True, True, True, True]


@pytest.mark.model
def test_english_balanced_sound_mask_and_heard(monkeypatch):
    from pronunciada import Scorer
    from pronunciada.report import coaching_block

    sc = Scorer("en", threads=2)
    res = sc.score("site/samples/en-learner-1.mp3", "It was a beautiful day for a float")
    original = res.to_dict()
    selected = []
    add_heard = coach._add_heard

    def capture(inputs, result, indices):
        selected.extend((int(inputs["word"][i]), int(inputs["k"][i])) for i in indices)
        add_heard(inputs, result, indices)

    monkeypatch.setattr(coach, "_add_heard", capture)
    items = coaching_block(res)["items"]
    assert set(selected) == {(i["word_index"], i["phone_index"]) for i in items
                             if i["kind"] in {"missing", "substitution"}}
    assert res.to_dict() == original

    model = coach.model()
    rule = model["by_lang"]["en"]["sound"]["balanced"]
    X = {"expected": ["t", "t", "x"], "initial": np.array([True, False, False])}
    cut = rule["thresholds"]["stop"]
    assert coach.sound_mask(X, np.array([cut, cut - 1e-5, rule["fallback"]]),
                            model, "en", "balanced").tolist() == [True, False, True]
