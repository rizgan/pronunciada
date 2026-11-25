"""Tone round 3: extra frame inputs, the selectable tone_zh_v3 model, reference-side leniency."""
import os
from pathlib import Path

import numpy as np
import pytest

from pronunciada.prosody import DATA, ProsodyScorer
from pronunciada.prosody import tone_leniency as TL


RETIRED = Path(os.environ.get("PRONUNCIADA_CACHE", os.path.expanduser("~/.cache/pronunciada"))) / "prosody" / "retired"


def model_stem(name):
    """Package model name, or the path stem of a model retired from the package (round 11:
    selectable as PRONUNCIADA_*_MODEL=<path stem>); skips the test if neither exists."""
    if (DATA / f"{name}.onnx").exists():
        return name
    if (RETIRED / f"{name}.onnx").exists():
        return str(RETIRED / name)
    pytest.skip(f"{name} neither in the package nor in {RETIRED}")
from pronunciada.prosody.features import N_LOWMEL, N_REG, lowmel_features, register_features


@pytest.mark.parametrize("T", [0, 3, 140, 1000])
def test_register_features_any_length(T):
    f = np.random.default_rng(0).normal(size=(T, 5)).astype(np.float32)
    f[:, 1] = 1.0
    r = register_features(f)
    assert r.shape == (T, N_REG) and np.isfinite(r).all()
    if T > 200:      # a constant offset is absorbed by the register
        r2 = register_features(np.c_[f[:, :1] + 2.0, f[:, 1:]])
        assert np.allclose(r[200:-200, 0], r2[200:-200, 0], atol=1e-4)


def test_lowmel_shape_and_speed():
    import time

    w = (np.random.default_rng(1).normal(size=160000) * 0.1).astype(np.float32)
    lowmel_features(w, 1000)
    t = time.perf_counter()
    x = lowmel_features(w, 1000)
    assert x.shape == (1000, N_LOWMEL) and np.isfinite(x).all()
    assert time.perf_counter() - t < 0.05


class _S:
    def __init__(self, text, tone, lex=None):
        self.text, self.tone, self.tone_lexical, self.tone_alts = text, tone, lex or tone, []


class _R:
    def __init__(self, syl):
        self.syllables = syl


def test_leniency_rules_keyed_on_reference_tone():
    ref = _R([_S("人", 2), _S("们", 5), _S("不", 2), _S("是", 4), _S("的", 5)])
    assert TL.extra_tones(ref, 1, ("neutral4",)) == {4}          # after tone 2
    assert TL.extra_tones(ref, 1, ()) == set()
    assert TL.extra_tones(ref, 2, ("yibu",)) == {4}
    assert TL.extra_tones(ref, 3, TL.RULES) == set()               # full tones untouched
    ref2 = _R([_S("你", 3), _S("的", 5)])
    assert TL.extra_tones(ref2, 1, ("neutral4",)) == set()        # after tone 3: low-rising, no
    ref3 = _R([_S("们", 1)])                                        # a changed tone never triggers
    assert TL.extra_tones(ref3, 0, TL.RULES) == set()


def test_leniency_env(monkeypatch):
    monkeypatch.setenv("PRONUNCIADA_TONE_LENIENCY", "none")
    assert TL.enabled({"leniency": ["neutral4"]}) == ()
    monkeypatch.delenv("PRONUNCIADA_TONE_LENIENCY")
    assert TL.enabled({"leniency": ["yibu"]}) == ("yibu",)
    assert TL.enabled({}) == TL.DEFAULT
    with pytest.raises(ValueError):
        TL.enabled({"leniency": ["nope"]})


def test_v3_selectable_and_runs(monkeypatch):
    from pronunciada.text import get_g2p

    monkeypatch.setenv("PRONUNCIADA_TONE_MODEL", model_stem("tone_zh_v3"))
    sc = ProsodyScorer("zh")
    assert sc.inputs == {"reg": True, "lowmel": True}
    ref = get_g2p("zh")("我们今天去学校")
    n = len(ref.phones)
    T = 200                                   # 2 s of audio; ZIPA frames 0..~97
    spans = [(int(3 + 90 * k / n), int(3 + 90 * (k + 1) / n)) for k in range(n)]
    t = np.arange(32000) / 16000
    wav = (0.2 * np.sin(2 * np.pi * (180 + 30 * np.sin(3 * t)) * t)).astype(np.float32)
    out = sc.score(wav, ref, spans)
    assert len(out) == len(ref.syllables) and all(0 <= d["tone_prob"] <= 1 for d in out if d)
    assert T > 0


SELECTABLE = ["tone_zh_v2", "tone_zh_v3", "tone_zh_v3z"]


@pytest.mark.parametrize("stem", SELECTABLE)
def test_selectable_tone_model_through_scorer(tmp_path, monkeypatch, stem):
    """Each selectable Mandarin tone model loads through the public Scorer with its own word
    calibrator (zh.<stem>.json, zh.json for the default) and scores a `say` clip."""
    import shutil
    import subprocess

    import os
    from pathlib import Path


    path_stem = model_stem(stem)
    if not _model_cached() or shutil.which("say") is None:
        pytest.skip("needs the cached ZIPA model and macOS `say`")
    out = tmp_path / "say.wav"
    text = "今天天气很好，我们一起去公园散步吧。"
    r = subprocess.run(["say", "-v", "Tingting", "-o", str(out), "--data-format=LEI16@16000", text],
                       capture_output=True)
    if r.returncode != 0:
        pytest.skip("voice unavailable")
    monkeypatch.setenv("PRONUNCIADA_TONE_MODEL", path_stem)
    from pronunciada.prosody import TONE_MODEL
    from pronunciada.scorer import Scorer

    sc = Scorer("zh", threads=2)
    assert sc.prosody is not None and sc.prosody.meta["model"]
    if stem != TONE_MODEL:
        assert sc.pros_cal.get("tone_model") == stem, "non-default model without its own calibrator"
    res = sc.score(str(out), text)
    assert not any("prosody failed" in w for w in res.warnings)
    tp = [s.tone_prob for w in res.words for s in w.syllables if s.tone_prob is not None]
    assert len(tp) >= 10 and all(0 <= p <= 1 for p in tp)
    assert all(0 <= w.prob <= 1 for w in res.words)


def _model_cached():
    from pronunciada.acoustic import model_cached

    return model_cached()
