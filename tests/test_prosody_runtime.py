"""ProsodyScorer runtime: end-to-end on `say`-generated clips, runtime-vs-training parity,
model size and latency. Needs the cached ZIPA model (skipped otherwise) and macOS `say`."""
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from pronunciada.prosody import DATA, ProsodyScorer
from pronunciada.prosody.features import context_input, frame_features, syllable_windows
from pronunciada.prosody.f0 import track


RETIRED = Path(os.environ.get("PRONUNCIADA_CACHE", os.path.expanduser("~/.cache/pronunciada"))) / "prosody" / "retired"


def model_stem(name):
    """Package model name, or the path stem of a model retired from the package (round 11:
    selectable as PRONUNCIADA_*_MODEL=<path stem>); skips the test if neither exists."""
    if (DATA / f"{name}.onnx").exists():
        return name
    if (RETIRED / f"{name}.onnx").exists():
        return str(RETIRED / name)
    pytest.skip(f"{name} neither in the package nor in {RETIRED}")



def _say(text, voice, tmp_path):
    if shutil.which("say") is None:
        pytest.skip("no `say`")
    out = tmp_path / "say.wav"
    r = subprocess.run(["say", "-v", voice, "-o", str(out), "--data-format=LEI16@16000", text],
                       capture_output=True)
    if r.returncode != 0 or not out.exists():
        pytest.skip(f"voice {voice} unavailable")
    from pronunciada.audio import load_audio

    return load_audio(str(out))


@pytest.fixture(scope="module")
def acoustic():
    if not _model_cached():
        pytest.skip("ZIPA model not cached")
    from pronunciada.acoustic import AcousticModel

    return AcousticModel(threads=2)


def _evidence(acoustic, wav, ref):
    from pronunciada.align import score_reference

    return score_reference(acoustic.posteriors(wav), ref)


def test_default_tone_model_is_v3z():
    from pronunciada import prosody

    assert prosody.TONE_MODEL == "tone_zh_v3z"
    assert all((DATA / f"{m}.onnx").exists() for m in ("tone_zh_v2", "tone_zh_v3z"))


def test_model_files_small():
    files = list(DATA.glob("*.onnx"))
    assert files
    assert sum(f.stat().st_size for f in DATA.iterdir()) <= 5 * 1024 * 1024


def test_unscored_languages():
    from pronunciada.text import get_g2p

    for lang, text in (("fr", "Bonjour tout le monde"), ("ja", "今日は良い天気です。")):
        ref = get_g2p(lang)(text)
        out = ProsodyScorer(lang).score(np.zeros(16000, np.float32), ref, [(0, 0)] * len(ref.phones))
        assert out == [{} for _ in ref.syllables]


def test_tone_end_to_end(acoustic, tmp_path):
    from pronunciada.text import get_g2p

    text = "今天天气很好，我们一起去公园散步吧。"
    wav = _say(text, "Tingting", tmp_path)
    ref = get_g2p("zh")(text)
    ev = _evidence(acoustic, wav, ref)
    sc = ProsodyScorer("zh")
    out = sc.score(wav, ref, ev)
    assert len(out) == len(ref.syllables)
    for o in out:
        assert len(o["tone_probs"]) == 5 and abs(sum(o["tone_probs"]) - 1) < 1e-4
        assert 1 <= o["tone_heard"] <= 5 and 0 <= o["tone_prob"] <= 1
    # a TTS voice says the tones correctly: most syllables should be accepted
    assert np.mean([o["tone_prob"] > 0.5 for o in out]) > 0.6
    # the same audio against a wrong-tone reference scores lower on the changed syllables
    import copy

    bad = copy.deepcopy(ref)
    for s in bad.syllables:
        s.tone = {1: 4, 2: 4, 3: 1, 4: 1, 5: 1}[s.tone]
        s.tone_alts = []
    out_bad = sc.score(wav, bad, ev)
    assert np.mean([o["tone_prob"] for o in out]) > np.mean([o["tone_prob"] for o in out_bad]) + 0.2


@pytest.mark.skipif(not (DATA / "stress.onnx").exists(), reason="stress model not exported")
def test_stress_end_to_end(acoustic, tmp_path):
    from pronunciada.text import get_g2p

    text = "The government announced a beautiful celebration yesterday."
    wav = _say(text, "Samantha", tmp_path)
    ref = get_g2p("en")(text)
    out = ProsodyScorer("en").score(wav, ref, _evidence(acoustic, wav, ref))
    for w in ref.words:
        s0, s1 = w.syllables
        for i in range(s0, s1):
            if s1 - s0 >= 2 and ref.syllables[i].stress == 1:
                assert 0 <= out[i]["stress_prob"] <= 1
            else:
                assert out[i]["stress_prob"] is None


def test_parity_with_training_pipeline(acoustic, tmp_path, monkeypatch):
    """Round-1 local model (tone_zh). Runtime (onnxruntime, features from the waveform) == training side (torch model on the
    features rebuilt from the stored pitch track the way bench.prosody.common does)."""
    torch = pytest.importorskip("torch")
    from bench.prosody.common import to_track
    from bench.prosody.prep import pack
    from bench.prosody.tone_eval import load_model, softmax
    from pronunciada.text import get_g2p

    monkeypatch.setenv("PRONUNCIADA_TONE_MODEL", model_stem("tone_zh"))
    try:
        net, _ = load_model(ProsodyScorer("zh").meta["model"])
    except FileNotFoundError:
        pytest.skip("training checkpoint not available")
    text = "这样的问题也得到了部分网友的回复"
    wav = _say(text, "Tingting", tmp_path)
    ref = get_g2p("zh")(text)
    ev = _evidence(acoustic, wav, ref)
    sc = ProsodyScorer("zh")
    out = sc.score(wav, ref, ev)
    spans = [(s.start_frame, s.end_frame) for s in ev.slots]
    tr = to_track(pack(track(wav)))          # training stores the track as float16
    f = frame_features(tr)
    w = syllable_windows(ref, spans, len(f))
    X = np.stack([context_input(f, w, i) for i in range(len(w))])
    with torch.no_grad():
        P = softmax(net(torch.tensor(X)).numpy(), sc.meta["temperature"])
    got = np.array([o["tone_probs"] for o in out])
    assert np.abs(got - P).max() < 0.02
    assert (got.argmax(1) == P.argmax(1)).mean() > 0.9


def test_latency_10s(acoustic, tmp_path):
    from pronunciada.text import get_g2p

    text = "今天天气很好，我们一起去公园散步吧。" * 3
    wav = _say(text, "Tingting", tmp_path)
    wav = np.tile(wav, int(np.ceil(160000 / len(wav))))[:160000]
    ref = get_g2p("zh")(text * int(np.ceil(160000 / len(wav))))
    ev = _evidence(acoustic, wav, ref)
    sc = ProsodyScorer("zh")
    sc.score(wav, ref, ev)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        sc.score(wav, ref, ev)
        ts.append(time.perf_counter() - t)
    assert min(ts) < 0.1          # budget 20 ms on an idle M3; loose bound for loaded machines


@pytest.mark.skipif(not (DATA / "tone_zh_v2.onnx").exists(), reason="v2 tone model not exported")
@pytest.mark.parametrize("stem", ["tone_zh", "tone_zh_v2"])
def test_both_tone_models_end_to_end(acoustic, tmp_path, monkeypatch, stem):
    """Both shipped Mandarin models (round-1 local, round-2 utterance-level) through the same
    interface; v2 must see the whole utterance (the meta flag `sequence`)."""
    from pronunciada.text import get_g2p

    monkeypatch.setenv("PRONUNCIADA_TONE_MODEL", model_stem(stem))
    text = "今天天气很好，我们一起去公园散步吧。"
    wav = _say(text, "Tingting", tmp_path)
    ref = get_g2p("zh")(text)
    ev = _evidence(acoustic, wav, ref)
    sc = ProsodyScorer("zh")
    assert sc.meta.get("sequence", False) == (stem == "tone_zh_v2")
    out = sc.score(wav, ref, ev)
    assert all(abs(sum(o["tone_probs"]) - 1) < 1e-4 and 0 <= o["tone_prob"] <= 1 for o in out)
    assert np.mean([o["tone_prob"] > 0.5 for o in out]) > 0.6


@pytest.mark.skipif(not (DATA / "tone_zh_v2.onnx").exists(), reason="v2 tone model not exported")
def test_v2_parity_with_training_pipeline(acoustic, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from bench.prosody.common import to_track
    from bench.prosody.prep import pack
    from bench.prosody.tone2 import load2, predict_utts, softmax
    from pronunciada.text import get_g2p

    monkeypatch.setenv("PRONUNCIADA_TONE_MODEL", "tone_zh_v2")
    sc = ProsodyScorer("zh")
    try:
        net, kind, _ = load2(sc.meta["model"])
    except FileNotFoundError:
        pytest.skip("training checkpoint not available")
    text = "这样的问题也得到了部分网友的回复"
    wav = _say(text, "Tingting", tmp_path)
    ref = get_g2p("zh")(text)
    ev = _evidence(acoustic, wav, ref)
    got = np.array([o["tone_probs"] for o in sc.score(wav, ref, ev)])
    tr = to_track(pack(track(wav)))
    f = frame_features(tr)
    w = syllable_windows(ref, [(s.start_frame, s.end_frame) for s in ev.slots], len(f))
    X = np.stack([context_input(f, w, i) for i in range(len(w))])
    P = softmax(predict_utts(net, kind, [X])[0], sc.meta["temperature"])
    assert np.abs(got - P).max() < 0.02 and (got.argmax(1) == P.argmax(1)).mean() > 0.9


def _model_cached():
    from pronunciada.acoustic import model_cached

    return model_cached()
