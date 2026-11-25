import os

import numpy as np
import pytest

from pronunciada.acoustic import AcousticModel, _download, greedy_tokens
from pronunciada.audio import fbank, num_fbank_frames
from pronunciada.text.ipa import load_tokens


def test_greedy_tokens_collapse():
    toks = load_tokens()
    ids = [0, 3, 10, 10, 0, 10, 11, 11, 3, 2, 12]
    lp = np.full((len(ids), 127), -10.0)
    lp[np.arange(len(ids)), ids] = 0.0
    out = greedy_tokens(lp)
    assert out == [(toks[10], 2, 4), (toks[10], 5, 6), (toks[11], 6, 8), (toks[12], 10, 11)]
    assert greedy_tokens(np.zeros((0, 127))) == []


def test_download_is_atomic(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(os.urandom(3 << 20))
    dest = tmp_path / "cache" / "m" / "f.bin"
    _download(src.as_uri(), str(dest))
    assert dest.read_bytes() == src.read_bytes()
    assert not os.path.exists(str(dest) + ".part")


def test_frame_count_formula():
    assert AcousticModel.num_output_frames(9) == 1
    assert AcousticModel.num_output_frames(456) == 224
    assert AcousticModel.num_output_frames(3) == 1      # padded to the 9-frame minimum


@pytest.mark.model
def test_posteriors_shape_timing_and_normalisation(acoustic_model, tts):
    wav = tts("Hello there, how are you?")
    lp = acoustic_model.posteriors(wav)
    T = num_fbank_frames(len(wav))
    assert lp.shape == ((T - 7) // 2, 127) and lp.dtype == np.float32
    np.testing.assert_allclose(np.logaddexp.reduce(lp, axis=1), 0.0, atol=1e-3)
    for n in (9, 10, 11, 57, 100, 101):
        out = acoustic_model.posteriors_from_features(fbank(wav)[:n])
        assert len(out) == (n - 7) // 2
    assert acoustic_model.frame_shift == pytest.approx(0.02)
    assert acoustic_model.frame_to_time(10) == pytest.approx(0.23)
    heard = "".join(t for t, _, _ in acoustic_model.greedy(lp))
    assert "h" in heard and "ɛ" in heard and "j" in heard


@pytest.mark.model
def test_onset_timing(acoustic_model, tts):
    wav = tts("Seven.")
    pad = np.zeros(16000, dtype=np.float32)
    lp = acoustic_model.posteriors(np.concatenate([pad, wav]))
    first = acoustic_model.greedy(lp)[0]
    onset = 1.0 + np.argmax(np.abs(wav) > 0.02) / 16000
    assert abs(acoustic_model.frame_to_time(first[1]) - onset) < 0.08


@pytest.mark.model
def test_short_audio(acoustic_model):
    for n in (0, 50, 1000):
        lp = acoustic_model.posteriors(np.zeros(n, dtype=np.float32))
        assert lp.shape == (1, 127) and np.isfinite(lp).all()


@pytest.mark.model
def test_batch_matches_single(acoustic_model, tts):
    wavs = [tts("Good morning."), tts("I would like a cup of coffee with milk, please."),
            tts("Good morning."), np.zeros(800, dtype=np.float32)]
    single = [acoustic_model.posteriors(w) for w in wavs]
    batch = acoustic_model.posteriors_batch(wavs)
    for s, b in zip(single, batch):
        assert s.shape == b.shape
        assert np.abs(s - b).max() < 1e-3
    # opt-in padded batching: same frames, same greedy tokens, small numeric differences
    padded = acoustic_model.posteriors_batch(wavs[:2], max_pad_ratio=10.0)
    for s, b in zip(single, padded):
        assert s.shape == b.shape
        assert (s.argmax(1) == b.argmax(1)).mean() > 0.95


@pytest.mark.model
@pytest.mark.parametrize("precision", ["fp16", "int8"])
def test_quantised_models_agree(acoustic_model, tts, precision):
    from pronunciada.acoustic import model_cached

    if not model_cached(acoustic_model.name, precision):
        pytest.skip(f"{precision} model not cached")
    wav = tts("I would like a cup of coffee with milk, please.")
    ref = acoustic_model.posteriors(wav)
    m = AcousticModel(acoustic_model.name, precision=precision, threads=2)
    lp = m.posteriors(wav)
    assert lp.shape == ref.shape
    assert (lp.argmax(1) == ref.argmax(1)).mean() > 0.9


@pytest.mark.model
def test_token_remap(acoustic_model):
    m = acoustic_model
    saved = m._remap
    try:
        perm = np.arange(127)[::-1].copy()
        perm[5] = -1                                  # a packaged token missing in the model
        m._remap = perm
        raw = np.log(np.full((2, 127), 1 / 127, dtype=np.float32))
        raw[:, 126] = 0.5                             # model column 126 -> packaged token 0
        out = m._fix(raw)
        assert out[0, 0] == 0.5 and out[0, 5] == -np.inf
    finally:
        m._remap = saved
