"""align.score_reference(..., wild_slots=...): explain-away pass with pre-wildcarded slots."""
import numpy as np
import pytest

from pronunciada.align import score_reference
from pronunciada.text import get_g2p
from pronunciada.text.ipa import token_ids


def _lp_for(seq, T_per=4, V=127, seed=0, noise=0.5):
    """Synthetic posteriors that 'say' the token sequence `seq` (blank between tokens)."""
    rng = np.random.default_rng(seed)
    rows = []
    for tid in seq:
        for _ in range(T_per - 1):
            rows.append(tid)
        rows.append(0)
    x = rng.normal(size=(len(rows) + 2, V)) * noise
    for t, tid in enumerate([0] + rows + [0]):
        x[t, tid] += 6.0
    return (x - np.log(np.exp(x).sum(1, keepdims=True))).astype(np.float32)


@pytest.fixture(scope="module")
def ref():
    return get_g2p("en")("sat on a mat")


def test_none_is_default(ref):
    ids = token_ids()
    lp = _lp_for([ids[c] for p in ref.phones for c in p.ipa])
    a = score_reference(lp, ref)
    b = score_reference(lp, ref, wild_slots=None)
    c = score_reference(lp, ref, wild_slots=[])
    for x, y, z in zip(a.slots, b.slots, c.slots):
        assert x == y == z


@pytest.mark.parametrize("wild", [[0], [2, 3], [len("sat on a mat") - 1]])
def test_fast_matches_exact(ref, wild):
    ids = token_ids()
    lp = _lp_for([ids[c] for p in ref.phones for c in p.ipa], seed=1)
    wild = [w for w in wild if w < len(ref.phones)]
    f = score_reference(lp, ref, wild_slots=wild)
    e = score_reference(lp, ref, wild_slots=wild, exact=True)
    assert f.L_can == pytest.approx(e.L_can, abs=1e-6)
    for i, (a, b) in enumerate(zip(f.slots, e.slots)):
        if i in wild:
            assert a.llr_sub == 0.0 and a.llr_del == 0.0
            continue
        assert a.llr_sub == pytest.approx(b.llr_sub, abs=1e-5)
        assert a.llr_del == pytest.approx(b.llr_del, abs=1e-5)
        assert (a.start_frame, a.end_frame) == (b.start_frame, b.end_frame)


def test_wildcarding_wrong_word_helps_neighbours():
    """Audio says 'the dog sat', reference says 'the elephant sat': wildcarding the wrong word
    must not lower the neighbours' evidence (it removes the forced misalignment)."""
    g = get_g2p("en")
    ids = token_ids()
    said = g("the dog sat")
    ref = g("the elephant sat")
    lp = _lp_for([ids[c] for p in said.phones for c in p.ipa], seed=2)
    first = score_reference(lp, ref)
    w = list(range(*ref.words[1].phones))
    second = score_reference(lp, ref, wild_slots=w)
    others = [i for i in range(len(ref.phones)) if i not in w]
    assert sum(second.slots[i].llr_sub for i in others) >= sum(first.slots[i].llr_sub for i in others) - 1e-6
    assert all(second.slots[i].alt_index == -2 for i in w if second.slots[i].end_frame > second.slots[i].start_frame)


def test_all_wild_does_not_crash(ref):
    ids = token_ids()
    lp = _lp_for([ids[c] for p in ref.phones for c in p.ipa], seed=3)
    ev = score_reference(lp, ref, wild_slots=range(len(ref.phones)))
    assert all(s.llr_sub == 0.0 for s in ev.slots)


def test_english_omission_recheck_keeps_prompt_indices_and_outer_scores(monkeypatch):
    from types import SimpleNamespace

    from pronunciada import Scorer
    from pronunciada import acoustic
    from pronunciada.coach import coaching

    monkeypatch.setenv("PRONUNCIADA_SENTENCE_HEAD", "0")
    monkeypatch.setattr(acoustic, "AcousticModel",
                        lambda *args, **kwargs: SimpleNamespace(frame_to_time=lambda f: f * 0.02))
    sc = Scorer("en", prosody=False, length=False, delivery=False, audio_check=False)
    reference = "the dog elephant sat at home"
    control = "the dog sat at home"
    ids = token_ids()
    lp = _lp_for([ids[c] for phone in sc.g2p(control).phones for c in phone.ipa])
    monkeypatch.setattr(sc, "_posteriors", lambda *_: (lp, None, None))
    audio = np.ones(len(lp) * 320, dtype=np.float32)
    with monkeypatch.context() as m:
        m.setattr(sc, "_recheck_omission", lambda cache, _: cache)
        baseline = sc.score(audio, reference)
    revised = sc.score(audio, reference)
    assert [w.text for w in revised.words] == reference.split()
    assert revised.prob == baseline.prob
    assert [w.omitted for w in revised.words] == [w.omitted for w in baseline.words]
    assert [i["word_index"] for i in coaching(revised) if i["kind"] == "missing_word"] == [2]
    assert revised.words[2].start == revised.words[2].end
    assert revised.words[2].present_prob == baseline.words[2].present_prob
    assert revised.words[0] == baseline.words[0]
    assert revised.words[4:] == baseline.words[4:]
    assert any(revised.words[i].phones != baseline.words[i].phones for i in (1, 3))
    assert all(a.end <= b.start for a, b in zip(revised.words, revised.words[1:]))
    assert sc.rescore(revised, 0.5).words == revised.words
