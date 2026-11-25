"""Round 8 / 11: selectable word-stress models load through the public Scorer with their own
channel calibrator and score a clip."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pronunciada.prosody import DATA


RETIRED = Path(os.environ.get("PRONUNCIADA_CACHE", os.path.expanduser("~/.cache/pronunciada"))) / "prosody" / "retired"


def model_stem(name):
    """Package model name, or the path stem of a model retired from the package (round 11:
    selectable as PRONUNCIADA_*_MODEL=<path stem>); skips the test if neither exists."""
    if (DATA / f"{name}.onnx").exists():
        return name
    if (RETIRED / f"{name}.onnx").exists():
        return str(RETIRED / name)
    pytest.skip(f"{name} neither in the package nor in {RETIRED}")


CASES = [  # env var, model stem, language, `say` voice, text
    ("PRONUNCIADA_STRESS_MODEL", "stress_v3", "en", "Samantha",
     "The government announced a remarkable economic recovery yesterday."),
    ("PRONUNCIADA_STRESS_MODEL", "stress", "en", "Samantha",
     "The government announced a remarkable economic recovery yesterday."),
]


def test_defaults_unchanged():
    from pronunciada import prosody

    assert prosody.STRESS_MODEL in ("stress", "stress_v3")
    assert prosody.kind_of("ja") is None and prosody.selected_model("ja") is None   # no pitch accent
    assert (DATA / f"{prosody.STRESS_MODEL}.onnx").exists() and (DATA / "stress.onnx").exists()


@pytest.mark.parametrize("env,stem,lang,voice,text", CASES)
def test_round8_candidate_through_scorer(tmp_path, monkeypatch, env, stem, lang, voice, text):
    path_stem = model_stem(stem)
    if not _model_cached() or shutil.which("say") is None:
        pytest.skip("needs the cached ZIPA model and macOS `say`")
    out = tmp_path / "say.wav"
    r = subprocess.run(["say", "-v", voice, "-o", str(out), "--data-format=LEI16@16000", text], capture_output=True)
    if r.returncode != 0:
        pytest.skip("voice unavailable")
    monkeypatch.setenv(env, path_stem)
    from pronunciada.scorer import Scorer

    sc = Scorer(lang, threads=2)
    assert sc.prosody is not None and sc.prosody.meta["model"]
    if sc.prosody.inputs.get("zemb"):
        assert sc.am.embeddings
    from pronunciada import prosody

    if stem != prosody.default_model(lang):
        assert sc.pros_cal and sc.pros_cal.get("prosody_model") == stem, "candidate without its own calibrator"
    else:
        assert sc.pros_cal and "prosody_model" not in sc.pros_cal
    res = sc.score(str(out), text)
    assert not any("prosody failed" in w for w in res.warnings)
    vals = [s.stress_prob for w in res.words for s in w.syllables if s.stress_prob is not None]
    assert vals and all(0 <= v <= 1 for v in vals)
    assert all(0 <= w.prob <= 1 for w in res.words)


def _model_cached():
    from pronunciada.acoustic import model_cached

    return model_cached()
