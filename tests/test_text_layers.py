"""The two text layers (docs/TEXT_LAYERS.md): layer 2 loads lazily, only the language's own packages
and files, and a missing layer 2 gives an error that names the extra."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "en": "The river carries 8 times as much water.",
    "de": "Der Zug fährt um 8 Uhr nach Berlin.",
    "es": "Tengo 31 años y vivo en Madrid.",
    "fr": "Le train part à 8 heures.",
    "it": "Il treno parte alle 8 per Roma.",
    "pt": "O comboio parte às 8 horas.",
    "ru": "Поезд отправляется в 8 часов.",
    "ja": "電車は8時に出発します。",
    "ko": "기차는 8시에 출발합니다.",
    "zh": "火车8点出发。",
}
LAYER2 = {"ja": ("pyopenjtalk", "pydantic"), "zh": ("jieba", "pypinyin", "opencc")}
NEVER = ("num2words", "sudachidict_core")
LEX = {"en": ("en",), "de": ("de", "en"), "es": ("es",), "fr": ("fr",), "it": ("it",), "pt": ("pt",), "ru": ("ru",)}

_CHILD = r"""
import json, sys
opened = []
sys.addaudithook(lambda e, a: opened.append(a[0]) if e == "open" and a and isinstance(a[0], str) else None)
BLOCK = set(sys.argv[2].split(",")) if sys.argv[2] else set()

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCK:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, Blocker())
import pronunciada
from pronunciada.core import phones_for
lang, text = sys.argv[1], sys.argv[3]
err = None
try:
    t = phones_for(text, lang)
    n = len(t["words"])
except ImportError as e:
    err, n = str(e), 0
print(json.dumps({"modules": sorted({m.split(".")[0] for m in sys.modules}), "pron": sorted(m for m in sys.modules
                  if m.startswith("pronunciada.text.")), "opened": opened, "error": err, "n": n}, ensure_ascii=False))
"""


def _child(lang, block=""):
    r = subprocess.run([sys.executable, "-c", _CHILD, lang, block, PROMPTS[lang]], cwd=ROOT, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("lang", list(PROMPTS))
def test_text_path_loads_only_its_own_layer_two(lang):
    if lang in LAYER2:
        pytest.importorskip(LAYER2[lang][0])
    rep = _child(lang)
    assert rep["error"] is None and rep["n"] > 0
    mods = set(rep["modules"])
    assert not mods & set(NEVER), sorted(mods & set(NEVER))
    for other, pkgs in LAYER2.items():
        if other != lang:
            assert not mods & set(pkgs), (other, sorted(mods & set(pkgs)))
    langs_loaded = {m.split(".")[2] for m in rep["pron"] if m.count(".") == 2} - {
        "base", "ipa", "_cjk_common", "_eu_common", "_eu_base", "_eu_lts", "numbers"}
    assert langs_loaded <= {lang} | ({"en"} if lang == "de" else set()), langs_loaded
    lex = {Path(p).name.split(".")[0].split("_")[0] for p in rep["opened"] if "/data/lex/" in p.replace("\\", "/")}
    assert lex <= set(LEX.get(lang, ())), lex
    if lang not in ("ja", "zh", "ko"):
        assert not [p for p in rep["opened"] if "/data/cjk/" in p.replace("\\", "/")]


@pytest.mark.parametrize("lang", ["ja", "zh"])
def test_missing_layer_two_names_the_extra(lang):
    rep = _child(lang, ",".join(LAYER2[lang]))
    assert rep["error"] and f'pronunciada[{lang}]' in rep["error"]
    assert ("phones_from_kana" if lang == "ja" else "phones_from_pinyin") in rep["error"]
    assert not set(rep["modules"]) & set(LAYER2[lang])


def test_layer_one_imports_nothing_heavy():
    code = ("import sys, pronunciada; from pronunciada import core, reading; "
            "print(sorted({m.split('.')[0] for m in sys.modules}))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    mods = set(eval(r.stdout.strip()))  # noqa: S307 - our own child's output
    assert not mods & {"numpy", "onnxruntime", "pyopenjtalk", "jieba", "pypinyin", "opencc", "num2words",
                       "sudachipy", "soundfile"}, mods


@pytest.mark.model
def test_scorer_without_layer_two_scores_targets_and_explains_text(tmp_path):
    code = r"""
import sys, json
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("pyopenjtalk", "sudachipy", "sudachidict_core"):
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, Blocker())
import numpy as np
from pronunciada.scorer import Scorer
from pronunciada.core import phones_from_kana, reference_from_target
sc = Scorer("ja", threads=1)
wav = (0.05 * np.sin(np.arange(32000) / 9.0)).astype(np.float32)
res = sc.score(wav, reference_from_target(phones_from_kana("デンシャ ワ"), "ja"))
try:
    sc.score(wav, "電車は")
    err = None
except ImportError as e:
    err = str(e)
print(json.dumps({"words": len(res.words), "err": err}))
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["words"] == 2 and 'pronunciada[ja]' in rep["err"]
