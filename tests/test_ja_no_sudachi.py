"""The Japanese front-end never uses Sudachi (the second dictionary pulled in by pyopenjtalk-plus)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["電車は八時に出発します。", "体全体に非常に迅速に送られ", "この種の人は論理的思考力に富み",
         # kanji outside Open JTalk's dictionary: pyopenjtalk-plus would tokenise these with Sudachi
         "𩸽を焼く", "彅の木", "人を嗤う"]

_RUN = r"""
import json, sys
BLOCK = set(sys.argv[1].split(",")) if sys.argv[1] else set()

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCK:
            raise ImportError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, Blocker())
built = []
from pronunciada.text.ja import import_pyopenjtalk
import_pyopenjtalk()
if "sudachipy" not in BLOCK:
    import sudachipy.dictionary as sd
    orig = sd.Dictionary
    def spy(*a, **k):
        built.append(1)
        return orig(*a, **k)
    sd.Dictionary = spy
from pronunciada.core import phones_for
out = [phones_for(t, "ja") for t in json.loads(sys.argv[2])]
print(json.dumps({"targets": out, "dictionary_built": len(built),
                  "loaded": sorted(m for m in sys.modules if m.split(".")[0] in ("sudachipy", "sudachidict_core"))},
                 ensure_ascii=False))
"""


def _run(block):
    pytest.importorskip("pyopenjtalk")
    r = subprocess.run([sys.executable, "-c", _RUN, block, json.dumps(TEXTS, ensure_ascii=False)], cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_sudachi_dictionary_is_never_loaded():
    res = _run("")
    assert res["dictionary_built"] == 0
    # pyopenjtalk-plus imports the sudachipy package at module level (unavoidable); the dictionary
    # package is never imported
    assert not any(m.startswith("sudachidict_core") for m in res["loaded"])


def test_front_end_works_without_sudachi_installed_and_gives_the_same_targets():
    with_pkg = _run("")
    without = _run("sudachipy,sudachidict_core")
    assert without["loaded"] == []
    assert without["targets"] == with_pkg["targets"]


def test_frontend_options_are_the_package_defaults_except_sudachi():
    pj = pytest.importorskip("pyopenjtalk")
    import inspect

    from pronunciada.text.ja import FRONTEND_OPTIONS

    sig = inspect.signature(pj.run_frontend)
    for k, v in FRONTEND_OPTIONS.items():
        default = sig.parameters[k].default
        if k == "use_sudachi_kanji_yomi":
            assert v is False and default is True
        else:
            assert v == default, k
    # every option of run_frontend except the text and the instance is pinned
    assert set(sig.parameters) - {"text", "jtalk"} == set(FRONTEND_OPTIONS)
