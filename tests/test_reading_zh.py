"""Layer 1 Mandarin: phones_from_pinyin (rules only) against the character path, and its isolation."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SENTENCES = ["火车八点出发。", "我不知道。", "今天天气很好，我们一起去公园散步吧。", "展览馆里有一只小老虎。",
             "他不是我的朋友，是不是？", "哪儿有一点儿水？", "你怎么样了", "我很好，你呢？",
             "一九九五年他去过北京一次。"]


def _pinyin_words(text):
    """The character path's own reading of `text` as pinyin-path input (citation tones, neutral
    marks, fixed marks, inner grouping of tone-3 runs, characters, phrase breaks)."""
    from pronunciada.text import get_g2p
    from pronunciada.text import zh as Z

    g = get_g2p("zh")
    cap = []
    orig = Z.yibu_sandhi
    Z.yibu_sandhi = lambda S: (cap.append(copy.deepcopy(S)), orig(S))[1]
    try:
        ref = g(text)
    finally:
        Z.yibu_sandhi = orig
    recs = [dict(s, _ph=i) for i, S in enumerate(cap) for s in S]
    out, k, prev_ph = [], 0, None
    for w in ref.words:
        n = w.syllables[1] - w.syllables[0]
        syl = []
        while k < len(recs) and (len(syl) < n or (recs[k]["ch"] == "儿" and recs[k]["base"] == "er"
                                                  and recs[k]["ci"] == recs[k]["wl"] - 1)):
            syl.append(recs[k])
            k += 1
        toks = []
        for s in syl:
            fx = "=" if s.get("fixed") else ""
            if s.get("neutral"):
                toks.append(fx + (("·" + s["base"] + str(s["lex"])) if s["lex"] != 5 else s["base"] + "5"))
            else:
                toks.append(fx + s["base"] + str(s["tone"]))
        cut = set()
        for p in range(len(syl) - 2):
            run = syl[p:p + 3]
            if all(x["tone"] == 3 and not x.get("fixed") for x in run):
                wd = "".join(x["ch"] for x in run)
                if g._freq.get(wd[1:], 0) > g._freq.get(wd[:2], 0):
                    cut.add(p + 1)
        out.append({"pinyin": "".join(("|" if i in cut else ("'" if i else "")) + t for i, t in enumerate(toks)),
                    "text": w.text, "char_span": list(w.char_span), "chars": [x["ch"] for x in syl],
                    "pause_before": not out or syl[0]["_ph"] != prev_ph})
        prev_ph = syl[0]["_ph"]
    return ref, out


@pytest.mark.parametrize("text", SENTENCES)
def test_pinyin_path_equals_character_path_on_its_own_reading(text):
    pytest.importorskip("jieba")
    pytest.importorskip("pypinyin")
    from pronunciada.core import phones_from_pinyin, target_from_reference

    ref, words = _pinyin_words(text)
    a = target_from_reference(ref)
    b = phones_from_pinyin(words, text)
    assert a["words"] == b["words"]


def _syl(t):
    return [(s["text"], [p["ipa"] for p in s["phones"]], s["tone"], s["tone_lexical"])
            for w in t["words"] for s in w["syllables"]]


def test_notation():
    from pronunciada.core import phones_from_pinyin

    a = phones_from_pinyin("ni3 hao3")
    b = phones_from_pinyin("nǐ hǎo")
    assert [x[1:] for x in _syl(a)] == [x[1:] for x in _syl(b)]
    assert [x[2] for x in _syl(phones_from_pinyin("ni3hao3"))] == [2, 3]          # one word: 3-3 sandhi
    assert [x[2] for x in _syl(phones_from_pinyin("ni3hao3", sandhi=False))] == [3, 3]
    t = _syl(phones_from_pinyin("zhi1·dao4"))
    assert (t[1][2], t[1][3]) == (5, 4)                                           # neutral, lexical 4
    assert [x[2] for x in _syl(phones_from_pinyin("xiao3|lao3hu3"))] == [3, 2, 3]
    assert [x[2] for x in _syl(phones_from_pinyin("zhan3lan3guan3"))] == [2, 2, 3]
    er = _syl(phones_from_pinyin("nar3"))
    assert len(er) == 1 and er[0][1][-1] == "ɻ"
    assert _syl(phones_from_pinyin("na3r")) == er
    assert [x[2] for x in _syl(phones_from_pinyin("bu4 shi4", text="不是"))] == [2, 4]   # 不 sandhi needs 不
    assert [x[2] for x in _syl(phones_from_pinyin("bu4 shi4"))] == [4, 4]
    assert [x[0] for x in _syl(phones_from_pinyin("lv4 nü3", text="绿女"))] == ["绿", "女"]
    with pytest.raises(ValueError):
        phones_from_pinyin("xyz3")


_ISO = r"""
import json, sys
opened = []
sys.addaudithook(lambda e, a: opened.append(a[0]) if e == "open" and a and isinstance(a[0], str) else None)
import pronunciada
t = pronunciada.phones_from_pinyin("huo3che1 ba1dian3 chu1fa1", text="火车八点出发。")
from pronunciada.core import reference_from_target
reference_from_target(t, "zh")
print(json.dumps({"modules": sorted(sys.modules), "opened": opened, "n": len(t["words"])}))
"""


def test_pinyin_path_loads_no_layer_two():
    r = subprocess.run([sys.executable, "-c", _ISO], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    mods = rep["modules"]
    bad = [m for m in mods if m.split(".")[0] in ("jieba", "pypinyin", "opencc", "pyopenjtalk", "sudachipy",
                                                    "num2words", "onnxruntime")]
    assert not bad, bad
    assert not [m for m in mods if m.startswith("pronunciada.text.") and m not in (
        "pronunciada.text.base", "pronunciada.text.ipa", "pronunciada.text._cjk_common")]
    data = [p.replace("\\", "/") for p in rep["opened"] if "/pronunciada/data/" in p.replace("\\", "/")]
    # only the 25-entry in-house Taiwan tone table, for the character rules
    assert all(p.endswith("data/cjk/zh_tw_tones.txt") or p.endswith("data/zipa_tokens.txt") for p in data), data
    assert rep["n"] == 3
