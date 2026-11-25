"""Layer 1 Japanese: phones_from_kana (rules only) against the dictionary path, and its isolation."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SENTENCES = ["電車は八時に出発します。", "きっと大丈夫です。", "今日は良い天気ですね。東京に行きました、3人で。",
             "ちょっと東京", "新しいコンピュータを買いました。", "場合によっては、ヴァイオリンを弾きます。",
             "金の価格の約10倍です。", "彼は学校へ行ったが、何も言わなかった。"]


def _reading(g, text):
    """The dictionary path's own reading: one word per written span, morphemes joined by |."""
    from pronunciada.reading import ja as R

    out = []
    for off, seg in g._segments(text, []):
        with g._lock:
            njd = g.run_frontend(seg)
        wl, pb, brk = [], [], True
        for x in njd:
            if x["mora_size"] > 0 and R.kana_moras(x["pron"]):
                wl.append(x)
                pb.append(brk)
                brk = False
            else:
                brk = True
        spans = g._spans([x["string"] for x in wl], seg, off)
        for i, (x, sp) in enumerate(zip(wl, spans)):
            if out and tuple(out[-1]["char_span"]) == sp and sp[1] > sp[0]:
                out[-1]["kana"] += "|" + x["pron"]
                out[-1]["read"] += "|" + x["read"]
                continue
            out.append({"kana": ("、" if pb[i] else "") + x["pron"], "read": x["read"],
                        "text": text[sp[0]:sp[1]] or x["string"], "char_span": list(sp)})
    return out


@pytest.mark.parametrize("text", SENTENCES)
def test_kana_path_equals_dictionary_path_on_its_own_reading(text):
    pytest.importorskip("pyopenjtalk")
    from pronunciada.core import phones_for, phones_from_kana
    from pronunciada.text import get_g2p

    g = get_g2p("ja")
    a = phones_for(text, "ja")
    b = phones_from_kana(_reading(g, text), text, devoicing="marks")
    assert a["words"] == b["words"]
    assert a["text"] == b["text"]


def test_kana_table_matches_open_jtalk_labels():
    pj = pytest.importorskip("pyopenjtalk")
    from pronunciada.reading.ja import KANA, kana_moras

    from pronunciada.reading.ja import mora_ph

    seen, out_of_sync = set(), []
    for mora in KANA:
        if mora[0] in "ァィゥェォャュョヮ":
            continue
        njd = pj.run_frontend("カ" + mora + "カ")
        # whatever Open JTalk read the input as, our table on its kana must give its labels
        moras = kana_moras("".join(x["pron"] for x in njd if x["mora_size"] > 0))
        lab, cur = [], ""
        for x in pj.make_label(njd):
            p = x.split("-")[1].split("+")[0]
            if p in ("sil", "pau"):
                continue
            if p in ("a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cl"):
                lab.append((cur, p if p in ("N", "cl") else p.lower()))
                cur = ""
            else:
                cur = p
        if len(moras) != len(lab):
            # Open JTalk's labels do not line up with its own kana (the dictionary path then warns
            # "labels out of sync"): ウァ is labelled u + a; our table keeps one mora [wa]
            out_of_sync.append(mora)
            continue
        prev = None
        for k, want in zip(moras, lab):
            got = mora_ph(k, prev)
            assert got == want, (mora, k, got, want)
            prev = got
            seen.add(k)
    assert len(seen) > 100
    assert len(out_of_sync) <= 3, out_of_sync


def test_inputs_and_styles():
    from pronunciada.core import phones_from_kana

    def ph(t):
        return [[p["ipa"] for s in w["syllables"] for p in s["phones"]] for w in t["words"]]

    a = phones_from_kana("キョー|ワ イー テンキ デス’。")
    b = phones_from_kana("きょう|は いい てんき です。", style="hiragana")
    c = phones_from_kana("きょうは いい てんき です。", style="hiragana")    # は inside a word: not a particle
    assert ph(a) == ph(b)
    assert ph(c)[0] == ["k", "j", "oː", "h", "a"]
    assert ph(phones_from_kana("ガッコー エ"))[0] == ["g", "a", "kk", "oː"]
    assert ph(phones_from_kana("シンブン"))[0] == ["ʃ", "i", "m", "b", "u", "ŋ"]
    assert ph(phones_from_kana("はし", style="hiragana")) == ph(phones_from_kana("ハシ"))
    t = phones_from_kana([{"kana": "トーキョー", "text": "東京"}, {"kana": "ニ", "text": "に"}], text="東京に")
    assert [w["char_span"] for w in t["words"]] == [[0, 2], [2, 3]] and t["text"] == "東京に"
    with pytest.raises(ValueError):
        phones_from_kana("キョー", style="romaji")


def test_devoicing_marks_and_rule():
    from pronunciada.core import phones_from_kana

    def drop(t):
        return [p["alts"] for w in t["words"] for s in w["syllables"] for p in s["phones"] if p["nucleus"]]

    marked = drop(phones_from_kana("デス’", devoicing="marks"))
    assert ["", -0.2] in marked[1]
    assert drop(phones_from_kana("デス")) == marked                          # rule: final す of です
    assert ["", -0.2] not in drop(phones_from_kana("デス", devoicing="none"))[1]


_ISO = r"""
import json, sys
opened = []
sys.addaudithook(lambda e, a: opened.append(a[0]) if e == "open" and a and isinstance(a[0], str) else None)
import pronunciada
t = pronunciada.phones_from_kana("キョー|ワ イー テンキ デス’。")
t2 = pronunciada.phones_from_kana("きょう|は いい てんき です。", style="hiragana")
from pronunciada.core import reference_from_target
reference_from_target(t, "ja")
print(json.dumps({"modules": sorted(sys.modules), "opened": opened, "n": len(t["words"])}))
"""


def test_kana_path_loads_no_layer_two():
    r = subprocess.run([sys.executable, "-c", _ISO], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    mods = rep["modules"]
    bad = [m for m in mods if m.split(".")[0] in ("pyopenjtalk", "sudachipy", "sudachidict_core", "pydantic",
                                                    "jieba", "pypinyin", "opencc", "num2words", "onnxruntime")]
    assert not bad, bad
    assert not [m for m in mods if m.startswith("pronunciada.text.") and m not in (
        "pronunciada.text.base", "pronunciada.text.ipa", "pronunciada.text._cjk_common")]
    assert not [p for p in rep["opened"] if "/data/lex/" in p.replace("\\", "/") or "/data/cjk/" in p.replace("\\", "/")]
    assert rep["n"] == 4
