"""Japanese, layer 1: a kana reading -> the target (pronunciada.core format), by rules only.

    from pronunciada import phones_from_kana
    phones_from_kana("キョー|ワ イー テンキ デス’。")                  # pronunciation kana (Open JTalk style)
    phones_from_kana("きょう|は いい てんき です。", style="hiragana")   # textbook hiragana
    phones_from_kana([{"kana": "キョー|ワ", "text": "今日は"}, ...], text="今日は良い天気です。")

No dictionary, no third-party package, no data file. The same mora -> phone rules build the targets
of the dictionary path (pronunciada.text.ja, Open JTalk), which only adds the reading itself: the
kana, the morpheme boundaries, the devoicing marks and the orthographic reading of each morpheme.

Input. Words are separated by white space (ASCII or U+3000); words matter for scoring (word
scores, the length channel's word adjacency). Inside a word, "|" separates morphemes (optional).
Punctuation (、。,.!?！？・「」 ...) between words is a pause, as in the dictionary path. A word may be
given as a dict {"kana": ..., "text": shown text, "read": orthographic kana per morpheme,
"char_span": [start, end] into `text`}.

Style "pron" (default): the reading as Open JTalk outputs it (NJD `pron`): long vowels as ー,
particles as pronounced (ワ, エ, ヲ), devoicing marks ’ (or ') after a devoiced mora.
Style "hiragana": ordinary orthography (hiragana or katakana); turned into pronunciation kana by
the rules in `orthography_to_pron` (particles, long vowels, ぢ / づ). Ambiguities a rule cannot
settle without a dictionary, and how to mark them, are listed there.

Devoicing (devoicing="auto"): with marks in the input they are used exactly; without, the rule in
`devoiced_by_rule` decides (it only changes a prior of an optional vowel drop).
"""
from __future__ import annotations

import re
import unicodedata

from ..text._cjk_common import Ph, Syl, Wd, assemble

# --------------------------------------------------------------------------- phone tables

# Open JTalk phoneme -> ZIPA consonant
CONS = {
    "k": "k", "g": "g", "s": "s", "sh": "ʃ", "z": "z", "j": "dʒ", "t": "t", "ch": "tʃ", "ts": "ts",
    "d": "d", "n": "n", "h": "h", "f": "f", "b": "b", "p": "p", "m": "m", "y": "j", "r": "r", "w": "w",
    "ky": "kj", "gy": "gj", "ny": "ɲ", "hy": "hj", "by": "bj", "py": "pj", "my": "mj", "ry": "rj",
    "v": "v", "dy": "dj", "ty": "tj",
}
# multi-token consonants are split into (consonant phone, glide phone)
SPLIT = {"hj": ("h", "j"), "kj": ("k", "j"), "gj": ("g", "j"), "bj": ("b", "j"), "pj": ("p", "j"), "mj": ("m", "j"),
         "rj": ("r", "j"), "dj": ("d", "j"), "tj": ("t", "j")}
VOW = {"a": "a", "i": "i", "u": "u", "e": "e", "o": "o"}
VOICELESS = {"k", "s", "sh", "t", "ch", "ts", "h", "f", "p", "ky", "hy", "py", "ty"}

SMALL = set("ャュョァィゥェォヮ")
MARKS = "’'"

# kana mora -> (Open JTalk consonant, vowel / "N" / "cl"), written from the kana syllabary; checked
# against Open JTalk's full-context labels on every mora of the practice text (tests/test_reading_ja.py)
_ROWS = {
    "": "アイウエオ", "k": "カキクケコ", "g": "ガギグゲゴ", "s": "サ_スセソ", "z": "ザ_ズゼゾ", "t": "タ__テト",
    "d": "ダ__デド", "n": "ナニヌネノ", "h": "ハヒ_ヘホ", "b": "バビブベボ", "p": "パピプペポ", "m": "マミムメモ",
    "y": "ヤ_ユ_ヨ", "r": "ラリルレロ", "w": "ワ____",
}
KANA: dict[str, tuple[str, str]] = {}
for _c, _row in _ROWS.items():
    for _k, _v in zip(_row, "aiueo"):
        if _k != "_":
            KANA[_k] = (_c, _v)
KANA.update({
    "シ": ("sh", "i"), "ジ": ("j", "i"), "チ": ("ch", "i"), "ヂ": ("j", "i"), "ツ": ("ts", "u"), "ヅ": ("z", "u"),
    "フ": ("f", "u"), "ヲ": ("", "o"), "ン": ("", "N"), "ッ": ("", "cl"), "ヴ": ("v", "u"),
    "ヰ": ("", "i"), "ヱ": ("", "e"), "ヮ": ("w", "a"),
    "ァ": ("", "a"), "ィ": ("", "i"), "ゥ": ("", "u"), "ェ": ("", "e"), "ォ": ("", "o"),
    "ャ": ("y", "a"), "ュ": ("y", "u"), "ョ": ("y", "o"),
})
# palatalised CV + small ャュョ (and ェ where it is a palatal)
for _base, _c in (("キ", "ky"), ("ギ", "gy"), ("ニ", "ny"), ("ヒ", "hy"), ("ビ", "by"), ("ピ", "py"), ("ミ", "my"),
                  ("リ", "ry"), ("シ", "sh"), ("ジ", "j"), ("チ", "ch"), ("ヂ", "j")):
    for _s, _v in (("ャ", "a"), ("ュ", "u"), ("ョ", "o")):
        KANA[_base + _s] = (_c, _v)
for _base, _c in (("シ", "sh"), ("ジ", "j"), ("チ", "ch")):
    KANA[_base + "ェ"] = (_c, "e")
KANA["イェ"] = ("y", "e")
# loanword combinations
for _base, _c in (("ウ", "w"), ("ヴ", "v"), ("フ", "f"), ("ツ", "ts")):
    for _s, _v in (("ァ", "a"), ("ィ", "i"), ("ェ", "e"), ("ォ", "o")):
        KANA[_base + _s] = (_c, _v)
KANA.update({
    "ティ": ("t", "i"), "ディ": ("d", "i"), "トゥ": ("t", "u"), "ドゥ": ("d", "u"),
    "テュ": ("ty", "u"), "デュ": ("dy", "u"), "スィ": ("s", "i"), "ズィ": ("z", "i"),
    "クァ": ("kw", "a"), "クィ": ("kw", "i"), "クェ": ("kw", "e"), "クォ": ("kw", "o"), "クヮ": ("kw", "a"),
    "グァ": ("gw", "a"), "グヮ": ("gw", "a"),
    "フュ": ("fy", "u"), "ヴュ": ("by", "u"), "ウォ": ("w", "o"),
})

_LAB = re.compile(r"\^(.+?)-(.+?)\+(.+?)=")


def to_katakana(s: str) -> str:
    """Hiragana -> katakana (other characters unchanged; ゔ -> ヴ)."""
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else ("ヴ" if c == "ゔ" else c) for c in s)


def kana_moras(pron: str) -> list[str]:
    """'キョーワ' -> ['キョ', 'ー', 'ワ'] (devoicing marks removed)."""
    pron = pron.replace("’", "").replace("'", "")
    out: list[str] = []
    for ch in pron:
        if ch in SMALL and out:
            out[-1] += ch
        elif "ァ" <= ch <= "ヺ" or ch == "ー":
            out.append(ch)
    return out


def marked_moras(pron: str) -> list[tuple[str, bool]]:
    """Moras with their devoicing mark (after the devoiced mora): 'デス’' -> [('デ', False), ('ス', True)]."""
    out: list[list] = []
    for ch in pron:
        if ch in MARKS:
            if out:
                out[-1][1] = True
        elif ch in SMALL and out:
            out[-1][0] += ch
        elif "ァ" <= ch <= "ヺ" or ch == "ー":
            out.append([ch, False])
    return [(k, m) for k, m in out]


def mora_ph(kana: str, prev: tuple[str, str] | None) -> tuple[str, str] | None:
    """(consonant, vowel) of one mora as Open JTalk labels it; ー takes the previous mora's vowel."""
    if kana == "ー":
        return ("", prev[1] if prev is not None and prev[1] != "cl" else "a")
    if kana in KANA:
        return KANA[kana]
    if len(kana) > 1 and kana[0] in KANA:          # unknown combination: the base mora
        return KANA[kana[0]]
    return None


def kana_ph_fallback(kana: str) -> tuple[str, str]:
    """Rough kana -> (consonant, vowel) from Unicode names: the dictionary path's fallback when Open
    JTalk's labels do not line up with its kana (kept as it was)."""
    if kana == "ン":
        return "", "N"
    if kana == "ッ":
        return "", "cl"
    if kana == "ー":
        return "", "a"
    try:
        name = unicodedata.name(kana[0])
    except ValueError:
        return "", "a"
    rom = name.split()[-1].lower()
    v = rom[-1] if rom[-1] in "aiueo" else "a"
    c = rom[:-1]
    return {"si": "sh", "ti": "ch", "tu": "ts", "hu": "f", "zi": "j"}.get(rom, c), v


# --------------------------------------------------------------------------- devoicing rule

def devoiced_by_rule(moras: list[tuple[str, str]], pause_after: list[bool], word_end: list[bool],
                     morph_text: list[str]) -> list[bool]:
    """Vowel devoicing without Open JTalk's labels: i / u after a voiceless consonant before another
    voiceless consonant (not across a pause, never twice in a row), and the final す of です / ます
    before a pause or at the end. Open JTalk also uses the accent, which a kana reading does not
    carry; the measured agreement is in docs/TEXT_LAYERS.md. A devoiced vowel only changes the prior
    of its optional drop (-0.2 instead of -0.7), never the canonical phones."""
    n = len(moras)
    out = [False] * n
    for i, (c, v) in enumerate(moras):
        if v not in ("i", "u") or c not in VOICELESS:
            continue
        if i > 0 and out[i - 1]:
            continue
        if not pause_after[i] and i + 1 < n and moras[i + 1][0] in VOICELESS:
            out[i] = True
        elif word_end[i] and morph_text[i] in ("デス", "マス") and c == "s":
            out[i] = True
    return out


# --------------------------------------------------------------------------- orthography rules

_O_ROW = set("オコゴソゾトドノホボポモヨロョ")
_E_ROW = set("エケゲセゼテデネヘベペメレ")
_U_ROW = set("ウクグスズツヅヌフブプムユルュ")
_A_ROW = set("アカガサザタダナハバパマヤラワャ")
_I_ROW = set("イキギシジチヂニヒビピミリ")


def orthography_to_pron(morph: str, *, particle: bool = False) -> str:
    """Textbook kana of one morpheme -> pronunciation kana (katakana), by rule:

    1. particles: a morpheme that is exactly は / へ (a word of its own, or after "|") is ワ / エ;
       を stays ヲ (Open JTalk's pron, said [o]);
    2. ぢ / づ -> ジ / ズ;
    3. long vowels inside a morpheme: an o-row mora + う or お, an e-row mora + い, and a mora followed
       by its own vowel (ああ, いい, うう, ええ) become V + ー (とうきょう -> トーキョー, おおきい -> オーキイ
       only when written as おおき|い, see below).
    Not decidable without a dictionary (mark them by hand):
    - は / へ inside a word as particles (こんにちは, または, あるいは): write わ / え, or put "|" before;
    - a vowel sequence across a morpheme boundary is not long (思う おも|う, 小売 こ|うり, 大きい おおき|い
      keeps the final い): put "|" between the morphemes;
    - a long vowel the spelling does not show (rare in native words): write ー;
    - number sound changes (いち -> いっ, ひゃく -> ぴゃく): write them as pronounced."""
    k = to_katakana(morph)
    if particle and k in ("ハ", "ヘ"):
        return "ワ" if k == "ハ" else "エ"
    k = k.replace("ヂ", "ジ").replace("ヅ", "ズ")
    out: list[str] = []
    for ch in k:
        prev = out[-1] if out else ""
        if prev and prev != "ー" and (
                (ch in "ウオ" and prev in _O_ROW) or (ch == "イ" and prev in _E_ROW)
                or (ch == "ウ" and prev in _U_ROW) or (ch == "ア" and prev in _A_ROW)
                or (ch == "イ" and prev in _I_ROW) or (ch == "エ" and prev in _E_ROW)):
            out.append("ー")
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- moras -> phones

def build_words(recs: list[dict], widx: dict[int, Wd]) -> None:
    """Mora records -> syllables of the words in `widx` (shared by the dictionary and kana paths).

    rec: w (morpheme index), kana (one mora), lab ((consonant, vowel, a1, a2, devoiced) or None),
    read (orthographic reading of the morpheme), first (first mora of the morpheme), pause (a pause
    before this mora)."""
    from ..ja_round19 import on as _r19

    merge_on = _r19("merge")
    n = len(recs)
    i = 0
    pending_gem: dict | None = None
    last_syl: Syl | None = None
    last_rec: dict | None = None
    while i < n:
        r = recs[i]
        lab = r["lab"]
        kana = r["kana"]
        cons, vow = (lab[0], lab[1]) if lab else kana_ph_fallback(kana)
        nxt = recs[i + 1] if i + 1 < n and not recs[i + 1]["pause"] else None
        nxt_cons = None
        if nxt is not None:
            nxt_cons = (nxt["lab"][0] if nxt["lab"] else kana_ph_fallback(nxt["kana"])[0])
        wd = widx[r["w"]]
        # っ: merge into the following mora (doubled consonant) when it has a consonant
        if vow == "cl":
            if nxt is not None and nxt_cons:
                pending_gem = {"kana": kana}
                i += 1
                continue
            # word/phrase-final っ (あっ): glottal stop closing the previous mora
            if last_syl is not None:
                last_syl.text += kana
                p = Ph("ʔ")
                p.alt("", -0.5)
                last_syl.phones.append(p)
            i += 1
            continue
        # round 19: a bare vowel mora repeating the previous mora's vowel inside the same word
        # is said as one long vowel (場合 バアイ [baːi], 思惟 シイ [ʃiː], 大きい オーキイ
        # [oːkiː]); written as ー would be, so the length channel judges it as long
        rep = (merge_on and not cons and vow in VOW and pending_gem is None and last_syl is not None
               and last_rec is not None and last_rec["w"] == r["w"] and not r["pause"]
               and not (lab is not None and lab[4])
               and next((p.ipa for p in last_syl.phones if p.nucleus), "") == VOW[vow])
        # ー (long vowel): merge into the previous mora's nucleus
        if (kana == "ー" or rep) and last_syl is not None and last_rec is not None and not r["pause"]:
            nuc = next(p for p in last_syl.phones if p.nucleus)
            if not nuc.ipa.endswith("ː"):
                base = nuc.ipa
                nuc.ipa = base + "ː"
                nuc.alts = [(a + "ː" if a and a[-1] not in "ː" else a, pr) for a, pr in nuc.alts
                            if a and not (rep and a[0] not in "aiueoɯ")]
                nuc.alt(base, -0.7)                  # shortened in fast speech
                spelled = None if rep else _spelled_long(last_rec, base)
                if spelled:
                    nuc.alt(spelled, -1.0)           # spelling-pronunciation おう -> [ou], えい -> [ei]
            last_syl.text += kana
            i += 1
            continue
        phones = _mora_phones(cons, vow, r, nxt_cons, lab, i, recs)
        text_m = kana
        if pending_gem is not None:
            text_m = pending_gem["kana"] + kana
            c0 = phones[0]
            if not c0.nucleus:
                single = c0.ipa
                c0.ipa = geminate(single)
                c0.alts = [(geminate(a), p) for a, p in c0.alts if a]
                c0.alt(single, -1.0)                 # weak / missing gemination
            pending_gem = None
        syl = Syl(text=text_m, phones=phones)
        wd.syls.append(syl)
        last_syl, last_rec = syl, r
        i += 1


def geminate(c: str) -> str:
    """ZIPA writes geminates as a doubled consonant: kk, tt, ss, ʃʃ; affricates ttʃ, tts."""
    if c in ("tʃ", "ts", "dʒ", "dz"):
        return "t" + c if c[0] == "t" else "d" + c
    return c[0] + c


def _spelled_long(rec: dict, vowel: str) -> str | None:
    read = rec.get("read", "")
    if vowel == "o" and "ウ" in read:
        return "ou"
    if vowel == "e" and "イ" in read:
        return "ei"
    return None


def _mora_phones(cons: str, vow: str, r: dict, nxt_cons, lab, i: int, recs) -> list[Ph]:
    out: list[Ph] = []
    if vow == "N":
        nas = moraic_nasal(nxt_cons, i + 1 < len(recs))
        p = Ph(nas, nucleus=True)
        for a in ("n", "m", "ŋ", "ɴ"):
            p.alt(a, -0.7)
        out.append(p)
        return out
    prev_vowel = i > 0 and recs[i - 1]["lab"] is not None and recs[i - 1]["lab"][1] in "aiueo"
    if cons:
        c = CONS.get(cons, cons)
        # ZIPA never writes ç / ɸ for Japanese: ひ -> h (49 %), ふ -> f (33 %, h 23 %)
        if cons == "h" and vow == "u":
            c = "f"
        if c in SPLIT:
            c0, g = SPLIT[c]
            p = Ph(c0)
            out.append(p)
            out.append(Ph(g))
        else:
            p = Ph(c)
            if cons == "g" and not r["first"]:
                p.alt("ŋ", -0.7)                    # nasal g, word-medial
            if cons == "r":
                p.alt("ɾ", -0.3)
                p.alt("l", -0.7)
                p.alt("d", -1.5)
            if cons == "z":
                p.alt("dz", -0.5)
            if cons == "j":
                p.alt("ʒ", -0.5)
            if c == "f":
                p.alt("ɸ", -0.3)
                p.alt("h", -0.5)
            if cons == "h" and vow == "i":
                p.alt("ç", -0.3)
            if cons == "w":
                p.alt("v", -0.5)                    # ZIPA hears わ / particle は as [va] 29 %
            out.append(p)
    v = Ph(VOW.get(vow, vow), nucleus=True)
    if vow == "u":
        v.alt("ɯ", -0.3)
    if lab is not None and lab[4]:
        v.alt("", -0.2)                             # devoiced vowel (desu, mashita)
    elif vow in ("i", "u") and cons in VOICELESS and nxt_cons in VOICELESS:
        v.alt("", -0.7)                             # devoicing environment
    if not cons and vow in ("i", "e") and prev_vowel:
        v.alt("j", -0.3)                            # ai -> [aj], ae -> [aj]
    if not cons and vow == "o" and r["kana"] == "ヲ":
        v.alt("wo", -1.0)
    out.append(v)
    return out


def moraic_nasal(nxt: str | None, has_next: bool) -> str:
    if not has_next or nxt is None:
        return "ŋ"
    if nxt in ("p", "b", "m", "py", "by", "my"):
        return "m"
    if nxt in ("k", "g", "ky", "gy"):
        return "ŋ"
    if nxt in ("t", "d", "n", "ts", "ch", "j", "z", "r", "ny", "ty", "dy", "ry"):
        return "n"
    return "ŋ"


# --------------------------------------------------------------------------- the kana path

PAUSE = set("、。，．,.!?！？・…‥「」『』（）()［］[]【】〈〉《》―—〜~:：;；\"“”‘")
_PUNCT = "".join(sorted(PAUSE))
_SPACE = re.compile(r"[ \t\n\r　]+")


def _tokens(reading) -> list[dict]:
    """Words of the input as dicts {kana, text, read, char_span, pause_before}."""
    items = reading if isinstance(reading, (list, tuple)) else [reading]
    out: list[dict] = []
    pause = True
    for it in items:
        if isinstance(it, dict):
            w = dict(it)
            if "kana" not in w:
                raise ValueError("a word dict needs 'kana'")
            k = str(w["kana"]).strip()
            left = k.lstrip(_PUNCT)
            core_k = left.rstrip(_PUNCT)
            w["kana"] = core_k
            w["pause_before"] = pause or len(left) < len(k)
            out.append(w)
            pause = len(core_k) < len(left)
            continue
        for chunk in _SPACE.split(str(it)):
            buf = ""
            for ch in chunk:
                if ch in PAUSE:
                    if buf:
                        out.append({"kana": buf, "pause_before": pause})
                        buf = ""
                    pause = True
                else:
                    buf += ch
            if buf:
                out.append({"kana": buf, "pause_before": pause})
                pause = False
    return out


def _locate(words: list[dict], text: str) -> None:
    pos = 0
    for w in words:
        if w.get("char_span") is not None:
            pos = int(w["char_span"][1])
            continue
        t = w["text"]
        j = text.find(t, pos) if t else -1
        if j >= 0:
            w["char_span"] = (j, j + len(t))
            pos = j + len(t)
        else:
            w["char_span"] = (pos, pos)


def reference_from_kana(reading, text: str | None = None, *, style: str = "pron", devoicing: str = "auto"):
    """The Reference (pronunciada.types) of a kana reading; see the module doc."""
    if style not in ("pron", "hiragana"):
        raise ValueError("style must be 'pron' or 'hiragana'")
    if devoicing not in ("auto", "marks", "rule", "none"):
        raise ValueError("devoicing must be 'auto', 'marks', 'rule' or 'none'")
    words = _tokens(reading)
    warnings: list[str] = []
    # morphemes: (word index, pron, read, devoicing marks given)
    morphs: list[tuple[int, str, str]] = []
    for wi, w in enumerate(words):
        kana = unicodedata.normalize("NFKC", w["kana"]) if any("ｦ" <= c <= "ﾟ" for c in w["kana"]) else w["kana"]
        parts = kana.split("|")
        reads = str(w["read"]).split("|") if w.get("read") is not None else None
        if reads is not None and len(reads) != len(parts):
            raise ValueError(f"word {w['kana']!r}: 'read' needs one entry per morpheme ('|')")
        for pi, part in enumerate(parts):
            if style == "hiragana":
                single = len(parts) == 1 and pi == 0
                pron = orthography_to_pron(part, particle=single or pi > 0)
                read = to_katakana(part) if reads is None else to_katakana(reads[pi])
            else:
                pron = to_katakana(part)
                read = to_katakana(reads[pi]) if reads is not None else pron.replace("’", "").replace("'", "")
            bad = [c for c in pron if not ("ァ" <= c <= "ヺ" or c == "ー" or c in MARKS)]
            if bad:
                warnings.append(f"skipped non-kana characters: {''.join(dict.fromkeys(bad))}")
            morphs.append((wi, pron, read))
        if w.get("text") is None:
            w["text"] = "".join(c for c in w["kana"] if c not in "|’'")
    has_marks = any(c in MARKS for _, p, _ in morphs for c in p)
    use_marks = devoicing == "marks" or (devoicing == "auto" and has_marks)
    # mora records
    recs: list[dict] = []
    prev = None
    for mi, (wi, pron, read) in enumerate(morphs):
        mm = marked_moras(pron)
        first_of_word = mi == 0 or morphs[mi - 1][0] != wi
        for j, (k, mark) in enumerate(mm):
            ph = mora_ph(k, prev)
            if ph is None:
                warnings.append(f"unknown kana: {k}")
                continue
            prev = ph
            recs.append({"w": mi, "kana": k, "lab": [ph[0], ph[1], 0, 1, mark if use_marks else False],
                         "read": read, "first": j == 0, "word": wi,
                         "pause": j == 0 and first_of_word and words[wi]["pause_before"], "morph": pron})
    if devoicing == "rule" or (devoicing == "auto" and not has_marks):
        mor = [(r["lab"][0], r["lab"][1]) for r in recs]
        pause_after = [i + 1 < len(recs) and recs[i + 1]["pause"] for i in range(len(recs))]
        word_end = [i + 1 == len(recs) or recs[i + 1]["w"] != recs[i]["w"] for i in range(len(recs))]
        mtext = ["".join(k for k, _ in marked_moras(r["morph"])) for r in recs]
        for r, d in zip(recs, devoiced_by_rule(mor, pause_after, word_end, mtext)):
            r["lab"][4] = d
    # words and their text
    if text is None:
        pieces = []
        for w in words:
            if w["pause_before"] and pieces:
                pieces.append("、")
            w["char_span"] = (sum(len(p) for p in pieces), sum(len(p) for p in pieces) + len(w["text"]))
            pieces.append(w["text"])
        text = "".join(pieces)
    else:
        _locate(words, text)
    wds = [Wd(text=w["text"], span=tuple(int(x) for x in w["char_span"])) for w in words]
    widx = {mi: wds[wi] for mi, (wi, _, _) in enumerate(morphs)}
    build_words(recs, widx)
    ref = assemble("ja", text, wds, warnings)
    ref.validate()
    return ref
