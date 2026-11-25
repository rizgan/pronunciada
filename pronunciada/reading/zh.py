"""Mandarin, layer 1: pinyin -> the target (pronunciada.core format), by rules only.

    from pronunciada import phones_from_pinyin
    phones_from_pinyin("huo3che1 ba1dian3 chu1fa1", text="火车八点出发。")
    phones_from_pinyin("nǐ hǎo")                                   # tone marks; sandhi applied: ní hǎo
    phones_from_pinyin("wo3 bu4 zhi1·dao4", text="我不知道。")      # · = neutral, written tone = lexical
    phones_from_pinyin([{"pinyin": "xiao3|lao3hu3", "text": "小老虎"}])   # | = inner grouping of a word

No dictionary and no third-party package. The same tone rules and pinyin -> phone tables build the
targets of the character path (pronunciada.text.zh: jieba, pypinyin, OpenCC), which only adds the
reading itself: the word segmentation, the pinyin, the neutral tones and the inner structure of
three-syllable words.

Input. Words are separated by white space; words matter (word scores, tone sandhi groups). Inside a
word, syllables are written together (tone numbers or tone marks; an apostrophe or hyphen may
separate them). Tones: 1-4 (digits or marks), 5 / 0 / no tone = neutral. A middle dot before a
syllable ("·dao4", the dictionary convention) = said neutral, the written tone is its lexical tone;
"=" before a syllable = its tone is fixed as written, no sandhi (the character path's lexicon
patterns, e.g. 怎么样 "=zen3'me5'=yang4": 怎 keeps tone 3 before 么).
Erhua: "nar3" or "na3r" (one syllable with a final r), or the syllable "er" written after the
character 儿 in `text` / "chars" (merged as the character path does). "v" or "u:" = ü.
Characters (optional): `text` (the prompt, aligned one Han character per syllable) or per word
{"pinyin": ..., "text": shown word, "chars": one string per syllable, "char_span": [start, end]}.
Characters switch on the character rules: 一 / 不 sandhi, 过 as a particle, the reduplication list,
Taiwan tone variants (pronunciada/data/cjk/zh_tw_tones.txt, a 25-entry in-house table).

sandhi=True (default): the written tones are citation tones (as in a dictionary) and the path
computes the surface tones: neutral tone, 一 / 不 (only with characters), third-tone sandhi over the
word grouping given ("|" inside a word marks where an inner group starts: "xiao3|lao3hu3" = A|BC;
without it a run of three tone-3 syllables inside one word is grouped AB|C),
plus the same tone alternatives as the character path. sandhi=False: the written tones are the
surface tones (tone_lexical = the written tone, or the dotted syllable's tone), no rule applied.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ..text._cjk_common import Ph, Syl, Wd, assemble

# --------------------------------------------------------------------------- pinyin -> ZIPA

INITIALS = {
    "b": "p", "p": "pʰ", "m": "m", "f": "f", "d": "t", "t": "tʰ", "n": "n", "l": "l",
    "g": "k", "k": "kʰ", "h": "x", "j": "tɕ", "q": "tɕʰ", "x": "ɕ",
    "zh": "ʈʂ", "ch": "ʈʂʰ", "sh": "ʂ", "r": "ʐ", "z": "ts", "c": "tsʰ", "s": "s",
}
_INIT_RE = re.compile(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*)$")

# final (after y/w/ü/iu/ui/un normalisation) -> (glide, nucleus, coda) in ZIPA convention
FINALS: dict[str, tuple[str, str, str]] = {
    "a": ("", "a", ""), "o": ("", "ɔ", ""), "e": ("", "ɤ", ""), "ê": ("", "ɛ", ""),
    "ai": ("", "aɪ", ""), "ei": ("", "eɪ", ""), "ao": ("", "ɑʊ", ""), "ou": ("", "oʊ", ""),
    "an": ("", "a", "n"), "en": ("", "ə", "n"), "ang": ("", "ɑ", "ŋ"), "eng": ("", "ɤ", "ŋ"),
    "ong": ("", "ʊ", "ŋ"), "er": ("", "ɑɻ", ""),
    "i": ("", "i", ""), "ia": ("j", "a", ""), "ie": ("j", "ɛ", ""), "iao": ("j", "ɑʊ", ""),
    "iou": ("j", "oʊ", ""), "ian": ("j", "ɛ", "n"), "in": ("", "i", "n"), "iang": ("j", "ɑ", "ŋ"),
    "ing": ("", "i", "ŋ"), "iong": ("j", "ʊ", "ŋ"), "io": ("j", "ɔ", ""),
    "u": ("", "u", ""), "ua": ("w", "a", ""), "uo": ("w", "ɔ", ""), "uai": ("w", "aɪ", ""),
    "uei": ("w", "eɪ", ""), "uan": ("w", "a", "n"), "uen": ("w", "ə", "n"), "uang": ("w", "ɑ", "ŋ"),
    "ueng": ("w", "ə", "ŋ"),
    "ü": ("", "y", ""), "üe": ("ɥ", "œ", ""), "üan": ("ɥ", "œ", "n"), "ün": ("", "y", "n"),
    "-i": ("", "ɯ", ""),      # apical vowel after z c s (ZIPA: ɯ ~70-90 %)
    "-ir": ("", "ɨ", ""),     # apical vowel after zh ch sh r
    # syllabic nasals / interjections
    "m": ("", "m", ""), "n": ("", "n", ""), "ng": ("", "ŋ", ""), "hm": ("", "m", ""), "hng": ("", "ŋ", ""),
}
_Y = {"yi": "i", "ya": "ia", "ye": "ie", "yao": "iao", "you": "iou", "yan": "ian", "yin": "in",
      "yang": "iang", "ying": "ing", "yong": "iong", "yo": "io", "yu": "ü", "yue": "üe", "yuan": "üan",
      "yun": "ün"}
_W = {"wu": "u", "wa": "ua", "wo": "uo", "wai": "uai", "wei": "uei", "wan": "uan", "wen": "uen",
      "wang": "uang", "weng": "ueng"}


def parse_pinyin(base: str) -> tuple[str, str] | None:
    """'zhong' -> ('zh', 'ong'); 'yu' -> ('', 'ü'); returns (initial, normalised final)."""
    base = base.replace("v", "ü").replace("u:", "ü")
    if base in ("m", "n", "ng", "hm", "hng"):
        return "", base
    if base in ("e", "ê", "eh"):
        return "", "ê" if base != "e" else "e"
    if base in _Y:
        return "", _Y[base]
    if base in _W:
        return "", _W[base]
    m = _INIT_RE.match(base)
    if not m:
        return None
    ini, fin = m.group(1), m.group(2)
    if ini in ("y", "w"):
        return None
    if ini in ("j", "q", "x") and fin.startswith("u"):
        fin = "ü" + fin[1:]
    if ini and fin == "iu":
        fin = "iou"
    elif ini and fin == "ui":
        fin = "uei"
    elif ini and fin == "un":
        fin = "uen" if ini not in ("j", "q", "x") else "ün"
    if fin == "i" and ini in ("z", "c", "s"):
        fin = "-i"
    elif fin == "i" and ini in ("zh", "ch", "sh", "r"):
        fin = "-ir"
    if ini in ("b", "p", "m", "f") and fin == "o":
        fin = "uo"           # bo po mo fo are [pwɔ] etc.
    if fin not in FINALS:
        return None
    return ini, fin


# --------------------------------------------------------------------------- tone rules

PARTICLES = set("的地得了着过吗呢吧啊呀嘛啦哦么们哇哪喽呗咯")
ERHUA_NOT = {"女儿", "婴儿", "儿女", "幼儿", "孤儿", "健儿", "宠儿", "胎儿", "少儿", "男儿", "患儿",
             "育儿", "孙儿", "混血儿", "新生儿", "儿", "托儿", "产儿", "病儿", "犬儿", "侄儿"}
REDUP_SANDHI = {"想想", "走走", "等等", "写写", "洗洗", "讲讲", "找找", "比比", "哪里", "小姐", "打扫"}

_TW: list = []


def tw_tones() -> dict[tuple[str, str], int]:
    """Taiwan tone variants: (character, pinyin with citation tone) -> Taiwan tone (in-house table)."""
    if not _TW:
        tw = {}
        path = Path(__file__).resolve().parent.parent / "data" / "cjk" / "zh_tw_tones.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                ch, py, t = line.split()
                tw[(ch, py)] = int(t)
        _TW.append(tw)
    return _TW[0]


def yibu_sandhi(S: list[dict]) -> None:
    """一 and 不 sandhi (character rules; syllables without a character are left alone)."""
    n = len(S)
    for i, s in enumerate(S):
        if s["ch"] not in ("一", "不") or s.get("neutral") or s.get("fixed"):
            continue
        nx = S[i + 1] if i + 1 < n else None
        nt = None
        if nx is not None:
            nt = nx["lex"] if nx["tone"] == 5 else nx["tone"]
        if s["ch"] == "不":
            if nt == 4:
                s["tone"] = 2
            continue
        prev = S[i - 1]["ch"] if i > 0 else ""
        if nx is None or (s["ci"] == s["wl"] - 1 and s["wl"] > 1):
            continue                  # 统一, 唯一, 之一, phrase-final
        if nx["ch"] in "百千万亿" and nx["ch"]:
            if prev and prev in "十零一二三四五六七八九〇":
                continue              # 二十一万
        elif prev and prev in "第零一二三四五六七八九十百千万亿两〇":
            continue                  # numbers, ordinals: 第一 十一 一百一十
        elif nx["ch"] and nx["ch"] in "零一二三四五六七八九十〇":
            continue                  # digit strings 一九九五
        if nx["ch"] and nx["ch"] in "月号":
            s["alts"].append(2 if nt == 4 else 4)
            continue
        s["tone"] = 2 if nt == 4 else 4


def third_tone_sandhi(S: list[dict], split3) -> None:
    """Third-tone sandhi on maximal chains of consecutive tone-3 syllables; word structure decides
    grouping. split3(chain segment) -> True for A+BC (小老虎), False for AB+C (展览馆)."""
    n = len(S)
    i = 0
    while i < n:
        if S[i]["tone"] != 3 or S[i].get("fixed"):
            i += 1
            continue
        j = i
        while j < n and S[j]["tone"] == 3 and not S[j].get("fixed"):
            j += 1
        if j - i >= 2:
            _t3_chain(S[i:j], split3)
        i = j


def _t3_chain(C: list[dict], split3) -> None:
    segs: list[list[dict]] = []
    for s in C:
        if segs and segs[-1][0]["w"] == s["w"]:
            segs[-1].append(s)
        else:
            segs.append([s])
    for k, seg in enumerate(segs):
        last = k == len(segs) - 1
        m = len(seg)
        if m == 1:
            tones, alts = [3], [[]]
        elif m == 2:
            tones, alts = [2, 3], [[], []]
        elif m == 3:
            if split3(seg):                                             # A+BC: 小老虎
                tones, alts = [3, 2, 3], [[2], [], []]
            else:                                                      # AB+C: 展览馆
                tones, alts = [2, 2, 3], [[], [], []]
        else:
            tones = [2] * (m - 1) + [3]
            alts = [[3] if q % 2 == 1 else [] for q in range(m - 1)] + [[]]
        if not last:
            tones[-1] = 2
            alts[-1] = alts[-1] + [3]
        for s, t, a in zip(seg, tones, alts):
            s["tone"] = t
            s["alts"].extend(a)


def after_sandhi_alts(S: list[dict], tw: dict | None) -> None:
    """3 + neutral syllable of underlying tone 3 (想想 xiángxiang, 姐姐 jiějie), and Taiwan usage:
    full tone instead of a lexical neutral tone, per-character Taiwan reading tones."""
    n = len(S)
    for i in range(n - 1):
        a, b = S[i], S[i + 1]
        if a["tone"] == 3 and b["tone"] == 5 and b["lex"] == 3 and a["w"] == b["w"]:
            if a["word"] in REDUP_SANDHI:
                a["tone"], alt = 2, 3
            else:
                alt = 2
            a["alts"].append(alt)
    for s in S:
        if s["tone"] == 5 and s["ch"] not in PARTICLES and s["lex"] != 5:
            s["alts"].append(s["lex"])
        if tw:
            t = tw.get((s["ch"], f"{s['base']}{s['lex']}"))
            if t is not None and s["tone"] == s["lex"]:
                s["alts"].append(t)


# --------------------------------------------------------------------------- syllables -> phones

def build_words(S: list[dict], wl_spans: dict[int, tuple[tuple[int, int], str]]) -> list[Wd]:
    """Syllable records -> words. wl_spans: word index -> ((start, end), shown text).
    record: ch, w, ci, wl, word, base, lex, tone, alts, shown, (erhua: merge into the previous
    syllable regardless of the character rule)."""
    words: dict[int, Wd] = {}
    prev_syl: Syl | None = None
    prev_info = None
    out: list[Wd] = []
    for s in S:
        wi = s["w"]
        if wi not in words:
            span, shown = wl_spans[wi]
            words[wi] = Wd(text=shown, span=span)
            out.append(words[wi])
        # erhua: word-final 儿 merges into the previous syllable of the same word
        if ((s.get("erhua") or (s["ch"] == "儿" and s["ci"] == s["wl"] - 1 and s["wl"] >= 2
                                and s["word"] not in ERHUA_NOT))
                and prev_syl is not None and prev_info is not None and prev_info["w"] == wi
                and s["base"] == "er"):
            prev_syl.text += s["shown"]
            last = prev_syl.phones[-1]
            if not last.nucleus and last.ipa in ("n", "j"):
                last.alt("", -0.3)
            er = Ph("ɻ")
            er.alt("ɑɻ", -0.7)
            er.alt("ə˞", -1.0)
            prev_syl.phones.append(er)
            continue
        parsed = parse_pinyin(s["base"])
        if parsed is None:
            continue
        phones = syl_phones(*parsed, s)
        attrs = {"tone": s["tone"], "tone_lexical": s["lex"],
                 "tone_alts": sorted({t for t in s["alts"] if t != s["tone"]})}
        syl = Syl(text=s["shown"], phones=phones, attrs=attrs)
        words[wi].syls.append(syl)
        # cross-syllable coda alternatives (assimilation to the next initial)
        if prev_syl is not None:
            coda_assim(prev_syl, phones)
        prev_syl, prev_info = syl, s
    return out


def syl_phones(ini: str, fin: str, s: dict) -> list[Ph]:
    glide, nuc, coda = FINALS[fin]
    out: list[Ph] = []
    if ini:
        p = Ph(INITIALS[ini])
        if ini == "r":
            p.alt("ɻ", -0.7)
        if ini == "h":
            p.alt("h", -1.0)
        out.append(p)
    if glide == "j" and not ini and fin in ("iou", "iong"):
        glide = "y"                # ZIPA writes you/yong as [yoʊ] [yʊŋ] (~95 %)
    if glide:
        g = Ph(glide)
        if glide == "y":
            g.alt("j", -0.5)
        elif glide == "j" and ini in ("j", "q", "x"):
            g.alt("", -1.0)        # glide absorbed by the palatal initial (千 tɕʰɛn)
        out.append(g)
    if fin == "eng" and ini in ("sh", "r", "h"):
        nuc = "ə"                  # ZIPA: ʂəŋ ʐəŋ xəŋ
    n = Ph(nuc, nucleus=True)
    if fin == "eng":
        n.alt("ɤ" if nuc == "ə" else "ə", -0.3)
    if fin in ("-i", "-ir"):
        n.alt("", -1.5)            # apical vowel swallowed in fast speech (是 → ʂ)
        n.alt("ɨ" if fin == "-i" else "ɯ", -0.5)
    if s["tone"] == 5 and nuc in ("ɤ", "a", "ɔ", "oʊ", "ə", "ɛ"):
        n.alt("ə", -0.7)          # neutral-tone vowel reduction
    out.append(n)
    if fin == "er":
        n.alt("ɑ", -1.0)
        n.alt("ə˞", -1.0)
    if coda:
        c = Ph(coda)
        c.alt("", -2.0)            # nasal coda realised as vowel nasalisation only
        if fin == "in":
            c.alt("ŋ", -1.0)       # -in / -ing merger, very common in native speech
        out.append(c)
    return out


def coda_assim(prev: Syl, nxt: list[Ph]) -> None:
    last = prev.phones[-1]
    if last.nucleus or last.ipa != "n" or not nxt:
        return
    first = nxt[0].ipa
    if first[0] in "pm":
        last.alt("m", -1.0)
    elif first[0] in "kx":
        last.alt("ŋ", -1.0)


# --------------------------------------------------------------------------- pinyin parsing

_MARKS = {"̄": 1, "́": 2, "̌": 3, "̀": 4}
_SEP = "'’-"


def _valid_base(b: str) -> bool:
    if b.endswith("r") and b not in ("er",) and len(b) > 1 and parse_pinyin(b[:-1]) is not None:
        return True
    return parse_pinyin(b) is not None or b == "er"


def _split_letters(s: str) -> list[str] | None:
    """Segment toneless pinyin letters into syllables (longest first, with backtracking)."""
    n = len(s)
    memo: dict[int, list[str] | None] = {n: []}

    def go(i):
        if i in memo:
            return memo[i]
        res = None
        for j in range(min(n, i + 7), i, -1):
            b = s[i:j]
            if _valid_base(b):
                rest = go(j)
                if rest is not None:
                    res = [b] + rest
                    break
        memo[i] = res
        return res

    return go(0)


def parse_syllables(word: str) -> list[tuple[str, int, bool, bool]]:
    """'zhi1·dao4' -> [('zhi', 1, False, False), ('dao', 4, True, False)]:
    (base, written tone 1-5, mark, erhua); mark "·" = said neutral (the written tone is lexical),
    "=" = tone fixed as written (no sandhi), "" = none."""
    w = unicodedata.normalize("NFD", word.strip().lower()).replace("ü", "ü").replace("u:", "ü")
    w = w.replace("v", "ü")
    out = []
    # pieces between separators and dots; digits close a syllable
    chunks: list[tuple[str, bool]] = []
    buf, dot = "", ""
    for ch in w:
        if ch in _SEP or ch in "·•=":
            if buf:
                chunks.append((buf, dot))
            buf, dot = "", ("·" if ch in "·•" else ch if ch == "=" else "")
            continue
        buf += ch
        if ch.isdigit():
            chunks.append((buf, dot))
            buf, dot = "", ""
    if buf:
        chunks.append((buf, dot))
    for chunk, dotted in chunks:
        digit = None
        if chunk[-1].isdigit():
            digit = int(chunk[-1])
            chunk = chunk[:-1]
        # tone marks -> (letters, tone per syllable)
        letters, marks = "", {}
        for ch in chunk:
            if ch in _MARKS:
                marks[len(letters) - 1] = _MARKS[ch]
            elif unicodedata.combining(ch):
                continue
            else:
                letters += ch
        letters = unicodedata.normalize("NFC", letters)
        if digit is not None:
            sy = [letters] if _valid_base(letters) else (_split_letters(letters) or [letters])
            if len(sy) > 1:
                for b in sy[:-1]:
                    out.append(_mk(b, 5, ""))
            # tone digit 'na3r' style: r after the digit is handled by the next chunk
            out.append(_mk(sy[-1], 5 if digit in (0, 5) else digit, dotted))
            continue
        if letters == "r" and out:                       # 'na3r'
            b, t, d, _ = out[-1]
            out[-1] = (b, t, d, True)
            continue
        sy = _split_letters(letters)
        if sy is None:
            raise ValueError(f"cannot read pinyin {chunk!r}")
        pos = 0
        for k, b in enumerate(sy):
            t = next((marks[q] for q in range(pos, pos + len(b)) if q in marks), 5)
            out.append(_mk(b, t, dotted if k == 0 else ""))
            pos += len(b)
    return out


def _mk(b, t, dotted):
    if b.endswith("r") and b != "er" and parse_pinyin(b) is None and parse_pinyin(b[:-1]) is not None:
        return (b[:-1], t, dotted, True)
    if parse_pinyin(b) is None and b != "er":
        raise ValueError(f"not a pinyin syllable: {b!r}")
    return (b, t, dotted, False)


# --------------------------------------------------------------------------- the pinyin path

def _is_han(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2FA1F
            or 0xF900 <= o <= 0xFAFF or ch == "〇")


_PUNCT = ",.;:!?，。；：！？、…—()（）「」“”\"《》"


def _words(reading) -> list[dict]:
    """Word dicts; "pause_before" marks a phrase break (punctuation in the string form)."""
    items = reading if isinstance(reading, (list, tuple)) else [reading]
    out = []
    brk = True
    for it in items:
        if isinstance(it, dict):
            if "pinyin" not in it:
                raise ValueError("a word dict needs 'pinyin'")
            w = dict(it)
            w["pause_before"] = bool(w.get("pause_before")) or brk or not out
            out.append(w)
            brk = False
            continue
        for tok in re.split(r"(\s+|[" + re.escape(_PUNCT) + r"])", str(it)):
            if not tok or tok.isspace():
                continue
            if tok in _PUNCT:
                brk = True
                continue
            out.append({"pinyin": tok, "pause_before": brk or not out})
            brk = False
    return out


def reference_from_pinyin(reading, text: str | None = None, *, sandhi: bool = True):
    """The Reference (pronunciada.types) of a pinyin reading; see the module doc."""
    words = _words(reading)
    warnings: list[str] = []
    S: list[dict] = []
    groups: dict[int, list[int]] = {}       # word -> inner group starts ("|")
    for wi, w in enumerate(words):
        parts = str(w["pinyin"]).split("|")
        syls = []
        starts = []
        for part in parts:
            starts.append(len(syls))
            syls += parse_syllables(part)
        groups[wi] = starts
        w["_syls"] = syls
    # characters: per word "chars", else aligned from `text`
    han_pos = [i for i, c in enumerate(text) if _is_han(c)] if text else []
    k = 0
    aligned = text is not None and not any("chars" in w for w in words)
    if aligned:
        need = sum(len(w["_syls"]) + sum(1 for s in w["_syls"] if s[3]) for w in words)
        if need != len(han_pos):
            warnings.append("characters of the text do not line up with the syllables; character rules off")
            aligned = False
    for wi, w in enumerate(words):
        syls = w["_syls"]
        chars: list[str] = []
        spans: list[tuple[int, int]] = []
        if "chars" in w:
            cs = list(w["chars"]) if isinstance(w["chars"], (list, tuple)) else list(str(w["chars"]))
            if len(cs) != len(syls):
                raise ValueError(f"word {w['pinyin']!r}: 'chars' needs one entry per syllable")
            chars = [str(c) for c in cs]
        elif aligned:
            for s in syls:
                n = 2 if s[3] else 1
                ps = han_pos[k:k + n]
                k += n
                chars.append("".join(text[p] for p in ps))
                spans.append((ps[0], ps[-1] + 1))
        if w.get("text") is None:
            w["text"] = "".join(chars) if chars else " ".join(
                b + ("r" if e else "") + str(t) for b, t, _, e in syls)
        if w.get("char_span") is None:
            w["char_span"] = (spans[0][0], spans[-1][1]) if spans else None
        word_chars = "".join(chars)
        for ci, (base, t, dotted, erhua) in enumerate(syls):
            ch = chars[ci][0] if chars and chars[ci] else ""
            lex = t
            tone = 5 if dotted == "·" else t
            rec = {"ch": ch, "w": wi, "ci": ci, "wl": len(syls) + sum(1 for s in syls if s[3]),
                   "word": word_chars, "base": base, "lex": lex, "tone": tone, "alts": [],
                   "shown": chars[ci][:1] if chars else base + str(t)}
            if not sandhi or dotted == "=":
                rec["fixed"] = True
            if tone == 5:
                rec["neutral"] = True
            S.append(rec)
            if erhua:
                S.append({"ch": "儿", "w": wi, "ci": ci + 1, "wl": rec["wl"], "word": word_chars, "base": "er",
                          "lex": 5, "tone": 5, "alts": [], "shown": chars[ci][1:] if chars else "r",
                          "erhua": True, "neutral": True})
        # ci with erhua expansion: renumber
    ci_by_w: dict[int, int] = {}
    for s in S:
        s["ci"] = ci_by_w.get(s["w"], 0)
        ci_by_w[s["w"]] = s["ci"] + 1
    # words: text and spans
    if text is None:
        pieces, pos = [], 0
        for w in words:
            w["char_span"] = (pos, pos + len(w["text"]))
            pieces.append(w["text"])
            pos += len(w["text"]) + 1
        text = " ".join(pieces)
    else:
        pos = 0
        for w in words:
            if w.get("char_span") is None:
                j = text.find(w["text"], pos) if w["text"] else -1
                w["char_span"] = (j, j + len(w["text"])) if j >= 0 else (pos, pos)
            pos = int(w["char_span"][1])
    spans = {wi: (tuple(int(x) for x in w["char_span"]), w["text"]) for wi, w in enumerate(words)}
    tw = tw_tones() if sandhi and any(s["ch"] for s in S) else None

    def split3(seg):                    # A|BC when the word marks a group start after A
        return seg[0]["ci"] + 1 in groups.get(seg[0]["w"], [0])

    # phrases (between punctuation): tone rules and sound assimilation stop at a phrase break, as in
    # the character path
    phrases: list[list[dict]] = []
    for s in S:
        if not phrases or (words[s["w"]]["pause_before"] and s["w"] != phrases[-1][-1]["w"]):
            phrases.append([])
        phrases[-1].append(s)
    out: list[Wd] = []
    for P in phrases:
        if sandhi:
            for i, s in enumerate(P):
                # aspect particle 过 (guo) vs verb guò: a one-character word after another word
                if (s["ch"] == "过" and s["ci"] == 0 and s["wl"] == 1 and s["tone"] != 5 and i > 0
                        and P[i - 1]["w"] == s["w"] - 1):
                    s["alts"].append(5)
            yibu_sandhi(P)
            third_tone_sandhi(P, split3)
            after_sandhi_alts(P, tw)
        out += build_words(P, spans)
    ref = assemble("zh", text, out, warnings)
    ref.validate()
    return ref
