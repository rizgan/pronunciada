"""Mandarin Chinese text front-end (Simplified and Traditional input).

Pipeline: original text -> (Traditional -> Simplified with OpenCC, length-preserving) ->
jieba words -> pypinyin (phrase-aware) -> lexical tones -> neutral tone / 一 / 不 / third-tone
sandhi -> ZIPA-convention phones (initial, glide, nucleus, coda) + connected-speech alternatives.

Everything after the script conversion runs on Simplified text, so equivalent Simplified and
Traditional prompts give identical phones and tones; only `text` fields keep the user's script.
"""
from __future__ import annotations

import re
import threading
import unicodedata

from ..reading.zh import (FINALS, INITIALS, PARTICLES, after_sandhi_alts, build_words,  # noqa: F401
                          parse_pinyin, third_tone_sandhi, yibu_sandhi)
from ..reading.zh import ERHUA_NOT as _ERHUA_NOT  # noqa: F401
from ..reading.zh import REDUP_SANDHI as _REDUP_SANDHI  # noqa: F401
from ..types import Reference
from ._cjk_common import Wd, assemble, data_lines, is_han, is_latin, nfkc_digit
from .base import G2P

# --------------------------------------------------------------------------- numbers

_DIG = "零一二三四五六七八九"
_MEASURE = set("个只件位名本条张次天岁种家口人辆台把双对项场倍份座艘架头匹间所棵枝支颗粒块片层趟遍回声句首部群批组周")


def _int_reading(n: int) -> str:
    if n == 0:
        return "零"
    units = [(10 ** 12, "万亿"), (10 ** 8, "亿"), (10 ** 4, "万")]
    for u, name in units:
        if n >= u:
            hi, lo = divmod(n, u)
            s = _int_reading(hi) + name
            if lo:
                s += ("零" if lo < u // 10 else "") + _int_reading(lo)
            return s
    s, zero = "", False
    for u, name in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d, n = divmod(n, u)
        if d:
            if zero and s:
                s += "零"
            s += _DIG[d] + name
            zero = False
        elif s:
            zero = True
    return s


def number_reading(tok: str, nxt: str, prev: str) -> str:
    """Chinese reading (Simplified) of a digit token like '2019', '3.5', '7,000', '45%'."""
    pct = tok.endswith("%")
    tok = tok.rstrip("%")
    tok_nc = tok.replace(",", "")
    if "." in tok_nc:
        ip, fp = tok_nc.split(".", 1)
        s = _int_reading(int(ip or "0")) + "点" + "".join(_DIG[int(c)] for c in fp if c.isdigit())
    elif (len(tok_nc) == 4 and nxt == "年") or len(tok_nc) > 12 or (len(tok_nc) > 1 and tok_nc[0] == "0"):
        s = "".join(_DIG[int(c)] for c in tok_nc)          # years, codes, phone numbers
    else:
        n = int(tok_nc)
        s = _int_reading(n)
        if 10 <= n < 20 and s.startswith("一十"):
            s = s[1:]
        if n == 2 and prev != "第" and (nxt in _MEASURE or nxt == "年"):
            s = "两"
        s = re.sub(r"^二(?=[千万亿])", "两", s)
    if pct:
        s = "百分之" + s
    return s


# --------------------------------------------------------------------------- tone tables

_ZI_FULL = set("原电分质中离光粒量因孔老庄孟荀墨韩君男女天太王弟学游才赤甲臣公仙骄棋种瓜莲")  # X子 with full tone 3
_ZHU = {"著名", "著作", "显著", "著称", "著述", "卓著", "土著", "编著", "名著", "原著", "巨著", "专著", "论著",
        "著者", "昭著", "著录", "拙著", "遗著", "新著", "译著", "合著", "所著", "大著",
        "彰著", "撰著", "杰著", "近著", "著译"}


INSTALL_HINT = ('the Mandarin text front-end (Chinese characters) needs the optional "zh" extra: '
                'pip install "pronunciada[zh]"; without it, pass pinyin (pronunciada.phones_from_pinyin) '
                "or phones (pronunciada.core.score_phones)")


class ZhG2P(G2P):
    lang = "zh"

    def __init__(self) -> None:
        import importlib.util
        import logging

        missing = []
        for m in ("jieba", "pypinyin", "opencc"):
            try:
                if importlib.util.find_spec(m) is None:
                    missing.append(m)
            except (ImportError, ValueError):
                missing.append(m)
        if missing:
            raise ImportError(INSTALL_HINT + f" (missing: {', '.join(missing)})")
        import jieba
        from opencc import OpenCC
        from pypinyin import Style, lazy_pinyin

        logging.getLogger("jieba").setLevel(logging.ERROR)
        jieba.setLogLevel(logging.ERROR)
        jieba.initialize()
        self._jieba = jieba
        self._freq = jieba.dt.FREQ
        self._lazy = lazy_pinyin
        self._style = Style.TONE3
        self._t2s = OpenCC("t2s")
        self._s2t = OpenCC("s2t")
        self._lock = threading.Lock()   # jieba / opencc keep small mutable caches
        self.neutral: dict[str, list[int] | None] = {}
        for parts in data_lines("zh_neutral.txt"):
            if parts[0].startswith("="):
                self.neutral[parts[0][1:]] = [int(x) for x in parts[1:]]
            else:
                for w in parts:
                    self.neutral.setdefault(w, None)
        self.tw: dict[tuple[str, str], int] = {}
        for ch, py, t in data_lines("zh_tw_tones.txt"):
            self.tw[(ch, py)] = int(t)
        self._cit: dict[str, tuple[str, int]] = {}
        for w in self.neutral:              # register neutral-tone words with jieba so they stay whole
            if w not in self._freq:
                jieba.add_word(w, 50)
        self("你好，今天天氣很好。我們一起去公園吧！2019年")   # warm-up

    # ----------------------------------------------------------------- helpers
    def citation(self, ch: str) -> tuple[str, int]:
        """Most common single-character reading (base, tone)."""
        r = self._cit.get(ch)
        if r is None:
            p = self._lazy(ch, style=self._style, neutral_tone_with_five=True)[0]
            r = (p[:-1], int(p[-1])) if p[-1:].isdigit() else (p, 5)
            self._cit[ch] = r
        return r

    def to_simplified(self, chunk: str) -> str:
        s = self._t2s.convert(chunk)
        if len(s) != len(chunk):
            s = "".join(self._t2s.convert(c)[:1] or c for c in chunk)
        if "著" in s:
            # Taiwan-standard text writes the particle/verb 着 as 著; OpenCC t2s keeps 著.
            # Keep 著 (zhù) only inside its own words (著名, 显著, 土著 ...).
            s = "".join("着" if c == "著" and s[max(0, i - 1):i + 1] not in _ZHU and s[i:i + 2] not in _ZHU
                        else c for i, c in enumerate(s))
        return s

    def _segment(self, simp: str) -> list[str]:
        out: list[str] = []
        for w in self._jieba.lcut(simp, HMM=True):
            # 慢慢地走 -> jieba sometimes glues the adverbial 地 to the next word
            if (len(w) > 1 and w[0] in "地的" and out and len(out[-1]) >= 2 and out[-1][-1] == out[-1][-2]):
                out.append(w[0])
                w = w[1:]
            out.append(w)
        return out

    # ----------------------------------------------------------------- main
    def __call__(self, text: str) -> Reference:
        warnings: list[str] = []
        with self._lock:
            phrases, trad = self._scan(text, warnings)
            words: list[Wd] = []
            for ph in phrases:
                words.extend(self._phrase(ph, text, trad, warnings))
        return assemble("zh", text, words, warnings)

    def _scan(self, text: str, warnings: list[str]):
        """Split into phrases (between punctuation); each phrase is a list of segments:
        ('han', simplified, [(orig_start, orig_end, shown_char)]) or ('num', simplified, span)."""
        trad = self.to_simplified(text) != text
        phrases: list[list] = []
        cur: list = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if is_han(ch):
                j = i
                while j < n and is_han(text[j]):
                    j += 1
                chunk = text[i:j]
                simp = self.to_simplified(chunk)
                cur.append(("han", simp, [(i + k, i + k + 1, chunk[k]) for k in range(len(chunk))]))
                i = j
            elif nfkc_digit(ch) is not None:
                j = i
                while j < n and (nfkc_digit(text[j]) is not None or (
                        unicodedata.normalize("NFKC", text[j]) in ".,%" and j + 1 <= n and (
                        unicodedata.normalize("NFKC", text[j]) == "%" or
                        (j + 1 < n and nfkc_digit(text[j + 1]) is not None)))):
                    if unicodedata.normalize("NFKC", text[j]) == "%":
                        j += 1
                        break
                    j += 1
                tok = unicodedata.normalize("NFKC", text[i:j])
                jn = j
                while jn < n and text[jn].isspace():
                    jn += 1
                nxt = self.to_simplified(text[jn]) if jn < n else ""
                prev = self.to_simplified(text[i - 1]) if i > 0 else ""
                if "," in tok and not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?%?", tok):
                    # a list separator, not a thousands separator: read the parts separately
                    parts = re.split(r"(,)", tok)
                    pos = i
                    for p in parts:
                        if p == ",":
                            phrases.append(cur) if cur else None
                            cur = []
                        elif p:
                            cur.append(("num", number_reading(p, nxt, prev), (pos, pos + len(p))))
                        pos += len(p)
                else:
                    cur.append(("num", number_reading(tok, nxt, prev), (i, j)))
                i = j
            elif is_latin(ch):
                j = i
                while j < n and (is_latin(text[j]) or text[j] in "'-"):
                    j += 1
                warnings.append(f"skipped non-Chinese text: {text[i:j]}")
                if cur:
                    phrases.append(cur)
                cur = []
                i = j
            else:
                cat = unicodedata.category(ch)
                if not ch.isspace() and cat[0] in "PSZ" or ch in "\n\r\t":
                    if cur:
                        phrases.append(cur)
                    cur = []
                elif not ch.isspace() and cat[0] not in "PSZC":
                    warnings.append(f"skipped unsupported character: {ch}")
                i += 1
        if cur:
            phrases.append(cur)
        return phrases, trad

    def _phrase(self, segs, text: str, trad: bool, warnings: list[str]) -> list[Wd]:
        # 1. words: (simplified word, [(start, end, shown)] per char, is_number)
        wl: list[tuple[str, list[tuple[int, int, str]], bool]] = []
        for kind, simp, info in segs:
            if kind == "han":
                pos = 0
                for w in self._segment(simp):
                    wl.append((w, info[pos:pos + len(w)], False))
                    pos += len(w)
            else:
                shown = self._s2t.convert(simp) if trad else simp
                if len(shown) != len(simp):
                    shown = simp
                s, e = info
                wl.append((simp, [(s, e, c) for c in shown], True))
        # 2. syllables with pinyin and lexical tones
        syls: list[dict] = []
        pys = self._lazy([w for w, _, _ in wl], style=self._style, neutral_tone_with_five=True,
                         errors=lambda x: ["?"] * len(x))
        k = 0
        for wi, (w, chars, isnum) in enumerate(wl):
            for ci, ch in enumerate(w):
                p = pys[k] if k < len(pys) else None
                k += 1
                if p is None or not p[-1:].isdigit():
                    if is_han(ch):
                        warnings.append(f"no reading for character: {chars[ci][2]}")
                    continue
                base, tone = p[:-1], int(p[-1])
                if ch == "一":
                    base, tone = "yi", 1
                elif ch == "不":
                    base, tone = "bu", 4
                syls.append({"ch": ch, "w": wi, "ci": ci, "wl": len(w), "word": w, "base": base, "lex": tone,
                             "tone": tone, "alts": [], "span": chars[ci][:2], "shown": chars[ci][2]})
            if len(pys) < k:
                break
        if not syls:
            return []
        self._tones(syls)
        return self._build(syls, wl, text, trad)

    # ----------------------------------------------------------------- tones
    def _tones(self, S: list[dict]) -> None:
        # lexical tone of neutral-toned dictionary readings: citation tone of the same syllable
        for s in S:
            if s["lex"] == 5:
                cb, ct = self.citation(s["ch"])
                if cb == s["base"] and ct != 5:
                    s["lex"] = ct
        # neutral tone (surface)
        n = len(S)
        for i, s in enumerate(S):
            w, ci, wl, ch = s["word"], s["ci"], s["wl"], s["ch"]
            neutral = None
            if w in self.neutral:
                pat = self.neutral[w]
                if pat is None:
                    neutral = ci == wl - 1
                else:
                    t = pat[ci] if ci < len(pat) else s["lex"]
                    if t == 5:
                        neutral = True
                    else:
                        if ch not in "一不":
                            s["tone"] = t
                        neutral = False
                        s["fixed"] = True
            elif s["tone"] == 5:
                neutral = True
                if ch == "子" and wl >= 2 and ci == wl - 1 and w[ci - 1] in _ZI_FULL:
                    neutral = False
                    s["tone"] = s["lex"] = 3
            elif ch == "们" or (ch == "子" and ci == wl - 1 and wl >= 2 and w[ci - 1] not in _ZI_FULL
                                and s["base"] == "zi"):
                neutral = True
            elif ch == "得" and s["base"] == "de" and 0 < ci < wl - 1:
                neutral = True            # V得C complement
            elif ch == "不" and wl == 3 and ci == 1:
                neutral = True            # V不C potential complement / A不A
            elif ch == "一" and wl == 3 and ci == 1 and w[0] == w[2]:
                neutral = True            # 看一看
            elif (ch == "地" and wl == 1 and 0 < i < n - 1 and S[i - 1]["w"] == s["w"] - 1
                  and S[i - 1]["wl"] >= 2 and S[i + 1]["w"] == s["w"] + 1):
                s["base"], neutral = "de", True   # adverbial marker: 有效地 寻找
            elif ch == "过" and ci == 0 and wl == 1 and i > 0 and S[i - 1]["w"] == s["w"] - 1:
                s["alts"].append(5)       # aspect particle 过 (guo) vs verb guò
            if ch == "不" and wl == 1 and 0 < i < n - 1 and S[i - 1]["ch"] == S[i + 1]["ch"]:
                neutral = True            # 是不是 / 好不好 (separate words)
            if ch == "一" and wl == 1 and 0 < i < n - 1 and S[i - 1]["ch"] == S[i + 1]["ch"]:
                neutral = True
            if neutral:
                s["tone"] = 5
                s["neutral"] = True
        yibu_sandhi(S)
        freq = self._freq

        def split3(seg):                                   # A+BC (小老虎) when BC is the likelier word
            w = "".join(x["ch"] for x in seg)
            return freq.get(w[1:], 0) > freq.get(w[:2], 0)

        third_tone_sandhi(S, split3)
        after_sandhi_alts(S, self.tw)

    # ----------------------------------------------------------------- phones
    def _build(self, S: list[dict], wl, text: str, trad: bool) -> list[Wd]:
        spans = {}
        for s in S:
            wi = s["w"]
            if wi not in spans:
                chars = wl[wi][1]
                span = (min(c[0] for c in chars), max(c[1] for c in chars))
                spans[wi] = (span, text[span[0]:span[1]])
        return build_words(S, spans)

def build() -> G2P:
    return ZhG2P()
