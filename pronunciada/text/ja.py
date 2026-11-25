"""Japanese text front-end: kanji/kana -> readings (pyopenjtalk-plus / Open JTalk) -> moras ->
ZIPA-convention phones. (Pitch accent is not scored; the full-context labels are read for the
mora segmentation and devoicing only.)

Unit structure: one RefSyllable per mora, except that ZIPA writes geminates as a doubled
consonant and long vowels as V+ː, so a っ mora is merged into the following mora ("ッカ" ->
[kk a]) and a long-vowel mora ー into the preceding one ("トー" -> [t oː]). ん is its own unit
whose nucleus is the (place-assimilated) nasal.

This is layer 2 (a dictionary: Open JTalk via pyopenjtalk-plus, the "ja" extra). It finds the
reading; the moras -> phones rules are layer 1 (pronunciada.reading.ja), shared with the kana input
path pronunciada.phones_from_kana.
"""
from __future__ import annotations

import re
import threading
import unicodedata

from ..reading.ja import (CONS, SPLIT, VOICELESS, VOW, build_words, kana_moras,  # noqa: F401
                          kana_ph_fallback)
from ..reading.ja import SMALL as _SMALL  # noqa: F401
from ..types import Reference
from ._cjk_common import Wd, assemble, is_latin
from .base import G2P

_LAB = re.compile(r"\^(.+?)-(.+?)\+(.+?)=")
_ACC = re.compile(r"/A:([\-\d]+|xx)\+(\d+|xx)\+(\d+|xx)")


# pyopenjtalk-plus run_frontend options, all written out so that a change of a default upstream cannot
# change our readings silently. Every value is the package default except use_sudachi_kanji_yomi: the
# Sudachi reading correction (a second, ~190 MB dictionary) is never used. The "nani" reading model
# (predict_nani, two small ONNX files inside pyopenjtalk-plus) and the other rule-based post-processing
# steps stay on.
FRONTEND_OPTIONS = {
    "run_marine": False, "use_vanilla": False, "use_tsqyomi": False, "use_sudachi_kanji_yomi": False,
    "predict_nani": True, "normalize_mode": "None", "iu_pronunciation": None, "use_read_as_pron": False,
    "revert_long_vowels": False, "revert_yotsugana": False,
}

INSTALL_HINT = ('the Japanese text front-end needs the optional "ja" extra: pip install "pronunciada[ja]"; '
                "without it, pass a kana reading (pronunciada.phones_from_kana) or phones "
                "(pronunciada.core.score_phones)")


class _NoSudachiTokenizer:
    """Stand-in for sudachipy.tokenizer.Tokenizer when sudachipy is not installed: pyopenjtalk-plus
    imports sudachipy at module level (and names this class in an annotation), but with FRONTEND_OPTIONS
    and JaG2P.run_frontend it never calls it."""

    class SplitMode:
        A = B = C = None


class _NoSudachiDictionary:
    """Stand-in for sudachipy.dictionary.Dictionary (also named in annotations): never constructed."""

    def __init__(self, *a, **k):
        raise RuntimeError("Sudachi is not used by Pronunciada (see pronunciada.text.ja)")


def _unihan_only(orig):
    def read_unknown_kanji(njd_features, text=None):
        return orig(njd_features, None)
    return read_unknown_kanji


def import_pyopenjtalk():
    """Import pyopenjtalk (pyopenjtalk-plus). Works whether or not sudachipy / sudachidict-core are
    installed: if sudachipy is missing, placeholder modules are registered only for the duration of the
    import (pyopenjtalk keeps its references; the placeholders are removed from sys.modules again)."""
    import importlib.util
    import sys
    import types

    if "pyopenjtalk" in sys.modules:
        return sys.modules["pyopenjtalk"]
    try:
        have = importlib.util.find_spec("pyopenjtalk") is not None
    except (ImportError, ValueError):
        have = False
    if not have:
        raise ImportError(INSTALL_HINT)
    stubs = {}
    try:
        sudachi = importlib.util.find_spec("sudachipy") is not None
    except (ImportError, ValueError):
        sudachi = False
    if not sudachi:
        pkg = types.ModuleType("sudachipy")
        dic = types.ModuleType("sudachipy.dictionary")
        tok = types.ModuleType("sudachipy.tokenizer")
        dic.Dictionary = _NoSudachiDictionary
        tok.Tokenizer = _NoSudachiTokenizer
        pkg.dictionary, pkg.tokenizer = dic, tok
        stubs = {"sudachipy": pkg, "sudachipy.dictionary": dic, "sudachipy.tokenizer": tok}
        sys.modules.update(stubs)
    try:
        import pyopenjtalk
    finally:
        for k, v in stubs.items():
            if sys.modules.get(k) is v:
                del sys.modules[k]
    return pyopenjtalk


class JaG2P(G2P):
    lang = "ja"

    def __init__(self) -> None:
        pyopenjtalk = import_pyopenjtalk()

        self._pj = pyopenjtalk
        self._lock = threading.Lock()   # Open JTalk instance is not re-entrant
        self("今日は良い天気ですね。東京に行きました、3人で。")

    def run_frontend(self, seg: str):
        """pyopenjtalk.run_frontend with FRONTEND_OPTIONS, Sudachi-free. Its unknown-kanji step
        (read_unknown_kanji) would tokenise the sentence with Sudachi when a kanji is missing from Open
        JTalk's dictionary; during our call it gets no text, so it uses only its Unihan per-character
        readings, exactly as pyopenjtalk-plus does for a call without text. Call with self._lock held."""
        pj = self._pj
        orig = getattr(pj, "read_unknown_kanji", None)
        if orig is None:
            return pj.run_frontend(seg, **FRONTEND_OPTIONS)
        pj.read_unknown_kanji = _unihan_only(orig)
        try:
            return pj.run_frontend(seg, **FRONTEND_OPTIONS)
        finally:
            pj.read_unknown_kanji = orig

    # ------------------------------------------------------------ main
    def __call__(self, text: str) -> Reference:
        warnings: list[str] = []
        words: list[Wd] = []
        for seg_start, seg in self._segments(text, warnings):
            words.extend(self._segment(seg, seg_start, text, warnings))
        return assemble("ja", text, words, warnings)

    @staticmethod
    def _segments(text: str, warnings: list[str]):
        """Cut out Latin-letter runs (skipped with a warning); yield (offset, chunk)."""
        i, n, start = 0, len(text), 0
        while i < n:
            if is_latin(text[i]):
                j = i
                while j < n and (is_latin(text[j]) or text[j] in "'-. "):
                    j += 1
                warnings.append(f"skipped non-Japanese text: {text[i:j].strip()}")
                if i > start:
                    yield start, text[start:i]
                start = i = j
            else:
                i += 1
        if start < n:
            yield start, text[start:n]

    def _segment(self, seg: str, off: int, text: str, warnings: list[str]) -> list[Wd]:
        if not seg.strip():
            return []
        with self._lock:
            njd = self.run_frontend(seg)
            labels = self._pj.make_label(njd) if any(x["mora_size"] > 0 for x in njd) else []
        # --- moras from the labels: (consonant, vowel/N/cl, a1, a2, devoiced)
        lab_moras: list[list] = []
        cur_c = ""
        for lab in labels:
            m = _LAB.search(lab)
            if not m:
                continue
            ph = m.group(2)
            if ph in ("sil", "pau"):
                cur_c = ""
                continue
            a = _ACC.search(lab)
            a1 = int(a.group(1)) if a and a.group(1) != "xx" else 0
            a2 = int(a.group(2)) if a and a.group(2) != "xx" else 1
            if ph in ("a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cl"):
                lab_moras.append([cur_c, ph.lower() if ph not in ("N", "cl") else ph, a1, a2, ph in "AIUEO"])
                cur_c = ""
            else:
                cur_c = ph
        # --- words
        wl, pause_before, brk = [], [], True
        for x in njd:
            if x["mora_size"] > 0 and kana_moras(x["pron"]):
                wl.append(x)
                pause_before.append(brk)
                brk = False
            else:
                brk = True
        total = sum(len(kana_moras(x["pron"])) for x in wl)
        if total != len(lab_moras):
            warnings.append("full-context labels out of sync; kana readings used for: " + seg.strip())
            lab_moras = []
        spans = self._spans([x["string"] for x in wl], seg, off)
        # --- build mora records over the whole segment
        recs: list[dict] = []
        k = 0
        for wi, x in enumerate(wl):
            km = kana_moras(x["pron"])
            read = x.get("read", "")
            for j, kana in enumerate(km):
                lm = lab_moras[k] if lab_moras else None
                k += 1
                recs.append({"w": wi, "kana": kana, "lab": lm, "read": read, "pos": x["pos"],
                             "first": j == 0, "string": x["string"], "pause": j == 0 and pause_before[wi]})
        return self._build(recs, wl, spans, text)

    @staticmethod
    def _spans(strings: list[str], seg: str, off: int) -> list[tuple[int, int]]:
        """Map analyser word strings back to offsets in the original text. Words the analyser
        rewrote (numbers -> kanji) get the gap between their found neighbours."""
        norm = [unicodedata.normalize("NFKC", c) for c in seg]
        flat = "".join(norm)
        idx = []                       # flat position -> seg index
        for i, c in enumerate(norm):
            idx.extend([i] * len(c))
        idx.append(len(seg))
        found: list[tuple[int, int] | None] = []
        pos = 0
        for s in strings:
            s2 = unicodedata.normalize("NFKC", s)
            j = flat.find(s2, pos) if s2 else -1
            gap = flat[pos:j] if j >= 0 else ""
            if j >= 0 and (j - pos <= 12 or not re.search(r"[\w]", gap)):
                found.append((idx[j], idx[j + len(s2)]))
                pos = j + len(s2)
            else:
                found.append(None)
        # fill gaps
        out: list[tuple[int, int]] = []
        for i, f in enumerate(found):
            if f is not None:
                out.append((f[0] + off, f[1] + off))
                continue
            prev_end = next((found[q][1] for q in range(i - 1, -1, -1) if found[q] is not None), 0)
            nxt = next((found[q][0] for q in range(i + 1, len(found)) if found[q] is not None), len(seg))
            while prev_end < nxt and seg[prev_end].isspace():
                prev_end += 1
            out.append((prev_end + off, max(prev_end, nxt) + off))
        return out

    # ------------------------------------------------------------ phones
    def _build(self, recs: list[dict], wl, spans, text: str) -> list[Wd]:
        words: list[Wd] = []
        wmap: dict[tuple[int, int], Wd] = {}
        widx: dict[int, Wd] = {}
        for wi in range(len(wl)):
            sp = spans[wi]
            if sp in wmap and sp[1] > sp[0]:
                widx[wi] = wmap[sp]     # analyser split one written token (e.g. digits): one word
                continue
            wd = Wd(text=text[sp[0]:sp[1]] or wl[wi]["string"], span=sp)
            wmap[sp] = widx[wi] = wd
            words.append(wd)
        build_words(recs, widx)
        return words


def build() -> G2P:
    return JaG2P()
