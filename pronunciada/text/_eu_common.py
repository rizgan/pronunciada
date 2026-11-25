"""Shared machinery for the European text front-ends (en de fr es it pt ru).

Pipeline used by every language module:

    text --tokenize()--> Tok (original text + char span + kind)
         --language expands/looks up--> W (working word: phones, syllables, stress)
         --cross-word rules add alts--> assemble() --> pronunciada.types.Reference

Lexicon entries are stored as space separated phones in ZIPA tokens; a nucleus may carry a
stress digit suffix ("aʊ1", "ə0", "ɪ2"; no digit = unstressed / language without stress).
"""
from __future__ import annotations

import gzip
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources

from ..types import RefPhone, RefSyllable, RefWord, Reference
from .ipa import VOWELS, is_valid

LEX_PKG = "pronunciada"
LEX_DIR = "data/lex"

# log-priors used for rule-based alternatives (documented in docs/text_eu_notes.md)
P_COMMON = math.log(0.5)     # very frequent optional process (weak forms, flapping, schwa drop)
P_MED = math.log(0.25)       # regular but less frequent
P_LOW = math.log(0.1)        # possible, rarer (accent-dependent, careful/strong forms)
P_RARE = math.log(0.03)      # dialectal (e.g. Caribbean /s/ aspiration)


# --------------------------------------------------------------------------- working types

@dataclass
class P:
    ipa: str
    nucleus: bool = False
    alts: list[tuple[str, float]] = field(default_factory=list)
    stress: int | None = None       # only meaningful on a nucleus

    def add_alt(self, ipa: str, prior: float) -> None:
        ipa = unicodedata.normalize("NFD", ipa)
        if ipa == self.ipa or (ipa and not is_valid(ipa)):
            return
        for i, (a, p) in enumerate(self.alts):
            if a == ipa:
                if prior > p:
                    self.alts[i] = (a, prior)
                return
        self.alts.append((ipa, min(0.0, prior)))


@dataclass
class W:
    text: str                        # original token
    span: tuple[int, int]
    phones: list[P] = field(default_factory=list)
    syls: list[tuple[int, int, int | None]] = field(default_factory=list)   # (start, end, stress)
    norm: str = ""                   # normalised spelling used for lookup
    oov: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class Tok:
    text: str
    span: tuple[int, int]
    kind: str                        # "word" | "num" | "sym"
    ordinal: str = ""                # ordinal suffix for numbers ("st", "e", "º" ...)
    num: str = ""                    # the digits part of a number token


# --------------------------------------------------------------------------- tokenisation

_APOS = "'’ʼ`´"
_LETTER = r"[^\W\d_]"
_MARK = "̀-ͯ"           # combining marks (NFD input, Russian stress accents)
_WORD = rf"{_LETTER}[{_MARK}]*(?:(?:[{_APOS}]|[-‐‑]|\.(?={_LETTER}\.))?{_LETTER}[{_MARK}]*)*[{_APOS}]?\.?"


def make_token_re(thousand_space: bool, ordinals: str) -> re.Pattern:
    grp = r"(?:[,.]\d{3})+" if not thousand_space else r"(?:[,.   ]\d{3})+"
    num = rf"\d+{grp}(?:[.,]\d+)?(?!\d)|\d+(?:[.,]\d+)?"
    ords = rf"(?:{ordinals})(?!{_LETTER})" if ordinals else "(?!)"
    return re.compile(
        rf"(?P<num>(?:{num}))(?P<ord>{ords})?|(?P<word>{_WORD})|(?P<sym>[%&$€£+=@§°])",
        re.UNICODE,
    )


def tokenize(text: str, rx: re.Pattern, abbrev_dot: frozenset[str] = frozenset()) -> list[Tok]:
    """Split text into word / number / symbol tokens with spans. A trailing '.' stays part of
    a word only for single letters ("U.N.", "J.") or when the word is a known abbreviation."""
    toks: list[Tok] = []
    for m in rx.finditer(text):
        if m.group("num") is not None:
            toks.append(Tok(m.group(0), (m.start(), m.end()), "num", m.group("ord") or "", m.group("num")))
        elif m.group("word") is not None:
            s, e = m.span()
            w = text[s:e]
            if w.endswith("."):
                core = w[:-1]
                if not ("." in core or len(core) == 1 or core.lower() in abbrev_dot):
                    w, e = core, e - 1
            # strip leading/trailing hyphens and apostrophes that are quote marks
            while w and w[0] in _APOS + "-‐‑":
                w, s = w[1:], s + 1
            if not w:
                continue
            toks.append(Tok(w, (s, e), "word"))
        else:
            toks.append(Tok(m.group(0), m.span(), "sym"))
    return toks


def fold_apos(s: str) -> str:
    for a in "’ʼ`´":
        s = s.replace(a, "'")
    return s.replace("‐", "-").replace("‑", "-")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def is_acronym(tok: str) -> bool:
    t = tok.replace(".", "")
    return len(t) >= 2 and t.isupper() and t.isalpha() and len(t) <= 6


# --------------------------------------------------------------------------- numbers

def _num2words():
    """The number speller: pronunciada.text.numbers.num2words (Apache-2.0, standard library only).
    Its strings are identical to those of the num2words package (LGPL) that earlier versions used
    when it was installed; that package is no longer used."""
    from .numbers import num2words as f

    return f


DECIMAL_WORD = {"en": "point", "de": "Komma", "fr": "virgule", "es": "punto", "it": "virgola",
                "pt_BR": "vírgula", "pt": "vírgula"}


def number_words(tok: Tok, lang: str, decimal_comma: bool, warnings: list[str]) -> list[str] | None:
    """Digits -> list of words in `lang` (None if the number cannot be read)."""
    _n2w = _num2words()
    s =(tok.num or tok.text).replace(" ", "").replace(" ", "").replace(" ", "")
    dec = "," if decimal_comma else "."
    grp = "." if decimal_comma else ","
    # a lone group separator followed by exactly 3 digits is a thousands separator
    if re.fullmatch(rf"\d{{1,3}}(\{grp}\d{{3}})+", s):
        s = s.replace(grp, "")
    elif dec in s and grp in s:
        s = s.replace(grp, "")
    elif grp in s:            # e.g. "3.5" in a decimal-comma language: still a decimal point
        s = s.replace(grp, dec)
    try:
        if dec in s:
            ip, fp = s.split(dec, 1)
            if lang == "ru":
                out = _n2w(float(f"{ip}.{fp}"), lang=lang)
            else:
                word = DECIMAL_WORD.get(lang, "point")
                if lang == "es" and "," in (tok.num or tok.text):
                    word = "coma"
                if lang == "en" or len(fp) > 2:
                    frac = " ".join(_n2w(int(d), lang=lang) for d in fp)
                else:
                    frac = " ".join([_n2w(0, lang=lang)] * (len(fp) - len(fp.lstrip("0"))) +
                                    ([_n2w(int(fp), lang=lang)] if fp.strip("0") else []))
                out = f"{_n2w(int(ip), lang=lang)} {word} {frac}"
        else:
            val = int(s)
            if tok.ordinal == "s" and lang == "en":          # 1990s, 80s
                base = _n2w(val, lang=lang, to="year") if len(s) == 4 else _n2w(val, lang=lang)
                out = re.sub(r"y$", "ies", base) if base.endswith("y") else base + "s"
            elif tok.ordinal:
                out = _n2w(val, lang=lang, to="ordinal")
            elif lang == "en" and len(s) == 4 and (1100 <= val <= 1999 or 2010 <= val <= 2099):
                out = _n2w(val, lang=lang, to="year")
            elif lang == "de" and len(s) == 4 and 1100 <= val <= 1999:
                out = _n2w(val // 100, lang="de") + "hundert" + (_n2w(val % 100, lang="de") if val % 100 else "")
            else:
                out = _n2w(val, lang=lang)
    except Exception:  # noqa: BLE001
        warnings.append(f"number_failed:{tok.text}")
        return None
    out = out.replace(",", " ")
    return [w for w in re.split(r"[\s]+", out) if w]


# --------------------------------------------------------------------------- lexicon i/o

def lex_path(name: str):
    return resources.files(LEX_PKG).joinpath(f"{LEX_DIR}/{name}")


def load_lexicon_dict(name: str) -> tuple[dict[str, str], dict[str, str]]:
    """gzip TSV: word \t pron [\t pron2 ...]. Values kept as the raw tail string (parsed lazily).
    Lines starting with '#' are metadata "#key \t value" (e.g. legal onsets)."""
    d: dict[str, str] = {}
    meta: dict[str, str] = {}
    with lex_path(name).open("rb") as fb, gzip.open(fb, "rt", encoding="utf-8") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            if k[:1] == "#":
                meta[k[1:]] = v
            elif k:
                d[k] = v
    return d, meta


def load_lexicon(name: str):
    """(lexicon, meta) of load_lexicon_dict, the lexicon as a SortedLexicon (same keys, values and
    order, a fraction of the memory) when the file is in the layout it needs (written by
    pronunciada/text/_eu_build.py: metadata lines first, then the words in increasing order);
    otherwise the dict."""
    lex = SortedLexicon.from_tsv(read_gz(name))
    if lex is None:
        return load_lexicon_dict(name)
    return lex, lex.meta


def read_gz(name: str) -> bytearray:
    """Decompressed bytes of the packaged gzip file `name`, decompressed in 1 MB pieces into one
    preallocated buffer: GzipFile.read() and zlib.decompress hold about twice the output while
    growing it, and the allocator keeps that peak as resident memory."""
    import zlib

    comp = lex_path(name).read_bytes()
    size = int.from_bytes(comp[-4:], "little") if len(comp) >= 18 else -1    # ISIZE (single member)
    if size >= 0:
        d = zlib.decompressobj(31)
        buf = bytearray(size)
        pos, data = 0, comp
        while data and pos <= size:
            piece = d.decompress(data, 1 << 20)
            buf[pos:pos + len(piece)] = piece
            pos += len(piece)
            data = d.unconsumed_tail
        if d.eof and not d.unused_data and pos == size == len(buf):
            return buf
    with lex_path(name).open("rb") as fb, gzip.open(fb, "rb") as f:     # anything unusual
        return bytearray(f.read())


def _positions(a, value: int, start: int, chunk: int = 1 << 20):
    """int32 positions >= start of byte `value` in uint8 array a (in pieces: no full-size mask)."""
    import numpy as np

    out = [np.flatnonzero(a[s:s + chunk] == value).astype(np.int32) + np.int32(s)
           for s in range(start, len(a), chunk)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int32)


class SortedLexicon(Mapping):
    """Read-only {word: raw pronunciation tail} over ONE buffer holding the TSV file ("word \\t
    tail" lines, words strictly increasing in UTF-8 byte order = Python str order), searched by
    bisection. A dict of the same file holds two Python str per entry; this holds the
    decompressed file plus three int32 offsets per entry."""

    def __init__(self, data, start: int, meta: dict[str, str]):
        import numpy as np

        self._b = data
        self.meta = meta
        a = np.frombuffer(data, dtype=np.uint8)
        nl = _positions(a, 10, start)
        lo = np.concatenate([np.array([start], np.int32), nl + np.int32(1)])
        hi = np.concatenate([nl, np.array([len(data)], np.int32)])
        if start >= len(data) or data[-1] == 10:          # a final newline ends the last line
            lo, hi = lo[:-1], hi[:-1]
        tabs = _positions(a, 9, start)
        j = np.searchsorted(tabs, lo)                     # first tab at or after each line start
        tab = hi.copy()                                   # no tab on the line: the key is the line
        has = j < len(tabs)
        tab[has] = np.minimum(tabs[j[has]], hi[has])
        self._lo, self._tab, self._hi = lo, tab, hi

    @classmethod
    def from_tsv(cls, raw, check: bool = False):
        """SortedLexicon of a TSV file, or None when it cannot stand in for load_lexicon_dict:
        a "\\r" (the text-mode reader translates it) or a body over 2 GB. The packaged lexicons
        are written sorted (_eu_build.write_lex); check=True also verifies that (non-empty keys,
        no "#" key, strictly increasing: what the bisection needs), as the tests do for every
        packaged file."""
        if b"\r" in raw or len(raw) >= 2**31:
            return None
        meta: dict[str, str] = {}
        i = 0
        while raw.startswith(b"#", i):
            j = raw.find(b"\n", i)
            j = len(raw) if j < 0 else j
            k, _, v = raw[i + 1:j].decode("utf-8").partition("\t")
            meta[k] = v
            i = j + 1
        lex = cls(raw, i, meta)
        if check:
            raw.decode("utf-8")                           # the text-mode reader's failure
            keys = [lex._key(q) for q in range(len(lex))]
            if any(not k or k.startswith(b"#") for k in keys) or \
                    any(keys[q] >= keys[q + 1] for q in range(len(keys) - 1)):
                return None
        return lex

    def _key(self, i: int) -> bytes:
        return self._b[int(self._lo[i]):int(self._tab[i])]

    def _find(self, w) -> int:
        if not isinstance(w, str):
            return -1
        try:
            k = w.encode("utf-8")
        except UnicodeEncodeError:
            return -1
        lo, hi = 0, len(self._lo)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._key(mid) < k:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < len(self._lo) and self._key(lo) == k else -1

    def _value(self, i: int) -> str:
        t, h = int(self._tab[i]), int(self._hi[i])
        return self._b[t + 1:h].decode("utf-8") if t < h else ""

    def __getitem__(self, w) -> str:
        i = self._find(w)
        if i < 0:
            raise KeyError(w)
        return self._value(i)

    def get(self, w, default=None):
        i = self._find(w)
        return default if i < 0 else self._value(i)

    def __contains__(self, w) -> bool:
        return self._find(w) >= 0

    def __len__(self) -> int:
        return len(self._lo)

    def __iter__(self):
        for i in range(len(self._lo)):
            yield self._key(i).decode("utf-8")

    def sorted_keys(self) -> "_SortedKeys":
        """The keys as a read-only sequence in increasing order (= sorted(dict)), decoded on
        access: what bisect.bisect_left and prefix scans need, without a list of every key."""
        return _SortedKeys(self)


class _SortedKeys(Sequence):
    def __init__(self, lex: SortedLexicon):
        self._lex = lex

    def __len__(self) -> int:
        return len(self._lex)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        return self._lex._key(i).decode("utf-8")


class SortedWords:
    """Membership test on a word list stored as ONE bytes object: the UTF-8 words in strictly
    increasing byte order (= Python's str order), separated by "\\n", searched by bisection.
    Same answers as frozenset(text.split("\\n")) for such a list, at about a tenth of its memory
    (the Spanish list: ~600 k words, ~85 MB as a frozenset of str, ~12 MB here)."""

    def __init__(self, data):
        import numpy as np

        if len(data) >= 2**31:
            raise ValueError("word list over 2 GB")
        self._b = data
        nl = _positions(np.frombuffer(data, dtype=np.uint8), 10, 0)
        self._lo = np.concatenate([np.zeros(1, np.int32), nl + np.int32(1)])
        self._hi = np.concatenate([nl, np.array([len(data)], np.int32)])

    def __len__(self) -> int:
        return len(self._lo)

    def _word(self, i: int) -> bytes:
        return self._b[int(self._lo[i]):int(self._hi[i])]

    def __contains__(self, w) -> bool:
        if not isinstance(w, str):
            return False
        try:
            k = w.encode("utf-8")
        except UnicodeEncodeError:          # lone surrogates: cannot be in a UTF-8 file
            return False
        lo, hi = 0, len(self._lo)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._word(mid) < k:
                lo = mid + 1
            else:
                hi = mid
        return lo < len(self._lo) and self._word(lo) == k

    def is_sorted(self) -> bool:
        """The precondition of the bisection: strictly increasing (checked by the tests)."""
        return all(self._word(i) < self._word(i + 1) for i in range(len(self._lo) - 1))


def parse_onsets(s: str) -> frozenset[tuple[str, ...]]:
    return frozenset(tuple(c.split("+")) for c in s.split(" ") if c)


def split_prons(raw: str) -> list[list[str]]:
    return [p.split(" ") for p in raw.split("\t") if p]


def parse_phone(tok: str) -> tuple[str, int | None]:
    if tok[-1] in "012":
        return tok[:-1], int(tok[-1])
    return tok, None


def is_nucleus_ipa(ipa: str) -> bool:
    return bool(ipa) and (ipa[0] in VOWELS or "̩" in ipa)


# --------------------------------------------------------------------------- syllables

def syllabify(phones: list[P], onsets: frozenset[tuple[str, ...]], max_onset: int = 3,
              default_stress: int | None = None) -> list[tuple[int, int, int | None]]:
    """Maximal-onset syllabification. Each syllable gets exactly one nucleus; a word without
    any vowel gets its most sonorous consonant marked as nucleus."""
    n = len(phones)
    nuc = [i for i, p in enumerate(phones) if p.nucleus]
    if not nuc:
        best = max(range(n), key=lambda i: _sonority(phones[i].ipa))
        phones[best].nucleus = True
        nuc = [best]
    bounds = [0]
    for a, b in zip(nuc, nuc[1:]):
        cl = phones[a + 1:b]
        j = len(cl)
        for k in range(len(cl) + 1):  # smallest k such that cl[k:] is a legal onset
            rest = tuple(p.ipa for p in cl[k:])
            if len(rest) <= 1 or (len(rest) <= max_onset and rest in onsets):
                j = k
                break
        bounds.append(a + 1 + j)
    bounds.append(n)
    out = []
    for si, i in enumerate(nuc):
        st = phones[i].stress
        out.append((bounds[si], bounds[si + 1], st if st is not None else default_stress))
    return out


_SON = {**{c: 1 for c in "ptkbdgqcɟʔ"}, **{c: 2 for c in "fvszʃʒθðxɣçχʁhɸβɕʑʂʐ"},
        **{c: 3 for c in "mnɲŋɱ"}, **{c: 4 for c in "lrɾɹʎɫɭɻʀ"}, **{c: 5 for c in "jwɥ"}}


def _sonority(ipa: str) -> int:
    return 6 if is_nucleus_ipa(ipa) else _SON.get(ipa[0], 2)


def word_from_prons(text: str, span, phones_tok: list[str], onsets, *, default_stress=None) -> W:
    ps = []
    for t in phones_tok:
        ipa, st = parse_phone(t)
        ps.append(P(ipa, is_nucleus_ipa(ipa), stress=st))
    w = W(text, span, ps)
    w.syls = syllabify(ps, onsets, default_stress=default_stress)
    return w


def resyllabify(w: W, onsets, default_stress=None) -> None:
    w.syls = syllabify(w.phones, onsets, default_stress=default_stress)


# --------------------------------------------------------------------------- alternatives

def align_alt(canon: list[str], alt: list[str]) -> list[str] | None:
    """Levenshtein-align `alt` onto `canon`; return, per canonical phone, the alt string
    realising it ("" = dropped; insertions are glued to the preceding phone, or the next one
    at the start). None if identical."""
    if canon == alt:
        return None
    n, m = len(canon), len(alt)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if canon[i - 1] == alt[j - 1] else (1 if _same_class(canon[i - 1], alt[j - 1]) else 1.5)
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub)
    out = [""] * n
    pending: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = 0 if canon[i - 1] == alt[j - 1] else (1 if _same_class(canon[i - 1], alt[j - 1]) else 1.5)
            if d[i][j] == d[i - 1][j - 1] + sub:
                out[i - 1] = alt[j - 1] + "".join(reversed(pending)); pending = []
                i -= 1; j -= 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            out[i - 1] = "".join(reversed(pending)); pending = []
            i -= 1
        else:
            pending.append(alt[j - 1]); j -= 1
    if pending:
        out[0] = "".join(reversed(pending)) + out[0]
    return out


def _same_class(a: str, b: str) -> bool:
    return is_nucleus_ipa(a) == is_nucleus_ipa(b)


def add_word_variant(w: W, alt_phones: list[str], prior: float) -> None:
    """Add a whole-word variant as per-phone alts (independent slots)."""
    canon = [p.ipa for p in w.phones]
    al = align_alt(canon, alt_phones)
    if al is None:
        return
    for p, a in zip(w.phones, al):
        if a != p.ipa and (a == "" or is_valid(a)):
            p.add_alt(a, prior)


# --------------------------------------------------------------------------- assembly

def assemble(lang: str, text: str, words: list[W], warnings: list[str]) -> Reference:
    ref = Reference(lang=lang, text=text, warnings=warnings)
    for w in words:
        if not w.phones:
            warnings.append(f"skipped:{w.text}")
            continue
        wi = len(ref.words)
        s0, p0 = len(ref.syllables), len(ref.phones)
        for (a, b, st) in w.syls:
            si = len(ref.syllables)
            ref.syllables.append(RefSyllable(word=wi, phones=(p0 + a, p0 + b), stress=st))
            for k in range(a, b):
                p = w.phones[k]
                ref.phones.append(RefPhone(p.ipa, wi, si, p.nucleus, [(x, float(y)) for x, y in p.alts]))
        ref.words.append(RefWord(w.text, (s0, len(ref.syllables)), (p0, len(ref.phones)), w.span))
    return ref


def check_phones(phones: list[str]) -> bool:
    return all(is_valid(p) for p in phones)
