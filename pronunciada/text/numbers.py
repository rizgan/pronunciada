"""Numbers spelled out as words, standard library only (Apache-2.0).

One function, `num2words(value, lang, to="cardinal")`, covering the calls the European front-ends
make (`pronunciada.text._eu_common.number_words`, `pronunciada.r20.number_alternatives`):

    lang      to                                   value
    en        cardinal, ordinal, year              int >= 0
    de fr it  cardinal, ordinal                    int >= 0
    es pt_BR  cardinal, ordinal                    int >= 0   (ordinal of 0 is "")
    ru        cardinal, ordinal                    int >= 0
    ru        cardinal                             float >= 0 (decimal fractions)
    fr_BE     cardinal, ordinal                    int >= 0

The strings are byte-identical to the ones the benchmarks and lexicon tests were produced with
(the LGPL package num2words 0.5.14), including its spelling choices where they differ from the
grammar of the language: see KNOWN_DIFFERENCES_FROM_GRAMMAR. The rules below are written from the
grammar of each language plus those observed choices; nothing here is taken from that package.

Range: every value below 10**18 is supported for every (lang, to). Above that:
    en de fr fr_BE it   supported below 10**36, ValueError from 10**36 (this module's own limit)
    es cardinal         OverflowError from 10**27
    es ordinal          see _es_ordinal (falls back to the cardinal from 10**18)
    pt_BR               OverflowError (cardinal) / KeyError (ordinal) from 10**18
    ru                  KeyError from 10**33
Negative numbers, other types and other (lang, to) pairs raise.

KNOWN_DIFFERENCES_FROM_GRAMMAR
------------------------------
Tuples (lang, to, example value, output of this function, grammatical form). The output is kept
as it is so the readings stay identical to the earlier benchmark runs; changing one changes the
reference pronunciation of every prompt that contains that number.
"""
from __future__ import annotations

import math
import re

__all__ = ["num2words", "KNOWN_DIFFERENCES_FROM_GRAMMAR"]

KNOWN_DIFFERENCES_FROM_GRAMMAR: list[tuple[str, str, int | float, str, str]] = [
    # English: British "and" ("one hundred and one") is kept; the commas are only written.
    ("en", "cardinal", 1100, "one thousand, one hundred", "one thousand one hundred"),
    # German: the count of a scale keeps the final s of "eins"; ordinals are all lower case and
    # a round multiple of a large scale is run together with the words before it.
    ("de", "cardinal", 101000, "einhunderteinstausend", "einhunderteintausend"),
    ("de", "cardinal", 101000000, "einhunderteins Millionen", "einhundertein Millionen"),
    ("de", "ordinal", 1000001, "eine million erste", "eine Million erste"),
    ("de", "ordinal", 1000001000000, "eine billionmillionste", "eine Billion einmillionste"),
    # French: the plural s of cents / vingts / millions stays in front of the ordinal ending.
    ("fr", "ordinal", 80, "quatre-vingtsième", "quatre-vingtième"),
    ("fr", "ordinal", 200, "deux centsième", "deux centième"),
    ("fr", "ordinal", 2000000, "deux millionsième", "deux millionième"),
    # Belgian French: scale nouns always plural, "cents" always plural, 80 without s, "et un" in 81.
    ("fr_BE", "cardinal", 80, "quatre-vingt", "quatre-vingts"),
    ("fr_BE", "cardinal", 81, "quatre-vingt et un", "quatre-vingt-un"),
    ("fr_BE", "cardinal", 201, "deux cents un", "deux cent un"),
    ("fr_BE", "cardinal", 200000, "deux cents mille", "deux cent mille"),
    ("fr_BE", "cardinal", 1000000, "un millions", "un million"),
    ("fr_BE", "ordinal", 200, "deux centsième", "deux centième"),
    # Spanish: no apocope of uno before mil / millones; ordinals of 11-19, 40, 400, 700, 800,
    # 10**9 and the failures near 10**15 and 10**18.
    ("es", "cardinal", 21000, "veintiuno mil", "veintiún mil"),
    ("es", "cardinal", 21000000, "veintiuno millones", "veintiún millones"),
    ("es", "cardinal", 101000000, "ciento uno millones", "ciento un millones"),
    ("es", "ordinal", 0, "", "(no ordinal)"),
    ("es", "ordinal", 11, "décimoprimero", "undécimo / decimoprimero"),
    ("es", "ordinal", 13, "décimo tercero", "decimotercero"),
    ("es", "ordinal", 40, "quadragésimo", "cuadragésimo"),
    ("es", "ordinal", 400, "cuadrigentésimo", "cuadringentésimo"),
    ("es", "ordinal", 700, "septigentésimo", "septingentésimo"),
    ("es", "ordinal", 800, "octigentésimo", "octingentésimo"),
    ("es", "ordinal", 21000, "veintiunomilésimo", "veintiunmilésimo"),
    ("es", "ordinal", 1000000001, "billonésimo primero", "milmillonésimo primero"),
    ("es", "ordinal", 999999999999999, "(RecursionError)", "novecientos noventa y nueve billonésimo ... noveno"),
    ("es", "ordinal", 999999999999999999, "(KeyError)", "novecientos noventa y nueve milbillonésimo ... noveno"),
    ("es", "ordinal", 1000000000000000000, "un trillón", "trillonésimo"),
    # Portuguese (Brazil): ", " instead of " e " after a scale word when a hundreds word follows
    # (only the first such place for each word, so trilhões is skipped after quatrilhões).
    ("pt_BR", "cardinal", 1101, "mil, cento e um", "mil cento e um"),
    ("pt_BR", "cardinal", 2200000000000, "dois trilhões, duzentos bilhões", "dois trilhões e duzentos bilhões"),
    ("pt_BR", "cardinal", 200200200200200200,
     "duzentos quatrilhões, duzentos trilhões e duzentos bilhões, duzentos milhões, duzentos mil e duzentos",
     "duzentos quatrilhões, duzentos trilhões, duzentos bilhões, duzentos milhões, duzentos mil e duzentos"),
    ("pt_BR", "ordinal", 0, "", "(no ordinal)"),
    ("pt_BR", "ordinal", 400, "quadrigentésimo", "quadringentésimo"),
    ("pt_BR", "ordinal", 600, "seiscentésimo", "sexcentésimo"),
    ("pt_BR", "ordinal", 700, "septigentésimo", "septingentésimo"),
    ("pt_BR", "ordinal", 800, "octigentésimo", "octingentésimo"),
    ("pt_BR", "ordinal", 2000, "segundo milésimo", "dois milésimos"),
    ("pt_BR", "ordinal", 1001000, "milionésimo primeiro milésimo", "milionésimo milésimo"),
    # Italian: 18 after cento loses its i, the accent of -tré comes and goes with each joining
    # step, the ordinal of 0 is "zero", and the plural of a scale is not used in ordinals.
    ("it", "cardinal", 118, "centodicotto", "centodiciotto"),
    ("it", "cardinal", 123, "centoventitre", "centoventitré"),
    ("it", "cardinal", 1000023, "un milione e ventitre", "un milione e ventitré"),
    ("it", "ordinal", 0, "zero", "(no ordinal)"),
    ("it", "ordinal", 2000000, "due milionesimo", "duemilionesimo"),
    # Russian: in an ordinal, a higher group whose count ends in a ten, teen or hundred has that
    # word as an ordinal too; in decimal fractions every один / два becomes feminine, also before
    # миллион; the reading fails when the float's repr uses an exponent (below 1e-4, from 1e16).
    ("ru", "ordinal", 10001, "десятый тысяч первый", "десять тысяч первый"),
    ("ru", "ordinal", 100001, "сотый тысяч первый", "сто тысяч первый"),
    ("ru", "cardinal", 1000000.5, "одна миллион целых пять десятых", "один миллион целых пять десятых"),
    ("ru", "cardinal", 2000000.5, "две миллиона целых пять десятых", "два миллиона целых пять десятых"),
    ("ru", "cardinal", 0.00001, "(ValueError)", "ноль целых одна стотысячная"),
]


def _check_int(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected a non-negative int, got {value!r}")
    if value < 0:
        raise ValueError(f"negative numbers are not supported: {value}")
    return value


def _groups(n: int, size: int = 1000) -> list[int]:
    """Groups of the number, lowest first: 1234567 -> [567, 234, 1]."""
    out = []
    while True:
        n, r = divmod(n, size)
        out.append(r)
        if not n:
            return out


_LIMIT_36 = 10 ** 36


def _limit(n: int, limit: int, exc=ValueError):
    if n >= limit:
        raise exc(f"{n} is too large (limit {limit})")


# =========================================================================== English

_EN_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
            "fifteen sixteen seventeen eighteen nineteen").split()
_EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_EN_SCALES = ("_ thousand million billion trillion quadrillion quintillion sextillion septillion "
              "octillion nonillion decillion").split()


def _en_below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    if r < 20:
        low = _EN_ONES[r]
    else:
        low = _EN_TENS[r // 10] + ("-" + _EN_ONES[r % 10] if r % 10 else "")
    if not h:
        return low
    return _EN_ONES[h] + " hundred" + (" and " + low if r else "")


def _en_cardinal(n: int) -> str:
    _limit(n, _LIMIT_36)
    if n < 1000:
        return _en_below_1000(n)
    k = (len(str(n)) - 1) // 3
    c, r = divmod(n, 1000 ** k)
    s = _en_below_1000(c) + " " + _EN_SCALES[k]
    if r:
        s += (" and " if r < 100 else ", ") + _en_cardinal(r)
    return s


_EN_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
           "nine": "ninth", "twelve": "twelfth"}


def _en_ordinal(n: int) -> str:
    s = _en_cardinal(n)
    cut = max(s.rfind(" "), s.rfind("-")) + 1
    head, last = s[:cut], s[cut:]
    if last in _EN_ORD:
        last = _EN_ORD[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + last


def _en_year(n: int) -> str:
    """Years read in pairs of digits: 1999 nineteen ninety-nine, 1905 nineteen oh-five,
    1900 nineteen hundred; round thousands, 2000-2009 and five or more digits as cardinals."""
    if n >= 10000:
        return _en_cardinal(n)
    high, low = divmod(n, 100)
    if not high or (high % 10 == 0 and low < 10):
        return _en_cardinal(n)
    if not low:
        return _en_cardinal(high) + " hundred"
    return _en_cardinal(high) + " " + ("oh-" + _EN_ONES[low] if low < 10 else _en_cardinal(low))


# =========================================================================== German

_DE_ONES = ("null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn "
            "fünfzehn sechzehn siebzehn achtzehn neunzehn").split()
_DE_TENS = "_ _ zwanzig dreißig vierzig fünfzig sechzig siebzig achtzig neunzig".split()
# long scale, one name per power of 1000 from 10**6 on: (singular, plural)
_DE_SCALES = [(w, w + ("n" if w.endswith("e") else "en")) for w in
              ("Million Milliarde Billion Billiarde Trillion Trilliarde Quadrillion Quadrilliarde "
               "Quintillion Quintilliarde").split()]


def _de_below_1000(n: int) -> str:
    """1..999 as one word; 1 alone is 'eins'."""
    h, r = divmod(n, 100)
    s = ("ein" if h == 1 else _DE_ONES[h]) + "hundert" if h else ""
    if r < 20:
        s += _DE_ONES[r] if r else ""
    else:
        u = r % 10
        s += (("ein" if u == 1 else _DE_ONES[u]) + "und" if u else "") + _DE_TENS[r // 10]
    return s


def _de_cardinal(n: int) -> str:
    _limit(n, _LIMIT_36)
    if n == 0:
        return "null"
    g = _groups(n)
    parts = []
    for k in range(len(g) - 1, 1, -1):
        c = g[k]
        if c:
            one, many = _DE_SCALES[k - 2]
            parts.append("eine " + one if c == 1 else _de_below_1000(c) + " " + many)
    low = ""
    if len(g) > 1 and g[1]:
        low = ("ein" if g[1] == 1 else _de_below_1000(g[1])) + "tausend"
    if g[0]:
        low += _de_below_1000(g[0])
    if low:
        parts.append(low)
    return " ".join(parts)


_DE_SCALE_WORDS = {w.lower(): s.lower() for s, p in _DE_SCALES for w in (s, p)}


def _de_ordinal(n: int) -> str:
    s = _de_cardinal(n).lower()
    if s in ("einhundert", "eintausend"):
        s = s[3:]
    last = s.rsplit(" ", 1)[-1]
    if last in _DE_SCALE_WORDS:
        # a round multiple of a large scale is written as one word: zweimillionste, millionste
        s = s[: len(s) - len(last)] + _DE_SCALE_WORDS[last]
        if s.endswith("eine " + _DE_SCALE_WORDS[last]):
            s = s[: len(s) - len(_DE_SCALE_WORDS[last]) - 5] + _DE_SCALE_WORDS[last]
        i = s.rfind(" ")
        if i >= 0:
            s = s[:i] + s[i + 1:]
        if s.endswith("e"):
            s = s[:-1]
        return s + "ste"
    for end, rep in (("eins", "erste"), ("drei", "dritte"), ("sieben", "siebte"), ("acht", "achte")):
        if s.endswith(end):
            return s[: len(s) - len(end)] + rep
    if s.endswith(("zig", "ßig", "hundert", "tausend")):
        return s + "ste"
    return s + "te"


# =========================================================================== French (France, Belgium)

_FR_ONES = ("zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze "
            "seize dix-sept dix-huit dix-neuf").split()
_FR_TENS = "_ _ vingt trente quarante cinquante soixante".split()
_FR_SCALES = ("million milliard billion billiard trillion trilliard quadrillion quadrilliard "
              "quintillion quintilliard").split()


def _fr_below_100(n: int, final: bool, be: bool) -> str:
    if n < 20:
        return _FR_ONES[n]
    t, u = divmod(n, 10)
    if be:
        tens = {7: "septante", 8: "quatre-vingt", 9: "nonante"}.get(t) or _FR_TENS[t]
        return tens + ("" if not u else " et un" if u == 1 else "-" + _FR_ONES[u])
    if t <= 6:
        return _FR_TENS[t] + ("" if not u else " et un" if u == 1 else "-" + _FR_ONES[u])
    if t == 7:
        return "soixante" + (" et onze" if u == 1 else "-" + _FR_ONES[10 + u])
    if t == 8 and not u:
        return "quatre-vingts" if final else "quatre-vingt"
    return "quatre-vingt-" + _FR_ONES[u + (10 if t == 9 else 0)]


def _fr_below_1000(n: int, final: bool, be: bool) -> str:
    """final: the group ends the number or comes before million / milliard (nouns), where
    'cents' and 'quatre-vingts' keep their plural s; before 'mille' they do not."""
    h, r = divmod(n, 100)
    if not h:
        return _fr_below_100(r, final, be)
    if h == 1:
        s = "cent"
    else:
        s = _FR_ONES[h] + " cent" + ("s" if be or (final and not r) else "")
    return s + (" " + _fr_below_100(r, final, be) if r else "")


def _fr_cardinal(n: int, be: bool = False) -> str:
    _limit(n, _LIMIT_36)
    if n == 0:
        return "zéro"
    g = _groups(n)
    parts = []
    for k in range(len(g) - 1, 1, -1):
        c = g[k]
        if c:
            parts.append(_fr_below_1000(c, True, be) + " " + _FR_SCALES[k - 2] + ("s" if c > 1 or be else ""))
    if len(g) > 1 and g[1]:
        parts.append("mille" if g[1] == 1 else _fr_below_1000(g[1], False, be) + " mille")
    if g[0]:
        parts.append(_fr_below_1000(g[0], True, be))
    return " ".join(parts)


def _fr_ordinal(n: int, be: bool = False) -> str:
    if n == 1:
        return "premier"
    s = _fr_cardinal(n, be)
    cut = max(s.rfind(" "), s.rfind("-")) + 1
    head, last = s[:cut], s[cut:]
    if last == "cinq":
        last = "cinqu"
    elif last == "neuf":
        last = "neuv"
    elif last.endswith("e"):
        last = last[:-1]
    return head + last + "ième"


# =========================================================================== Spanish

_ES_ONES = ("cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince "
            "dieciséis diecisiete dieciocho diecinueve veinte veintiuno veintidós veintitrés veinticuatro "
            "veinticinco veintiséis veintisiete veintiocho veintinueve").split()
_ES_TENS = "_ _ _ treinta cuarenta cincuenta sesenta setenta ochenta noventa".split()
_ES_HUNDREDS = ("_ ciento doscientos trescientos cuatrocientos quinientos seiscientos setecientos "
                "ochocientos novecientos").split()
# long scale, one name per power of a million: (singular, plural)
_ES_SCALES = [("millón", "millones"), ("billón", "billones"), ("trillón", "trillones"),
              ("cuatrillón", "cuatrillones")]


def _es_below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    if r < 30:
        low = _ES_ONES[r]
    else:
        low = _ES_TENS[r // 10] + (" y " + _ES_ONES[r % 10] if r % 10 else "")
    if not h:
        return low
    if h == 1 and not r:
        return "cien"
    return _ES_HUNDREDS[h] + (" " + low if r else "")


def _es_below_million(n: int) -> str:
    k, r = divmod(n, 1000)
    if not k:
        return _es_below_1000(r)
    s = "mil" if k == 1 else _es_below_1000(k) + " mil"
    return s + (" " + _es_below_1000(r) if r else "")


def _es_cardinal(n: int) -> str:
    _limit(n, 10 ** 27, OverflowError)
    g = _groups(n, 10 ** 6)
    parts = []
    for k in range(len(g) - 1, 0, -1):
        c = g[k]
        if c:
            one, many = _ES_SCALES[k - 1]
            parts.append("un " + one if c == 1 else _es_below_million(c) + " " + many)
    if g[0] or not parts:
        parts.append(_es_below_million(g[0]))
    return " ".join(parts)


_ES_ORD_ONES = "_ primero segundo tercero cuarto quinto sexto séptimo octavo noveno".split()
_ES_ORD_TENS = ("_ décimo vigésimo trigésimo quadragésimo quincuagésimo sexagésimo septuagésimo "
                "octogésimo nonagésimo").split()
_ES_ORD_HUNDREDS = ("_ centésimo ducentésimo tricentésimo cuadrigentésimo quingentésimo sexcentésimo "
                    "septigentésimo octigentésimo noningentésimo").split()
_ES_ORD_SCALES = {10 ** 3: "milésimo", 10 ** 6: "millonésimo", 10 ** 9: "billonésimo",
                  10 ** 12: "trillonésimo", 10 ** 15: "cuadrillonésimo"}


def _es_ordinal(n: int) -> str:
    """The count in front of a scale is the cardinal glued to it (dosmilésimo); a count of one is
    left out. The scale is found from the floating-point logarithm base 1000, which rounds up for the
    last 4 values below 10**15 (count 0: the reading never ends, RecursionError, also for every larger
    value whose last 15 digits are one of those) and for the last 4928 values below 10**18 (no such
    scale: KeyError); kept so the same values fail. From 10**18 the cardinal is used."""
    if n >= 10 ** 18:
        return _es_cardinal(n)
    return _es_ord(n, 0)


def _es_ord(n: int, depth: int) -> str:
    if depth > 200:
        raise RecursionError("maximum recursion depth exceeded")
    if n == 0:
        return ""
    if n < 10:
        return _ES_ORD_ONES[n]
    if n < 20:
        if n == 10:
            return "décimo"
        if n < 13:
            return "décimo" + _ES_ORD_ONES[n - 10]
        return "décimo " + _ES_ORD_ONES[n - 10]
    if n < 100:
        t, u = divmod(n, 10)
        return _ES_ORD_TENS[t] + (" " + _ES_ORD_ONES[u] if u else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ES_ORD_HUNDREDS[h] + (" " + _es_ord(r, depth + 1) if r else "")
    p = 10 ** (int(math.log(n, 1000)) * 3)
    word = _ES_ORD_SCALES[p]
    c, r = divmod(n, p)
    s = ("" if c == 1 else _es_cardinal(c)) + word
    return s + (" " + _es_ord(r, depth + 1) if r else "")


# =========================================================================== Portuguese (Brazil)

_PT_ONES = ("zero um dois três quatro cinco seis sete oito nove dez onze doze treze catorze quinze "
            "dezesseis dezessete dezoito dezenove").split()
_PT_TENS = "_ _ vinte trinta quarenta cinquenta sessenta setenta oitenta noventa".split()
_PT_HUNDREDS = ("_ cento duzentos trezentos quatrocentos quinhentos seiscentos setecentos oitocentos "
                "novecentos").split()
_PT_SCALES = [("milhão", "milhões"), ("bilhão", "bilhões"), ("trilhão", "trilhões"),
              ("quatrilhão", "quatrilhões")]
# scale words after which " e " becomes ", " when a hundreds word follows and some later word has
# an "e" in it ("mil, cento e um", "um bilhão, duzentos milhões"). Only the first occurrence of
# "<scale> e" is changed, in this order, so "trilhões e" inside "quatrilhões e" is the one changed
# ("duzentos quatrilhões, duzentos trilhões e duzentos bilhões").
_PT_COMMA_AFTER = ("mil", "milhão", "milhões", "bilhão", "bilhões", "trilhão", "trilhões", "quatrilhão",
                   "quatrilhões")


def _pt_below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    if r < 20:
        low = _PT_ONES[r]
    else:
        low = _PT_TENS[r // 10] + (" e " + _PT_ONES[r % 10] if r % 10 else "")
    if not h:
        return low
    if h == 1 and not r:
        return "cem"
    return _PT_HUNDREDS[h] + (" e " + low if r else "")


def _pt_cardinal(n: int) -> str:
    _limit(n, 10 ** 18, OverflowError)
    g = _groups(n)
    parts = []
    for k in range(len(g) - 1, 0, -1):
        c = g[k]
        if not c:
            continue
        if k == 1:
            parts.append("mil" if c == 1 else _pt_below_1000(c) + " mil")
        else:
            one, many = _PT_SCALES[k - 2]
            parts.append("um " + one if c == 1 else _pt_below_1000(c) + " " + many)
    if g[0] or not parts:
        parts.append(_pt_below_1000(g[0]))
    s = " e ".join(parts)
    for w in _PT_COMMA_AFTER:
        if re.match(r".*%s e \w*entos? (?=.*e)" % w, s):
            s = s.replace(w + " e", w + ",", 1)
    return s


_PT_ORD_ONES = "_ primeiro segundo terceiro quarto quinto sexto sétimo oitavo nono".split()
_PT_ORD_TENS = ("_ décimo vigésimo trigésimo quadragésimo quinquagésimo sexagésimo septuagésimo "
                "octogésimo nonagésimo").split()
_PT_ORD_HUNDREDS = ("_ centésimo ducentésimo tricentésimo quadrigentésimo quingentésimo seiscentésimo "
                    "septigentésimo octigentésimo nongentésimo").split()
_PT_ORD_SCALES = ("_ milésimo milionésimo bilionésimo trilionésimo quadrilionésimo").split()


def _pt_ord_below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    t, u = divmod(r, 10)
    words = [w for w in (_PT_ORD_HUNDREDS[h] if h else "", _PT_ORD_TENS[t] if t else "",
                         _PT_ORD_ONES[u] if u else "") if w]
    return " ".join(words)


def _pt_ordinal(n: int) -> str:
    """Every group is an ordinal followed by the ordinal of its scale (segundo milésimo); a leading
    count of one is left out (milésimo primeiro)."""
    if n >= 10 ** 18:
        raise KeyError(18)
    g = _groups(n)
    parts = []
    for k in range(len(g) - 1, -1, -1):
        c = g[k]
        if not c:
            continue
        if k == 0:
            parts.append(_pt_ord_below_1000(c))
        elif c == 1 and k == len(g) - 1:
            parts.append(_PT_ORD_SCALES[k])
        else:
            parts.append(_pt_ord_below_1000(c) + " " + _PT_ORD_SCALES[k])
    return " ".join(parts)


# =========================================================================== Italian

_IT_ONES = ("zero uno due tre quattro cinque sei sette otto nove dieci undici dodici tredici "
            "quattordici quindici sedici diciassette diciotto diciannove").split()
_IT_TENS = "_ _ venti trenta quaranta cinquanta sessanta settanta ottanta novanta".split()
# one name per power of 1000 from 10**6 on: (singular, plural)
_IT_SCALES = [(w, w[:-1] + "i") for w in
              ("milione miliardo bilione biliardo trilione triliardo quadrilione quadriliardo "
               "quintilione quintiliardo").split()]


def _it_flip_tre(s: str) -> str:
    """The accent of a final -tre (ventitré) is switched at every step that joins a number to the
    words before it: a word ending in -tre gets it, one ending in -tré loses it, plain 'tre'
    never has it. So 23 is 'ventitré', 123 'centoventitre', 1123 'millecentoventitré'."""
    out = []
    for w in s.split(" "):
        if w.endswith("tré"):
            w = w[:-1] + "e"
        elif w.endswith("tre") and w != "tre":
            w = w[:-1] + "é"
        out.append(w)
    return " ".join(out)


_IT_ELIDE = re.compile(r"[aeiou](?=ott)")


def _it_below_million(n: int) -> str:
    if n < 20:
        return _IT_ONES[n]
    if n < 100:
        t, u = divmod(n, 10)
        tens = _IT_TENS[t]
        if u in (1, 8):                     # ventuno, ventotto: the vowel of the tens is elided
            tens = tens[:-1]
        return _it_flip_tre(tens + (_IT_ONES[u] if u else ""))
    if n < 1000:
        h, r = divmod(n, 100)
        s = ("" if h == 1 else _IT_ONES[h]) + "cento" + (_it_below_million(r) if r else "")
        # a vowel before "ott" is elided: centotto, centottanta, and also centodicotto (118)
        return _it_flip_tre(_IT_ELIDE.sub("", s))
    k, r = divmod(n, 1000)
    s = "mille" if k == 1 else _it_below_million(k).replace("tré", "tre") + "mila"
    return _it_flip_tre(s + (_it_below_million(r) if r else ""))


def _it_big(n: int) -> tuple[str, bool]:
    """(words, single part). Parts are joined with ', ', the last one with ' e '."""
    if n < 10 ** 6:
        return _it_below_million(n), True
    k = (len(str(n)) - 1) // 3
    c, r = divmod(n, 1000 ** k)
    one, many = _IT_SCALES[k - 2]
    s = "un " + one if c == 1 else _it_flip_tre(_it_below_million(c)) + " " + many
    if not r:
        return s, True
    rest, single = _it_big(r)
    return s + (" e " if single else ", ") + _it_flip_tre(rest), False


def _it_cardinal(n: int) -> str:
    _limit(n, _LIMIT_36)
    return _it_big(n)[0]


_IT_ORD = "zero primo secondo terzo quarto quinto sesto settimo ottavo nono decimo".split()


def _it_ordinal(n: int) -> str:
    if n <= 10:
        return _IT_ORD[n]
    s = _it_cardinal(n)
    if s.endswith(("tre", "tré")):
        return s[:-1] + "eesimo"
    if s.endswith("sei"):
        return s + "esimo"
    if s.endswith("mila"):
        return s[:-4] + "millesimo"
    return s[:-1] + "esimo"


# =========================================================================== Russian

_RU_ONES = ("ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать "
            "тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать").split()
_RU_TENS = "_ _ двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто".split()
_RU_HUNDREDS = "_ сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот".split()
# one name per power of 1000: (form after 1, after 2-4, after 5-20 / 0)
_RU_SCALES = [("тысяча", "тысячи", "тысяч")] + [(w, w + "а", w + "ов") for w in
              ("миллион миллиард триллион квадриллион квинтиллион секстиллион септиллион октиллион "
               "нониллион").split()]


def _ru_form(n: int) -> int:
    if 11 <= n % 100 <= 14:
        return 2
    return 0 if n % 10 == 1 else 1 if 2 <= n % 10 <= 4 else 2


def _ru_below_1000(n: int, fem: bool) -> list[str]:
    h, r = divmod(n, 100)
    out = [_RU_HUNDREDS[h]] if h else []
    if r >= 20:
        out.append(_RU_TENS[r // 10])
        r %= 10
    if r or not out:
        w = _RU_ONES[r]
        if fem and r in (1, 2):
            w = "одна" if r == 1 else "две"
        out.append(w)
    return out


def _ru_limit(n: int):
    if n >= 10 ** 33:
        raise KeyError(11)


def _ru_cardinal(n: int) -> str:
    _ru_limit(n)
    if n == 0:
        return "ноль"
    g = _groups(n)
    out: list[str] = []
    for k in range(len(g) - 1, -1, -1):
        c = g[k]
        if c:
            out += _ru_below_1000(c, k == 1)
            if k:
                out.append(_RU_SCALES[k - 1][_ru_form(c)])
    return " ".join(out)


_RU_ORD_ONES = "нулевой первый второй третий четвёртый пятый шестой седьмой восьмой девятый".split()
_RU_ORD_TEENS = ("десятый одиннадцатый двенадцатый тринадцатый четырнадцатый пятнадцатый шестнадцатый "
                 "семнадцатый восемнадцатый девятнадцатый").split()
_RU_ORD_TENS = "_ _ двадцатый тридцатый сороковой пятидесятый шестидесятый семидесятый восьмидесятый девяностый".split()
_RU_ORD_HUNDREDS = ("_ сотый двухсотый трёхсотый четырёхсотый пятисотый шестисотый семисотый восьмисотый "
                    "девятисотый").split()
# genitive stems glued in front of a scale ordinal: двадцатиоднотысячный, стотысячный
_RU_GEN_ONES = "_ одно двух трёх четырёх пяти шести семи восьми девяти".split()
_RU_GEN_TEENS = ("десяти одиннадцати двенадцати тринадцати четырнадцати пятнадцати шестнадцати "
                 "семнадцати восемнадцати девятнадцати").split()
_RU_GEN_TENS = "_ _ двадцати тридцати сорока пятидесяти шестидесяти семидесяти восьмидесяти девяносто".split()
_RU_GEN_HUNDREDS = "_ сто двухсот трёхсот четырёхсот пятисот шестисот семисот восьмисот девятисот".split()


_RU_ORD_OF_WORD = dict(zip(_RU_ONES[10:], _RU_ORD_TEENS))
_RU_ORD_OF_WORD.update(zip(_RU_TENS[2:], _RU_ORD_TENS[2:]))
_RU_ORD_OF_WORD.update(zip(_RU_HUNDREDS[1:], _RU_ORD_HUNDREDS[1:]))


def _ru_ord_below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    if not r:
        return _RU_ORD_HUNDREDS[h]
    out = [_RU_HUNDREDS[h]] if h else []
    if r < 10:
        out.append(_RU_ORD_ONES[r])
    elif r < 20:
        out.append(_RU_ORD_TEENS[r - 10])
    elif r % 10 == 0:
        out.append(_RU_ORD_TENS[r // 10])
    else:
        out += [_RU_TENS[r // 10], _RU_ORD_ONES[r % 10]]
    return " ".join(out)


def _ru_genitive(n: int) -> str:
    if n == 1:
        return ""
    h, r = divmod(n, 100)
    s = _RU_GEN_HUNDREDS[h] if h else ""
    if 10 <= r < 20:
        return s + _RU_GEN_TEENS[r - 10]
    t, u = divmod(r, 10)
    return s + (_RU_GEN_TENS[t] if t else "") + (_RU_GEN_ONES[u] if u else "")


def _ru_ordinal(n: int) -> str:
    """Groups above the last one are cardinals (a count of one is only the scale noun: миллион
    тысячный; a count ending in a ten, teen or hundred ends in an ordinal: десятый тысяч первый);
    the last group is an ordinal, or a compound adjective when it counts a scale."""
    _ru_limit(n)
    if n == 0:
        return _RU_ORD_ONES[0]
    g = _groups(n)
    j = next(k for k in range(len(g)) if g[k])
    out: list[str] = []
    for k in range(len(g) - 1, j, -1):
        c = g[k]
        if c:
            if c == 1:
                out.append(_RU_SCALES[k - 1][0])
            else:
                words = _ru_below_1000(c, k == 1)
                # a count ending in a ten, a teen or a hundred has that word as an ordinal too:
                # десятый тысяч первый (10001), сто двадцатый миллионов первый
                words[-1] = _RU_ORD_OF_WORD.get(words[-1], words[-1])
                out += words
                out.append(_RU_SCALES[k - 1][_ru_form(c)])
    if j == 0:
        out.append(_ru_ord_below_1000(g[0]))
    else:
        stem = _RU_SCALES[j - 1][0]
        stem = stem[:-1] + "н" if stem.endswith("а") else stem + "н"
        out.append(_ru_genitive(g[j]) + stem + "ый")
    return " ".join(out)


def _ru_fem(words: str) -> str:
    return " ".join("одна" if w == "один" else "две" if w == "два" else w for w in words.split(" "))


_RU_FRAC_LOW = {1: "десят", 2: "сот"}
_RU_FRAC_PREFIX = ("", "десяти", "сто")


def _ru_float(value: float) -> str:
    """Decimal fraction read from the shortest repr of the float: 2.30 -> 2.3 'две целых три
    десятых'. A repr in exponent notation (below 1e-4, from 1e16) raises ValueError."""
    s = repr(value)
    ip, _, fp = s.partition(".")
    i, f = int(ip), int(fp)
    _ru_limit(i)
    _ru_limit(f)
    whole = _ru_fem(_ru_cardinal(i)) + " " + ("целая" if _ru_form(i) == 0 else "целых")
    d = len(fp)
    if d in _RU_FRAC_LOW:
        stem = _RU_FRAC_LOW[d]
    else:
        q, r = divmod(d, 3)
        scale = _RU_SCALES[q - 1][0]
        stem = _RU_FRAC_PREFIX[r] + (scale[:-1] + "н" if scale.endswith("а") else scale + "н")
    frac = _ru_fem(_ru_cardinal(f)) + " " + stem + ("ая" if _ru_form(f) == 0 else "ых")
    return whole + " " + frac


# =========================================================================== entry point

_FUNCS = {
    ("en", "cardinal"): _en_cardinal, ("en", "ordinal"): _en_ordinal, ("en", "year"): _en_year,
    ("de", "cardinal"): _de_cardinal, ("de", "ordinal"): _de_ordinal,
    ("fr", "cardinal"): _fr_cardinal, ("fr", "ordinal"): _fr_ordinal,
    ("fr_BE", "cardinal"): lambda n: _fr_cardinal(n, True), ("fr_BE", "ordinal"): lambda n: _fr_ordinal(n, True),
    ("es", "cardinal"): _es_cardinal, ("es", "ordinal"): _es_ordinal,
    ("pt_BR", "cardinal"): _pt_cardinal, ("pt_BR", "ordinal"): _pt_ordinal,
    ("it", "cardinal"): _it_cardinal, ("it", "ordinal"): _it_ordinal,
    ("ru", "cardinal"): _ru_cardinal, ("ru", "ordinal"): _ru_ordinal,
}


def num2words(value, lang: str, to: str = "cardinal") -> str:
    """`value` spelled out in `lang` ('en', 'de', 'fr', 'fr_BE', 'es', 'it', 'pt_BR', 'ru') as a
    'cardinal', 'ordinal' or (English only) 'year'. Russian cardinals also take a float."""
    f = _FUNCS.get((lang, to))
    if f is None:
        raise NotImplementedError(f"no {to!r} numbers for language {lang!r}")
    if lang == "ru" and to == "cardinal" and isinstance(value, float):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"unsupported value: {value!r}")
        return _ru_float(value)
    return f(_check_int(value))
