"""Write docs/phones.json and docs/PHONES.md: the canonical phone inventory, marks and normalisation
per language, i.e. what a client-side G2P must produce for pronunciada.core.score_phones.

    python scripts/export_phones.py            # rewrite both files
    python scripts/export_phones.py --check    # exit 1 if either file is out of date

Sources (nothing is measured on audio): the token file pronunciada/data/zipa_tokens.txt, the
normalisation tables of pronunciada/text/ipa.py, the inventory of the language front-ends on FLEURS
train prompts committed in pronunciada/data/coach/inventory.json, and the front-ends' output on the
worked-example sentences below (needs every front-end installed: requirements-text-*.txt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LANGS = ["en", "de", "es", "fr", "it", "pt", "ru", "ja", "ko", "zh"]
EXAMPLES = {
    "en": ["The river carries 8 times as much water.", "She thought about the weather."],
    "de": ["Der Zug fährt um acht Uhr nach Berlin.", "Wir möchten zwei Brötchen."],
    "es": ["Tengo 31 años y vivo en Madrid.", "El perro corre por la calle."],
    "fr": ["Le train part à huit heures.", "Les enfants ont un livre."],
    "it": ["Il treno parte alle otto per Roma.", "Abbiamo mangiato la pizza."],
    "pt": ["O trem parte às oito horas.", "Ela não quer café."],
    "ru": ["Поезд отправляется в восемь часов.", "Мы читаем книгу."],
    "ja": ["電車は八時に出発します。", "きっと大丈夫です。"],
    "ko": ["기차는 여덟 시에 출발합니다.", "책을 읽고 싶어요."],
    "zh": ["火车八点出发。", "我不知道。"],
}
# where each language's symbol choices are made (a client G2P must reproduce these files' rules)
FRONTEND = {
    "en": "pronunciada/text/en.py (English.rewrite: CMUdict ARPAbet -> ZIPA symbols, stress digits); "
          "OOV words: pronunciada/text/_eu_lts.py; shared EU pipeline pronunciada/text/_eu_base.py, _eu_common.py",
    "de": "pronunciada/text/de.py (German.rewrite: ipa-dict de -> ZIPA symbols, compound splitting, "
          "length from ː); English loanwords via en; shared EU pipeline _eu_base.py, _eu_common.py",
    "es": "pronunciada/text/es.py (rules(): orthographic rules; Spanish.rewrite; round-20 glide alternatives "
          "when es_glide is on); shared EU pipeline _eu_base.py, _eu_common.py",
    "fr": "pronunciada/text/fr.py (French: ipa-dict fr_FR, liaison and schwa as \"\" alternatives; no stress); "
          "shared EU pipeline _eu_base.py, _eu_common.py",
    "it": "pronunciada/text/it.py (Italian.rewrite: WikiPron it, stress_from_end, doubled consonants as length); "
          "shared EU pipeline _eu_base.py, _eu_common.py",
    "pt": "pronunciada/text/pt.py (rules(): Brazilian rules + WikiPron corrections); shared EU pipeline",
    "ru": "pronunciada/text/ru.py (rules(): palatalisation, reduction, voicing by stress position; "
          "Russian.rewrite); shared EU pipeline",
    "ja": "pronunciada/text/ja.py (JaG2P: Open JTalk readings -> moras; one syllable per mora, っ merged "
          "into the next mora as a doubled consonant, long vowels as Vː)",
    "ko": "pronunciada/text/ko.py (KoG2P: standard pronunciation rules from Hangul; tense flag on "
          "RefPhone.tense, the symbol stays plain); Hangul arithmetic pronunciada/hangul.py",
    "zh": "pronunciada/text/zh.py (ZhG2P: OpenCC -> jieba -> pypinyin -> sandhi; one syllable per hanzi; "
          "tones on the syllable)",
}


def _marks(refs) -> dict:
    stress, tones, length, tense, alts, drop = set(), set(), set(), set(), 0, 0
    nuc_multi = 0
    for r in refs:
        for s in r.syllables:
            if s.stress is not None:
                stress.add(int(s.stress))
            if s.tone is not None:
                tones.add(int(s.tone))
        for p in r.phones:
            if p.length is not None:
                length.add(p.length)
            if p.tense is not None:
                tense.add(bool(p.tense))
            alts += bool(p.alts)
            drop += any(a == "" for a, _ in p.alts)
            nuc_multi += p.is_nucleus and len(p.ipa) > 1
    return {"stress_values": sorted(stress), "tone_values": sorted(tones), "length_values": sorted(length),
            "tense_values": sorted(tense), "phones_with_alternatives": alts, "phones_droppable": drop}


def _word_line(w: dict) -> str:
    syls = []
    for s in w["syllables"]:
        ph = " ".join(f"({p['ipa']})" if p["nucleus"] else p["ipa"] for p in s["phones"])
        tag = []
        if s.get("stress") is not None:
            tag.append(f"s{s['stress']}")
        if s.get("tone") is not None:
            tag.append(f"t{s['tone']}")
        for p in s["phones"]:
            if p.get("length"):
                tag.append(f"{p['ipa']}:{p['length']}")
            if p.get("tense") is not None:
                tag.append(f"{p['ipa']}:{'tense' if p['tense'] else 'lax'}")
        syls.append(ph + (f" /{','.join(tag)}" if tag else ""))
    return " . ".join(syls)


def build() -> tuple[dict, str]:
    from pronunciada.core import phones_for, reference_from_target
    from pronunciada.text import ipa

    toks = list(ipa.load_tokens())
    inv = json.loads((ROOT / "pronunciada/data/coach/inventory.json").read_text(encoding="utf-8"))["phones"]
    data = {"format": "pronunciada-phones", "version": 1,
            "tokens": toks, "special_tokens": list(ipa.SPECIAL),
            "modifiers": sorted(ipa.MODIFIERS), "vowel_letters": sorted(ipa.VOWELS),
            "fold": dict(sorted(ipa.FOLD.items())), "languages": {}}
    for lang in LANGS:
        tg = [phones_for(t, lang) for t in EXAMPLES[lang]]
        refs = [reference_from_target(t, lang) for t in tg]
        data["languages"][lang] = {
            "inventory": sorted(inv[lang]), "inventory_source": "pronunciada/data/coach/inventory.json",
            "marks": _marks(refs), "frontend": FRONTEND[lang],
            "examples": [{"text": EXAMPLES[lang][i], "words": [{"text": w["text"], "phones": _word_line(w),
                                                                  "alternatives": len(w["alternatives"])}
                                                                 for w in t["words"]]}
                         for i, t in enumerate(tg)]}
    return data, _markdown(data)


def _markdown(d: dict) -> str:
    L = ["# Phones: what a client-side G2P must produce",
         "",
         "Generated by `scripts/export_phones.py` from the token file, `pronunciada/text/ipa.py`,",
         "`pronunciada/data/coach/inventory.json` and the front-ends' output on the example sentences;",
         "machine-readable copy: `docs/phones.json`. Do not edit by hand. The target format itself is",
         "described in `pronunciada/core.py`.",
         "",
         "For Japanese and Mandarin a client does not need its own G2P: `pronunciada.phones_from_kana`",
         "and `pronunciada.phones_from_pinyin` (layer 1, rules only, no dictionary) turn a kana or pinyin",
         "reading into exactly this target; docs/TEXT_LAYERS.md gives the input formats and how closely",
         "they match the dictionary front-ends.",
         "",
         "## Symbols",
         "",
         f"The backbone emits {len(d['tokens'])} tokens (`pronunciada/data/zipa_tokens.txt`, `<symbol> <id>`",
         "per line). Ids 0-3 are not sounds: " + ", ".join(f"`{t}`" for t in d["special_tokens"]) + ".",
         "Every other token is one Unicode code point. A target phone (`ipa`) is a string of one or more",
         "of these code points: a base symbol followed by its modifiers (for example `tʰ`, `aː`, `tʃ`,",
         "`aʊ`, `ɜ˞`); affricates and diphthongs are written as two base symbols when the language's",
         "front-end treats them as one sound. `Reference.validate` (`pronunciada/types.py`) rejects a phone",
         "with a code point outside the token list or starting with a modifier.",
         "",
         "Modifiers (attach to the preceding base symbol, `pronunciada.text.ipa.MODIFIERS`): "
         + " ".join(f"`{m}` (U+{ord(m):04X})" for m in d["modifiers"]) + ".",
         "",
         "Vowel letters (a phone whose first code point is one of these is a vowel,",
         "`pronunciada.text.ipa.is_vowel`; read by the length and word-step channels): "
         + " ".join(d["vowel_letters"]) + ".",
         "",
         "## Normalisation a client G2P must apply",
         "",
         "1. Unicode NFD, then put `ç` back together (`pronunciada.text.ipa.normalize`).",
         "2. Apply the universal fold table code point by code point (`pronunciada.text.ipa.FOLD`);",
         "   an empty value deletes the code point (stress marks, tie bars, syllable dots, tone letters,",
         "   diacritics ZIPA does not use). Code points outside the token list are dropped.",
         "3. Split into phones with `pronunciada.text.ipa.segment` (base + trailing modifiers), then merge",
         "   units exactly as the language's front-end does (file references per language below).",
         "4. Stress is not a phone mark: it is the syllable's `stress` (1 primary, 2 secondary,",
         "   0 unstressed, null = the language has no word stress). Tone (Mandarin) is the syllable's",
         "   `tone` after sandhi (1-4, 5 neutral) with `tone_lexical` and `tone_alts`. Length is `ː`",
         "   inside the symbol plus the phone's `length` (\"long\" / \"short\") where the prompt fixes it.",
         "   Korean tense obstruents keep the plain symbol and set `tense`.",
         "5. Syllables: exactly one phone per syllable has `nucleus: true`; words are lists of syllables.",
         "",
         "Fold table (`from` -> `to`, empty = deleted):",
         "",
         "| from | to |", "|---|---|"]
    for k, v in d["fold"].items():
        k_ = k.replace("|", "\\|")
        L.append(f"| `{k_}` (" + " ".join(f"U+{ord(c):04X}" for c in k) + f") | `{v}` |" if v else
                 f"| `{k_}` (" + " ".join(f"U+{ord(c):04X}" for c in k) + ") | (deleted) |")
    L += ["", "## Which target fields the scorer reads", "",
          "| field | read by | languages |", "|---|---|---|",
          "| phone `ipa` | alignment lattice, features, native statistics, per-sound heads (classes keyed by symbol) "
          "| all |",
          "| phone `alts` | alignment lattice (acceptable variants; empty string = may be dropped), identity classes "
          "| all |",
          "| phone `nucleus` | features, length, delivery, prosody windows, identity classes | all |",
          "| phone `tense` | contrast heads, Korean identity classes | ko |",
          "| phone `length` | length channel (else derived from `ː` in the symbol) | ja, it, de |",
          "| syllable `stress` | features (`stress_f`), stress channel, German length rule and identity "
          "| en de es it pt ru |",
          "| syllable `tone`, `tone_lexical`, `tone_alts` | tone channel and its leniency rules | zh |",
          "| syllable `text` | 一 / 不 tone rule (zh); spelled vowel and 의 (ko) | zh, ko |",
          "| word spans, word `text` | word / sentence pooling; `word_text` feature column | all |",
          "| word `char_span` + prompt `text` | word adjacency in the length channel | ja, it, de |",
          "| `warnings` entries `oov:<word>` | German word recalibration (`pronunciada/word_lang.py`) | de |",
          "| word `alternatives` | round-20 word-level second pass (`pronunciada/r20.py`) | es (on by default) |",
          "", "## Per language", "",
          "Inventory = the canonical phones the front-end produced on FLEURS train prompts. Marks = the",
          "values seen on the example sentences. Examples: syllables separated by ` . `, the nucleus in",
          "parentheses, `/s1` stress, `/t3` tone, `x:long` length, `x:tense` Korean tense flag;",
          "`alternatives` = number of round-20 candidate readings carried by the word.", ""]
    for lang, x in d["languages"].items():
        m = x["marks"]
        L += [f"### {lang}", "",
              f"Front-end: {x['frontend']}.", "",
              f"Inventory ({len(x['inventory'])}): " + " ".join(x["inventory"]), "",
              f"Marks: stress {m['stress_values'] or 'none'}; tone {m['tone_values'] or 'none'}; "
              f"length {m['length_values'] or 'none'}; tense {m['tense_values'] or 'none'}; "
              f"phones with alternatives {m['phones_with_alternatives']}, of which may be dropped "
              f"{m['phones_droppable']}.", ""]
        for ex in x["examples"]:
            L += [f"Example: {ex['text']}", "", "| word | phones | alternatives |", "|---|---|---|"]
            for w in ex["words"]:
                L.append(f"| {w['text']} | {w['phones']} | {w['alternatives']} |")
            L.append("")
    return "\n".join(L).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    data, md = build()
    js = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    pj, pm = ROOT / "docs/phones.json", ROOT / "docs/PHONES.md"
    if a.check:
        ok = pj.exists() and pj.read_text(encoding="utf-8") == js and pm.exists() and pm.read_text(encoding="utf-8") == md
        print("up to date" if ok else "out of date: run python scripts/export_phones.py")
        return 0 if ok else 1
    pj.write_text(js, encoding="utf-8")
    pm.write_text(md, encoding="utf-8")
    print(f"wrote {pj.relative_to(ROOT)} and {pm.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
