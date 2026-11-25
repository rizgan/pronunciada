# Text in two layers

Scoring needs the target sounds of the prompt. Pronunciada gets them in two layers:

- **Layer 1, the core, always installed.** No dictionary, no heavy or copyleft package, nothing
  that cannot be ported to a phone or a browser. It scores from target phones
  (`pronunciada.core.score_phones`) and builds those targets from a *reading* by rules:
  `pronunciada.phones_from_kana` (Japanese kana), `pronunciada.phones_from_pinyin` (Mandarin
  pinyin), plus our own number speller (`pronunciada/text/numbers.py`) and the Korean rules.
- **Layer 2, optional and explicit.** Turning raw text into a reading needs dictionaries: Open JTalk
  for Japanese kanji, jieba / pypinyin / OpenCC for Chinese characters, the bundled pronunciation
  lexicons for the European languages. The packages are extras (`pip install "pronunciada[ja]"`,
  `"pronunciada[zh]"`), imported only when a call needs them and never by layer 1.

`Scorer(lang).score(audio, text)` works exactly as before when the extras are installed (that is
what the benchmarks and tests use). Without the extra, a `Scorer` still scores a Reference or a
target, and text raises an `ImportError` that names the extra and the layer-1 alternatives:

    the Japanese text front-end needs the optional "ja" extra: pip install "pronunciada[ja]"; without
    it, pass a kana reading (pronunciada.phones_from_kana) or phones (pronunciada.core.score_phones)

## What each language needs

Installed sizes measured in fresh virtual environments (`uv venv`, Python 3.12, macOS arm64,
`du -sk` of site-packages, 2026-11-23). The core includes NumPy, ONNX Runtime,
kaldi-native-fbank, soundfile and the pronunciada package itself (36.9 MB, of which 10 MB are the
European lexicons); the acoustic model is downloaded separately at first use.

| install | site-packages | added by the extra |
|---|---|---|
| `pip install pronunciada` (core) | 149,484 KiB (153 MB) | |
| core + `[zh]` | 191,920 KiB | +42,436 KiB (jieba 37.5 MB, pypinyin 3.6 MB, opencc 1.2 MB) |
| core + `[ja]` | 473,628 KiB | +324,144 KiB |
| core + `[ja]`, then `pip uninstall sudachipy sudachidict-core` | 271,292 KiB | +121,808 KiB |
| core + `[all]` (zh, ja, mcp) | 535,984 KiB | |
| (former `[numbers]`: num2words 0.5.14 + docopt) | | 752 KiB, no longer used |

| language | layer 1 (no extra) | layer 2 |
|---|---|---|
| ja | kana reading -> target (`phones_from_kana`); target phones | kanji / mixed text: `[ja]` (pyopenjtalk-plus, Open JTalk + its 103 MB dictionary) |
| zh | pinyin -> target (`phones_from_pinyin`); target phones | Chinese characters: `[zh]` (jieba, pypinyin, OpenCC) |
| ko | Hangul text (in-house rules, no data file), numbers; target phones | none (Hanja is skipped with a warning) |
| en de es fr it pt ru | target phones; numbers spelled by our own rules | text: the bundled lexicons (`pronunciada/data/lex/`, 10 MB inside the core package, read only by the text path); no extra to install |

The European lexicons stay inside the core package: they are plain data files, not a package, and
are opened only by the text path of their own language (tests/test_text_layers.py). Moving them
into separate distributions is possible later; it would need extra packages on PyPI.

### The second Japanese dictionary

`pyopenjtalk-plus` declares `sudachipy` and `sudachidict-core` (about 200 MB, Apache-2.0) as hard
requirements and imports `sudachipy` at module level. Pronunciada never uses them: the front-end
switches off the Sudachi reading correction and keeps the unknown-kanji step on its Unihan
readings, so the Sudachi dictionary is never loaded (tests/test_ja_no_sudachi.py, which also
checks that the targets are identical with both packages blocked). Packaging alone cannot drop
them: pip has no way for an extra to exclude a dependency's requirement, the original
`pyopenjtalk` publishes only a source archive (needs a compiler and downloads its dictionary at
first use), and `pyopenjtalk-prebuilt` stops at Python 3.11 and an older API. After installing the
`ja` extra, `pip uninstall sudachipy sudachidict-core` frees 202 MB and changes nothing.
Measured effect of not using Sudachi (5,076 practice sentences): 25 of 75,334 words read
differently (0.033 %); see docs/text_cjk_notes.md.

## Input formats

### Target phones (all languages)

The JSON target of `pronunciada.core` (format and fields: `pronunciada/core.py`, docs/PHONES.md).
`phones_for(text, lang)` produces it with layer 2; `score_phones(samples, 16000, target, lang)`
scores it.

### Kana (Japanese): `phones_from_kana(reading, text=None, *, style="pron", devoicing="auto")`

    phones_from_kana("キョー|ワ イー テンキ デス’。")
    phones_from_kana("きょう|は いい てんき です。", style="hiragana")
    phones_from_kana([{"kana": "トーキョー", "text": "東京"}, {"kana": "ニ", "text": "に"}], text="東京に")

- Words are separated by spaces; `|` separates morphemes inside a word (optional; it matters for
  long vowels across morphemes and for the word-medial g ~ ŋ alternative); punctuation between
  words is a pause. A word dict may carry `text` (shown), `char_span` and `read` (orthographic kana
  per morpheme).
- `style="pron"`: pronunciation kana as Open JTalk writes it (`pron`): ー for long vowels,
  particles as said (ワ, エ, ヲ), `’` after a devoiced mora.
- `style="hiragana"`: textbook orthography, turned into pronunciation kana by rule: a word or
  morpheme that is exactly は / へ is the particle ワ / エ; ぢ / づ -> ジ / ズ; inside a morpheme,
  o-row + う / お, e-row + い and a repeated vowel become a long vowel. Cannot be decided without a
  dictionary, so mark them: は / へ inside a word used as particles (こんにちは, または: write わ / え or
  put `|` before), vowel sequences across a morpheme boundary (おも|う, おおき|い: put `|`), number
  sound changes (いっ, ぴゃく: write them as said).
- `devoicing`: `"marks"` uses the `’` marks, `"rule"` a rule (i / u between voiceless consonants,
  final す of です / ます), `"none"` neither, `"auto"` marks if the input has any, else the rule.
  Devoicing only sets the prior of an optional vowel drop (-0.2 instead of -0.7).

### Pinyin (Mandarin): `phones_from_pinyin(reading, text=None, *, sandhi=True)`

    phones_from_pinyin("huo3che1 ba1dian3 chu1fa1", text="火车八点出发。")
    phones_from_pinyin("nǐ hǎo")
    phones_from_pinyin([{"pinyin": "xiao3|lao3hu3", "text": "小老虎"}])

- Words are separated by spaces (tone sandhi groups and word scores follow them); syllables inside
  a word are written together, optionally separated by `'` or `-`; punctuation is a phrase break
  (sandhi and sound assimilation stop there, as in the character path).
- Tones: digits 1-4 or tone marks; 5, 0 or none = neutral. `·` before a syllable = said neutral with
  the written tone as its lexical tone (`zhi1·dao4`); `=` before a syllable = tone fixed as written
  (no sandhi, for lexicalised patterns such as 怎么样 `=zen3'me5'=yang4`). `|` inside a word marks
  an inner group (`xiao3|lao3hu3` = A|BC; without it a run of three third tones inside a word is
  grouped AB|C). Erhua: `nar3` / `na3r`, or the syllable `er` under the character 儿. `v` = ü.
- Characters (optional, `text` aligned one Han character per syllable, or `chars` per word) turn on
  the character rules: 一 / 不 sandhi, 过 as a particle, reduplicated verbs, Taiwan tone variants
  (a 25-entry in-house table, `pronunciada/data/cjk/zh_tw_tones.txt`). Traditional characters are
  shown as given; converting them is layer 2.
- `sandhi=True`: the tones are citation tones and the path computes the surface tones with the
  character path's rules. `sandhi=False`: the tones are already the surface tones and are used as
  given (lexical tone = written tone, no tone alternatives).

## Parity with the dictionary paths (practice data only)

Each dictionary path's own reading was fed into the layer-1 path and the targets compared.

Japanese, 5,076 sentences (FLEURS ja train 805 + validation 135, Common Voice ja dev-half 4,136),
74,250 words:

| kana input | sentences exactly equal | words exactly equal | what differs |
|---|---|---|---|
| pronunciation kana with `’` marks (`devoicing="marks"`) | 5,067 (99.82 %) | 74,236 (99.98 %) | 6 Open JTalk label / kana mismatches (the dictionary path then falls back to rough kana), 7 ai/ae alternative after a skipped Latin run, 1 pause inside one written token |
| pronunciation kana without marks (rule) | 4,045 (79.7 %) | 72,875 (98.1 %) | devoicing prior only (rule agrees with Open JTalk on 99.16 % of moras); canonical phones equal in 5,073 sentences |
| textbook hiragana of Open JTalk's `read`, morphemes marked | 3,311 (65.2 %) | 66,761 (89.9 %) | canonical phones equal in 4,546 sentences (89.6 %); devoicing 1,281 words, canonical phones 407 (number sound changes, lexical は = wa, くけ -> っけ), word split 234 (でしょ + う written as two words), syllable text 530 |

Scores on the 20 Japanese practice clips (fp32, every number of the result compared): with marks
20 / 20 clips identical (29,231 values, max |difference| 0); without marks 12 / 20 identical, max
0.486 on a per-sound feature, 0.125 on a word probability; hiragana 9 / 20 identical.

Mandarin, 6,877 sentences (FLEURS zh train 1,416 + validation 150, Common Voice zh-CN dev-half
5,311), 137,129 syllables:

| pinyin input | phones | tone | tone_lexical | tone_alts | sentences exactly equal |
|---|---|---|---|---|---|
| citation tones, characters, `|` grouping, `=` fixed | 100 % | 100 % | 100 % | 100 % | 6,852 (99.6 %; the other 25 differ only in a shown character: the feed passed the simplified form) |
| same without `|` / `=` | 100 % | 137,114 | 100 % | 137,114 | 6,837 |
| citation tones, no characters | 100 % | 136,926 (一 / 不 need the character) | 100 % | 136,973 | 6,518 |
| surface tones, `sandhi=False` | 100 % | 100 % | 131,670 | 133,741 | 3,072 |

Scores on the 20 Mandarin practice clips: full input 20 / 20 identical (24,516 values, max 0); no
characters 17 / 20 (max 0.133 on a tone probability); surface tones 11 / 20 (max 0.883 on a tone
probability, 0.049 on a word probability).

Isolation (fresh processes, tests/test_reading_ja.py, test_reading_zh.py, test_text_layers.py):
importing `pronunciada` and calling `phones_from_kana`, `phones_from_pinyin` or `score_phones`
loads no layer-2 package and opens no lexicon (pinyin opens only the 25-line Taiwan tone table when
characters are given); each language's text path loads only its own packages and lexicon files.

## Numbers

Digits in European prompts are spelled by `pronunciada/text/numbers.py` (cardinals, ordinals,
English years, Russian decimals; French Belgian forms for the round-20 alternatives), written from
the grammar of each language. Its strings are identical to those of the num2words package the
benchmarks were run with: 0 differences on all integers 0-9,999, a dense sample to 10^18
(about 240,000 values per language and form), 222,490 Russian decimals, every number token of the
practice text (300 en, 311 de, 288 es, 304 fr, 295 it, 303 pt, 287 ru tokens, plus the round-20
alternatives) and the 1,857 number tokens of the held-out prompts (expansions compared, no audio
read). With every change of this split after the Sudachi removal, the scores of the practice
clips are exactly identical (bench/opt/parity.py, 200 clips of ten languages, fp32 and int8:
max |difference| 0 on every score and posterior). Where the old package is not what the grammar says (for example es "veintiuno mil",
de "einhunderteinstausend"), the old output is kept so scores stay identical; the list is
`KNOWN_DIFFERENCES_FROM_GRAMMAR` in that module. Korean (`text/ko.py`), Mandarin (`text/zh.py`)
and Japanese (Open JTalk) never used num2words.

For a user who did not have num2words installed before, digits used to be dropped from the target
with a `no_num2words:<token>` warning; they are now read out. That changes those users' scores for
prompts with digits, as intended.

## Not possible without layer 2

- Japanese text with kanji (or any text whose reading is not given): the reading, pitch-accent
  phrases, devoicing marks and the morpheme boundaries come from Open JTalk.
- Chinese characters: word segmentation, polyphone readings, the neutral-tone word list, the
  word-frequency choice of the three-syllable sandhi structure and Traditional -> Simplified conversion.
- European text: the lexicon readings and stress (the letter-to-sound fallbacks are trained on the
  lexicons too).
- Japanese and Chinese numbers written in digits: read by Open JTalk / the Chinese reading rules of
  the text path; in kana or pinyin input, write numbers as they are said.
