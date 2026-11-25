# CJK text front-ends (zh, ja, ko): notes, provenance, known gaps

Modules: `pronunciada/text/zh.py`, `ja.py`, `ko.py`, shared helpers `pronunciada/text/_cjk_common.py`,
hand-written tables in `pronunciada/data/cjk/`. Tests: `tests/test_text_cjk_{zh,ja,ko}.py` (offline).
Runtime packages: `requirements/text-cjk.txt`.

## Provenance and licenses

| component | used for | license |
|---|---|---|
| pypinyin 0.55.0 (code + bundled pinyin/phrase data) | phrase-aware readings, polyphones | MIT |
| jieba 0.42.1 (code + `dict.txt`) | Mandarin word segmentation, word-frequency for 3-3-3 sandhi structure | MIT |
| opencc-python-reimplemented 0.1.7 (+ OpenCC dictionaries) | Traditional -> Simplified normalization | Apache-2.0 |
| pyopenjtalk-plus 0.4.1.post10 | Japanese readings, accent, devoicing (`run_frontend`, `make_label`) | MIT |
| ├ Open JTalk (C library, bundled) | morphological analysis / NJD / accent phrases | Modified BSD |
| ├ dictionary (NAIST-jdic + UniDic-derived entries) | lexicon + accent types | BSD-3 (NAIST); UniDic under its BSD-3 option |
| ├ Unihan kJapanese map, CJKVI variants map | unknown-kanji fallbacks | Unicode-3.0, MIT |
| ├ HTS voice "Mei" (shipped in the wheel, **not used**) | TTS only | CC-BY-3.0 |
| sudachipy 0.7.0 + sudachidict-core (dependency of pyopenjtalk-plus) | **not used**: installed by pyopenjtalk-plus, never loaded (may be uninstalled) | Apache-2.0 |
| pydantic / pydantic-core / annotated-types / typing-inspection | dependencies of pyopenjtalk-plus | MIT |
| `data/cjk/zh_neutral.txt`, `data/cjk/zh_tw_tones.txt` | neutral-tone words, Taiwan tone variants | written for pronunciada (Apache-2.0 with the project) |
| Korean rules, number readings, all ZIPA mapping tables | in-house code | project license |

No GPL code or data is imported at runtime. Original `pyopenjtalk` has no macOS-arm64/py3.12 wheel
(source build needs a compiler; it failed here), so `pyopenjtalk-plus` (same API, prebuilt wheels,
bundled dictionary) is used. Its own footprint is ~112 MB (103 MB Open JTalk dictionary); the
~200 MB of sudachipy + sudachidict-core it pulls in are unused (below) and can be uninstalled.

Two layers (docs/TEXT_LAYERS.md): everything after the reading (moras -> phones for Japanese, tone
rules and pinyin -> phones for Mandarin, all of Korean) is layer 1, rules only, in
`pronunciada/reading/` and `pronunciada/text/ko.py`; the packages above are layer 2 and only find the
reading. `pronunciada.phones_from_kana` / `phones_from_pinyin` take a reading directly.
Data for tuning: FLEURS validation split (CC-BY-4.0), dev only.

## Method

Conventions were tuned against ZIPA (`zipa-small-crctc-500k`) greedy output on FLEURS validation
(native speech), per-phone confusion counts in context. Token error rate (`scripts/g2p_eval.py`, n=150):

| lang | first version | final | notes |
|---|---|---|---|
| zh | 0.088 | **0.073** | remaining: aspiration loss, -n/-ŋ merger, model noise |
| ja | 0.293 (n=266) | **0.297** (n=150; 0.289 on all 266) | v1 already used ZIPA-style ʃ tʃ dʒ, Vː, doubled geminates; rest is model noise, u/i deletion |
| ko | 0.467 (n=226) | **0.363** (n=150; 0.367 on all 226) | ZIPA's Korean output is noisy (vowels, aspiration, h) |

`g2p_eval.py` only scores canonical phones; alternatives do not change these numbers.

## Mandarin

- Layer 1 / layer 2 split: jieba (words), pypinyin (readings, polyphones), OpenCC (script),
  the neutral-tone word list and the 子 / 们 / 得 / 地 neutral rules find the reading (layer 2,
  `text/zh.py`); 一 / 不 sandhi, third-tone sandhi, the reduplication and Taiwan tone alternatives,
  erhua and pinyin -> phones are layer 1 (`reading/zh.py`), shared with `phones_from_pinyin`.
- Script: the whole prompt is converted with OpenCC `t2s` (length-preserving; per-character
  fallback), then everything runs on Simplified. `text` fields keep the user's script; number
  readings are shown in the user's script. Taiwan-standard 著 (= 着) is mapped back to 着 except
  inside 著-words (著名, 显著, 土著 ...). Mismatch Simplified vs Traditional on the 150 unique FLEURS
  validation sentences: s2t 0/150, s2tw 0/150 (8/150 before the 著 fix), s2hk 0/150; plus 0/75 on
  25 hand-written polyphone-heavy sentences x 3 configs. Residual risk: 看著書 vs 著書 ambiguity
  (resolved towards 着).
- Units: word = jieba word (with neutral-tone words registered); syllable = one hanzi; erhua 儿
  (word-final, not 女儿/婴儿 ...) merges into the previous syllable (`text` "哪儿", extra phone ɻ,
  alts ɑɻ / ə˞; n/j coda may drop). Digits -> Chinese reading (years and codes digit by digit,
  两 before measure words, decimals, %, thousands separators); Latin -> warning, skipped.
- Phones (ZIPA conventions found): zh/ch/sh/r = ʈʂ ʈʂʰ ʂ ʐ; apical vowel ɨ after retroflexes, **ɯ**
  after z/c/s (ZIPA 70-90 %); ü = y, üe/üan = ɥœ(n); you/yong = **y**oʊ / **y**ʊŋ (95 %); ao ɑʊ,
  ou oʊ, ai aɪ, ei eɪ, ang ɑŋ, ong ʊŋ, eng ɤŋ (əŋ after sh/r/h), er ɑɻ, bo/po/mo/fo = pwɔ ....
  Diphthongs are one nucleus phone; glide and coda are separate phones.
- Tones: `tone_lexical` = citation tone of the reading (for lexically neutral syllables the
  citation tone of that syllable). `tone` = surface: 一/不 sandhi (numbers, ordinals, word-final 一
  excluded; A一A, A不A, V不C neutral), 3-3 sandhi on chains with jieba word structure deciding
  grouping ([AB]C 2-2-3, A[BC] 3-2-3 with alt 2; across word boundaries 2 with alt 3), neutral tone
  from pypinyin + a hand list (276 words, 子/们/particles, V得C, adverbial 地 after reduplication).
  Taiwan: every lexical (non-particle) neutral tone gets the full tone as `tone_alts`; ~25
  characters with Taiwan tone differences (期 qí, 危 wéi, 质 zhí, 识 shì, 息 xí ...) likewise.
  Reliability: not measured against gold tone labels (none available offline); polyphone accuracy
  is that of pypinyin's phrase dictionary; neutral-tone coverage is the weakest part.
- Alternatives: coda n -> m / ŋ before labials / velars, nasal coda dropped (nasalised vowel),
  -in -> -ing merger, neutral-tone vowel -> ə, apical vowel dropped (是 -> ʂ), j after
  palatals dropped (千 tɕʰɛn), r ~ ɻ, x ~ h, eng ɤ ~ ə, er ~ ɑ.
- Taiwan segmental readings that cannot be expressed as same-slot alternatives (left Mainland):
  和 hé/hàn, 垃圾 lājī/lèsè, 液 yè/yì, 携 xié/xī, 暂 zàn/zhàn, 血 xuè/xiě, 熟 shú/shóu,
  头发 fa (tone handled), 企/期/危 (tone handled). De-retroflexion (zh->z) common in Taiwan is
  intentionally **not** an alternative (it is also a typical learner error).

## Japanese

- pyopenjtalk-plus `run_frontend` + `make_label`; per-mora consonant/vowel, devoicing (capital
  vowels) from the labels. Every `run_frontend` option is written out (`FRONTEND_OPTIONS` in
  `text/ja.py`): the package defaults (the "nani" reading model and the rule-based post-processing
  stay on) except `use_sudachi_kanji_yomi=False`. The unknown-kanji step, which would tokenise the
  sentence with Sudachi for a kanji missing from Open JTalk's dictionary, gets no sentence text during
  our call and uses its Unihan per-character readings. So the Sudachi dictionary is never loaded, and
  the front-end also works with sudachipy and sudachidict-core uninstalled (it then registers
  placeholder modules for the duration of `import pyopenjtalk` only). Effect on practice text
  (5,076 sentences: FLEURS train + validation, Common Voice dev half): 25 of 75,334 words change
  reading (0.033 %, 25 sentences = 0.49 %); by our reading of the 25, Sudachi was right in 6, Open
  JTalk in 13, 5 are the same sound written イ / ウ instead of ー, 1 unclear. Full quality gate:
  pass, only Japanese rows moved (largest +0.009 / -0.0002).
- Kana input (layer 1, `pronunciada/reading/ja.py`): the same moras -> phones builder on a kana
  reading; Open JTalk's devoicing marks (’) in its `pron` agree with its labels on every mora of the
  practice text (7,336 devoiced, 0 disagreements), so pronunciation kana with marks reproduces the
  dictionary path exactly (99.98 % of words; the rest: label / kana mismatches inside Open JTalk,
  Latin runs, a pause inside one written token). Numbers in kana input must be written in kana. (Until round 15 a per-mora high/low pitch label was also derived from
  the accent fields; pitch accent is no longer scored and the label was removed.)
- Units: mora, except ZIPA conventions force two merges: っ joins the **following** mora (doubled
  consonant kk / tt / ss / ʃʃ / ttʃ / tts; alt single consonant) - this can put the geminate into
  the next word (待っ|て -> て has "ッテ"); ー joins the **preceding** mora (nucleus Vː; alts short
  V, spelling forms ou / ei). Phrase-final っ -> ʔ (may drop). ん is its own unit,
  nucleus = place-assimilated nasal (m / n / ŋ; ŋ before vowels and at the end), alts n m ŋ ɴ.
- ZIPA conventions found: し ʃ, ち tʃ, じ dʒ, つ ts, ひ **h** (ZIPA never writes ç), ふ **f**
  (never ɸ), ら-row r, long vowels Vː, いく after a vowel often j.
- Alternatives: devoiced i/u may be dropped (labeled -0.2, environment -0.7), g ~ ŋ word-medially,
  r ~ ɾ / l / d, z ~ dz, w ~ v (ZIPA hears は/わ as va 29 %), を ~ wo, ai/ae -> aj, ou/ei spelling.
- Word `text` = original spans; numbers read by Open JTalk (7,000 -> ナナセン) keep the digit span
  as one word. Latin runs are cut out before analysis (warning).
- Open JTalk prints C-level warnings to stderr for odd input (e.g. text starting with ー).

## Korean

- In-house: Hangul decomposition, then rules on (coda, next onset) pairs: ㅎ aspiration / ㅎ
  deletion, liaison (clusters split, ㅅ tensed: 값이 갑씨), palatalisation (같이, 굳이, 닫히다),
  coda neutralisation (ㄺ -> ㄹ before ㄱ, 밟 exception), ㄹ rules (신라 실라, 설날 설랄, 종로 종노,
  국립 궁닙), nasalisation, tensification after obstruents and ㄵ/ㄻ/ㄼ/ㄾ stems.
  Within an eojeol these are canonical; across eojeol boundaries inside a phrase they are
  alternatives (liaison with voiced/flapped coda, nasalisation, tensification, aspiration,
  ㄴ-insertion 할 일 -> ll).
- ZIPA conventions found: tense consonants are written as plain stops (ㄲ k 63 %, ㄸ t 72 %, ㅃ p
  79 %, ㅆ s 70 %) - so tensification is mostly invisible acoustically to this model; ㄹ onset r;
  ㅓ ɔ; ㅐ = ㅔ = e; ㅗ o (u before ㅇ coda: 공 kuŋ 67-84 %); ㅡ is never ɯ in ZIPA output: u after
  labials, ə after velars/ㅅ, e elsewhere (alts ɯ ə u ɨ e); ㅂ between voiced sounds b; 시 ʃi.
- Alternatives: voiced/voiceless lax stops, ㅈ ~ dʒ / ts, h-weakening, word-initial ㅁ/ㄴ
  denasalised to b/d (ZIPA b 55 %), weak aspiration, ㅐ/ㅔ, ㅢ (ɯi / i / e for genitive 의),
  ㄴ dropped after ㅁ (습니다), ㅓ ~ ʌ / o, ㅗ ~ u.
- Not handled (needs morphology): ㄴ-insertion inside compounds (솜이불), tensification after
  verb-stem ㄴ/ㅁ (신고 [신꼬]) and Sino-Korean ㄹ+ㄷㅅㅈ (갈등), 맛없다-type liaison inside a word,
  native-Korean numbers with counters (3개 -> 삼개, should be 세 개; Sino-Korean is the default by
  spec). Hanja in Korean text is skipped with a warning.

## Environment note

`scripts/g2p_eval.py` fails in this venv with `NotADirectoryError` from `importlib.resources` in
`pronunciada/text/ipa.py`: the editable install was made while `pronunciada/` had no `__init__.py`, so the
editable finder registers `pronunciada` as a namespace package. Run it with `PYTHONPATH=.` or
reinstall (`uv pip install -e .`). The CJK modules read their tables via `__file__` paths and are
unaffected.
