# European text front-ends (en de fr es it pt ru)

Code: `pronunciada/text/{en,de,fr,es,it,pt,ru}.py` on top of shared helpers
`_eu_common.py` (tokenizer, numbers, syllabifier, alternative alignment, `Reference` assembly),
`_eu_base.py` (the common G2P driver), `_eu_lts.py` (learned letter-to-sound fallback and a
suffix stress model) and `_eu_build.py` (offline lexicon builder, never imported at runtime).
Tests: `tests/test_text_eu.py` (offline).

## Pipeline

1. Tokenize (letters incl. apostrophes / inner hyphens / dotted acronyms, numbers with
   language-specific separators and ordinal suffixes, a few symbols). `RefWord.text` is the token
   exactly as written, `char_span` its offsets.
2. Expand: abbreviations (small tables), acronyms (letter names), digits (`pronunciada/text/numbers.py`,
   our own speller, string-identical to the num2words package used before; years in en/de; decimals spelled digit by digit in en), symbols (% & $ € ...).
   French and Italian elided clitics (`l'`, `dell'`, `qu'` ...) stay inside the written word.
3. Pronounce each item: lexicon -> hyphen/apostrophe pieces -> inflection fallback (it, ru, pt)
   -> compound split into lexicon words (de; and number words) -> rules / learned letter-to-sound
   table, with a warning `oov:<word>`.
4. Language `rewrite()` maps dictionary notation onto ZIPA conventions (see below); `build_word()`
   adds per-word alternatives, `connected()` adds cross-word ones.
5. Maximal-onset syllabification with legal onsets learned from the lexicon (word-initial clusters,
   >= 5 occurrences) plus their rewritten forms; every syllable has exactly one nucleus (a word
   without a vowel, e.g. Russian в/к/с, gets its consonant as nucleus).

Priors used for alternatives: `P_COMMON = ln 0.5`, `P_MED = ln 0.25`, `P_LOW = ln 0.1`,
`P_RARE = ln 0.03`. The canonical form always has prior 0.

## Results on FLEURS validation (scripts/g2p_eval.py, ZIPA small crctc 500k, greedy)

"first" = the first working version (straight dictionary / rule conversion, measured with the same
metric on the cached greedy outputs); "now" = official script `--n 150`; "all" = every validation
utterance. OOV = share of words that needed rules / letter-to-sound.

| lang | first | now (`--n 150`) | all utts | utts | OOV words | lexicon (shipped) | stress source |
|---|---|---|---|---|---|---|---|
| en | 14.4 % (150) | **12.7 %** | 13.2 % | 394 | 1.2 % | CMUdict 126 k, 0.95 MB | CMUdict digits |
| es | 17.5 % | **11.9 %** | 12.0 % | 408 | 3.8 % (not in es_MX list; all by rules) | rules + 596 k word list, 1.4 MB | orthographic rules |
| it | 13.2 % | **11.1 %** | 11.5 % | 391 | 7.3 % | WikiPron 81 k, 0.6 MB | accents / ɛɔ / suffix rules |
| pt | 35.7 % | **22.4 %** | 22.5 % | 386 | 6.8 % (all by rules) | rules + 57 k flags, 0.18 MB | orthographic rules |
| ru | 52.0 % | **27.8 %** | 27.5 % | 356 | 6.2 % | 381 k stress forms + 57 k irregular, 1.55 MB | lexicon, stem, suffix model |
| de | 41.1 % | **29.5 %** | 28.9 % | 363 | 2.7 % | ipa-dict 661 k, 4.2 MB | ipa-dict marks |
| fr | 19.2 % | **17.0 %** | 18.3 % | 289 (all there is) | 3.2 % | ipa-dict 243 k, 1.1 MB | none |

0 failures in every run. Speed: build 0.1-0.45 s, 0.3-0.6 ms per FLEURS sentence after warm-up.
The residual errors are dominated by ZIPA's own inconsistency between equally native renderings
(es/pt e~ɛ per speaker, pt/es r~ɾ, ru a~ɑ and ʲ, de ʀ~ʁ~ɾ and vowel length, en l~l̴, ɜ˞~ə˞):
these are covered by alternatives but count as errors in this canonical-only metric.

## Convention decisions (measured, canonical <- what ZIPA writes for natives)

* **en** stressed ʌ -> ə (62 %), r-colored schwa -> ɜ˞ (54 %), coda l -> l̴ (55 %); weak forms
  "a" [ɪ] (61 %), "an" [æn] (84 %), "was" [wɑz] (88 %), "with" [wɪθ], "on" [ɔn]. Flaps are
  written t by ZIPA (92 %), so flapping is an alternative only.
* **es** the tap is written r (86 %) and the trill ɾ (71 %) - swapped labels, kept as ZIPA does
  (pero/perro remain distinct); ll -> ʎ (67-90 %), y -> j; rising diphthong after a consonant
  written as one phone "ie", "ia" (ZIPA i 63 %); b d g stay stops (ZIPA never writes β ð ɣ).
  e ~ ɛ is speaker-dependent (per-utterance share of ɛ: quartiles 4 % / 70 % / 100 %), so e is
  canonical and ɛ an alternative everywhere.
* **it** ɔ -> o (62 %); ʎʎ ɲɲ ʃʃ and geminate affricates written single; de-/ne- articulated
  prepositions and "e" with ɛ.
* **pt** nasal vowels = oral vowel + n/m (ZIPA 53-57 %; tilde as alternative), stressed -ão -> [o n],
  "não" [now]; final -o stays o (83 %), final -e -> i; lh -> l (60 %); #r -> ɾ, rr -> ʁ, coda r -> r;
  coda l -> l (72-80 %, w alternative); "de" [di], "em" [in], "e" [i].
* **ru** weak reduction: ɐ/ə -> a, unstressed е -> ɛ, и -> i, у -> u; stressed е -> ɛ;
  л -> ɭ / ɭʲ; soft т/д -> ts(ʲ) / dz(ʲ) (affricated); ч -> ʈʂ; щ -> ʂʲ; palatalisation written
  only before non-front vowels (ZIPA marks ʲ before е/я/ё/ю, rarely before и or a consonant);
  no regressive voicing inside words (также [takʐa]); clitic а -> ɑ.
* **de** r -> ʀ; syllabic n̩ m̩ l̩ -> ən əm əl (69 %); long vowel + ɐ kept (deːɐ, vaːɐ), short
  vowel + ɐ -> V ʀ, -er -> əʀ; tense vowels written long even unstressed; aɪ -> ai, aʊ -> au,
  aː -> ɑː; final -e -> eː (49 % vs ə 37 %); function words take ZIPA's modal form
  (mit [miːt], um [uːm], es [eːs], und [ʊnd] ...), dictionary forms as alternatives.
* **fr** ZIPA follows the first reading of CharsiuG2P "fra": "est" [ɛst] (77 %; s and t droppable),
  "un" [yn] (84 %; œ̃ ɛ̃ alternatives). Liaison: obligatory contexts (determiners, pronouns,
  monosyllabic prepositions/adverbs, c'est, quand ...) canonical with a rare "" alternative;
  optional ones (est, sont, ont, pas, plus ...) as a consonant with a "" alternative; h aspiré list.

Systematic ZIPA "spelling" effects that are *not* natural speech (French final s/t heard in
"pas", "plus", "fait"; German long vowels) were not added as rules beyond the frequent function
words: that is the job of the learned variant layer.

## Connected-speech alternatives (rule based)

* en: weak/strong forms of ~40 function words; flapping (t/d -> ɾ, d) inside words and across
  boundaries; final t/d dropping in clusters; glottal t before consonants; h-dropping in weak
  pronouns/auxiliaries; syllabic n/l/m (schwa drop); -ing -> -in; "the" -> [ði] before vowels;
  l/l̴, ɜ˞/ə˞, cot-caught.
* es: synalepha (final vowel "" / glide), b d g -> β ð ɣ after vowels/continuants, final -d loss,
  syllable-final s -> h / "" (Caribbean, P_RARE), x -> h, yeísmo variants (j ʎ ʝ ʒ ʃ dʒ), e/ɛ.
* it: raddoppiamento sintattico after trigger words and oxytones (doubled consonant), intervocalic
  s/z, ts/dz, open/closed e o, degemination (P_LOW), vowel elision across words.
* pt: final unstressed vowel variants (i~e~ɛ, o~u, a~ɐ, final i ""), tʃ/dʒ -> t/d, strong-r variants,
  coda r variants and final r loss, coda s -> ʃ, l vocalisation, nasal coda "", vowel sandhi.
* ru: reduction variants (a~ɐ~ə~ɑ, i~ɪ~ɛ, ...), voicing assimilation across word boundaries,
  preposition в/к/с voicing by the next word, optional assimilative softening, palatalisation
  not heard, intervocalic j "".
* de: schwa drop in -en/-em/-el, r variants and vocalisation, optional glottal stop before
  vowel-initial words, short unstressed tense vowels, s/z, ç/x, final d/t, ʃt/ʃp -> st/sp,
  weak forms (und, ist, nicht, ein ...), final t dropping before consonants.
* fr: liaison (see above), schwa elision (clitics P_COMMON, inside words P_MED), final-cluster
  simplification (-tre, -ble), optional final schwa after -e, e/ɛ o/ɔ ø/œ a/ɑ, ʁ/ʀ.

## Stress (RefSyllable.stress)

* en: CMUdict digits (reliable for listed words); LTS output keeps the first predicted primary.
  Monosyllabic function words 0.
* de: ipa-dict ˈ/ˌ marks (reliable); unmarked monosyllabic content words (ipa-dict omits the mark)
  get primary; OOV compounds from the splitter: first element primary, later elements secondary
  (number words too: einhundertzwanzig); letter-to-sound OOV keeps its first predicted primary.
* es: orthographic rule (accent mark, else penultimate for -V/-n/-s, else final) - reliable.
* pt: orthographic rule (accents, tilde, ending classes) - reliable for native words.
* it: written final accent; else the syllable with ɛ/ɔ in the WikiPron transcription (only stressed
  syllables can be open); else penultimate unless a curated antepenultimate suffix applies
  (-abile, -ibile, -ico/-ica after a consonant, -issimo, -esimo, -ologo ...). Weak for
  proparoxytone verb forms (parlano) and nouns like macchina.
* ru: acute accent or ё in the prompt; else the lexicon (unreduced vowel of the Wiktionary
  transcription, majority over variants; 380 k forms); else the stress of lexicon forms sharing
  the stem (when it falls on the stem, >= 70 % agreement); else a suffix model (84.5 % on 5 %
  held-out lexicon forms). Clitics unstressed.
* fr: None everywhere.

## Data sources and licenses

| file (pronunciada/data/lex) | source | license | shipped size |
|---|---|---|---|
| en.tsv.gz, en_lts.tsv.gz | CMUdict (github.com/cmusphinx/cmudict) | BSD-2-Clause style | 0.86 MB + 0.09 MB |
| de.tsv.gz, de_lts.tsv.gz | ipa-dict `de` (github.com/open-dict-data/ipa-dict) | MIT | 4.2 MB + 0.04 MB |
| fr.tsv.gz, fr_lts.tsv.gz | ipa-dict `fr_FR` | MIT | 1.1 MB + 0.02 MB |
| es_words.txt.gz | ipa-dict `es_MX` word list only | MIT | 1.4 MB |
| it.tsv.gz, it_lts.tsv.gz | WikiPron `ita_latn_broad` (Wiktionary) | **CC BY-SA** | 0.57 MB + 0.03 MB |
| pt.tsv.gz | WikiPron `por_latn_bz_broad` (word list + open-vowel flags) | **CC BY-SA** | 0.18 MB |
| ru.tsv.gz, ru_stress.tsv.gz | WikiPron `rus_cyrl_narrow` (stress index, irregular forms) | **CC BY-SA** | 1.5 MB + 0.03 MB |

All lexicons ship gzip-compressed in the package (each <= 5 MB, total 13 MB); nothing is
downloaded at runtime. Attribution text: `pronunciada/data/lex/SOURCES-eu.txt`.

CharsiuG2P (MIT code) was inspected: its `dicts/*.tsv` are compilations of ipa-dict (MIT),
WikiPron (CC BY-SA) and sources of unknown license, so the underlying sources were used directly
instead. No GPL data or code: espeak-ng was not used at all. Digits are spelled by
`pronunciada/text/numbers.py` (Apache-2.0, written for this project); the LGPL package num2words
that earlier versions imported when installed is no longer used.

**License flag for the architect:** it / pt / ru lexicons are Wiktionary-derived CC BY-SA (not GPL,
but share-alike). If the release must be permissive-only, these three files can be moved to a
separately-licensed download; the code then still works with rules only (it: rules + LTS would
need retraining on a permissive source; pt and ru fall back to rules / the suffix stress model).

Rebuild: `python -m pronunciada.text._eu_build <lang> <source file>` with the sources above
(cmudict.dict; ipa-dict data/{de,fr_FR,es_MX}.txt; WikiPron data/scrape/tsv/{ita_latn_broad,
por_latn_bz_broad,rus_cyrl_narrow}.tsv).

## Known weaknesses

* en: OOV letter-to-sound is approximate (~54 % words fully right incl. stress on training
  words; proper names worst); CMUdict's first variant is used for homographs (read/lead), the
  second only as aligned per-phone alternatives; ER0 before a vowel stays ɜ˞ ("arrive").
* es: foreign words and names go through Spanish rules (Washington, Shyam); x is [ks] except a
  short list (México, Texas ...); e/ɛ is unpredictable in ZIPA's output (speaker trait).
* it: stress of words without ɛ/ɔ evidence is rule-based (penultimate + curated suffixes): wrong
  for many proparoxytones (parlano, macchina, -ere verbs); 7 % OOV (verb forms missing from
  Wiktionary) use the letter-to-sound table; s/z and e/ɛ choice follows the dictionary's first
  variant; geminates are kept although ZIPA often writes them single.
* pt: pure rules apart from open ɛ/ɔ flags: x (ʃ/ks/z/s) irregularities, hiatus vs diphthong,
  and closed/open e o in unflagged words can be wrong; ZIPA's TER stays high because of r/ɾ and
  e/ɛ/i variation.
* ru: OOV stress (6 % of words) is a guess (stem lookup, then 84.5 % suffix model); Wiktionary
  variant selection can pick a casual form; palatalisation canonical rules are a compromise
  (ZIPA marks it ~50 % of the time); ZIPA's o~ʌ and a~ɑ are unpredictable.
* de: long compounds split by min-pieces DP can split wrongly (Staubecken); ZIPA's German output
  mixes two conventions (ʀ/ai/x/long vowels vs ʁ/aɪ/ç/short) per speaker, hence the high TER.
* fr: liaison lists are hand-made and context-free (no syntax: plural noun + adjective liaison
  is not modeled); h aspiré list is short; "est"/"un" follow ZIPA's label conventions; numbers
  like "vingt et un" lack the liaison t.
* All: abbreviations are small tables; only the symbols % & $ € £ + = @ ° § are verbalised; a
  Russian number is always nominative (pronunciada/text/numbers.py), so case agreement after prepositions is wrong.

## Changes needed in frozen files

* `pronunciada/text/ipa.py::normalize()` NFD-decomposes `ç` (U+00E7, a ZIPA token) into `c` + U+0327
  and drops the cedilla, so any IPA containing ç comes out as c. The builder works around it; a
  fix in `normalize` would be `unicodedata.normalize("NFD", ipa).replace("ç", "ç")`.
