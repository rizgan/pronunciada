# Design

Pronunciation scoring for read-aloud prompts, built on the ZIPA universal phone recognizer
(ONNX). Input: audio, the text the learner was asked to read, and the language. Output:
calibrated 0–1 acceptance probabilities per phoneme, per word and for the sentence, plus tone
and stress results, all controlled by one `strictness` parameter.

## Verified facts about the model (measured on this machine)

- Default model: `anyspeech/zipa-small-crctc-500k` (64 M params, **CC-BY-4.0**). `model.onnx` 248 MB,
  `model.fp16.onnx` 126 MB, `model.int8.onnx` 67 MB. Cached at `~/.cache/pronunciada/models/<name>/`.
  The `-ns-` variants are CC-BY-**NC**-SA: research comparison only, never the default.
- Inputs `x` float32 `[N, T, 80]`, `x_lens` int64 `[N]`. Outputs `log_probs` `[N, T', 127]`
  (already log-softmax) and `log_probs_len`. Output rate ≈ 49 frames/s (20 ms).
- Features: Kaldi fbank, 16 kHz, 80 mel bins, 25 ms / 10 ms, `dither=0`, `snip_edges=False`,
  waveform in [-1, 1] float (lhotse `Fbank` defaults). `kaldi-native-fbank` reproduces it without torch.
- Vocabulary: 127 tokens = `<blk>`(0) `<sos/eos>`(1) `<unk>`(2) `▁`(3) + single IPA characters and
  modifiers (`ʰ ʲ ʷ ʼ ː ˞ ˠ ˤ` and combining nasal/unreleased/voiceless/syllabic/dental/velarised/apical).
  **No tone and no stress symbols** – tones and stress need a separate pitch/energy path.
  Greedy output starts with `▁`; treat `▁` as blank-equivalent.
- Speed on an M3 Pro CPU (4 threads): 3.4 s clip → 35 ms fp32, 22 ms int8.
- Output is not very spiky (blank on ~35–40 % of frames), so frame timings are usable.

## Pipeline

```
text ──► text front-end (per language) ──► Reference: words › syllables › target sounds
                                             (+ expected tone / stress, + allowed variants)
audio ─► fbank ─► ZIPA ONNX ─► frame log-posteriors lp[T,127]
                                   │
        Reference + lp ──► lattice aligner ──► per-sound evidence + timings
        audio + timings ──► prosody path (pitch, energy, duration) ──► tone / stress evidence
        evidence ──► calibrators (sound → word → sentence) ──► probabilities @ strictness
```

Package layout (owner in brackets):

```
pronunciada/types.py            data contract (architect; frozen)
pronunciada/text/ipa.py, base.py  token conventions, registry (architect; frozen)
pronunciada/text/<lang>.py      one front-end per language, exposes build() -> G2P
pronunciada/audio.py            load / resample / fbank
pronunciada/acoustic.py         ZIPA ONNX wrapper
pronunciada/lattice.py          Reference -> token lattice
pronunciada/align.py            lattice CTC forward / Viterbi, per-sound evidence
pronunciada/variants.py         native-variant tables (learned), applied to a Reference
pronunciada/calibrate.py        calibrators + strictness
pronunciada/prosody/            f0, tone, stress
pronunciada/scorer.py           public API
bench/                      datasets, corruption protocol, metrics, Azure runner, experiment scripts
```

## 1. Text front-end

`get_g2p(lang)(text) -> Reference` (see `pronunciada/types.py`). Rules:

- Target sounds are written in **ZIPA's own conventions**, i.e. what the model emits for native
  speakers of that language, not textbook IPA. The objective check for a front-end is the phone
  error rate between its reference and ZIPA's greedy output on native speech (FLEURS); where ZIPA
  is systematically different from the dictionary, the reference follows ZIPA.
- Affricates and diphthongs are one `RefPhone` with several tokens when the language treats them
  as one sound.
- Every syllable has exactly one nucleus. Stress / tone live on the syllable.
- Natural connected speech must never be punished. Front-ends add rule-based `alts` for the
  well-known optional processes of their language (weak forms, linking, liaison, elision,
  flapping, assimilation across word boundaries, vowel devoicing, sandhi …). A second, automatic
  layer (`pronunciada.variants`) adds every realization that native speakers produce often enough.
- No GPL code or data in the runtime path (espeak-ng may be used as an offline dev cross-check only).

## 2. Acoustic evidence (`lattice.py`, `align.py`)

The Reference becomes a **lattice** ("sausage"): slot *i* = target sound *i* with its alternatives
(canonical first; an empty alternative = the sound may be dropped). Each alternative is a chain of
token states; standard CTC topology applies along every path (optional blank between tokens,
blank mandatory between identical consecutive tokens, optional blank at both ends). Blank emission
= `logaddexp(lp[:,0], lp[:,3])`. Alternative priors are added as arc weights.

Per slot *i* we compute, with full-sum (forward) scores:

| name | definition | meaning |
|---|---|---|
| `L_can` | log P(audio \| lattice) | — |
| `L_wild_i` | same lattice, but slot *i*'s token states emit `log(1 − p_blank(t))` (any speech sound) and slot *i* may also be skipped | "anything or nothing was said here" |
| `llr_sub_i` | `L_can − L_wild_i` (≤ 0) | main evidence: how much worse the allowed realizations explain the audio than "anything" |
| `L_del_i` | lattice with slot *i* removed | — |
| `llr_del_i` | `L_can − L_del_i` | > 0: the sound is present; < 0: it was probably skipped |

Plus from the Viterbi path of the canonical lattice: start/end frame, which alternative was used,
duration, peak posterior of the target inside the span, and `heard` = the collapsed greedy tokens
overlapping the span (for feedback: "sounded like X").

This is alignment-free scoring: it does not trust a single hard alignment, which matters with CTC
models. Exactness first (a batched reference implementation checked against brute force on tiny
inputs), then a windowed/banded version (reuse global forward α and backward β, recompute only a
local window around the slot) validated against the exact one.

Budget: evidence for a 10 s / 100-sound utterance in ≤ 30 ms on a laptop CPU; pure numpy.

## 3. Native variants (`variants.py`)

Per language, run the aligner over a few hours of native read speech with the wildcard trick, and
collect, for each target sound (with coarse context: word-initial / final / medial, neighbouring
class), what natives actually produced. Any substitution or deletion with a native rate above a
threshold becomes an allowed alternative with prior = log(rate). This single mechanism absorbs
dictionary-vs-model convention gaps, allophones, and connected-speech reductions for every
language, and is what keeps the architecture language-general. Tables ship as small JSON files.

Also stored per (language, sound): mean / std of `llr_sub`, `llr_del`, duration on native speech,
used to express a learner's evidence relative to natives.

## 4. Calibration and strictness (`calibrate.py`)

- **Sound level.** Features: `llr_sub` (raw, per-frame, native-z-scored), `llr_del`, duration z,
  peak posterior, sound class, position, utterance-level mean evidence. Model: logistic regression
  first; a small boosted-tree model only if it clearly wins on held-out data.
- Two calibrated targets are learned from human-rated data: `P(lenient ok)` = a rater would call it
  at least understandable, and `P(strict ok)` = a rater would call it fully correct.
  `strictness s ∈ [0,1]` interpolates in log-odds: `logit P_s = (1−s)·logit P_lenient + s·logit P_strict`.
  Default 0.5. So the knob has a concrete meaning at both ends and stays calibrated in between.
- **Transfer to languages without human ratings.** Features are native-relative, so the English-
  trained calibrator transfers; a per-language bias is then fitted so that native speech is accepted
  at the same target rate in every language, and checked with the synthetic-error protocol below.
- **Word level.** Inputs: its sounds' log-odds (mean, min, share below 0.5, count), stress / tone
  probabilities, omission evidence → `P(word ok)`. **Sentence level**: word log-odds aggregates,
  completeness, pacing → `P(sentence ok)`. Each level has its own calibrator, trained on the
  human word / sentence ratings of speechocean762 and checked with reliability curves (ECE).

## 5. Tones and stress (`prosody/`)

ZIPA has no tone or stress symbols, so these use a small separate model fed by the audio and the
syllable timings from the aligner:

- Frame features at 10 ms: log-F0 (speaker-normalised), voicing, log-energy, plus the fbank frames.
- One small network family ("ProsodyNet", conv + GRU, well under 1 M params, exported to ONNX) that
  looks at a syllable with its neighbours and predicts the syllable's class:
  Mandarin tone (1-4, neutral), stress level (en, de, ru, it, es, pt).
- Trained on native speech with labels from the text front-end; scored as
  `P(expected class | audio)` then calibrated like sounds. Mandarin tone sandhi is applied in the
  reference (`tone` = surface tone) so correct sandhi is never punished.
- French and Korean have no word-level stress or tone: nothing is scored there. Japanese pitch
  accent is not scored either: it was tried (rounds 1, 8, 13; results/round13_pitch_ja.md) and
  removed in round 15, because dictionary accent labels could not be separated from natives' own
  variation.

## 6. Benchmarks

| purpose | data | metric |
|---|---|---|
| human agreement (English) | speechocean762 test | correlation with human scores at sound / word / sentence level; ECE |
| real learner errors (English) | L2-ARCTIC (if obtainable), EpaDB | detection AUC / F1 |
| all 10 languages | FLEURS test (native) + **controlled errors**: the same audio scored against a reference with a known wrong sound / word / tone / stress | native acceptance rate; error-detection AUC; ECE |
| Traditional script | zh-TW speech if obtainable; otherwise the same audio with Traditional text | identical reference and scores |
| tones | held-out native Mandarin speakers | tone accuracy; wrong-tone detection AUC |
| speed | all of the above | ms per utterance, real-time factor, model size |

The same clips and the same protocol are sent to Azure Pronunciation Assessment when
`AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` are set; otherwise we compare against published Azure
numbers and say so. Controlled errors are a proxy for real learner errors and are reported as such.

## Conventions for contributors

- Python ≥ 3.10; runtime deps limited to numpy, onnxruntime, kaldi-native-fbank, soundfile
  plus small permissively-licensed per-language text packages. Training-only deps (torch, sklearn,
  datasets …) stay out of the runtime path.
- Large data and model files go under `~/.cache/pronunciada/` (override with `PRONUNCIADA_CACHE`), never in the repo.
  The model cache keeps one backbone file per precision: the first load builds the copy that
  declares the extra encoder outputs (`model[.int8].emb-enc0-…-final.onnx`, plus a small `.json`
  sidecar with sizes and SHA-256), checks it against the download and then removes the download and
  older copies (`pronunciada/acoustic.py`, `backbone_model_file`). `PRONUNCIADA_PREOPT=1` also saves
  ONNX Runtime's optimized graph (faster later loads, a second model-sized file);
  `PRONUNCIADA_KEEP_DOWNLOAD=1` keeps the download. A model folder reached through a symbolic link
  is never cleaned.
- Tests under `tests/`, runnable offline except those marked `@pytest.mark.network`.

## 7. What ships (round 18), where it differs from the plan above

Sections 1–6 are the plan the code was built against. What a `Scorer(lang).score(audio, text)`
call actually does today:

- **Per-language output adapter** (`adapter.py`, `data/adapters/`): a small correction on the
  frozen model's frame scores (`lp' = log_softmax(lp + f(lp))`), trained with CTC on FLEURS train.
- **Native variants (§3): off.** Only the front-ends' rule-based alternatives are used; the
  per-sound native statistics (`data/native/`) are features.
- **Sound probability**: boosted trees walked in numpy (`calibrate.py`). English: `sound_B`
  (evidence + sound identity); the reported 0–2 score and the coaching inputs come from
  `sound_E` (adds pooled ZIPA activations). Other languages: universal `sound_U` + a
  per-language bias. The *shown* sound log-odds are then capped by contrast evidence: per-contrast
  caps on the evidence (`data/calib/contrast_caps.json`: zh, en, ru, ja, de) and attribute-head
  caps (`contrast_heads.py`: ko aspiration and lax / tense, ru soft / hard, zh aspiration).
- **Word probability** = `sounds_prob × P_prosody × present_prob` × (ja) length factor ×
  (zh, ja) contrast factor:
  - `sounds_prob`: the word model of §4 (`data/calib/word.npz`, speechocean762 train + FLEURS
    validation planted errors since the license-clean refit of 2026-11-17); it reads the uncapped
    sound outputs.
  - `P_prosody`: tone (zh) or stress (en de es it pt ru) through a word-level calibrator whose
    leniency costs at most one point of native word acceptance on FLEURS validation.
  - `present_prob`: the word was said at all (`data/calib/omission.json`).
  - length factor (`length.py`, `data/length/ja.json`): Japanese long / short vowels and っ,
    from alignment timing plus the length head.
  - contrast factor (`contrast_word.py`): lowers a word whose lowest contrast evidence is below a
    native quantile; adopted for zh (aspiration) and ja (voicing) only.
  - Korean identity factor (`ko_identity.py`, round 19): a small head on the pooled activations
    names the Korean sound class that was said (7 vowels, glides, 18 onsets, 7 codas). The word's
    log-odds are shifted by a linear step in its margins ("expected vs best other class", relative
    to class x position native thresholds), the aligner's extra-sound (`llr_ins`) and missing-sound
    evidence and its duration against the expected one; the step can lower or raise (cap +3), and
    a step <= -1 also marks the word mispronounced.
  - Japanese (round 19, `ja_round19.py`; `PRONUNCIADA_JA_ROUND19=0` restores round 18):
    - the round-19 length model (`length_r19.py`, `data/length/ja_r19.json`) replaces the length
      factor's inputs;
    - instead of the length factor, a capped two-sided step moves the word log-odds
      (`word_step.py`); its evidence is the identity head (`ja_identity.py`), the length logit and
      extra heard sounds (`word_match.py`, and `llr_ins` of `align.py`);
    - the front-end merges repeated vowel morae into one long vowel;
    - the length labels judge consonants across word boundaries.

    Details: results/round19_japanese.md.
- **Word 0–10 score**: `word_reg_r6` (adds word-pooled ZIPA activations).
- **Sentence**: four regression heads (accuracy / fluency / prosodic / total) and the acceptance
  probability, the round-2/4 heads (`sentence_r4.npz`) in every language. The round-5 heads with
  delivery inputs (`sentence.npz`) were used for English and Portuguese until the license-clean
  refit; re-fitted on the clean word log-odds they fail their own native guard (median native
  prosodic score drops > 0.3) in every language, so config `sentence_model` now selects
  `sentence_r4` everywhere (results/licence_refit.md). Mandarin accuracy / prosodic are
  recalibrated on OMPAL dev (`data/calib/sentence_zh_recal.json`).
  English (round 19, config `sentence_head`): the four 0–10 scores are 0.65 × `sentence_r19`
  (trees on the round-4 inputs + delivery + F0 summary + a projection of the std-pooled outputs
  of every encoder stack, which passes the native guard where the mean-pooled ones did not)
  + 0.35 × `seqhead_en.onnx` (`sentence_head.py`, three averaged 205 k-parameter transformers
  over per-sound fused stacks and aligner evidence). The same head's sound and word outputs are
  blended into the reported English 0–2 sound score (0.3) and 0–10 word score (0.4); every
  probability and the coaching inputs are unchanged (results/round19_english.md).
  `PRONUNCIADA_SENTENCE_HEAD=0` restores `sentence_r4` and the previous sound / word scores.
- **Tones and stress** (`prosody/`): the default tone model `tone_zh_v3z` reads the pitch contour,
  a low-frequency spectrum patch and per-syllable ZIPA activations (`tone_zh_v2` = pitch only,
  selectable); the stress model `stress_v3` likewise (`stress` = the previous one, selectable).
  French, Japanese and Korean have no scored stress or tone; Japanese pitch accent was tried
  (rounds 1, 8, 13) and removed in round 15.
- **Coaching** (`coach.py`, `coach_zh.py`, `coach_attr.py`, `delivery_coach.py`, `tips.py`) and the
  **audio-quality gate** (`quality.py`) read only what the score call cached; no score depends on
  them. Schema and thresholds: docs/COACHING.md.

Selectable but off by default: `Scorer(..., explain_away=True)` (round 3 second pass; it lowered
held-out English correlations); `coaching(..., delivery="all")` adds the `slow` / `flat_pitch`
flags. Environment switches exist for paired before / after benchmarks and keep the shipped
behavior when unset: `PRONUNCIADA_TONE_MODEL`, `PRONUNCIADA_STRESS_MODEL`,
`PRONUNCIADA_TONE_LENIENCY`, `PRONUNCIADA_CONTRAST_CAPS=0`, `PRONUNCIADA_HEAD_CAPS=0`,
`PRONUNCIADA_WORD_CONTRAST=0`, `PRONUNCIADA_PT_NASAL=0`, `PRONUNCIADA_KO_IDENTITY=0`.

Tried and not shipped (write-ups in `results/`): the prompt-match channel (round 9; its code now
lives in `bench/calib/match_channel.py`), outside encoders for the sentence heads (round 6),
Italian and German length channels (round 15), Korean aspiration and Italian mid-vowel caps
(round 16), word-level contrast factors for en / ru / de / ko (round 18), a wider training mix for
the universal sound model (round 2).

Held-out discipline: nothing is fitted or selected on speechocean762 test, FLEURS test, OMPAL test
or the cv-real test half; L2-ARCTIC and EpaDB (both splits) are not used for any shipped fit or
selection at all. Rounds 2–18 were each designed after an
earlier held-out evaluation had been seen and then fitted / selected on train, validation or dev
data only; every results file says so. The current numbers are in results/SCOREBOARD.md.
