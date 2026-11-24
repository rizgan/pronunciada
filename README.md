<p align="center">
  <img src="docs/assets/banner.jpg" alt="Pronunciada: on-device pronunciation assessment with cloud-level accuracy. The waveform of a learner saying 'It was a beautiful day for a float', cut at the word boundaries, with the probability given to each word." width="100%">
</p>

# Pronunciada

[![PyPI](https://img.shields.io/pypi/v/pronunciada)](https://pypi.org/project/pronunciada/)
[![CI](https://github.com/moglearn/pronunciada/actions/workflows/ci.yml/badge.svg)](https://github.com/moglearn/pronunciada/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

[Website](https://pronunciada.dev) · [PyPI](https://pypi.org/project/pronunciada/) · [Hugging Face](https://huggingface.co/kimitonatsu/pronunciada) · [Benchmarks](#benchmarks) · [Scoreboard](results/SCOREBOARD.md) · [How it works](#how-it-works)

Pronunciada is an open-source Python library for on-device pronunciation assessment with cloud-level accuracy. It takes a recording and the text the speaker was reading, and returns the probability (0 to 1) that a human rater would accept each phoneme, each word and the sentence as a whole.

It covers English, Mandarin, Japanese, Spanish, French, German, Italian, Portuguese, Russian and Korean.

- It matches or beats Microsoft Azure Pronunciation Assessment in 77 of 84 comparisons on public test sets: significantly better in 40, not significantly different in 37 and significantly worse in 7.
- It runs on the device. Scoring a 3.5-second recording takes about 50 ms on an Apple M3 Pro, on CPU with ONNX Runtime and no PyTorch dependency. The audio does not leave the machine. The quantized model is edge-optimized: 67 MB on disk and about 0.3 GB of memory.
- It is built on [ZIPA](https://github.com/lingjzhu/zipa), an open multilingual phone recognizer with 64M parameters (248 MB, or 67 MB quantized). It is significantly more accurate than ZIPA alone on 45 of 60 tests and significantly less accurate on none.

The numbers, and the tests where Azure is ahead, are under [Benchmarks](#benchmarks).

A 45-second overview, with sound:

https://github.com/user-attachments/assets/f5932a3a-322f-442c-852b-ef7416d50f58

## Quick start

```bash
pip install pronunciada
pronunciada check clip.wav "It was a beautiful day for a float" --lang en
```

![The command above running in a terminal and printing the scores for each word and sound.](docs/assets/cli.gif)

<details>
<summary>Output as text</summary>

```text
~ Close  sentence 75% · accuracy 7.7/10 · fluency 8.1/10   en · strictness 0.5

  It was a beautiful day for a float
           ~~~~~~~~~         ~ ^^^^^

  beautiful  63%  b 92%  [j→b 67%]  u 66%  t 58%  [ə 57%]  f 84%  ə 64%
                  [l̴ 65%]
  a          79%  [ɪ 64%]
  float      10%  f 83%  [l 27%]  [oʊ 35%]  t 56%

  → Let the air pass along the sides of the tongue for /l/ in “float”.
```

</details>

This is the output for [`site/samples/en-learner-1.mp3`](site/samples/en-learner-1.mp3), a recording from the speechocean762 test set. The corpus annotators marked the /l/ of "float" as dropped. `^` marks a word below 0.5 and `~` a word between 0.5 and 0.8.

The same call from Python:

```python
import pronunciada

result = pronunciada.score("clip.wav", "It was a beautiful day for a float", lang="en")

print(f"{result.prob:.2f}")
# 0.75
for word in result.words:
    print(word.text, f"{word.prob:.2f}")
# It 0.87
# was 0.96
# a 0.97
# beautiful 0.63
# day 0.94
# for 0.93
# a 0.79
# float 0.10
for phone in result.words[7].phones:
    print(phone.ipa, f"{phone.prob:.2f}", repr(phone.heard))
# f 0.83 None
# l 0.27 ''
# oʊ 0.35 'ɔ'
# t 0.56 None
```

`heard` is set when a phoneme scores below 0.5. It is what the recognizer output in that span, or an empty string if it output nothing there.

The model is downloaded to `~/.cache/pronunciada` on first use.

## Benchmarks

All comparisons are with the Azure Pronunciation Assessment REST API (region `westus`, November 2026, one request per recording). Both systems received the same audio and reference text. An asterisk in a chart, and bold in a table, marks the better system where the difference is significant (paired bootstrap over speakers, p < 0.05). The charts are generated from the scoreboard by `scripts/graphics/make_charts.py`. Confidence intervals and every other table are in [results/SCOREBOARD.md](results/SCOREBOARD.md).

### Summary

![Head-to-head comparisons with Azure by language. Significantly ahead, no significant difference, significantly behind: English 11, 5, 3. Mandarin 5, 11, 0. Japanese 3, 1, 2. Spanish 3, 2, 1. French 5, 1, 0. German 3, 4, 0. Italian 3, 3, 0. Portuguese 3, 2, 1. Russian 3, 3, 0. Korean 1, 5, 0.](docs/assets/chart-tally.png)

The scoreboard has 84 comparisons with Azure, each one metric on one held-out test set. Pronunciada is significantly ahead in 40, not significantly different in 37 (ahead in 20 of those) and significantly behind in 7. The seven are English sentence prosody, the native false-alarm rate in English, Japanese, Spanish and Portuguese, and two tests on Common Voice recordings (English native versus non-native speakers, Japanese accepted versus rejected clips). The sections below show the main tests, starting with the one that covers all ten languages.

For the other pronunciation systems on the market, commercial and open source, see [results/industry_snapshot.md](results/industry_snapshot.md).

### All ten languages

Learner recordings with expert annotation are public for English and Mandarin only, so the test that covers all ten languages uses native speech: 150 FLEURS test recordings per language, scored against a reference in which one word has been replaced by a word that differs by a single phoneme. The figure is the area under the ROC curve for separating the replaced word from the unchanged words.

![Changed-word detection in ten languages, Pronunciada and Azure. English 0.981 and 0.963, Mandarin 0.975 and 0.966, Japanese 0.968 and 0.919, Spanish 0.990 and 0.942, French 0.986 and 0.937, German 0.960 and 0.887, Italian 0.979 and 0.970, Portuguese 0.963 and 0.938, Russian 0.900 and 0.909, Korean 0.944 and 0.937.](docs/assets/chart-languages.png)

This test measures whether a system detects a mismatch between audio and text. It does not measure accuracy on learner speech.

<details>
<summary>Table</summary>

| Language | Pronunciada | Azure |
| --- | --- | --- |
| English | **0.981** | 0.963 |
| Mandarin | 0.975 | 0.966 |
| Japanese | **0.968** | 0.919 |
| Spanish | **0.990** | 0.942 |
| French | **0.986** | 0.937 |
| German | **0.960** | 0.887 |
| Italian | 0.979 | 0.970 |
| Portuguese | **0.963** | 0.938 |
| Russian | 0.900 | 0.909 |
| Korean | 0.944 | 0.937 |

</details>

### English learners

![Pearson correlation with expert scores on speechocean762, Pronunciada and Azure. Phoneme 0.685 and 0.538, word 0.609 and 0.589, sentence accuracy 0.748 and 0.673, sentence fluency 0.779 and 0.708, sentence prosody 0.772 and 0.827, sentence total 0.762 and 0.796.](docs/assets/chart-english.png)

speechocean762 test set: 2,500 recordings from 125 Mandarin-speaking learners, each scored by five experts. The figure is the Pearson correlation between system score and expert score.

<details>
<summary>Table</summary>

| Level | Pronunciada | Azure |
| --- | --- | --- |
| Phoneme | **0.685** | 0.538 |
| Word | 0.609 | 0.589 |
| Sentence accuracy | **0.748** | 0.673 |
| Sentence fluency | 0.779 | 0.708 |
| Sentence prosody | 0.772 | **0.827** |
| Sentence total | 0.762 | 0.796 |

</details>

### Mispronunciation detection

![Area under the ROC curve for detecting mispronunciations, Pronunciada and Azure. L2-ARCTIC 0.712 and 0.631, EpaDB 0.721 and 0.587, Mandarin words with any error 0.792 and 0.704, Mandarin words with tone errors 0.778 and 0.679.](docs/assets/chart-detection.png)

Area under the ROC curve for separating the phonemes or words that annotators marked as mispronounced from the rest. L2-ARCTIC (speakers of six first languages) and EpaDB (Spanish-speaking learners) were not used to train or tune any part of Pronunciada. The Mandarin rows are from the OMPAL test set: 21 French-speaking learners of Taiwan Mandarin, every character marked by teachers.

On speechocean762 test (47,041 phonemes, 18.6% marked as errors), with both systems thresholded to flag the same 36% of the marked phonemes, 65.9% of Pronunciada's flags are phonemes the experts marked, against 44.0% of Azure's.

<details>
<summary>Tables</summary>

| English, phoneme level (AUC) | Pronunciada | Azure |
| --- | --- | --- |
| L2-ARCTIC test | **0.712** | 0.631 |
| EpaDB test | **0.721** | 0.587 |

| Mandarin, OMPAL test | Pronunciada | Azure |
| --- | --- | --- |
| Word, any error (AUC) | **0.792** | 0.704 |
| Word, tone error only (AUC) | **0.778** | 0.679 |
| Word, consonant or vowel error (AUC) | 0.856 | 0.822 |
| Sentence accuracy (Pearson) | **0.634** | 0.363 |
| Sentence fluency (Pearson) | **0.590** | 0.343 |

</details>

Azure was not tuned on any of this data. Pronunciada's calibration models were fitted on speechocean762 train, FLEURS train and validation, and the OMPAL development speakers. Every figure above is from a held-out split.

### On-device latency

Median time for one `Scorer.score` call on 100 speechocean762 recordings (mean length 3.5 s), Apple M3 Pro, CPU only.

| Model | 4 threads | 1 thread |
| --- | --- | --- |
| fp32 (248 MB) | 50 ms | 100 ms |
| int8 (67 MB) | 40 ms | 65 ms |

Peak memory of the Python process while scoring is about 0.5 GB with the fp32 model and 0.3 GB with int8 (English, one thread, Apple M3 Pro; 480 MB and 294 MB measured). The int8 model is optimized for edge devices: a 67 MB file on one CPU thread. There are no packaged iOS or Android builds yet.

For recordings of about ten seconds (150 FLEURS test recordings per language), the median is 0.30 s to 0.44 s on one thread, depending on the language. That is about the same rate as the one-thread figure above, 100 ms for 3.5 seconds of audio. A request to Azure's REST API with the same recordings and text took 1.8 s to 3.0 s from the same laptop, upload included. The per-language figures are in the [scoreboard](results/SCOREBOARD.md#speed-size-and-what-runs-where).

### Compared with ZIPA alone

![Pronunciada and the unmodified ZIPA model on the learner tests. English phonemes 0.685 and 0.434, words 0.609 and 0.394, sentences 0.748 and 0.583 (Pearson). L2-ARCTIC 0.712 and 0.659, EpaDB 0.721 and 0.638, Mandarin any error 0.792 and 0.602, Mandarin tone errors 0.778 and 0.552 (AUC).](docs/assets/chart-zipa.png)

To measure what this project adds to the open model, the same held-out tests were run on the unmodified ZIPA model scored with the standard goodness-of-pronunciation method: the mean log-posterior of each expected phoneme over its force-aligned frames, with no adapters, calibration, or tone and stress models. Over 60 test rows, Pronunciada is significantly ahead in 45 and not significantly different in 15. ZIPA alone is not significantly ahead in any. Repeating the correlation rows with a rank-based measure (Spearman) leaves every one of them significantly ahead, with smaller gaps at the phoneme and word level (0.469 against 0.309 for phonemes).

The difference is smallest where the task is easy for both, such as detecting a clearly different word in the prompt (AUC 0.99 for both), and largest for Mandarin tones, which ZIPA does not output. For reference, ZIPA alone scores 0.659 and 0.638 at detecting marked errors on L2-ARCTIC and EpaDB, where Azure scores 0.631 and 0.587, and 0.434 in phoneme-level agreement with expert scores, where Azure scores 0.538. The differences between ZIPA alone and Azure were not tested for significance. All rows are in [results/zipa_baseline.md](results/zipa_baseline.md).

## Supported languages

![English, Chinese, Japanese, Spanish, French, German, Italian, Portuguese, Russian and Korean, each written in its own script.](docs/assets/languages.jpg)

| Language | Code | Install | Also scored |
| --- | --- | --- | --- |
| English | `en` | `pronunciada` | word stress |
| Mandarin, Simplified and Traditional | `zh` | `pronunciada[zh]` | tone of each syllable, aspiration |
| Japanese | `ja` | `pronunciada[ja]` | vowel and consonant length |
| Spanish | `es` | `pronunciada` | word stress |
| French | `fr` | `pronunciada` | |
| German | `de` | `pronunciada` | word stress |
| Italian | `it` | `pronunciada` | word stress |
| Portuguese (Brazil) | `pt` | `pronunciada` | word stress |
| Russian | `ru` | `pronunciada` | word stress, palatalization |
| Korean | `ko` | `pronunciada` | aspirated and tense consonants |

`pip install "pronunciada[all]"` installs every language. The Japanese extra adds about 300 MB of dictionaries.

## Usage

### Python

On NVIDIA GPUs, pass `device="cuda"` to `Scorer` (or `--device cuda` to `check`,
`score`, or `serve`). Use an environment with the GPU build of ONNX Runtime and
matching CUDA and cuDNN libraries. `precision="fp16"` (CLI: `--precision fp16`)
selects the existing half-precision export. The device setting applies to the
acoustic model. See the [ONNX Runtime CUDA setup guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

To run this checkout with CUDA on Windows, create a separate Python environment
and use the GPU build of ONNX Runtime as the sole ONNX Runtime distribution:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]" "numpy>=1.24" "kaldi-native-fbank>=1.18" "soundfile>=0.12"
.\.venv\Scripts\python.exe -m pronunciada serve --device cuda --precision fp16
```

Create a `Scorer` once when scoring more than one recording:

```python
from pronunciada import Scorer

scorer = Scorer("zh")                    # or Scorer("en", precision="int8")
result = scorer.score("clip.wav", "我家有四口人")

for word in result.words:
    for syllable in word.syllables:
        print(syllable.text, syllable.tone_expected, syllable.tone_heard, syllable.tone_prob)
```

`strictness` (0 to 1, default 0.5) moves every probability between a lenient and a strict calibration. `scorer.rescore(result, strictness=0.8)` applies a new value without running the acoustic model again.

`result.coaching()` returns the detected errors as dictionaries, ordered by severity. Each has the word, the expected phoneme, the phoneme recognized when one can be named, and the articulatory features that differ:

```python
for item in result.coaching():
    print(item["kind"], item["word"], item["expected_ipa"], item["heard_ipa"], item["diff_keys"])
# substitution float l None ['lateral']
# substitution beautiful l̴ None ['lateral']
# ...
```

`pronunciada.tips.tips()` turns these into the sentences shown in the command-line report. Each tip has a stable key so an application can substitute its own wording or translation. The schema is in [docs/COACHING.md](docs/COACHING.md).

The result also carries timestamps for every phoneme and word, fluency and completeness scores, omitted and inserted words, and warnings when the recording is clipped, noisy, very quiet or mostly silence.

### Command line

```bash
pronunciada check clip.wav "我家有四口人" --lang zh
pronunciada check clip.wav "Buenos días" --lang es --json
pronunciada langs
```

Options are listed in [docs/CLI.md](docs/CLI.md).

### HTTP server

```bash
pronunciada serve --port 8765
curl -s -F audio=@clip.wav --form-string "text=Good morning" --form-string lang=en http://127.0.0.1:8765/v1/score
```

The server is for local use. It has no authentication. See [docs/SERVER.md](docs/SERVER.md).

### Azure-compatible output

`to_azure` converts a result to the JSON layout returned by Azure Pronunciation Assessment. `pronunciada serve --azure-compat` accepts Azure's short-audio REST request and answers in the same layout.

```python
from pronunciada.compat.azure import to_azure

payload = to_azure(pronunciada.score("clip.wav", "Good morning", lang="en"))
payload["NBest"][0]["PronunciationAssessment"]["AccuracyScore"]
```

The field names match. The score distributions do not, so thresholds chosen for Azure need to be chosen again. Field-by-field notes are in [docs/AZURE_COMPAT.md](docs/AZURE_COMPAT.md).

### MCP server

```bash
pip install "pronunciada[mcp]"
pronunciada mcp
```

This starts a Model Context Protocol server on stdio with one tool, `assess_pronunciation(audio_path, reference_text, language)`. Client configuration is in [docs/MCP.md](docs/MCP.md).

## How it works

![How one word is scored. The word is "float", said without its l. Expected sounds f, l, oʊ, t. Recognized instead: nothing for l, and ɔ for oʊ. Probabilities 0.83, 0.27, 0.35, 0.56.](docs/assets/figure-word.png)

1. The reference text is converted to a phoneme sequence with per-language dictionaries and rules. This includes Mandarin tone sandhi and French liaison.
2. ZIPA produces frame-level posteriors over IPA phones from the audio alone. A small per-language adapter corrects its systematic confusions.
3. For each expected phoneme, the scorer compares the likelihood that it was produced with the likelihood of a substitution or a deletion. Each likelihood is a CTC forward score summed over all alignments, so the result does not depend on a single forced alignment.
4. Gradient-boosted calibrators fitted on expert-scored speech map this evidence to acceptance probabilities for phonemes, words and sentences.
5. Tone, stress, length and a few consonant contrasts are not recoverable from the phone posteriors. Separate small classifiers read ZIPA's intermediate layers for these.

All models trained for this project together add about 10 MB to the package. The design is described in [docs/DESIGN.md](docs/DESIGN.md).

## Limitations

- Accuracy on learner speech has been measured for English and Mandarin only. The other eight languages are evaluated on native speech with altered reference text and on Common Voice recordings accepted or rejected by volunteer validators.
- English sentence prosody correlates less well with expert scores than Azure's (0.772 against 0.827).
- At the default strictness, about 3% of a native speaker's words score below 0.6, against about 2% for Azure, in English, Japanese, Spanish and Portuguese.
- In Mandarin, a low tone probability on one syllable does not always lower the word probability. Applications that grade tones should read `syllable.tone_prob` directly.
- Coaching tips were tuned on English and Mandarin learners. On native FLEURS recordings, between 9% and 43% of sentences (depending on language) still receive at least one pronunciation tip.
- Italian geminate consonants and Portuguese nasal vowels are detected poorly. Japanese pitch accent is not scored. Details are in [results/contrast_audit.md](results/contrast_audit.md).
- The reference text is required. For spontaneous speech, transcribe first and pass the transcript.
- There are no iOS, Android or WebAssembly builds.

## Reproducing the benchmarks

```bash
pip install -e ".[all,bench]"
python -m bench.azure_pa run speechocean762 --split test   # needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
python -m bench.eval_english      # speechocean762, L2-ARCTIC, EpaDB
python -m bench.eval_languages    # FLEURS tests in ten languages
python -m bench.scoreboard        # regenerates results/SCOREBOARD.md
```

The corpora are public. Azure's responses are not redistributed, so the comparison needs an Azure Speech resource. `bench.azure_pa` sends each recording once and caches the response under `~/.cache/pronunciada/azure`. Download instructions, the remaining commands and run times are in [docs/benchmarks_data.md](docs/benchmarks_data.md).

## Background

Pronunciation assessment is usually bought as a cloud service. The learner's recording is uploaded, the service bills by the amount of audio, and the application waits for the round trip. Pronunciada was written at [Moglearn](https://moglearn.com), a language-learning app, to do the same job on the learner's device.

ZIPA, an open phone recognizer, already ran on a CPU in many languages. Used directly, it agrees with expert phoneme scores less well than Azure does (0.434 against 0.538 on speechocean762). This project tested whether established scoring methods and the sound rules of each language, added on top of ZIPA, could close that gap without training a large model. The system combines the recognizer with lattice-based scoring, per-language phonology and small calibration models.

## Development

Development ran in rounds. Each round set a target in advance, selected changes on development data and was evaluated once on held-out data. The write-ups for all rounds are in [results/](results/), including approaches that were tried and dropped: Japanese pitch accent, a second acoustic model, and training on synthetic speech.

## License

The code is licensed under Apache-2.0. ZIPA is licensed under CC BY 4.0 and is downloaded at first use. The Italian, Portuguese and Russian pronunciation dictionaries are derived from Wiktionary and remain under CC BY-SA. [docs/LICENSING.md](docs/LICENSING.md) and [NOTICE](NOTICE) list every data file with the corpus it was fitted on and its license.

Azure is a trademark of Microsoft. This project is not affiliated with or endorsed by Microsoft.

## Citation

See [CITATION.cff](CITATION.cff).

## Acknowledgements

This project depends on [ZIPA](https://github.com/lingjzhu/zipa) (Zhu et al., ACL 2025) and was evaluated on speechocean762, OMPAL, FLEURS, Common Voice, L2-ARCTIC, EpaDB and AISHELL.
