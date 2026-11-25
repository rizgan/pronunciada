---
license: other
license_name: apache-2.0-and-cc-by-sa-4.0
license_link: https://github.com/moglearn/pronunciada/blob/main/docs/LICENSING.md
library_name: pronunciada
base_model: anyspeech/zipa-small-crctc-500k
base_model_relation: adapter
language:
  - en
  - zh
  - ja
  - es
  - fr
  - de
  - it
  - pt
  - ru
  - ko
tags:
  - pronunciation-assessment
  - mispronunciation-detection
  - speech
  - language-learning
  - onnx
  - on-device
---

<p align="center">
  <img src="https://raw.githubusercontent.com/moglearn/pronunciada/main/docs/assets/banner.jpg" alt="Pronunciada: on-device pronunciation assessment with cloud-level accuracy." width="100%">
</p>

# Pronunciada

On-device pronunciation assessment with cloud-level accuracy. Open source · 10 languages · 71 MB

[Website](https://pronunciada.dev) · [GitHub](https://github.com/moglearn/pronunciada) · [PyPI](https://pypi.org/project/pronunciada/) · [Scoreboard](https://github.com/moglearn/pronunciada/blob/main/results/SCOREBOARD.md)

Pronunciada scores a language learner's pronunciation sound by sound, in 10 languages, on the learner's own device. It is open source, and it is the [state of the art for on-device pronunciation assessment](https://pronunciada.dev/benchmarks/). Against Microsoft Azure's cloud service it comes out ahead in all 10 languages: across 84 public tests it is significantly better on 48 and worse on 2.

It needs no GPU and no network. A sentence is scored in about 40 ms on a laptop CPU with a 71 MB model, and the audio never leaves the machine.

**[Benchmarks on real learner speech →](https://pronunciada.dev/benchmarks/)** Pronunciada against Azure, OpenPronounce, Lingolix, TonePerfect and Chivox on the same recordings.

![Key figures: 48 significant wins and 2 losses against Azure, 10 languages, a 71 MB small model, 41 ms per sentence on a laptop CPU.](https://raw.githubusercontent.com/moglearn/pronunciada/main/docs/assets/stats.jpg)


## What is in this repository

Pronunciada uses [ZIPA](https://huggingface.co/anyspeech/zipa-small-crctc-500k) unchanged as its phone recognizer. This repository holds the small models that Pronunciada adds on top of it, about 26 MB in total:

| Folder | Contents |
| --- | --- |
| `adapters/` | One output adapter per language. It corrects ZIPA's systematic confusions for that language. |
| `calib/` | Gradient-boosted calibrators that turn the acoustic evidence into acceptance probabilities for phonemes, words and sentences. |
| `prosody/` | Mandarin tone and word stress models (ONNX). |
| `length/` | Japanese vowel length. |
| `coach/` | Small classifiers on ZIPA's intermediate layers for consonant contrasts, such as Korean tense consonants. They feed the coaching tips. |
| `native/` | Accepted native variants per language. |

The same files ship inside the pip package, so there is nothing to download from here. This copy matches version 1.1.0. ZIPA itself is downloaded from the anyspeech repository at first use.

## Use

```bash
pip install pronunciada
pronunciada check clip.wav "It was a beautiful day for a float" --lang en
```

```python
from pronunciada import Scorer

scorer = Scorer("en")
result = scorer.score("clip.wav", "It was a beautiful day for a float")
for word in result.words:
    print(word.text, round(word.prob, 2))
```

## Results

Most language-learning apps buy their pronunciation scores from a cloud service, and the most widely used one is Microsoft Azure Pronunciation Assessment. Pronunciada was compared with it on 84 public tests, with the same recordings and text given to both.

![Head-to-head comparisons with Azure by language. Significantly ahead, no significant difference, significantly behind: English 13, 4, 2. Mandarin 6, 10, 0. Japanese 4, 2, 0. Spanish 4, 2, 0. French 5, 1, 0. German 3, 4, 0. Italian 3, 3, 0. Portuguese 4, 2, 0. Russian 3, 3, 0. Korean 3, 3, 0.](https://raw.githubusercontent.com/moglearn/pronunciada/main/docs/assets/chart-tally.png)

Pronunciada is significantly better on 48 of the tests and Azure on 2. The other 34 show no significant difference. In each of the 10 languages Pronunciada wins more tests than it loses. Azure's two wins are both in English: it marks a native speaker's words down a little less often, and it is better at telling native from non-native speakers in one test on Common Voice recordings.

The most direct tests use recordings of learners that experts have graded by hand, which are public for English and Mandarin. Pronunciada agrees with the experts more closely than Azure on individual sounds, on words and on sentence accuracy. On English intonation the two are level.

It is also better at finding the sounds and the Mandarin tones that teachers marked as wrong.

Pronunciada is built on [ZIPA](https://github.com/lingjzhu/zipa), an open speech model, and adds the scoring around it. On the same learner tests it is significantly more accurate than ZIPA alone on 48 of 60, and less accurate on none.

These results use the full 260 MB model. The 71 MB model is slightly behind it, with 47 wins and 4 losses against Azure. On an Apple M3 Pro it scores a 3.4-second sentence in about 40 ms with four CPU threads and 65 ms with one, and uses about 0.3 GB of memory. A request to Azure for a nine-second sentence took about two seconds from the same laptop.

Every test and its confidence interval is in the [scoreboard](https://github.com/moglearn/pronunciada/blob/main/results/SCOREBOARD.md). The figures are for version 1.1.0.

[Full real-learner benchmarks against OpenPronounce, Lingolix, TonePerfect and Chivox →](https://pronunciada.dev/benchmarks/)

## How it works

![Architecture. Audio goes into ZIPA, which is frozen. Its output passes through Pronunciada's adapter, lattice scoring and calibration to give phoneme, word and sentence scores. The text goes through the lexicon and G2P into lattice scoring, and prosody heads read ZIPA and feed calibration.](https://raw.githubusercontent.com/moglearn/pronunciada/main/docs/assets/pipeline.jpg)

Pronunciada does not retrain a speech model. [ZIPA](https://github.com/lingjzhu/zipa), an open phone recognizer, is used exactly as released for all 10 languages. It listens to the recording and reports the sounds it hears, without seeing the text.

Everything Pronunciada adds sits around ZIPA and comes to 36 MB. A dictionary and the sound rules of each language turn the text into the sounds a speaker should produce. A small adapter corrects ZIPA's habits in each language. For every expected sound, the scorer weighs three explanations: it was said, it was replaced by another sound, or it was dropped. Calibration models fitted on expert-scored speech turn that evidence into the probabilities you get back, and separate small models read tone, stress and length from ZIPA's inner layers.

The design is described in [docs/DESIGN.md](https://github.com/moglearn/pronunciada/blob/main/docs/DESIGN.md).

## Limitations

- The text being read is required. For free speech, transcribe first and pass the transcript.
- Agreement with expert scores has been measured for English and Mandarin only. For the other 8 languages the evidence is native speech with altered text, and Common Voice recordings judged by volunteers.
- A native English speaker's words are marked down a little more often than by Azure: about 3% score below 0.6, against about 2%.
- Coaching tips still appear on 9% to 40% of native speakers' sentences, depending on the language.
- Italian double consonants and Portuguese nasal vowels are detected poorly, and Japanese pitch accent is not scored ([results/contrast_audit.md](https://github.com/moglearn/pronunciada/blob/main/results/contrast_audit.md)).
- There are no iOS, Android or WebAssembly builds.

## License

The code and most of these files are Apache-2.0. The Italian, Portuguese and Russian adapters (`adapters/it.npz`, `pt.npz`, `ru.npz`) were trained towards pronunciation dictionaries derived from Wiktionary and are CC BY-SA 4.0. ZIPA is CC BY 4.0 (Zhu et al., ACL 2025). The corpora each file was fitted on are listed in [docs/LICENSING.md](https://github.com/moglearn/pronunciada/blob/main/docs/LICENSING.md).

Azure is a trademark of Microsoft. This project is not affiliated with or endorsed by Microsoft.

## Citation

See [CITATION.cff](https://github.com/moglearn/pronunciada/blob/main/CITATION.cff).
