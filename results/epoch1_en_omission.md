# Epoch 1 | English omissions and neighbouring words

Baseline `5d7ded89a7580aacd9c1267cc0371e30782df7c6`. CPU ZIPA int8, two threads, Python 3.12.0, NumPy 2.5.5 and ONNX Runtime GPU package 1.32.0 using its CPU provider. Cached `model.int8.onnx` SHA-256 `d0e28b68164e8b1fbd6105100c01798828aa0855000ce9bbbd1a2cec233adf13`. The default FP32 source graph was unavailable locally and its download failed DNS. The service at 127.0.0.1:8765 refused the connection. All results below use the local API or CLI.

The English scorer checks the least-present word when its first-pass presence probability is below 0.5. The alternative removes its slots in a crop bounded by the closest unaffected phones, up to 200 frames and 32 reference phones. It can update at most two adjacent words, and only when the crop favours deletion, frames actually shift, and existing presence and word-logit evidence does not decline. The omitted word keeps its original presence score and gets a zero-length ordered location when an alternative is adopted. Sentence, non-adjacent word and first-pass prosody, length and auxiliary embedding evidence are kept. In particular, frame-aligned stress and length remain estimates from the first pass even where a reported sound span moves. The global wildcard pass stays available for its existing opt-in benchmarks and language settings.

## Evaluation

SpeechOcean762 train was used for development (60 clips, 49 speakers). The first candidate improved false alerts but missed one annotated adjacent error on the original test group (7 to 6 of 8) and caused 15 adjacent timestamp overlaps over 50 ms. In clip `so762-004570129` the OVEN vowel annotation was AH→OW, and the candidate removed all OVEN feedback despite making no change to its phone boundaries. That variant was rejected. The revised eligibility and timestamp constraints were checked on fresh test speakers disjoint from train and from the 88 speakers in that first test group. This run used 200 clips from 22 speakers, 200 original prompts, 200 prompts with one unsaid word inserted and 101 prompts with a near substitution. The script uses seed 20261125 and records the manifest ID and prompt for every pair.

The unit is one word per clip and prompt. False neighbours have accuracy 10, no marked phonetic error, and no balanced coaching on the matched original prompt. A coaching flag is any item at the scored word index. Adjacent real errors have explicit phone annotations. Near substitutions compare prompt identity against the recorded original text. They test the robustness of the same-audio counterfactual, without claiming a teacher-validated articulation label. Intervals shown are paired speaker-cluster bootstrap with 2,000 draws at seed 20261125.

| Fresh test outcome | Baseline | Candidate | Change, 95% interval when applicable |
| --- | ---: | ---: | --- |
| Matched correct neighbour gets coaching | 121/295 | 87/295 | −11.53 pp [−15.67, −7.55] |
| Matched correct neighbour scores below 0.6 | 77/295 | 61/295 | −5.42 pp [−8.46, −2.98] |
| Inserted word receives `missing_word` | 179/200 | 179/200 | No discordant pairs |
| Annotated adjacent error gets coaching | 18/19 | 18/19 | Same item kind, phone index, expected and heard on all 19 pairs |
| Annotated adjacent error scores below 0.6 | 14/19 | 14/19 | No discordant pairs |
| Near substitute gets coaching | 78/101 | 78/101 | No discordant pairs |
| Near substitute scores below 0.6 | 62/101 | 62/101 | No discordant pairs |
| Correct original word gets coaching | 162/1,105 | 162/1,105 | No discordant pairs |
| Annotated original-word error gets coaching | 31/38 | 31/38 | No discordant pairs |
| Inserted-prompt boundary overlaps >50 ms | 0/1,228 | 0/1,228 | No discordant pairs |

The real-deletion cohort comprises 27 words in 13 clips from four test speakers. Every canonical phone has an explicit zero-based `<DEL>` mark with matching canonical ARPABET after removing the stress digit. Prompt and label tokens match and each clip has at least three words. This is a whole-word deletion proxy based on corpus annotations. Both scorers flag 22/27 omissions, 5/5 annotated neighbouring errors and 1/7 good neighbours, with no change in coaching content on the five annotated errors. That cohort and the 19 annotated errors in the fresh counterfactual cohort place limits on the evidence for preserving rare errors.

Median total latency in the sequential matched benchmark was 85.40→74.02 ms on original prompts, 27.23→35.55 ms on inserted prompts and 26.82→26.66 ms on near substitutions. The candidate's separately instrumented recheck medians were 0.074, 8.771 and 0.076 ms respectively. Each original prompt ran first on its clip, so its total includes first-time acoustic inference. Environment load makes sequential end-to-end differences noisier than the stage timer.

## Reproduction

From the candidate checkout in PowerShell. Use a writable path for the untracked output; the JSONL outputs include locally cached corpus annotations.

```powershell
$py = 'C:\Users\Dell\Documents\pronunciada\.venv-gpu\Scripts\python.exe'
$source = (Get-Location).Path
$out = Join-Path $env:TEMP 'pronunciada-epoch1'
$base = Join-Path $out 'baseline'
New-Item -ItemType Directory -Path $out,$base -Force | Out-Null
$archive = Join-Path $out 'baseline.tar'
git archive --format=tar --output=$archive 5d7ded89a7580aacd9c1267cc0371e30782df7c6
tar -xf $archive -C $base
$collector = Join-Path $source 'bench/round20_native_fa/epoch1_omission.py'
$summary = Join-Path $source 'bench/round20_native_fa/epoch1_omission_summary.py'
$excluded = Join-Path $out 'first-test.jsonl'
$env:PYTHONPATH = $base
& $py $collector test $excluded --n 180
foreach ($name in @('baseline', 'candidate')) {
    $env:PYTHONPATH = if ($name -eq 'baseline') { $base } else { $source }
    & $py $collector train (Join-Path $out "$name-train.jsonl") --n 60
    & $py $collector test (Join-Path $out "$name-fresh.jsonl") --n 200 --exclude-speakers-from $excluded
    & $py $collector test (Join-Path $out "$name-real.jsonl") --real
}
foreach ($group in @('train', 'fresh', 'real')) {
    & $py $summary (Join-Path $out "baseline-$group.jsonl") (Join-Path $out "candidate-$group.jsonl")
}
& $py -m pytest -q tests/test_align_explain.py tests/test_scorer.py tests/test_coach.py
```

Targeted tests yielded 96 passes and 16 model-dependent skips. The new regression exercises the normal scorer and cached rescore with fixed acoustic posteriors. Actual recorded audio was scored by the evaluator and CLI.

Run the following from each checkout for the same recorded clip, `so762-001030143`. Indices are zero-based.

```powershell
$clip = "$env:USERPROFILE/.cache/pronunciada/data/speechocean762/audio/train/001030143.flac"
& $py -m pronunciada check $clip 'BILL LIKES THE P E CLASS' --lang en --coach --sensitivity balanced --json --precision int8 --threads 2
& $py -m pronunciada check $clip 'BILL LIKES THE APPLE P E CLASS' --lang en --coach --sensitivity balanced --json --precision int8 --threads 2
```

The control has no tips in either checkout. The edited baseline gives the missing-word tip for APPLE (3) and an extra instruction to retry P (4). The revised scorer gives the APPLE tip alone. Lower-ranked coaching items for THE (2) and E (5) remain. A second reproducible control is `so762-000010011`, `WE CALL IT BEAR` versus `WE CALL APPLE IT BEAR`, with APPLE at index 2. For this clip the inserted baseline already gives just the missing-word tip.
