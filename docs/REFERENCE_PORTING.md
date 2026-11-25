# Porting the scoring post-processing

`pronunciada/reference/` is a second implementation of the alignment, evidence and calibration
steps, written with the Python standard library only (no NumPy) so that it can be translated line
by line to C++, Swift, Kotlin or Rust. It gives the same numbers as the production code
(`pronunciada/align.py`, `pronunciada/lattice.py`, `pronunciada/calibrate.py`); the tests in
`tests/test_reference_module.py` check this.

## What it covers

| step | reference | production |
|---|---|---|
| token file, phone symbol check | `tokens.py` | `text/ipa.py` |
| target sounds -> CTC lattice | `lattice.py` | `lattice.py` |
| emission table (token, blank, wildcard columns) | `align.emission_table` | `align.emission_table` |
| scaled forward / backward, stitched L_wild, L_del, L_ins | `align._fast_scores` | `align._fast_scores` |
| exact log-domain pass (fallback) | `align.exact_scores`, `forward_log` | same names |
| Viterbi spans, greedy decode, per-sound evidence | `align.score_slots`, `score_phones` | `align.score_slots`, `score_reference` |
| tree ensembles, Platt / isotonic maps, log-odds | `calibrate.TreeEnsemble`, `Head`, `LevelModel` | `calibrate.py` |
| strictness blend, logit, sigmoid | `calibrate.blend_logit`, `blend` | `calibrate.py` |
| omission probability, prosody-channel word probability | `calibrate.present_prob`, `prosody_prob` | `scorer._present_prob`, `_pros_prob` |
| per-sound feature table, calibrator input rows, word table | `features.phone_table`, `matrix`, `word_table` | `features.py`, `variants.NativeStats` |
| evidence -> universal sound log-odds (+ bias) -> word log-odds | `pipeline.universal_scores` | first part of `Scorer._calibrate` |
| .npz reader | `npz.py` | `numpy.load` |

Input: the adapted log-posteriors `lp[T][127]` (backbone output after the language's adapter,
float32 values; frame t starts at 0.02 t + 0.03 s), the target phones
`[{"ipa": str, "alts": [[ipa, log_prior], ...]}]` (an empty `ipa` in `alts` means the sound may be
dropped), and two switches set by the scorer: `insertion` (extra-sound evidence; Korean always,
Japanese with its word step, languages whose identity rule reads it) and `wild_slots` (the
explain-away pass, German).

Not covered, and why:

- Before the evidence: audio decoding, resampling, fbank (kaldi-native-fbank), the backbone (ONNX
  Runtime) and the adapter (a small float32 matrix product whose last bits depend on the BLAS
  library; `pronunciada/adapter.py`).
- After the universal sound model and the word model: the language's own sound model where
  `calib/config.json` names one (English: `sound_B`), the round-16 contrast caps (a 6-term dot
  product evaluated by BLAS, whose last bits depend on the library), the round-2 word regression,
  the sentence table and sentence models (they read delivery and F0 features), and the
  per-language word factors of `Scorer._assemble` other than the omission formula.
- Everything that reads encoder activations (identity and contrast heads, word-score projection,
  English sound embeddings, sentence delivery projection), the three small ONNX models
  (`prosody/stress_v3.onnx`, `prosody/tone_zh_v3z.onnx`, `calib/seqhead_en.onnx`), the F0 tracker,
  the delivery features, the length channel and the per-language word factors in
  `scorer._assemble`. These need the extra-output backbone on the device (ONNX Runtime) or further
  porting.

## Arithmetic

All numbers are IEEE-754 binary64 (`double`). `exp`, `log`, `log1p` are the C library functions
(NumPy returned the same bits as the C library on the test machine). Where Python would raise
(log 0, exp overflow, division by 0) `numeric.py` returns the IEEE result, as C does.

To match production to the last bit, a port must also follow production's order of additions. The golden vectors were written on macOS arm64 and reproduce to the last bit there. On other platforms the C math library returns different last bits, and the repository's tests then allow a difference of 1e-5.
Every sum in the reference names its order (`numeric.py`):

| order | where |
|---|---|
| left to right from 0.0 | wildcard emission (columns 1, 2, 4..126); arc contributions per state in arc order; deletion and insertion sums over frames (more than one slot); Viterbi has no sums |
| NumPy pairwise (`sum_pairwise`: blocks of up to 128 with 8 partial sums, recursive halves above 128) | per-frame scale `c` and `d` of the forward / backward pass; `L_can` final sum; exact-pass state sums and block totals; wildcard chain sums over frames; tree leaf values over trees; mean in the omission formula |
| `x[0] + pairwise(x[1:])` (`sum_first_pw`) | per-slot sums of bypass, inflow and outflow arcs (production uses `np.add.reduceat`) |

Two more details: `np.interp` (isotonic maps) is evaluated by production's NumPy build as one fused
multiply-add, `fma(slope, x - x_j, y_j)`; a port should call `fma()` (the reference computes it
exactly with rational arithmetic). `peak_post` is `exp` of a float32 value in float32: production's
result equals C `expf`; the Python reference rounds the double `exp` to float32, which differs from
`expf` by one float32 unit in about 0.6 % of inputs. A port should use `expf`.

Constants: blank emission = logaddexp(lp[0], lp[3]) (`<blk>` or the word marker); wildcard
emission floor -50; reported log(0) stand-in -1e6; llr of an impossible clip -1000, llr_del clipped
to [-1000, 1000]; exact recompute when L_wild < L_can - 1e-6 or a stitched value is not finite;
reachability slack 2 x (longest alternative in tokens) + 2; probability clip [1e-4, 1 - 1e-4];
strictness clipped to [-0.5, 1.5].

## Measured agreement with production

fp32 backbone, all 200 clips of the practice list (`bench/opt/clips.json`, 20 FLEURS train clips
per language, ten languages); evidence in three modes per clip (plain, insertion on, first word
wildcarded); calibration on the universal sound model (three heads), the word model and the
omission formula; feature table and universal / word log-odds computed end to end by the
reference from the posteriors:

| quantity | max abs difference, every language |
|---|---|
| L_can, llr_sub, llr_del, llr_ins | 0 |
| span start / end, alternative index, heard, free tokens | identical |
| peak_post | 5.96e-8 (one float32 unit; see above) |
| sound P(lenient), P(strict), 0-2 regression, their log-odds | 0 |
| word log-odds (lenient, strict) | 0 |
| omission probability, strictness blend | 0 |
| feature table (every column) | 5.96e-8, only in `peak_post` and the columns computed from it |
| universal sound log-odds with bias (lenient, strict), 0-2 regression; word log-odds from the reference's own table | 0 |

The tests assert these bounds on the golden vectors and on one further practice clip per language.

## Golden vectors

`tests/data/reference_golden/<lang>.json.gz` (ten files, 1.40 MB in total), written by
`scripts/make_reference_golden.py` from the two shortest practice clips of each language.
Each file is gzip-compressed UTF-8 JSON:

```
{"format": 1, "lang": "en", "tokens": [127 symbols],
 "clips": [{"id", "text", "T", "lp_scale": 256,
            "lp_q": base64 of T x 127 little-endian int16, row-major (lp = q / 256),
            "phones": [{"ipa", "alts"}], "word_phones": [[start, end]],
            "cases": [{"name", "frames": [t0, t1], "phones": [p0, p1], "insertion",
                       "wild_slots", "exact", "expected": evidence}],
            "sound_model": {"file", "rows": [[...]], "expected": [{head: value}]},
            "word_model":  {"file", "rows": [[...]], "expected": [{head: value}]},
            "omission":    {"file", "llr_del": [[...]], "expected": [...]},
            "dur", "target": {"words", "syllables", "phones"},
            "universal": {"expected": {"sound": {"z_len", "z_str", "reg"},
                                       "word": {"z_len", "z_str"}}}}]}
```

The stored posteriors are the real ones clipped below at -24 and rounded to 1/256 nat, so they
are exact in float32 and double; every expected value was computed on the stored values. The
cases per clip: `plain`, `insertion`, `wild_word0` (whole clip), `exact_crop` (first 80 frames and
5 sounds, exact pass), `fast_crop_vs_exact` (same crop, fast pass with insertion) and `too_short`
(6 frames, up to 12 sounds: the canonical lattice cannot fit). `evidence` has `L_can`,
`n_frames`, `frame_shift`, `free_tokens` ([token, start, end]), `llr_ins` (list or null) and
`slots` with `llr_sub`, `llr_del`, `start_frame`, `end_frame`, `alt_index`, `peak_post`, `heard`,
`n_frames`. Non-finite floats are the strings "NaN", "Infinity", "-Infinity". Calibration rows
are in the model's column order (meta `num`, then `cat`); they were built by production from the
stored posteriors. `universal` is `pipeline.universal_scores` on the `plain` evidence.

## Files a port loads

| file | format | size |
|---|---|---|
| `pronunciada/data/zipa_tokens.txt` | text, `<symbol> <id>` per line | 0.8 KB |
| `pronunciada/data/calib/sound_U.npz` (the universal sound model, `calib/config.json` key `universal_sound_model`) | zip of .npy arrays + JSON meta | 291 KB |
| `pronunciada/data/calib/word.npz` | same | 119 KB |
| `pronunciada/data/calib/omission.json` | JSON (`coef`, `clip`) | 8.3 KB |
| `pronunciada/data/calib/config.json` | JSON (per-language logit `bias`) | 3.9 KB |
| `pronunciada/data/calib/phone_dist.npz` | 127 x 127 float32 distances + token list | 13 KB |
| `pronunciada/data/native/<lang>.json` | JSON native statistics (count, mean, std per phone / position / class) | 93-227 KB |

The .npy and tree array layouts are described in `pronunciada/reference/npz.py` and
`pronunciada/reference/calibrate.py`.
