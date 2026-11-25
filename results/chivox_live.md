# Chivox API measured by us, against Pronunciada

**Audio: Chivox received MP3 (libmp3lame, 64 kbit/s CBR, 16 kHz mono, encoded locally by ffmpeg from the 16 kHz mono PCM16 WAV); Pronunciada and Azure scored the original audio.** Chivox reads inline audio as MP3 whatever is sent (live smoke test 2026-11-23: WAV timed out, MP3 was scored), so MP3 is the only working input; at 64 kbit/s the coding loss is negligible for speech.

- **speechocean762**: 299 answered, 0 rejected, 0 not sent, of 299 items selected (125 of 125 speakers answered); versions chivox-eval-1.0.0_0.0.80.2026.6.22.17:51:12_eng.snt.G4.N1.0.13 (299); received 2026-11-23T18:18:38Z to 2026-11-23T18:44:37Z (UTC).
  - **SAMPLE**: these rows rest on 299 of the 2500 clips of the set (a seeded, speaker-balanced pilot). Read the intervals, not the point values; they describe this sample, not the whole set.
- **ompal**: 299 answered, 0 rejected, 0 not sent, of 299 items selected (21 of 21 speakers answered); versions chivox-eval-1.0.0_0.0.80.2026.6.22.17:51:12_chn.snt.G4.A2.0.8 (299); received 2026-11-23T18:19:17Z to 2026-11-23T19:39:31Z (UTC).
  - **SAMPLE**: these rows rest on 299 of the 745 clips of the set (a seeded, speaker-balanced pilot). Read the intervals, not the point values; they describe this sample, not the whole set.

Ledger (every attempt counted, also failures; per set): session set-up (0 cost): 0 attempts, 0 points booked, HTTP statuses {'200': 2}; speechocean762: 303 attempts, 303 points booked, HTTP statuses {'200': 301}; ompal: 300 attempts, 300 points booked, HTTP statuses {'200': 299, 'network error': 1}

| set | metric | n | Chivox [95% CI] | Pronunciada [95% CI] | Chivox − Pronunciada [95% CI], p | Azure [95% CI] |
|---|---|---|---|---|---|---|
| speechocean762 test | sounds, Pearson | 5679 | — | 0.672 [0.585, 0.735] | — | 0.512 [0.431, 0.582] |
| speechocean762 test | words, Pearson | 1909 | 0.517 [0.425, 0.594] | 0.570 [0.472, 0.646] | -0.053 [-0.133, +0.025], p 0.183 | 0.566 [0.469, 0.647] |
| speechocean762 test | low words (expert accuracy <= 6), AUC | 1909 | 0.882 [0.851, 0.908] | 0.908 [0.880, 0.933] | -0.026 [-0.056, +0.003], p 0.087 | 0.907 [0.877, 0.933] |
| speechocean762 test | sentence accuracy, Pearson | 299 | 0.529 [0.436, 0.609] | 0.733 [0.649, 0.799] | -0.204 [-0.294, -0.112], p < 0.001 | 0.672 [0.569, 0.756] |
| speechocean762 test | sentence total, Pearson | 299 | 0.644 [0.556, 0.715] | 0.753 [0.674, 0.816] | -0.109 [-0.181, -0.038], p 0.004 | 0.802 [0.732, 0.856] |
| speechocean762 test | sentence fluency, Pearson | 299 | 0.647 [0.550, 0.732] | 0.822 [0.764, 0.866] | -0.175 [-0.248, -0.109], p < 0.001 | 0.706 [0.590, 0.787] |
| speechocean762 test | sentence prosodic, Pearson | 299 | 0.492 [0.372, 0.602] | 0.806 [0.744, 0.854] | -0.313 [-0.414, -0.218], p < 0.001 | 0.840 [0.786, 0.879] |
| speechocean762 test | sentence accuracy vs the vendor's secondary score, Pearson | 299 | 0.532 [0.434, 0.617] | — | — | — |
| OMPAL test | word_auc[all] primary: any error | 3022 | 0.690 [0.645, 0.734] | 0.805 [0.770, 0.840] | -0.115 [-0.155, -0.077], p < 0.001 | 0.695 [0.641, 0.749] |
| OMPAL test | word_auc[all] (a) tone-only error | 3022 | 0.660 [0.609, 0.714] | 0.802 [0.762, 0.844] | -0.142 [-0.187, -0.098], p < 0.001 | 0.678 [0.615, 0.739] |
| OMPAL test | word_auc[all] (a) tone-only error, diagnostic channel (tone) | 3022 | 0.765 [0.729, 0.799] | 0.874 [0.845, 0.902] | -0.109 [-0.137, -0.080], p < 0.001 | — |
| OMPAL test | word_auc[all] (b) consonant or vowel error | 3022 | 0.828 [0.751, 0.897] | 0.815 [0.759, 0.873] | +0.013 [-0.061, +0.069], p 0.627 | 0.773 [0.684, 0.854] |
| OMPAL test | word_auc[all] (b) consonant or vowel error, diagnostic channel (sounds) | 3022 | 0.834 [0.748, 0.905] | 0.867 [0.829, 0.905] | -0.032 [-0.117, +0.027], p 0.348 | — |
| OMPAL test | word_auc[learners] primary: any error | 2571 | 0.675 [0.633, 0.719] | 0.776 [0.752, 0.798] | -0.100 [-0.131, -0.069], p < 0.001 | 0.658 [0.614, 0.698] |
| OMPAL test | word_auc[learners] (a) tone-only error | 2571 | 0.644 [0.593, 0.697] | 0.773 [0.741, 0.804] | -0.128 [-0.164, -0.090], p < 0.001 | 0.639 [0.584, 0.691] |
| OMPAL test | word_auc[learners] (b) consonant or vowel error | 2571 | 0.819 [0.744, 0.889] | 0.789 [0.736, 0.845] | +0.030 [-0.039, +0.085], p 0.318 | 0.745 [0.658, 0.826] |
| OMPAL test | hit[primary: any error] FA on native correct units 5 % | 419 | 0.189 [0.080, 0.561] | 0.790 [0.655, 0.901] | -0.601 [-0.711, -0.310], p 0.001 | 0.656 [0.496, 0.840] |
| OMPAL test | hit[primary: any error] FA on native correct units 10 % | 419 | 0.366 [0.189, 0.637] | 0.871 [0.778, 0.941] | -0.505 [-0.629, -0.295], p 0.001 | 0.742 [0.633, 0.870] |
| OMPAL test | hit[(a) tone-only error] FA on native correct units 5 % | 345 | 0.122 [0.039, 0.506] | 0.788 [0.654, 0.907] | -0.667 [-0.764, -0.378], p 0.001 | 0.629 [0.464, 0.832] |
| OMPAL test | hit[(a) tone-only error] FA on native correct units 10 % | 345 | 0.302 [0.139, 0.589] | 0.875 [0.780, 0.946] | -0.573 [-0.691, -0.348], p 0.001 | 0.720 [0.596, 0.865] |
| OMPAL test | hit[(b) consonant or vowel error] FA on native correct units 5 % | 74 | 0.500 [0.252, 0.849] | 0.797 [0.634, 0.918] | -0.297 [-0.509, +0.010], p 0.069 | 0.780 [0.569, 0.922] |
| OMPAL test | hit[(b) consonant or vowel error] FA on native correct units 10 % | 74 | 0.663 [0.402, 0.896] | 0.851 [0.724, 0.958] | -0.189 [-0.406, +0.000], p 0.069 | 0.841 [0.705, 0.945] |
| OMPAL test | hit[primary: any error] FA on learner correct units 5 % | 419 | 0.123 [0.069, 0.175] | 0.123 [0.088, 0.146] | -0.000 [-0.057, +0.063], p 0.978 | 0.107 [0.082, 0.141] |
| OMPAL test | hit[primary: any error] FA on learner correct units 10 % | 419 | 0.208 [0.155, 0.281] | 0.237 [0.184, 0.296] | -0.029 [-0.091, +0.056], p 0.546 | 0.194 [0.155, 0.286] |
| OMPAL test | hit[(a) tone-only error] FA on learner correct units 5 % | 345 | 0.070 [0.032, 0.097] | 0.097 [0.035, 0.134] | -0.027 [-0.081, +0.038], p 0.382 | 0.060 [0.038, 0.092] |
| OMPAL test | hit[(a) tone-only error] FA on learner correct units 10 % | 345 | 0.142 [0.100, 0.224] | 0.206 [0.127, 0.287] | -0.064 [-0.129, +0.031], p 0.180 | 0.143 [0.099, 0.262] |
| OMPAL test | hit[(b) consonant or vowel error] FA on learner correct units 5 % | 74 | 0.370 [0.208, 0.557] | 0.247 [0.111, 0.387] | +0.123 [-0.032, +0.304], p 0.132 | 0.324 [0.208, 0.471] |
| OMPAL test | hit[(b) consonant or vowel error] FA on learner correct units 10 % | 74 | 0.514 [0.339, 0.686] | 0.378 [0.241, 0.528] | +0.135 [-0.031, +0.286], p 0.115 | 0.432 [0.289, 0.577] |
| OMPAL test | sentence[learners] accuracy pearson | 257 | 0.391 [0.175, 0.561] | 0.589 [0.432, 0.699] | -0.199 [-0.327, -0.056], p 0.004 | 0.214 [0.023, 0.416] |
| OMPAL test | sentence[learners] accuracy spearman | 257 | 0.391 [0.158, 0.579] | 0.585 [0.423, 0.704] | -0.195 [-0.324, -0.068], p 0.002 | 0.281 [0.082, 0.448] |
| OMPAL test | sentence[learners] fluency pearson | 257 | 0.620 [0.486, 0.717] | 0.558 [0.418, 0.677] | +0.062 [-0.041, +0.165], p 0.271 | 0.243 [0.075, 0.402] |
| OMPAL test | sentence[learners] fluency spearman | 257 | 0.622 [0.469, 0.736] | 0.589 [0.445, 0.714] | +0.033 [-0.067, +0.128], p 0.590 | 0.318 [0.136, 0.482] |
| OMPAL test | sentence[learners] prosody pearson | 257 | 0.392 [0.209, 0.544] | 0.523 [0.333, 0.674] | -0.131 [-0.293, +0.018], p 0.085 | 0.078 [-0.111, 0.272] |
| OMPAL test | sentence[learners] prosody spearman | 257 | 0.405 [0.199, 0.569] | 0.511 [0.309, 0.671] | -0.105 [-0.277, +0.027], p 0.119 | 0.142 [-0.065, 0.345] |
| OMPAL test | sentence[learners] accuracy, secondary pearson | 257 | 0.375 [0.142, 0.557] | 0.188 [0.010, 0.315] | +0.187 [-0.012, +0.419], p 0.067 | 0.007 [-0.163, 0.174] |
| OMPAL test | sentence[learners] accuracy, secondary spearman | 257 | 0.382 [0.128, 0.578] | 0.246 [0.073, 0.396] | +0.136 [-0.053, +0.311], p 0.164 | 0.046 [-0.132, 0.222] |

Intervals: 10,000 resamples of speakers with replacement (hit rates: 2,000), every system on the same resamples; p two-sided, paired. Only the items the vendor answered are compared; the Azure column comes from cached responses on the same items. Field mapping and assumptions: results/chivox_harness.md.
