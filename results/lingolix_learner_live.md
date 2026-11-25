# Lingolix API measured by us, against Pronunciada

Sentence score: the `overall` Lingolix serves, otherwise the mean of its word scores (the convention of the Lingolix comparison on speechocean762); the per-item file records which applied.

- **heroico**: 600 answered, 0 rejected, 0 not sent, of 600 items selected (48 of 48 speakers answered); versions 2026-11 (600); received 2026-11-23T18:35:16Z to 2026-11-23T18:45:15Z (UTC).

Ledger (every attempt counted, also failures; per set): heroico: 601 attempts, 21.9611 minutes booked, HTTP statuses {'401': 1, '200': 600}

| set | metric | n | Lingolix [95% CI] | Pronunciada [95% CI] | Lingolix − Pronunciada [95% CI], p | Azure [95% CI] |
|---|---|---|---|---|---|---|
| Heroico / USMA | USMA, native vs learner, prompt-matched (primary) | 450 | 0.603 [0.505, 0.697] | 0.669 [0.504, 0.818] | -0.066 [-0.213, +0.100], p 0.394 | 0.581 [0.437, 0.714] |
| Heroico / USMA | USMA learners vs USMA + HEROICO natives, prompt-matched (secondary) | 600 | 0.541 [0.441, 0.634] | 0.618 [0.491, 0.735] | -0.077 [-0.186, +0.042], p 0.204 | 0.542 [0.431, 0.646] |

Intervals: 10,000 resamples of speakers with replacement (hit rates: 2,000), every system on the same resamples; p two-sided, paired. Only the items the vendor answered are compared; the Azure column comes from cached responses on the same items. Field mapping and assumptions: results/lingolix_learner_live.md.
