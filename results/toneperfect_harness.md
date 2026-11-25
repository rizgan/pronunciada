# TonePerfect benchmark runner

> **Update, 25 November 2026.** The runner has since been used live: a 48-credit pilot on 23 November, then all 745 OMPAL test recordings and 1,725 speechocean762 test recordings (one Starter month). Results: `results/tables/learner_benchmark_sample_toneperfect_tables.md` and [learner_benchmark.md](learner_benchmark.md). The text below describes the runner as it was built.

`bench/third_party/toneperfect_live.py` uses the same shared machinery as the Chivox runner: `live_core.py` and `live_eval.py`. Tests are in `tests/test_toneperfect_live.py` and use a fake transport only. Nothing has been sent to TonePerfect, and no account or key was used.

## What the documentation says (read 2026-11-23, public)

Sources: https://api.toneperfect.app/docs, /terms/, /privacy/, /changelog/, /benchmark/ and /pricing. The reference adapter is `vendors.toneperfect` / `toneperfect_words` in lingolix/pronunciation-benchmark (MIT, `main` f8750aa), read from the local copy of that repository.

| topic | documented |
|---|---|
| endpoint | `POST https://api.toneperfect.app/v1/assess`, `multipart/form-data`. Fields: `audio` (file: WAV, MP3, M4A, Ogg or WebM; 0.15 to 120 s; at most 8,000,000 bytes), `text` (up to 2,000 characters; Mandarin up to 600), `language` (locale). Optional fields: `task` and `pinyin`. |
| locales | `zh-CN` (the default), `en-US`, `fr-FR`, `es-419`, `de-DE`, `it-IT`, `pt-BR`, `ru-RU`, `ko-KR`, `ja-JP` |
| authentication | `Authorization: Bearer <key>`. Live keys start `tp_live_`. Test keys (`tp_test_`) return "a fixed example result instead of scoring the audio", with `"test": true` in the body and `X-TonePerfect-Test: true` in the headers. |
| headers | `Idempotency-Key`: results are kept 24 h; the same key with the same body is replayed and never charged twice. `TonePerfect-Version` takes 1 or 2; 2 has been the default since 2026-11-17. `TonePerfect-Fallback: off` returns a 503 instead of the backup engine. |
| response (version 2, full body) | `model_version`, `response_version`, `assessable`, `reason`, `scores` {`overall`, `pronunciation`, `initial`, `final`, `tone` (zh), `integrity`, `fluency`, `rhythm`}, `words[]` {`word`, `score`, `decision`, `omitted`, `phones[]`, `p_error`, …}, `syllables[]` (zh: `score` and `components` {`initial`, `final`, `tone`}), `billing` {`credits`, `balance_credits`}, `raw`. Scores are on a 0–100 display scale; null means not measured. `fluency` and `rhythm` are "reserved and null in every language with the current engine". Japanese and Korean are scored at word level only. Mandarin has no word-level decision: decisions are given per initial, final and tone. |
| decisions | `review` (strong evidence of an error), `uncertain`, `no_correction`, `content_mismatch_suspected` (comes with `review`), `unassessable`, `partial`, `legacy_correction`. The docs say any other value means "no decision". |
| errors | 401 `invalid_api_key`. 402 `subscription_required` / `subscription_past_due` / `insufficient_credits` / `payment_balance_due`. 408 `upload_timeout`. 409 `idempotency_conflict` / `request_pending` / `idempotency_expired`. 413, 415, 422 (incl. `invalid_audio` and `invalid_assessment`). 429 `rate_limit` / `account_busy` / `free_capacity` / `uploads_in_progress`. 503 `gateway_busy` / `engine_unavailable`. 504 `result_held`. Silent or unusable audio returns 200 with `assessable: false`, free of charge. |
| limits | 60 assessments per minute per account. 2 assessments in progress at a time. A 26 s proxy limit, after which the result is held and fetched with the same idempotency key. A shared daily capacity of 1,500 free credits across all accounts. |
| billing | Credits: 1 for up to 15 s of decoded audio, 2 for up to 30 s, then +1 per further started 15 s, at most 8. Failed, unassessable and rejected requests are free. 50 free credits once per account. Plans per month: Starter $10 for 2,500 credits, Builder $50 for 15,000, Scale $150 for 50,000. Packs ($10 for 2,500 at list price) need an active plan. There is no overage: when credits run out the API answers 402. |
| retention | Terms: "API audio is not stored and is not used to train models." Privacy: audio is scored in memory and discarded when the response is sent; results are kept 24 h under the idempotency key and then deleted; usage records (time, locale, duration, credits, status) are kept for billing; refused requests are kept 90 days without audio, text or key. Processing happens with Hetzner (Nuremberg and Helsinki), and requests pass through Netlify (global CDN, USA) in transit. |
| consent | **Terms §2: recordings must be "made with their consent"** (the speakers'), with parental or school consent for children where the law requires it, and no identification of people or inference of their origin. The owner has to decide whether this condition is met for public research corpora recorded under their own consent procedures (speechocean762, OMPAL, Heroico / USMA, Common Voice clips in the Lingolix benchmark). |
| benchmarking | No clause restricts benchmarking or publishing comparisons. Terms §3: results are "not a certification". The vendor's own benchmark page reports results on Lingolix test 1 (odd-numbered clips; it says the even-numbered ones were used during development) and on 100 OMPAL utterances ("we have used other OMPAL recordings during development"). |

The public documentation is enough to build the runner: endpoint, fields, headers, schema, errors and the billing rule are all documented. The docs do not publish the test-key fixture body; the runner refuses test keys and test answers, so it does not need it.

## Field mapping (fixed from the documentation before any response existed)

| level | field | notes |
|---|---|---|
| sentence total | `scores.overall` | speechocean762 total; the Heroico sentence score |
| sentence accuracy | `scores.pronunciation` | speechocean762 and OMPAL accuracy rows |
| completeness | `scores.integrity` | kept in the JSON only |
| sentence fluency | `scores.fluency` | documented null, so the row shows "—" |
| sentence prosodic | `scores.rhythm` (en); `scores.tone` (OMPAL prosody row) | rhythm is documented null. Tone is the nearest documented Mandarin sentence score; Azure's row uses PronScore |
| secondary | `scores.overall` vs teacher accuracy (OMPAL) | the slot Azure's PronScore takes in results/ompal.md |
| word | `words[k].score` | reduced as the Lingolix adapter does (see below) |
| sound (en) | `words[k].phones[j].score`, phoneme = `.expected` | the "experimental" rows, aligned per word to the human ARPAbet phones |
| Mandarin unit | `syllables[i]` ↔ the i-th Han character | word row = `score`; tone channel = `components.tone.score`; sounds channel = min(`components.initial.score`, `components.final.score`). If the syllable count differs from the character count, `words[k]` is located in the text instead |
| error flag, own sets | `decision` ∈ {`review`, `uncertain`} | the docs' headline rule. `no_correction`, `partial`, `unassessable`, `legacy_correction` and any other value mean no flag |
| error flag, Lingolix tables | flagged unless `decision` ∈ {`no_correction`, `partial`} | the benchmark adapter's convention, kept as instructed |
| unassessable | `assessable: false` | a final, free rejection: counted, not scored, never re-sent |

Word reduction (from `vendors.toneperfect_words`): a word with a null score and decision `partial` has a phonetic symbol the engine does not cover. It takes the score the inner scorer computed anyway (`raw.words[k].score`, same order), and `partial` is not counted as an error call. An omitted word scores 0 (decision `review`).

The flag mapping is a choice, written down before any response. The vendor's benchmark page counts a word as flagged "unless its `decision` is `no_correction` or `partial`", which is the adapter's rule, while its headline figures use "review or uncertain".

Version: `TonePerfect-Version: 2` is sent explicitly, and `TonePerfect-Fallback: off`, so a backup-engine answer (version 1, an older model) never enters the tables; the 503 that comes back instead is retried. A version 2 answer whose `response_version` is not 2 is retried, not cached. The cache file name carries `model_version` and `response_version`, for example `…__mandarin-human-v2-…+scoremap-5__rv2.json`.

## What is sent

Every allowed set is sent, and nothing else; any other set is refused.

- **speechocean762 TEST**: 2,500 clips. Prompt = the scored words joined by spaces (the Lingolix all-2,500 convention). Locale `en-US`.
- **OMPAL TEST**: 745 clips, Simplified prompt (OpenCC t2s), `zh-CN`.
- **Heroico / USMA test selection**: 600 clips, `es-419`. A small pilot is ordered prompt by prompt, alternating USMA natives and learners, so the prompt-matched metric still has pairs.
- **Lingolix benchmark v2026.10, test 1**: 200 clips per language for en, fr, es, de, ja and ko. The WAV files of the benchmark release are sent byte for byte, with the manifest's `target` text.
- **Lingolix benchmark v2026.10, test 2**: its 500 speechocean762 clips. This is the identical request to the speechocean762 items, so they are sent once and share the cache.

Pronunciada's rule for the Lingolix sets is unchanged: their held-out clips are scored, never inspected or tuned on.

## Cost (credits from the real decoded durations)

| set | clips | clips > 15 s | credits, full | default pilot | USD for the full set at plan rates (Scale to Starter) |
|---|---|---|---|---|---|
| speechocean762 | 2,500 | 0 | 2,500 | 12 | $7.50 to $10.00 |
| OMPAL | 745 | 9 | 754 | 12 | $2.26 to $3.02 |
| Heroico / USMA | 600 | 0 | 600 | 12 | $1.80 to $2.40 |
| Lingolix test 1 (6 × 200) | 1,200 | 0 | 1,200 (200 per language) | 12 (2 per language) | $3.60 to $4.80 |
| Lingolix test 2 | 500 | 0 | 0 extra (part of speechocean762) | 0 | — |
| **all** | 5,045 requests | 9 | **5,054** | **48** | **$15.16 to $20.22** |

The default pilot uses 48 credits and fits the account's 50 free credits. A credit cap of 50 leaves room for 2 retries; a retry with the same idempotency key is not charged twice by TonePerfect, but the runner books it anyway.

After the 50 free credits, a full run needs 5,004 credits. A Builder month ($50, 15,000 credits) covers it. Starter ($10, 2,500) plus one $10 pack gives 5,000 + 50 = 5,050, which is 4 credits short, so plan for Starter plus two packs ($30), or leave out a few OMPAL clips longer than 15 s. The shared daily cap of 1,500 free credits applies only to free credits (429 `free_capacity`, which stops the run; rerun the next day).

## Commands

```
export TONEPERFECT_API_KEY=tp_live_...     # never in a file; tp_test_ keys are refused
python -m bench.third_party.toneperfect_live --plan                         # full run: requests, credits, cached
python -m bench.third_party.toneperfect_live --pilot --plan                 # the 48-credit pilot
python -m bench.third_party.toneperfect_live --pilot --yes --credits-cap 50 # pilot
python -m bench.third_party.toneperfect_live --pilot --report-only          # results/toneperfect_live.md + tables/toneperfect_live.json
python -m bench.third_party.toneperfect_live --yes --credits-cap 5054       # full run (resumes; cached items skipped)
python -m bench.third_party.toneperfect_live --datasets lingolix_test1_en lingolix_test1_fr ... --yes --credits-cap N   # one set at a time
python -m bench.third_party.toneperfect_live --report-only                  # full report
python -m bench.third_party.toneperfect_live --dry-run --pilot              # rehearsal against the local fake
```

## Evaluation (`--report-only`)

- **speechocean762, OMPAL and Heroico.** The same tables as the Chivox report (see results/chivox_harness.md). Heroico uses the learner suite's prompt-matched concordance of native vs learner with the sentence score: Pronunciada `scores["total"]`, Azure PronScore, TonePerfect `scores.overall`. USMA only is the primary row; adding the HEROICO natives is secondary. All of these use 10,000 speaker resamples, paired.
- **Lingolix benchmark tables.** Test 1: word AUC and FA@80 under the v2026.10 rules, with report.py's 1,000 utterance resamples, paired against Pronunciada's cached answers (`lingolix/responses/pronunciada/`). Test 2: word and sentence Pearson with 10,000 speaker resamples. Next to them, our own measured Lingolix answers on the same items (point values) where they are cached. These rows can replace the TonePerfect column of results/industry_snapshot.md, which today shows figures Lingolix published for an older TonePerfect engine. That file is not edited here.
- **Reporting.** Only answered items are compared. Rejected and failed requests are counted. A pilot report marks each set as a sample.

## Safety

The safety design is the same as the Chivox runner's:

- Dry run is the default. A live send needs `--yes`, `--credits-cap N` and a `tp_live_` key in `TONEPERFECT_API_KEY`.
- Each attempt reserves its credits, computed from the decoded duration by the documented rule, under a lock before it is sent.
- A fsynced ledger is kept at `$PRONUNCIADA_CACHE/third_party/toneperfect_live/ledger.jsonl`, including the `X-TonePerfect-Credits-Charged` header when it is present.
- Every answer is cached the moment it arrives and is never re-sent.
- The run stops at once on 401, 402, 403, 429 `free_capacity` and test-key answers.
- Retries are bounded, at most 4 attempts with backoff and Retry-After, and happen only for 5xx, 504 `result_held`, 429 `rate_limit` / `account_busy` / `uploads_in_progress`, 409 `request_pending` and 408. Requests go out one at a time, at most one every 2 s.
- The idempotency key is derived from the request hash, so a retry is always the same request.
- Recordings outside 0.15–120 s or over 8 MB are not sent. There are none in these sets.

## For the owner to read or switch in the console

1. **Speaker consent (Terms §2).** Decide whether the corpora's own consent and licences (Apache-2.0, CC BY 4.0, OpenSLR 39, CC0 Common Voice) satisfy "made with their consent" for uploading to a third-party API.
2. **No training switch is needed.** The terms say audio is not stored or used for training. Results are deleted after 24 h. Account data export and deletion are in the dashboard.
3. **Plan.** The pilot fits the free credits. The full run needs a paid plan; Builder covers it in one month. Credits bought on a plan can only be spent while a plan is paid.
4. **Keys.** Create a `tp_live_` key. A `tp_test_` key makes the runner refuse to start.
5. **Data path.** Requests pass through Netlify's CDN (USA) in transit and are scored in Germany and Finland.

Environment variable: **`TONEPERFECT_API_KEY`**.
