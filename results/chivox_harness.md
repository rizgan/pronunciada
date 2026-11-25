# Chivox benchmark runner (built and tested in dry-run mode only)

`bench/third_party/chivox_live.py`, with the shared machinery in `bench/third_party/live_core.py` (sending, budget, cache, ledger) and `bench/third_party/live_eval.py` (the scoreboard metrics). Tests: `tests/test_chivox_live.py`, fake transport only. No request has been sent to Chivox. No account was created and no key was used.

## What the documentation says (read 2026-11-23, all public, no login)

Sources: https://api-portal.cloud.chivox.com/global, /docs, /pricing, /runtime, /solutions/function-calling, /faq, /products/*, /legal/terms and /legal/privacy (both updated 2026-06-01). Also the vendor's public repository https://github.com/chivox-developer/chivox-speech-eval-mcp (MIT for its docs and examples; last commit 2026-06-12), files `docs/api-reference.md` and `examples/quick-test/python/test_eval.py`.

| topic | documented |
|---|---|
| endpoint | MCP over Streamable HTTP, JSON-RPC 2.0: `POST https://mcp-global.cloud.chivox.com/` (`initialize`, `tools/list`, `tools/call`). Function-calling host `fc-global.cloud.chivox.com` (`GET /v1/functions`, `POST /v1/call`), but its request body is not documented, so the runner does not use it. Streaming WebSockets exist and are not used. |
| authentication | `Authorization: Bearer <api key>` on every request. One key, no secret and no signature on the self-serve (B2C) path. The repository also describes a B2B JWT scheme (`cvx_ak_` / `cvx_sk_`, HS256), which the portal does not offer to this account. |
| tools used | `en_sentence_eval` (English read-aloud sentence) and `cn_sentence_eval` (Mandarin sentence). Arguments: `ref_text`, plus one of `audio_base64` or `audio_url`. Optional: `accent` (1 British, 2 American, 3 neutral, default 3), `rank` (4 or 100, default 100), `voiced` (en, default 1 strict), `age_group` (zh, `child` or `adult`, default adult), `attachAudioUrl` (default 0). The live `tools/list` schema has no `age_group`; see the smoke test below. |
| Mandarin text | Only Chinese characters appear in the examples (`"今天天气真好"`, Simplified). No pinyin input is documented for `cn_sentence_eval`. |
| audio | "MP3, WAV, etc.", recommended 16 kHz mono; the FAQ lists mp3 / wav / m4a / ogg / aac / pcm. Max 50 MB per file. No maximum duration is published. |
| response | The scoring JSON is the text of `result.content[0]`. Fields: `overall` ("weighted roll-up"), `accuracy` ("phoneme / syllable correctness"), `pron` ("articulation quality"), `integrity` ("did the user read every word?"), `fluency.overall` / `.speed` / `.pause`, `details[i].char` / `.score`, `details[i].phone[j].phoneme` / `.score` / `.dp_type`, `phoneme_error {expected, actual}` (English), and `tone_ref` / `tone_detected` (Mandarin). Scale 0–100. No version field. |
| errors | 400 malformed; 401 missing or bad token; 403 "API Key invalid, disabled, or quota exhausted"; 404 session not found; 409 invalid state; a "structured 429" when a key's cap is reached (/runtime); JSON-RPC error `-32000` "Evaluation failed: …". |
| limits | /docs "technical guardrails": 3 parallel scoring workers, 10 pending jobs per key, 50 MB per upload, audio TTL 5 min. No rate limit (QPS) is published. |
| billing | Points: 1 point per word, phrase or sentence evaluation; 2 per paragraph; 0 for a failed call. Free: 600 points, valid 30 days, no card. Paid, per evaluation: Standard $0.0038, Advanced $0.0035 (+10 % bonus points), Flagship $0.0032 (+20 %), packs from $19.90 / $99.90 / $199.90. Points expire 30 days after they land (/pricing). The terms instead say "as stated in the console", so the two disagree. |
| retention | /docs: "Audio is processed, scored, and dropped. We retain the resulting JSON for 30 days." /runtime: "zero audio retention", streaming TTL 0 s. /docs Limits: a 500 MB "temporary audio cache" and a 5-minute audio TTL. Privacy policy: content is kept "as long as needed to provide the Service and configured retention settings". Processing in the United States "and other countries". Region-locked tenants (US, EU, SG) only on enterprise plans. |
| training | Privacy: "We do not use your evaluation content to train public models unless you opt in to a separate program disclosed in the console." Terms §7 grants a licence to process inputs and outputs "to provide and improve the Service". |
| benchmarking | Neither the terms nor the privacy policy restricts benchmarking or publishing comparisons. Terms §9: scores are "informational and not a guarantee of accuracy". Terms §4 forbids processing children's voice data without a lawful basis (none of the audio here is children's speech). |

The public documentation was enough to build the client: single Bearer key, endpoint, arguments, response fields and billing unit. It left two points open, the audio format and the exact answer wrapper. The live smoke test settled the first and part of the second.

## Live smoke test (2026-11-23, run by the coordinator, not by this runner)

- **Session.** `initialize` with the single Bearer key works; the server is `chivox-eval` 1.0.0.
- **Audio format: inline audio must be MP3.**
  - A 16 kHz mono PCM16 WAV in `audio_base64` got HTTP 200 with a tool error (`isError: true`) after about 30 s (once after 2 s): `评测失败: 请求错误: map[id:53000 msg:core is timeout]`. This happened on three attempts, including one with a plain 44-byte header and lower-case text.
  - The same clip as MP3 (mono, 16 kHz) was scored in 2 s.
  - The answer echoes `"audio": {"audioType": "mp3", "channel": 1, "sampleBytes": 2, "sampleRate": 16000}`: the server labels inline audio as MP3 whatever is sent.
- **Answer shape.** The answer is a JSON text in `result.content[0].text`.
  - Top-level keys: `app`, `audio`, `ccode`, `eof`, `request` and `result`.
  - `request` holds `coreType: "en.sent.score"`, `accent`, `attachAudioUrl`, `rank`, `refText`, `tokenId` and `voiced`.
  - `result` holds `accuracy`, `delaytime` and `details: [{accent, beginindex, char, dur, end, endindex, …}]`.
  - The full body was not saved, so the runner still checks the first real answer against the mapping below. On a mismatch it stops and lists the keys it received at the top level, in the scores object and in `details[0]`.
- **Input schemas (`tools/list`, free).**
  - `en_sentence_eval`: `ref_text` (required), `audio_base64`, `audio_url`, `accent` (1 British, 2 American, 3 either; default 3), `rank` (4 or 100; default 100), `attachAudioUrl` (default 0), `voiced` (0 lenient, 1 strict; default 1).
  - `cn_sentence_eval`: `ref_text` (required), `audio_base64`, `audio_url`, `rank`, `attachAudioUrl`.
  - There is **no `age_group`** and no audio-format argument. 17 tools exist; the runner uses only these two.
- **Ledger.** It holds 4 booked attempts for item 001220068: 3 timed out, and 1 was scored by the diagnostic but not cached. Chivox charges nothing for failed calls, so about 1 of the 600 free points is used.

**What changed in the runner after the smoke test:**

1. **MP3.** Audio is now sent as MP3, encoded locally and deterministically by ffmpeg from the 16 kHz mono PCM16 WAV:
   - Arguments: `ffmpeg -nostdin -hide_banner -loglevel error -f wav -i pipe:0 -map_metadata -1 -fflags +bitexact -flags:a +bitexact -ac 1 -ar 16000 -c:a libmp3lame -b:a 64k -write_xing 0 -id3v2_version 0 -f mp3 pipe:1`. That is libmp3lame, 64 kbit/s CBR, 16 kHz mono, no tags; the build used here is ffmpeg 7.1.1.
   - The request hash covers the WAV and the request options, which include these MP3 settings.
   - Every cached record stores the MP3's sha256 and size, the ffmpeg version and the arguments.
   - If ffmpeg is missing, the runner stops before sending.
   - **Chivox therefore received MP3 at 64 kbit/s while Pronunciada and Azure scored the original audio**, because MP3 is the only inline format Chivox reads. At 64 kbit/s the coding loss is negligible for speech. The report header says so.
2. **Arguments.** Only arguments in the live schema are sent, each at its documented default. English: `accent` 3, `rank` 100, `voiced` 1, `attachAudioUrl` 0. Mandarin: `rank` 100, `attachAudioUrl` 0. `age_group` was dropped.
3. **Schema check.** After `initialize`, a free `tools/list` call stores the two input schemas in the run log. The run stops before any audio if a required argument the runner does not send appears, or if one it sends has disappeared.
4. **Timeouts.** A `core is timeout` tool error is retried with a low limit: 2 attempts in all. If the first two items of a run both end that way, the run stops with a message saying the request format is wrong, not that the service is slow.
5. **Tool errors are not answers.** Nothing is cached for a tool error or a JSON-RPC error; the body is kept under `errors/` for diagnosis, and the item will be sent again by a later run. A scored answer is still cached the moment it arrives.

## Real answer layout (smoke test of 2026-11-23, structure only)

Read from the two answers the smoke test cached: one speechocean762 clip and one OMPAL clip. Only the structure was used (field names, types, nesting), never how the scores relate to the human labels. No raw response text is committed.

- **Envelope.** The tool's JSON text has the top-level keys `app`, `audio` (`audioType` "mp3", `channel`, `sampleBytes`, `sampleRate`), `ccode`, `eof`, `request` (the request as the engine saw it, with `coreType` `en.sent.score` / `cn.sent.raw`), `result`, `time`, `tokenId` and `uuid`. The scores are under `result`.
- **Engine.** `result.res` names the engine (English `eng.snt.G4.N1.0.13`, Mandarin `chn.snt.G4.A2.0.8`) and `result.version` gives its build. Both go into the version string and the cache file name, after the MCP server's `chivox-eval-1.0.0`.
- **English `result`.**
  - Sentence scores: `overall`, `accuracy`, `pron`, `integrity` (integers 0–100) and `fluency` {`overall`, `pause`, `speed`}.
  - `rhythm` {`overall`, `sense`, `stress`, `tone`}, which the docs don't mention.
  - Other fields: `info` {`clip`, `snr`, `tipId`, `trunc`, `volume`}, timing fields, `rank`, `res`, `version`.
  - `details` has **one entry per word**: `char` (the word in lower case), `score`, `beginindex` / `endindex` (character positions in the reference text), `start` / `end` / `dur`, `fluency`, and per-word `stressscore` / `stressref`, `tonescore` / `toneref`, `sensescore` / `senseref`, `pause_score` / `pause_ref`, `accent`, `indict`.
  - **There is no `phone` list in English answers.**
- **Mandarin `result`.**
  - Sentence scores: the same fields as English, plus `tone` (an integer sentence tone score), `phn`, and `chars` {`chn_input`}. There is no `rhythm`.
  - `details` has **one entry per character**: `char` (the syllable in pinyin), `chn_char` (the character), `score`, `overall`, `pron`, `phn`, `tone` (an integer), `tonescore`, `confidence` (5 integers), `group`, `beginindex` / `endindex`, timing fields, and `phone`.
  - `phone` holds the initial and the final as {`char`, `score`}. A syllable without an initial has a single entry.
  - The documented `tone_ref` / `tone_detected` do not appear.

A synthetic example in this layout (values made up), Mandarin, one character:

```json
{"result": {"overall": 72, "accuracy": 70, "pron": 74, "integrity": 100, "tone": 65,
            "fluency": {"overall": 80, "pause": 0, "speed": 100}, "res": "chn.snt.…", "version": "…",
            "details": [{"char": "hai", "chn_char": "孩", "score": 61, "tone": 2, "tonescore": 40,
                         "phone": [{"char": "h", "score": 50}, {"char": "ai", "score": 70}]}]}}
```

**How the parser reads it now.** The choices written down from the docs are kept; only how each field is read changed.

- **Words.** The Mandarin word unit is located by `chn_char`, left to right in the Simplified text sent. The text is converted one character for one character from the Traditional prompt, so the span is the OMPAL unit span by position, the same span matching `bench/eval_ompal.unit_table` uses for the other systems. Characters the service leaves out, such as punctuation, are skipped. The report counts entries, Han characters and placed entries; both real answers placed every entry (20 of 20 in the OMPAL clip).
- **Phones.** A phone is labelled `char` as well as the documented `phoneme`.
- **Documented choices the real layout makes impossible:**
  - **English sound level.** English answers have no phones, so **English has no sound row**. The vendor's word scores are not passed off as sound scores; the row shows "—" for Chivox.
  - **Mandarin tone channel.** `tone_ref` / `tone_detected` don't exist, so the tone channel is the syllable's **`tonescore`**, the per-syllable tone information the answers do carry. The integer `tone` field is kept in the cache and not used. The equality rule remains only as a fallback when `tonescore` is absent.
- **Prosody-like rows.** The docs listed none. The OMPAL prosody row now uses the sentence `tone` score, as asked. For speechocean762 I added `rhythm.overall`: it is the only rhythm / intonation score the English answers carry. I chose it from the field's name and structure before any score was read. Drop it if you prefer "—" there.
- **The two cached answers stay valid.** Their cache keys are unchanged, so `--limit 299 --plan` now shows 596 to send (598 minus the 2 cached), with 6 points in the ledger.

## Field mapping (fixed from the documentation before any response existed)

| level | speechocean762 row | OMPAL row | Chivox field |
|---|---|---|---|
| sentence total | total | — | `overall` |
| sentence accuracy | accuracy | accuracy | `accuracy` |
| sentence fluency | fluency | fluency | `fluency.overall` |
| sentence prosodic | prosodic | prosody | none documented; real answers: `rhythm.overall` (en), sentence `tone` (zh) |
| secondary | vs expert accuracy | vs teacher accuracy (the slot Azure's PronScore takes in results/ompal.md) | `pron` (en), `overall` (zh) |
| completeness | JSON only | JSON only | `integrity` |
| word | word Pearson, low-word AUC | word unit (one character) | `details[i].score` |
| sound | sound Pearson: none (real English answers have no phones) | sounds channel = min phone score of the unit (initial, final) | `details[i].phone[j].score` |
| tone | — | tone channel: `details[i].tonescore` (docs: 100 if `tone_ref` = `tone_detected`, else 0; those fields don't exist in real answers, so only a fallback) | `tonescore` |

The brief asked for sentence total mapped to overall and sentence accuracy mapped to pronunciation / accuracy. Chivox documents both `accuracy` ("phoneme / syllable correctness") and `pron` ("articulation quality"). Accuracy is mapped to `accuracy`, which is the closer definition of the experts' accuracy score. `pron` is reported as the secondary row.

**Alignment.**
- English words: the human words are lined up with `details[].char` by the Lingolix benchmark's word alignment (`lingolix.align`). A human word with no Chivox word gets 0, the lowest score, and is counted.
- English phones: each human phone (ARPAbet mapped to IPA) is aligned with that word's `phone[]` using `bench.eval_azure_so762.align`, the same alignment the Azure rows use. An unaligned phone takes its word's score and is counted.
- Mandarin: each `details[i].char` is located left to right in the Simplified prompt. Because the conversion is one character for one character, its span is the OMPAL unit span.

## What is sent

- **speechocean762 TEST**: all 2,500 clips (Apache-2.0). Prompt = the scored words joined by single spaces (`lingolix.tier2x_items`, the convention of the Lingolix all-2,500 run), so the human word labels line up. `en_sentence_eval` with `accent` 3, `rank` 100, `voiced` 1 and `attachAudioUrl` 0, all the documented defaults sent explicitly.
- **OMPAL TEST**: 745 clips including the 3 native speakers (CC BY 4.0; `bench.eval_ompal.records("test")`). `cn_sentence_eval` with `ref_text` in Simplified characters, `rank` 100 and `attachAudioUrl` 0. The conversion uses OpenCC t2s, the converter of the Azure zh-CN run; the corpus prompts are Traditional, and Chivox documents only Simplified examples.
- **Audio**: decoded locally to 16 kHz mono PCM16 WAV, then encoded locally to MP3 by ffmpeg (64 kbit/s CBR, 16 kHz mono, bit-exact, no tags; see the smoke-test section). It is sent inline as base64. The same file always gives the same WAV, and with the same ffmpeg build the same MP3.
- **Nothing else.** Every other set is refused, including L2-ARCTIC, EpaDB, Speech Accent Archive, Heroico, LeaP and Common Voice.

## Safety

- **Dry run is the default.** A live send needs `--yes`, `--points-cap N` and `CHIVOX_API_KEY`. Without all three it prints the plan and sends nothing.
- **The cap.** Every attempt books 1 point under a lock before it is sent, retries included. Points already in the ledger count (the cap is a ledger total), and the first refusal ends the run. `initialize` and `tools/list` are not evaluations; they are logged at 0 points.
- **Ledger**: `$PRONUNCIADA_CACHE/third_party/chivox_live/ledger.jsonl`. Each attempt line (UTC time, set, item id, request hash, cost, unit) is fsynced before sending. Each result line records the HTTP status.
- **Cache**: every scored answer is written atomically the moment it arrives, to `responses/<h[:2]>/<h>__<item>__<server name-version>.json`. Here `h` hashes the WAV, text, language and request options (MP3 settings included). An answered request is never sent again. Tool errors are never cached as answers.
- **Stopping**: the run stops at once on 401, 402, 403, any 429 without Retry-After, or a JSON-RPC or tool error that mentions quota, points, balance, credits or the key.
- **Retries**: at most 4 attempts with exponential backoff, for 5xx, 408 and "upstream", "capacity" or timeout messages only. `core is timeout` gets at most 2 attempts, and the run stops if the first two items both end that way. 404 (session expired) re-initializes and retries.
- **Pace**: requests go out at most one every 2 s, one at a time.
- **The key** is read only from `CHIVOX_API_KEY`. The transport refuses any host but `mcp-global.cloud.chivox.com`. Every file written is scrubbed for the key, and the Authorization header is never stored.

## Cost (from the real item lists)

| run | requests | points | USD at paid rates (Flagship to Standard) |
|---|---|---|---|
| smoke test (`--limit 1`) | 2 | 2 | — (free points) |
| pilot (`--pilot`: 300 speechocean762 + 300 OMPAL) | 600 | 600 | — (the 600 free points; 1 is already used, so run `--limit 299` = 598) |
| full (2,500 + 745) | 3,245 | 3,245 | $10.38 to $12.33 |

The pilot sample is seeded (seed 0) and speaker-balanced: one clip per speaker per round, in a seeded speaker order. It covers all 125 speechocean762 speakers and all 21 OMPAL speakers. A smaller `--limit` is a prefix of the same order, so the smoke test's 2 items count towards the pilot.

Because every retry also books a point, a pilot stops early if anything is retried. Chivox does not charge failed calls, so in that case raise the cap by the number of retries the ledger shows, without buying anything. After the free points, the full run needs about 2,650 more points: a Standard pack plus a top-up, or an Advanced pack.

**The cap is a ledger total.** The ledger already holds 4 attempts from the smoke test. About 599 free points remain, because only the diagnostic's one scored call was charged; check the balance in the console.

The full 300 + 300 pilot needs 600 points, 1 more than remains. Run 299 + 299 instead (`--limit 299`): that is 598 points, and it already includes the smoke test's 2 items, since a smaller `--limit` is a prefix of the same order. The report marks it as a sample.

## Commands

```
export CHIVOX_API_KEY=...        # the user's key; never in a file
python -m bench.third_party.chivox_live --plan                              # full run: requests, points, cached
python -m bench.third_party.chivox_live --limit 299 --plan                  # pilot plan (598 requests)
python -m bench.third_party.chivox_live --limit 1 --yes --points-cap 6      # smoke test: 1 English + 1 Mandarin (ledger 4 + 2)
python -m bench.third_party.chivox_live --limit 299 --yes --points-cap 603  # pilot: ledger 4 + 599 free points
python -m bench.third_party.chivox_live --limit 299 --report-only           # results/chivox_live.md + results/tables/chivox_live.json
python -m bench.third_party.chivox_live --yes --points-cap N                # full run, N = ledger total + 2,647 (resumes; cached items are skipped)
python -m bench.third_party.chivox_live --report-only                       # full report
python -m bench.third_party.chivox_live --dry-run --pilot                   # rehearsal against the local fake (no network)
```

Run all of them from the repository root with the project virtualenv.

## Evaluation (`--report-only`)

The report uses only the items Chivox answered.

- **Pronunciada.** On speechocean762, the scoreboard's own outputs (`bench.eval_english.so762_items` over the cached `api_run` pickle; sound model B, `word_reg`). On OMPAL, the latest `bench.eval_ompal` ours run (the report names the cache directory).
- **Azure.** Cached responses only: speechocean762 `aligned_test.jsonl` and OMPAL zh-TW.
- **speechocean762 metrics.** Pearson with the experts for sounds, words and the sentence (accuracy, total, fluency, prosodic). Low-word AUC: expert word accuracy ≤ 6 (build_tier2.py's "mispronounced") against the word score.
- **OMPAL metrics.** The rows of results/ompal.md: word AUC (any error, tone-only, consonant or vowel; all speakers and learners only), the tone and sounds channels, hit rate at 5 % and 10 % false alarms (native or learner thresholds), and learner sentence Pearson and Spearman.
- **Intervals.** 10,000 speaker resamples with every system on the same resamples, giving Chivox − Pronunciada with a 95 % interval and a two-sided p. The hit rates use 2,000 resamples.
- **Counts.** Answered, rejected and failed requests, ledger attempts and HTTP statuses.
- **Samples.** A pilot report says on each set that it is a sample of n of N clips and that the intervals describe that sample.
- **Output.** `results/chivox_live.md` and `results/tables/chivox_live.json`. A dry run writes under `DRYRUN_FAKE/` with a "DRY RUN, FAKE DATA" banner, never to `results/`.
- **Cost of the report.** About 90 s on one core for the pilot.

## Show the user before any audio is sent

1. **Training opt-in.** The privacy policy mentions "a separate program disclosed in the console" for training on evaluation content. Ask the user to check the console and leave it off (opt-in, so off by default).
2. **Retention.** Results JSON is kept 30 days per /docs; the privacy policy refers to "configured retention settings". Ask whether the console has a retention setting, and set it to the shortest.
3. **Region.** No region choice on self-serve plans; data may be processed in the US "and other countries". The audio is public research corpora (speechocean762 Apache-2.0, OMPAL CC BY 4.0), no user data.
4. **Points expiry.** The free 600 points expire 30 days after they land.
5. **Status of the service.** The /docs changelog says the service is "invite-only" and "ships from local builds", while /global offers self-serve sign-up. The user's account and key exist, so this is a note only.
6. **Key caps.** /runtime mentions scoped keys and per-key monthly caps. Setting a cap of 600 points on the key in the console gives a second, server-side limit.

Environment variable: **`CHIVOX_API_KEY`** (the single key the console issued). The documented self-serve authentication needs nothing else: no app id, secret, region or product id.
