"""TonePerfect pronunciation API measured by us, against Pronunciada (and Lingolix where measured).

    export TONEPERFECT_API_KEY=tp_live_...                              # environment only; tp_test_ refused
    python -m bench.third_party.toneperfect_live --plan                 # full run: requests and credits
    python -m bench.third_party.toneperfect_live --pilot --plan         # the 50-credit pilot
    python -m bench.third_party.toneperfect_live --pilot --yes --credits-cap 50
    python -m bench.third_party.toneperfect_live --yes --credits-cap N  # full run (N from --plan)
    python -m bench.third_party.toneperfect_live --report-only [--pilot]
    python -m bench.third_party.toneperfect_live --dry-run --limit 10   # whole pipeline on a local fake

Documentation (read 2026-11-23, public): https://api.toneperfect.app/docs, /terms/, /privacy/,
/changelog/, /benchmark/, /pricing. Reference adapter: vendors.toneperfect / toneperfect_words of
lingolix/pronunciation-benchmark (MIT, main f8750aa). Write-up: results/toneperfect_harness.md.

What is sent, and nothing else (live_core.check_datasets): speechocean762 TEST (2,500 clips,
Apache-2.0; prompt = scored words joined by spaces, the Lingolix all-2,500 convention); OMPAL TEST
(745 clips, CC BY 4.0; Simplified prompt, OpenCC t2s, zh-CN); the Heroico / USMA test selection of
the learner suite (600 clips; OpenSLR 39 states Apache-2.0; es-419); the Lingolix open benchmark
v2026.10 (test 1: 200 Common Voice clips per language for en fr es de ja ko, its WAV files byte for
byte, its `target` text; test 2: its 500 speechocean762 clips, the same request as the speechocean762
items, so they share the cache). Pronunciada's rule for the Lingolix sets is unchanged: their
held-out clips are scored, never inspected or tuned on.

Request: POST https://api.toneperfect.app/v1/assess, multipart `audio` (16 kHz mono PCM16 WAV
decoded locally, or the benchmark's WAV), `text`, `language` (locale); headers
`Authorization: Bearer <key>`, `Idempotency-Key: pronunciada-<request hash>` (the same request
always has the same key, so a retry is never charged twice), `TonePerfect-Version: 2` (pinned; the
default since 2026-11-17) and `TonePerfect-Fallback: off` (the backup engine answers in version 1
with an older model: refused, a 503 is retried instead). Full (not compact) body.

Billing: credits; 1 credit up to 15 s of decoded audio, 2 up to 30 s, +1 per further started 15 s,
at most 8 (120 s maximum); failed, unassessable and rejected requests are free (/docs, /pricing).
Every attempt reserves its credits against --credits-cap before it is sent.

FIELD MAPPING, fixed from the documentation before any response existed (0-100 display scale; null =
not measured):
    sentence total        <- scores.overall
    sentence accuracy     <- scores.pronunciation
    sentence completeness <- scores.integrity
    sentence fluency      <- scores.fluency   (documented "reserved and null with the current engine")
    sentence prosodic     <- scores.rhythm    (likewise null) ; OMPAL prosody row <- scores.tone (the
                             nearest documented sentence score; Azure's row uses PronScore)
    secondary             <- OMPAL: scores.overall vs teacher accuracy (the PronScore slot); none on
                             speechocean762 (overall is the total row)
    word                  <- words[k].score, reduced exactly as the Lingolix adapter does: a word with
                             score null and decision "partial" takes the inner score raw.words[k].score;
                             an omitted word scores 0 (decision "review")
    sound (en)            <- words[k].phones[j].score, phoneme = .expected ("experimental" rows)
    Mandarin units        <- syllables[i] for the i-th Han character: score (word row),
                             components.tone.score (tone channel), min(components.initial.score,
                             components.final.score) (sounds channel); if the syllable count differs
                             from the character count, words[k] located in the text with its score
    Heroico sentence      <- scores.overall
    error flag            <- own sets: decision in {review, uncertain} (the docs' headline rule;
                             content_mismatch_suspected comes with review); no_correction, partial,
                             unassessable, legacy_correction and any other value = no flag (the docs:
                             other values mean "no decision"). Lingolix-benchmark tables: the
                             adapter's convention, flagged unless decision is no_correction or partial.
    assessable: false     <- a final, free rejection (counted, not scored, never re-sent).

ASSUMPTIONS checked on every answer (`validate`; a breach stops the run after the answer is cached):
  T1  scores.overall is a number in [0, 100] when assessable; word scores are numbers in [0, 100] or null.
  T2  `words` follows the text; for non-Japanese / Korean / Mandarin text at least half of the
      sentence's words appear among words[].word.
  T3  response_version is 2 and model_version a string (both recorded; the cache key includes both).
  T4  test keys: a body with "test": true or the header X-TonePerfect-Test stops the run (fixed fixture).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys

from bench.third_party import live_core as C

NAME = "TonePerfect"
UNIT = "credit"
KEY_ENV = "TONEPERFECT_API_KEY"
CAP_FLAG = "--credits-cap"
WRITEUP = "toneperfect_harness.md"
LINGOLIX_SETS = tuple(f"lingolix_test1_{x}" for x in ("en", "fr", "es", "de", "ja", "ko")) + ("lingolix_test2",)
ALLOWED = ("speechocean762", "ompal", "heroico") + LINGOLIX_SETS
# 50 free credits: 12 + 12 + 12 + 6 x 2 = 48 one-credit clips (every clip of these sets is <= 15 s
# in practice; --plan shows the exact credits from the decoded durations; the cap enforces it)
PILOT = {"speechocean762": 12, "ompal": 12, "heroico": 12, **{f"lingolix_test1_{x}": 2 for x in
                                                              ("en", "fr", "es", "de", "ja", "ko")},
         "lingolix_test2": 0}
FULL_SIZE = {"speechocean762": 2500, "ompal": 745, "heroico": 600, "lingolix_test2": 500,
             **{f"lingolix_test1_{x}": 200 for x in ("en", "fr", "es", "de", "ja", "ko")}}
HOST = "api.toneperfect.app"
URL = f"https://{HOST}/v1/assess"
LOCALE = {"en": "en-US", "fr": "fr-FR", "es": "es-419", "de": "de-DE", "ko": "ko-KR", "ja": "ja-JP", "zh": "zh-CN"}
VERSION = "2"
OPTIONS = {"endpoint": URL, "encoding": "multipart/form-data", "fields": ["text", "language"], "file_field": "audio",
           "content_type": "audio/wav", "TonePerfect-Version": VERSION, "TonePerfect-Fallback": "off",
           "response": "full"}
MIN_S, MAX_S, MAX_BYTES = 0.15, 120.0, 8_000_000
PRICES = {"source": "https://api.toneperfect.app/pricing (read 2026-11-23)", "currency": "USD", "free_credits": 50,
          "plans": {"Starter": {"usd": 10, "credits": 2500}, "Builder": {"usd": 50, "credits": 15000},
                    "Scale": {"usd": 150, "credits": 50000}},
          "packs_list_price": {"2500": 10, "10000": 40, "25000": 100}, "overage": "none (402 insufficient_credits)"}
FLAG_DECISIONS = ("review", "uncertain")
MAPPING = {
    "sentence_total": "scores.overall", "sentence_accuracy": "scores.pronunciation",
    "sentence_completeness": "scores.integrity", "sentence_fluency": "scores.fluency (documented null)",
    "sentence_prosodic": "scores.rhythm (documented null); OMPAL prosody row: scores.tone",
    "secondary": {"ompal": "scores.overall vs teacher accuracy", "speechocean762": None},
    "word": "words[k].score; null + decision partial -> raw.words[k].score (Lingolix adapter); omitted -> 0",
    "sound_en": "words[k].phones[j].score (phoneme = expected)",
    "mandarin_units": "syllables[i]: score / components.tone.score / min(components.initial, components.final)",
    "heroico_sentence": "scores.overall",
    "flag_own_sets": "decision in {review, uncertain}",
    "flag_lingolix_tables": "decision not in {no_correction, partial} (vendors.toneperfect_words)",
    "unassessable": "assessable false -> rejected (free), counted, not scored",
    "scale": "0-100", "source": "https://api.toneperfect.app/docs, read 2026-11-23"}


def api_lang(item: C.Item) -> str:
    return LOCALE[item.lang]


def cost(seconds: float) -> float:
    """1 credit up to 15 s, 2 up to 30 s, +1 per further started 15 s, at most 8."""
    return float(min(8, max(1, math.ceil(seconds / 15.0 - 1e-9))))


def over_limit(item: C.Item, audio: bytes) -> str | None:
    if item.seconds < MIN_S or item.seconds > MAX_S:
        return f"{item.seconds:.2f} s outside 0.15-120 s"
    if len(audio) > MAX_BYTES:
        return "over 8 MB"
    if len(item.text) > (600 if item.lang == "zh" else 2000):
        return "text too long"
    return None


def endpoint_desc() -> str:
    return f"POST {URL} (TonePerfect-Version {VERSION})"


def price_note(credits: float) -> str:
    p = PRICES["plans"]
    return (f"Free: {PRICES['free_credits']} credits once per account. Plans (USD/month): " +
            ", ".join(f"{k} {v['usd']} for {v['credits']}" for k, v in p.items()) +
            f"; at plan rates this run is USD {credits * 150 / 50000:.2f} (Scale) to {credits * 10 / 2500:.2f} "
            f"(Starter) worth of credits. No overage: a short balance answers 402. Source: {PRICES['source']}.")


def check_key(key: str) -> str | None:
    if key.startswith("tp_test_"):
        return "TONEPERFECT_API_KEY is a test key (tp_test_): it returns a fixed fixture, not a score; use a tp_live_ key"
    if not key.startswith("tp_live_"):
        return "TONEPERFECT_API_KEY does not start with tp_live_"
    return None


def policy() -> C.Policy:
    return C.Policy(rps=0.5, timeout_s=200.0)


def real_transport():
    return C.RequestsTransport((HOST,))


def fake_transport():
    return FakeTransport()


def new_session() -> dict:
    return {}


def ensure_session(s, **kw) -> None:
    return None


def session_info(s) -> dict:
    return {}


def send(s, item: C.Item, audio: bytes, h: str, *, transport, key: str, policy: C.Policy) -> C.Response:
    headers = {"Authorization": f"Bearer {key}", "Idempotency-Key": f"pronunciada-{h[:48]}",
               "TonePerfect-Version": VERSION, "TonePerfect-Fallback": "off", "Accept": "application/json"}
    return transport.post(URL, headers=headers, data={"text": item.text, "language": api_lang(item)},
                          files={"audio": (f"{C.slug(item.id)}.wav", audio, "audio/wav")}, timeout=policy.timeout_s)


def ledger_extra(resp: C.Response) -> dict:
    h = {k.lower(): v for k, v in resp.headers.items()}
    out = {}
    if "x-toneperfect-credits-charged" in h:
        out["credits_charged"] = h["x-toneperfect-credits-charged"]
    if "idempotent-replayed" in h:
        out["replayed"] = h["idempotent-replayed"]
    return out


_FATAL_429 = ("free_capacity",)
_RETRY_429 = ("rate_limit", "account_busy", "uploads_in_progress")


def classify(resp: C.Response, s=None) -> tuple[str, str]:
    st = resp.status
    body = resp.json() if resp.body else None
    code = ((body or {}).get("error") or {}).get("code") if isinstance(body, dict) else None
    msg = f"{code or ''} {str(((body or {}).get('error') or {}).get('message', ''))[:200]}".strip() \
        if isinstance(body, dict) else resp.body[:200].decode("utf-8", "replace")
    test_hdr = any(k.lower() == "x-toneperfect-test" and str(v).lower() == "true" for k, v in resp.headers.items())
    if test_hdr or (isinstance(body, dict) and body.get("test") is True):
        return "fatal", "a test-key answer (fixed fixture, not a score); use a tp_live_ key"
    if st == 401:
        return "fatal", f"key rejected (401 {msg})"
    if st in (402, 403):
        return "fatal", f"billing / subscription ({st} {msg}); nothing more will be sent"
    if st == 429:
        if code in _FATAL_429:
            return "fatal", f"429 {msg}: today's shared free capacity is used up; rerun tomorrow"
        return "retry", f"429 {msg}"
    if st == 409:
        return ("retry", f"409 {msg}") if code == "request_pending" else ("fail", f"409 {msg}")
    if st in (408, 425) or st >= 500:
        return "retry", f"HTTP {st} {msg}"
    if st in (413, 415, 422):
        return "rejected", f"HTTP {st} {msg}"
    if st == 400:
        return "rejected", f"HTTP 400 {msg} (a proxy 400 has no JSON body: request over 8 MiB)"
    if not 200 <= st < 300:
        return "fail", f"HTTP {st} {msg}"
    if not isinstance(body, dict):
        return "fatal", "a 2xx answer that is not a JSON object; the adapter must be checked"
    if str(body.get("response_version")) != VERSION:
        return "retry", f"response_version {body.get('response_version')!r} (fallback?) instead of {VERSION}"
    if body.get("assessable") is False:
        return "rejected", f"assessable false: {body.get('reason')}"
    return "answered", "ok"


def version(resp: C.Response, s=None) -> str:
    b = resp.json() or {}
    return f"{b.get('model_version') or 'unknown'}__rv{b.get('response_version')}"


# ── reading an answer ──────────────────────────────────────────────────────────

def toneperfect_words(raw: dict) -> list[dict]:
    """vendors.toneperfect_words of lingolix/pronunciation-benchmark (MIT), verbatim in effect: a word
    with score null and decision "partial" takes the inner score raw.words[k].score; "partial" and
    "no_correction" are not error calls."""
    inner = (raw.get("raw") or {}).get("words") or []
    words = raw.get("words") or []
    out = []
    for k, w in enumerate(words):
        score = w.get("score")
        if score is None and len(inner) == len(words):
            score = inner[k].get("score")
        out.append({"text": w["word"], "score": (score or 0) / 100, "scored": score is not None,
                    "error_type": None if w.get("decision") in ("no_correction", "partial") else "Mispronunciation"})
    return out


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_body(b: dict, text: str, lang: str) -> None:
    def bad(a, msg):
        raise C.ResponseFormatError(f"TonePerfect answer does not match assumption {a}: {msg}. Check the adapter "
                                    "(bench/third_party/toneperfect_live.py) against the cached answer.")

    if b.get("test") is True:
        bad("T4", "a test-key fixture")
    if str(b.get("response_version")) != VERSION or not isinstance(b.get("model_version"), str):
        bad("T3", f"response_version {b.get('response_version')!r}, model_version {b.get('model_version')!r}")
    sc = b.get("scores") or {}
    if not _num(sc.get("overall")) or not 0 <= sc["overall"] <= 100:
        bad("T1", f"scores.overall = {sc.get('overall')!r}")
    words = b.get("words")
    if not isinstance(words, list):
        bad("T2", "no `words` list")
    for k, w in enumerate(words):
        if not isinstance(w, dict) or not isinstance(w.get("word"), str):
            bad("T2", f"words[{k}] has no `word` string")
        if w.get("score") is not None and (not _num(w["score"]) or not 0 <= w["score"] <= 100):
            bad("T1", f"words[{k}].score = {w['score']!r}")
    if lang not in ("ja", "ko", "zh") and text.strip():
        from bench.third_party import lingolix as L
        want = [L.key(t) for t in text.split() if L.key(t)]
        got = {L.key(w["word"]) for w in words}
        hit = sum(t in got for t in want)
        if want and hit < 0.5 * len(want):
            bad("T2", f"only {hit} of {len(want)} words of the sentence appear in `words`")


def validate(resp: C.Response, item: C.Item) -> None:
    validate_body(resp.json(), item.text, item.lang)


def body_of(rec: dict) -> dict:
    return json.loads(rec["body"])


def _comp(syl: dict, name: str):
    c = (syl.get("components") or {}).get(name) or {}
    return c.get("score") if _num(c.get("score")) else None


def view(rec: dict, item: C.Item) -> dict | None:
    b = body_of(rec)
    if b.get("assessable") is False:
        return None
    validate_body(b, item.text, item.lang)
    sc = b.get("scores") or {}
    red = toneperfect_words(b)
    words = []
    for w, r in zip(b.get("words") or [], red):
        ph = [(p.get("expected"), float(p["score"])) for p in w.get("phones") or []
              if isinstance(p, dict) and _num(p.get("score")) and p.get("dimension") is None and p.get("expected")]
        words.append({"text": r["text"], "score": 100.0 * r["score"], "phones": ph,
                      "flag": w.get("decision") in FLAG_DECISIONS, "decision": w.get("decision")})
    sent = {"total": sc.get("overall"), "accuracy": sc.get("pronunciation"), "completeness": sc.get("integrity"),
            "fluency": sc.get("fluency"), "prosodic": sc.get("rhythm") if item.lang != "zh" else sc.get("tone"),
            "secondary": sc.get("overall") if item.lang == "zh" else None}
    out = {"sentence": sent, "words": words}
    if item.lang == "zh":
        han = [i for i, ch in enumerate(item.text) if "一" <= ch <= "鿿"]
        syl = [s for s in b.get("syllables") or [] if isinstance(s, dict)]
        units = []
        if syl and len(syl) == len(han):
            for i, s in zip(han, syl):
                seg = [x for x in (_comp(s, "initial"), _comp(s, "final")) if x is not None]
                units.append((i, i + 1, s.get("score") if _num(s.get("score")) else None, _comp(s, "tone"),
                              min(seg) if seg else None))
        else:
            from bench.eval_ompal import word_char_spans
            for w, sp in zip(words, word_char_spans(item.text, [w["text"] for w in words])):
                if sp:
                    units.append((sp[0], sp[1], w["score"], None, None))
        out["zh_units"] = units
    return out


def record(rec: dict, item_id: str) -> dict:
    """A benchmark response record (run.py's shape) for the Lingolix-benchmark metrics."""
    b = body_of(rec)
    return {"id": item_id, "vendor": "toneperfect", "words": toneperfect_words(b), "raw": b,
            "latency_ms": rec.get("latency_ms")}


# ── Lingolix-benchmark tables ──────────────────────────────────────────────────

def extra_evaluation(A: dict, n_boot: int) -> dict:
    """Test 1 (word AUC, FA@80; v2026.10 rules, report.py's 1,000 utterance resamples, paired) and
    test 2 (word / sentence Pearson; 10,000 speaker resamples, paired) on the items TonePerfect
    answered, next to Pronunciada's cached answers and Lingolix's measured ones where they exist."""
    from bench.third_party import lingolix as L
    from bench.third_party import lingolix_live as LL

    out = {}
    have = [d for d in LINGOLIX_SETS if d in A and A[d]["views"]]
    if not have:
        return out
    T = L.tests()
    for d in have:
        test = "tier2" if d == "lingolix_test2" else "tier1_" + d.rsplit("_", 1)[1]
        recs_tp = A[d]["records"]
        man = [it for it in T[test] if it["id"] in recs_tp]
        pron = L.read_responses(test)
        try:
            lx, _ = LL.lingolix_records(LL.benchmark_jobs([test[6:] if test != "tier2" else "en"], [test]),
                                        LL.live_root(False))
            lx = lx.get(test, {})
        except Exception as e:                 # Lingolix's measured column is optional
            lx = {}
            out.setdefault("_notes", []).append(f"{d}: Lingolix measured answers not read ({type(e).__name__})")
        row = {"n_items": len(man), "toneperfect": L.metrics(man, recs_tp),
               "pronunciada": L.metrics(man, pron) if pron else None,
               "lingolix_measured": L.metrics([it for it in man if it["id"] in lx], lx) if lx else None}
        if pron:
            row["auc_delta"] = LL.paired_units(man, recs_tp, pron, L.roc_auc)
            row["fa80_delta"] = LL.paired_units(man, recs_tp, pron, lambda l, s: L.fpr_at_recall(l, s, 0.80))
        if test == "tier2" and pron:
            _, spk = L.tier2x_items()
            row["word_pearson_speakers"] = LL.speaker_bootstrap(man, recs_tp, pron, "word", spk, n=n_boot)
            row["sentence_pearson_speakers"] = LL.speaker_bootstrap(man, recs_tp, pron, "utt", spk, n=n_boot)
        out[f"lingolix:{d}"] = row
    return out


def extra_markdown(R: dict) -> list[str]:
    rows = [(k, v) for k, v in R.items() if k.startswith("lingolix:")]
    if not rows:
        return []
    f = C.fmt
    lines = ["## Lingolix open benchmark v2026.10 (v2026.10 rules)", "",
             "| test | metric | n | TonePerfect [95% CI] | Pronunciada [95% CI] | TonePerfect − Pronunciada [95% CI], p | "
             "Lingolix measured |", "|---|---|---|---|---|---|---|"]
    for k, v in rows:
        tp, pr, lx = v["toneperfect"], v.get("pronunciada") or {}, v.get("lingolix_measured") or {}
        name = k.split(":", 1)[1]
        if name != "lingolix_test2":
            for m, ci, dk in (("roc_auc", "roc_auc_ci", "auc_delta"), ("fpr_at_recall_80", "fpr_at_recall_80_ci",
                                                                       "fa80_delta")):
                d = v.get(dk) or {}
                dv = (tp.get(m, math.nan) - pr.get(m, math.nan)) if pr else math.nan
                lines.append(f"| {name} | {'word AUC' if m == 'roc_auc' else 'FA@80'} | {v['n_items']} | {f(tp.get(m))} "
                             f"[{f((tp.get(ci) or [None])[0])}, {f((tp.get(ci) or [None, None])[1])}] | {f(pr.get(m))} | "
                             f"{f(dv)} [{f((d.get('delta_ci') or [None])[0])}, {f((d.get('delta_ci') or [None, None])[1])}], "
                             f"p {f(d.get('p'))} | {f(lx.get(m))} |")
        else:
            for which, key in (("word Pearson", "word_pearson_speakers"), ("sentence Pearson", "sentence_pearson_speakers")):
                b = v.get(key) or {}
                fs = b.get("full_sample") or [math.nan, math.nan]
                ci_a, ci_b, dci = b.get("pronunciada_ci") or [None] * 2, b.get("lingolix_ci") or [None] * 2, \
                    b.get("delta_ci") or [None] * 2
                lxv = lx.get("pcc_word" if which.startswith("word") else "pcc_utterance")
                lines.append(f"| {name} | {which} | {v['n_items']} | {f(fs[0])} [{f(ci_a[0])}, {f(ci_a[1])}] | "
                             f"{f(fs[1])} [{f(ci_b[0])}, {f(ci_b[1])}] | {f(fs[0] - fs[1])} [{f(dci[0])}, {f(dci[1])}], "
                             f"p {f(b.get('p'))} | {f(lxv)} |")
    lines += ["", "Test 1: report.py's metrics and 1,000 utterance resamples (paired); TonePerfect's error flag is "
              "the benchmark adapter's (flagged unless decision is no_correction or partial). Test 2: 10,000 speaker "
              "resamples, paired. Lingolix measured: our own Lingolix answers (results/lingolix_live.md) on the same "
              "items, point values only."]
    return lines


# ── the local fake ─────────────────────────────────────────────────────────────

def fake_body(text: str, locale: str, audio_sha: str, seconds: float) -> dict:
    """A version 2 full body in the documented shape (/docs "Response structure"), random scores."""
    rng = random.Random(hashlib.sha256(f"{audio_sha}|{text}|{locale}".encode()).hexdigest())
    zh = locale == "zh-CN"
    toks = [c for c in text if "一" <= c <= "鿿"] if zh else re.findall(r"[\w']+", text)
    words, syls = [], []
    for i, t in enumerate(toks):
        s = rng.uniform(20, 60) if rng.random() < 0.12 else rng.uniform(70, 100)
        dec = "review" if s < 50 else ("uncertain" if s < 65 else "no_correction")
        w = {"word": t, "score": round(s, 1), "start_ms": 0, "end_ms": 0, "decision": None if zh else dec,
             "omitted": False, "phones": [], "p_error": round(1 - s / 100, 3)}
        if zh:
            tone = rng.randint(1, 4)
            comp = {k: {"expected": "x", "heard": "x", "agrees_with_prompt": s > 55, "score": round(min(100, s + rng.uniform(-10, 10)), 1),
                        "confidence": 0.9, "decision": "no_correction" if s > 55 else "review"} for k in ("initial", "final")}
            comp["tone"] = {"expected": tone, "heard": tone if s > 55 else tone % 4 + 1, "score": round(s, 1),
                            "confidence": 0.9, "decision": "no_correction" if s > 55 else "review"}
            syls.append({"index": i, "expected": {"pinyin": "x", "tone": tone}, "score": round(s, 1), "components": comp})
        else:
            w["expected_ipa"] = t.lower()
            w["phones"] = [{"expected": ch, "heard": ch, "score": round(min(100, max(0, s + rng.uniform(-15, 10))), 1),
                            "operation": "match", "diagnosis_status": "experimental"} for ch in t.lower()[:6]]
        words.append(w)
    m = sum(w["score"] for w in words) / len(words) if words else 0.0
    return {"id": f"fake-{audio_sha[:12]}", "language": locale[:2], "locale": locale, "task": "general",
            "model_version": "FAKE-dry-run+scoremap-5", "response_version": 2, "status": "scored", "assessable": True,
            "reason": None, "scores": {"overall": round(m, 1), "pronunciation": round(m - 1, 1),
                                       "integrity": 100, "fluency": None, "rhythm": None,
                                       **({"tone": round(m, 1), "initial": round(m, 1), "final": round(m, 1)} if zh else {})},
            "words": words, "syllables": syls, "duration_seconds": round(seconds, 2),
            "billing": {"credits": cost(seconds), "balance_credits": 0}}


class FakeTransport:
    def __init__(self, script: dict | None = None):
        self.calls: list[dict] = []
        self.script = dict(script or {})

    def post(self, url, *, headers, json_body=None, data=None, files=None, timeout=60.0) -> C.Response:
        self.calls.append({"url": url, "headers": dict(headers), "data": dict(data or {}),
                           "n_bytes": len(files["audio"][1])})
        n = len(self.calls)
        if n in self.script:
            return self.script[n]
        audio = files["audio"][1]
        b = fake_body(data["text"], data["language"], hashlib.sha256(audio).hexdigest(), C.wav_seconds(audio))
        return C.Response(200, {"Content-Type": "application/json", "TonePerfect-Version": VERSION,
                                "X-Request-Id": f"req_fake_{n}"}, json.dumps(b, ensure_ascii=False).encode())


def main(argv=None) -> int:
    return C.main(sys.modules[__name__], argv)


if __name__ == "__main__":
    sys.exit(main())
