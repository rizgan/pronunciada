"""Chivox pronunciation assessment (global developer portal) measured by us, against Pronunciada.

    export CHIVOX_API_KEY=...                                          # read from the environment only
    python -m bench.third_party.chivox_live --plan                     # what a full run sends and costs
    python -m bench.third_party.chivox_live --pilot --plan             # the 300 + 300 pilot
    python -m bench.third_party.chivox_live --limit 1 --yes --points-cap N      # smoke test (N = ledger + 2)
    python -m bench.third_party.chivox_live --pilot --yes --points-cap N        # the pilot (N = ledger + 600)
    python -m bench.third_party.chivox_live --yes --points-cap N                # full run (N = ledger + 3245)
    python -m bench.third_party.chivox_live --report-only [--pilot]    # results/chivox_live.md from the cache
    python -m bench.third_party.chivox_live --dry-run --limit 20       # the whole pipeline on a local fake

Documentation (read 2026-11-23, all public, no login): https://api-portal.cloud.chivox.com/global,
/docs, /pricing, /runtime, /legal/terms, /legal/privacy, and the vendor's MIT repository
github.com/chivox-developer/chivox-speech-eval-mcp (docs/api-reference.md, examples/quick-test/python).
Write-up: results/chivox_harness.md.

What is sent. Only speechocean762 TEST (2,500 clips, Apache-2.0; prompt = the scored words joined by
single spaces, the Lingolix all-2,500 convention, bench.third_party.lingolix.tier2x_items) and the
OMPAL TEST half (745 clips incl. natives, CC BY 4.0; bench.eval_ompal.records("test")). Any other set
is refused (live_core.check_datasets). Audio: decoded locally to 16 kHz mono 16-bit PCM WAV, then
encoded locally by ffmpeg to MP3 (libmp3lame, 64 kbit/s CBR, 16 kHz mono, bit-exact flags, no tags;
FFMPEG_ARGS) and sent inline as `audio_base64`. Live smoke test 2026-11-23: the server labels inline
audio as MP3 whatever is sent (the answer echoes audioType "mp3"); WAV got the tool error "core is
timeout" three times, the same clip as MP3 was scored in 2 s. So Chivox receives MP3 while
Pronunciada and Azure scored the original audio. The request hash covers the WAV and OPTIONS (which
name the MP3 settings); each cached record holds the MP3's sha256, the ffmpeg version and arguments.

Request (MCP over Streamable HTTP, JSON-RPC 2.0; docs/api-reference.md §4):
    POST https://mcp-global.cloud.chivox.com/      Authorization: Bearer $CHIVOX_API_KEY
    1. initialize (once per session; 0 points; the server's name and version are recorded; live:
       chivox-eval 1.0.0)
    2. notifications/initialized
    3. tools/list (free): the two input schemas go into the run log; the run stops before any audio
       if a required argument we do not send appears or one we send has disappeared
    4. per clip: tools/call {"name": "en_sentence_eval" | "cn_sentence_eval",
                             "arguments": {"ref_text", "audio_base64", ...}}
       Only arguments of the live schema, each at its documented default, sent explicitly:
       English: accent 3 (British or American), rank 100, voiced 1 (strict), attachAudioUrl 0.
       Mandarin: rank 100, attachAudioUrl 0; ref_text = the OMPAL prompt in Simplified characters
       (OpenCC t2s, the converter the Azure zh-CN run used; one character for one character).
    The evaluation JSON is the text of result.content[0] (examples/quick-test/python/test_eval.py).

Billing: points; 1 point per word / phrase / sentence evaluation, 2 per paragraph, 0 for a failed call
(/pricing). Both tools used here are sentence evaluations: 1 point per request. The cap counts 1 point
for EVERY attempt (retries included), so the ledger is an upper bound.

FIELD MAPPING, fixed from the documentation before any response existed (docs "Response fields"):
    sentence total        <- overall            ("Weighted roll-up")
    sentence accuracy     <- accuracy           ("Phoneme / syllable correctness")
    sentence fluency      <- fluency.overall
    sentence completeness <- integrity          (reported in the JSON only; no table row uses it)
    sentence prosodic     <- none documented. Real answers (2026-11-23) carry one: Mandarin `tone` (the
                             sentence tone score, the prosody-like row asked for) and English
                             `rhythm.overall` (an object {overall, sense, stress, tone}, not in the docs)
    secondary             <- pron (speechocean762: vs expert accuracy), overall (OMPAL: vs teacher
                             accuracy, the slot Azure's PronScore takes in results/ompal.md)
    word                  <- details[i].score   (English: one entry per word, `char` = the word in lower
                             case; Mandarin: one per character, `char` = pinyin, `chn_char` = character)
    sound                 <- details[i].phone[j].score (Mandarin: initial and final, labelled `char`; real
                             English answers carry NO phones, so English has no sound row)
    tone (Mandarin)       <- details[i].tonescore (the documented tone_ref / tone_detected do not exist in
                             real answers; used only if tonescore is absent); the Mandarin "sounds"
                             channel = min phone score of the syllable (initial, final).
All scores are on 0-100 ("all 0-100 unless noted"). Layout of the real answers: results/chivox_harness.md.

ASSUMPTIONS the documentation does not settle; checked on every answer (`validate`); a real answer
that breaks one stops the run after it is cached, so fixing the adapter costs nothing:
  C1  The MCP endpoint above is the scoring path for audio_base64 (the function-calling host
      fc-global.cloud.chivox.com/v1/call has no documented body and is not used).
  C2  Inline audio is MP3 (settled live 2026-11-23, see above). A "core is timeout" tool error is
      retried once (2 attempts in all); if the first two items of a run both end that way the run
      stops: the request format is wrong. A tool error is never cached as an answer.
  C3  The scores sit at the top of the tool's JSON or under its `result` (live: top-level app, audio,
      ccode, eof, request, result; result has accuracy and details). A mismatch message lists the keys
      received. overall (and accuracy / pron / integrity when present) is a number in [0, 100]; fluency is a
      number or an object with `overall`.
  C4  details is a list of objects with `char` (string) and `score` (number in [0, 100]) in the order
      of the text; at least half of the reference words (English) / characters (Mandarin) appear.
  C5  details[i].phone, when present, is a list of objects with a label (`phoneme` in the docs, `char`
      in real answers) and `score` in [0, 100].
  C6' details[i].tonescore, when present, is a number in [0, 100].
  C6  No version field is documented; the version string is the MCP serverInfo name + version of
      the session, plus any top-level `version` / `engine_version` / `res` field of the result.
  C7  Quota / billing errors: HTTP 401, 402, 403 ("API Key invalid, disabled, or quota exhausted"),
      429 ("structured 429" when a key cap is reached; any 429 without Retry-After stops the run),
      or a JSON-RPC error whose message names quota / points / balance / credits. They stop the run.
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
import re
import sys

from bench.third_party import live_core as C

NAME = "Chivox"
UNIT = "point"
KEY_ENV = "CHIVOX_API_KEY"
CAP_FLAG = "--points-cap"
WRITEUP = "chivox_harness.md"
ALLOWED = ("speechocean762", "ompal")
PILOT = {"speechocean762": 300, "ompal": 300}            # 600 points = the free allowance
FULL_SIZE = {"speechocean762": 2500, "ompal": 745}
HOST = "mcp-global.cloud.chivox.com"
URL = f"https://{HOST}/"
PROTOCOL = "2025-03-26"
TOOLS = {"en": "en_sentence_eval", "zh": "cn_sentence_eval"}
# Every argument the tools/list schema offers besides the audio and the text, sent explicitly at its
# documented default so the configuration is on record (tools/list, seen live 2026-11-23).
ARGS = {"en": {"accent": 3, "rank": 100, "voiced": 1, "attachAudioUrl": 0}, "zh": {"rank": 100, "attachAudioUrl": 0}}
# Inline audio must be MP3: the live smoke test (2026-11-23) showed the server labels audio_base64 as
# MP3 whatever is sent (the answer echoes audioType "mp3"); a WAV timed out ("core is timeout") and
# the same clip as MP3 was scored in 2 s. Encoded locally from the 16 kHz mono PCM16 WAV.
MP3 = {"codec": "libmp3lame", "bitrate": "64k", "mode": "CBR", "sample_rate": 16000, "channels": 1}
FFMPEG_ARGS = ["-nostdin", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
               "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", "-ac", "1", "-ar", "16000",
               "-c:a", "libmp3lame", "-b:a", "64k", "-write_xing", "0", "-id3v2_version", "0", "-f", "mp3", "pipe:1"]
OPTIONS = {"endpoint": URL, "transport": "MCP Streamable HTTP, JSON-RPC 2.0 tools/call", "tools": TOOLS, "args": ARGS,
           "audio": "audio_base64 of an MP3 encoded locally by ffmpeg from the 16 kHz mono PCM16 WAV",
           "mp3": MP3, "ffmpeg_args": FFMPEG_ARGS, "zh_text": "Simplified (OpenCC t2s)"}
REPORT_NOTE = ("**Audio: Chivox received MP3 (libmp3lame, 64 kbit/s CBR, 16 kHz mono, encoded locally by ffmpeg "
               "from the 16 kHz mono PCM16 WAV); Pronunciada and Azure scored the original audio.** Chivox reads "
               "inline audio as MP3 whatever is sent (live smoke test 2026-11-23: WAV timed out, MP3 was scored), "
               "so MP3 is the only working input; at 64 kbit/s the coding loss is negligible for speech.")
TIMEOUT_MSG = "core is timeout"            # the tool error a wrongly formatted request gets
TIMEOUT_ATTEMPTS = 2
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PRICES = {"source": "https://api-portal.cloud.chivox.com/pricing (read 2026-11-23)", "currency": "USD",
          "per_sentence": {"Standard": 0.0038, "Advanced": 0.0035, "Flagship": 0.0032},
          "free_points": 600, "free_valid_days": 30}
MAPPING = {
    "sentence_total": "overall", "sentence_accuracy": "accuracy", "sentence_fluency": "fluency.overall",
    "sentence_completeness": "integrity",
    "sentence_prosodic": {"speechocean762": "rhythm.overall (real answers; not in the docs)",
                          "ompal": "tone (sentence tone score)"},
    "secondary": {"speechocean762": "pron", "ompal": "overall"},
    "word": "details[i].score (en: one entry per word, `char` = word; zh: one per character, `chn_char`)",
    "sound": "details[i].phone[j].score (zh: initial and final; en: real answers carry no phones -> no sound row)",
    "tone": "details[i].tonescore (Mandarin, diagnostic tone channel); the docs' tone_ref == tone_detected "
            "is used only when tonescore is absent",
    "mandarin_sounds_channel": "min details[i].phone[j].score (initial, final)",
    "real_layout": "read from the two answers of the 2026-11-23 smoke test (structure only)",
    "scale": "0-100", "source": "https://api-portal.cloud.chivox.com/docs (Response fields), read 2026-11-23"}

_QUOTA = re.compile(r"quota|insufficient|\bpoints?\b|balance|credit|exhaust|billing|payment|cap (?:reached|exceeded)",
                    re.I)
_AUTH = re.compile(r"unauthori[sz]ed|api[ _-]?key|invalid key|forbidden|permission|jwt|token", re.I)
_TRANSIENT = re.compile(r"timeout|timed out|upstream|capacity|busy|temporar|unavailable|try again", re.I)
_SESSION = re.compile(r"session", re.I)


def api_lang(item: C.Item) -> str:
    return item.lang


def cost(seconds: float) -> float:
    return 1.0


def over_limit(item: C.Item, audio: bytes) -> str | None:
    return "over 50 MB" if len(audio) * 4 / 3 > MAX_UPLOAD_BYTES else None


def endpoint_desc() -> str:
    return f"POST {URL} (MCP tools/call {TOOLS['en']} / {TOOLS['zh']})"


def price_note(points: float) -> str:
    p = PRICES["per_sentence"]
    return (f"Free allowance: {PRICES['free_points']} points, valid {PRICES['free_valid_days']} days. At the published "
            f"per-sentence prices this run would cost USD {points * p['Flagship']:.2f}-{points * p['Standard']:.2f} "
            f"(Flagship-Standard, before any bonus points). Source: {PRICES['source']}.")


def check_key(key: str) -> str | None:
    return None if len(key) >= 8 else f"{KEY_ENV} looks too short to be a key"


def policy() -> C.Policy:
    return C.Policy(rps=0.5)


def real_transport():
    return C.RequestsTransport((HOST,))


def fake_transport():
    return FakeTransport()


# ── MCP plumbing ───────────────────────────────────────────────────────────────

def new_session() -> dict:
    return {}


def session_info(s: dict) -> dict:
    return {"server": s.get("server"), "protocol": s.get("protocol"), "has_session_id": bool(s.get("id")),
            "tool_schemas": s.get("tools")}


# ── MP3 encoding ───────────────────────────────────────────────────────────────

_MP3: dict = {}
_FFV: list = []


def ffmpeg_version() -> str:
    import shutil
    import subprocess

    if not _FFV:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise C.Fatal("ffmpeg is not installed (needed to send MP3 to Chivox: inline audio must be MP3); "
                          "nothing sent. Install it, e.g. `brew install ffmpeg`")
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=30).stdout
        _FFV.append(out.splitlines()[0] if out else "ffmpeg (version unknown)")
    return _FFV[0]


def encode_audio(item: C.Item, wav: bytes) -> bytes:
    """The bytes sent: the 16 kHz mono PCM16 WAV encoded to MP3 by ffmpeg (FFMPEG_ARGS, bit-exact flags,
    no tags), so the same WAV always gives the same MP3 with the same ffmpeg build."""
    import hashlib as _h
    import shutil
    import subprocess

    k = _h.sha256(wav).hexdigest()
    if k in _MP3:
        return _MP3[k]
    ffmpeg_version()
    r = subprocess.run([shutil.which("ffmpeg"), *FFMPEG_ARGS], input=wav, capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        raise C.Fatal(f"ffmpeg could not encode {item.id} to MP3: {r.stderr.decode('utf-8', 'replace')[:200]}")
    if len(_MP3) > 64:
        _MP3.clear()
    _MP3[k] = r.stdout
    return r.stdout


def record_extra(item: C.Item, sent: bytes) -> dict:
    return {"sent_audio": {"format": "mp3", "sha256": hashlib.sha256(sent).hexdigest(), "bytes": len(sent),
                           "ffmpeg": ffmpeg_version(), "ffmpeg_args": FFMPEG_ARGS, **MP3}}


def _headers(key: str, s: dict) -> dict:
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
         "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": PROTOCOL}
    if s.get("id"):
        h["Mcp-Session-Id"] = s["id"]
    return h


def rpc_messages(resp: C.Response) -> list[dict]:
    """JSON-RPC messages of a Streamable HTTP answer: a JSON body, or server-sent events."""
    ctype = next((v for k, v in resp.headers.items() if k.lower() == "content-type"), "") or ""
    text = resp.body.decode("utf-8", "replace")
    if "text/event-stream" in ctype or text.lstrip().startswith(("event:", "data:")):
        out, buf = [], []
        for line in text.splitlines() + [""]:
            if line.startswith("data:"):
                buf.append(line[5:].lstrip())
            elif not line.strip() and buf:
                try:
                    m = json.loads("\n".join(buf))
                    out += m if isinstance(m, list) else [m]
                except ValueError:
                    pass
                buf = []
        return [m for m in out if isinstance(m, dict)]
    m = resp.json()
    if isinstance(m, list):
        return [x for x in m if isinstance(x, dict)]
    return [m] if isinstance(m, dict) else []


def rpc_reply(resp: C.Response, want_id) -> dict | None:
    msgs = rpc_messages(resp)
    for m in msgs:
        if m.get("id") == want_id and ("result" in m or "error" in m):
            return m
    return next((m for m in msgs if "result" in m or "error" in m), None)


def ensure_session(s: dict, *, transport, key: str, budget: C.Budget, policy: C.Policy) -> None:
    if s.get("ready"):
        return
    body = {"jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                       "clientInfo": {"name": "pronunciada-bench", "version": "1"}}}
    if budget.ledger is not None:
        budget.ledger.append({"kind": "session", "utc": C.utc_now(), "vendor": NAME, "method": "initialize",
                              "cost": 0, "unit": UNIT})
    try:
        r = transport.post(URL, headers=_headers(key, {}), json_body=body, timeout=policy.timeout_s)
    except C.TransientError as e:
        raise C.Fatal(f"Chivox initialize failed ({e}); nothing scored, rerun later") from None
    if budget.ledger is not None:
        budget.ledger.append({"kind": "session_result", "utc": C.utc_now(), "vendor": NAME, "status": r.status})
    if r.status in (401, 402, 403):
        raise C.Fatal(f"Chivox refused the key at initialize (HTTP {r.status}); check {KEY_ENV}")
    if not 200 <= r.status < 300:
        raise C.Fatal(f"Chivox initialize answered HTTP {r.status}; run stopped")
    m = rpc_reply(r, 0) or {}
    if "error" in m:
        raise C.Fatal(C.scrub(f"Chivox initialize error: {str(m['error'])[:200]}", key))
    res = m.get("result") or {}
    sid = next((v for k, v in r.headers.items() if k.lower() == "mcp-session-id"), None)
    info = res.get("serverInfo") or {}
    s.update(id=sid, server={"name": info.get("name"), "version": info.get("version")},
             protocol=res.get("protocolVersion"), next_id=1)
    try:
        transport.post(URL, headers=_headers(key, s), json_body={"jsonrpc": "2.0", "method": "notifications/initialized"},
                       timeout=policy.timeout_s)
    except C.TransientError:
        pass
    # tools/list (free): the live input schemas; stop before any audio if ours no longer fit
    if budget.ledger is not None:
        budget.ledger.append({"kind": "session", "utc": C.utc_now(), "vendor": NAME, "method": "tools/list",
                              "cost": 0, "unit": UNIT})
    try:
        r = transport.post(URL, headers=_headers(key, s), json_body={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                           timeout=policy.timeout_s)
    except C.TransientError as e:
        raise C.Fatal(f"Chivox tools/list failed ({e}); nothing sent, rerun later") from None
    if budget.ledger is not None:
        budget.ledger.append({"kind": "session_result", "utc": C.utc_now(), "vendor": NAME, "status": r.status})
    m = rpc_reply(r, 1) or {}
    tools = {t.get("name"): t.get("inputSchema") or {} for t in ((m.get("result") or {}).get("tools") or [])
             if isinstance(t, dict)}
    s["tools"] = {name: tools.get(name) for name in TOOLS.values()}
    problems = check_schemas(s["tools"])
    if problems:
        raise C.Fatal("Chivox tools/list does not fit the requests this runner sends, nothing sent: " + "; ".join(problems))
    s["ready"] = True


def sent_arguments(lang: str) -> set:
    return {"ref_text", "audio_base64", *ARGS[lang]}


def check_schemas(tools: dict) -> list[str]:
    """Problems: a tool missing, a required argument we do not send, an argument we send that the
    schema does not list."""
    out = []
    for lang, name in TOOLS.items():
        sc = tools.get(name)
        if not sc:
            out.append(f"tool {name} not listed")
            continue
        props = set((sc.get("properties") or {}))
        req = set(sc.get("required") or [])
        sent = sent_arguments(lang)
        if req - sent:
            out.append(f"{name} requires {sorted(req - sent)}, which this runner does not send")
        if sent - props:
            out.append(f"{name} no longer lists {sorted(sent - props)}")
    return out


def request_body(s: dict, item: C.Item, audio: bytes) -> dict:
    s["next_id"] = s.get("next_id", 1) + 1
    s["last_id"] = s["next_id"]
    args = {"ref_text": item.text, "audio_base64": base64.b64encode(audio).decode("ascii"), **ARGS[item.lang]}
    return {"jsonrpc": "2.0", "id": s["last_id"], "method": "tools/call",
            "params": {"name": TOOLS[item.lang], "arguments": args}}


def send(s: dict, item: C.Item, audio: bytes, h: str, *, transport, key: str, policy: C.Policy) -> C.Response:
    return transport.post(URL, headers=_headers(key, s), json_body=request_body(s, item, audio), timeout=policy.timeout_s)


def ledger_extra(resp: C.Response) -> dict:
    return {}


def _tool_text(m: dict) -> tuple[bool, str]:
    res = m.get("result") or {}
    parts = [c.get("text", "") for c in res.get("content") or [] if isinstance(c, dict) and c.get("type") == "text"]
    return bool(res.get("isError")), "\n".join(parts)


def _message_kind(msg: str) -> str:
    """A tool / JSON-RPC error is never an answer: nothing is cached for it (fail, retry or stop)."""
    if TIMEOUT_MSG in msg:
        return "retry"                      # at most TIMEOUT_ATTEMPTS attempts (attempt_limit)
    if _QUOTA.search(msg) or _AUTH.search(msg):
        return "fatal"
    if _TRANSIENT.search(msg):
        return "retry"
    if _SESSION.search(msg) and re.search(r"not found|expired|invalid", msg, re.I):
        return "session"
    return "fail"


def attempt_limit(detail) -> int:
    return TIMEOUT_ATTEMPTS if detail and TIMEOUT_MSG in str(detail) else 99


def after_item(s: dict, item: C.Item, outcome, detail) -> None:
    """Stop when the first two items of a run both end in "core is timeout": that is a wrong request
    format (seen live with WAV), not a slow service."""
    s.setdefault("first", [])
    if len(s["first"]) < 2:
        s["first"].append(outcome == "failed" and TIMEOUT_MSG in str(detail or ""))
        if len(s["first"]) == 2 and all(s["first"]):
            raise C.Fatal(f"the first two items both failed with '{TIMEOUT_MSG}' ({TIMEOUT_ATTEMPTS} attempts each): "
                          "the request format is wrong (audio not MP3?), not a slow service; run stopped")


def classify(resp: C.Response, s: dict | None = None) -> tuple[str, str]:
    """answered | rejected (final, cached) | retry | session (re-initialize, retry) | fatal | fail."""
    st = resp.status
    snippet = resp.body[:300].decode("utf-8", "replace")
    if st in (401, 402, 403):
        return "fatal", {401: "key missing or rejected (401)", 402: "payment required (402)",
                         403: "key invalid, disabled, quota exhausted, or no permission for the tool (403)"}[st]
    if st == 404:
        return "session", "session not found (404)"
    if st == 429:
        if C.retry_after_s(resp.headers) is not None and not _QUOTA.search(snippet):
            return "retry", "rate limited (429 with Retry-After)"
        return "fatal", "429: key cap or quota reached (structured 429, /runtime); nothing more will be sent"
    if st in (408, 425) or st >= 500:
        return "retry", f"HTTP {st}"
    if st == 400:
        return "fail", f"HTTP 400: {snippet}"
    if not 200 <= st < 300:
        return "fail", f"HTTP {st}: {snippet}"
    m = rpc_reply(resp, (s or {}).get("last_id"))
    if m is None:
        return "fatal", "a 2xx answer without a JSON-RPC result (C1); the adapter must be checked"
    if "error" in m:
        msg = json.dumps(m["error"], ensure_ascii=False)[:300]
        return _message_kind(msg), f"JSON-RPC error: {msg}"
    is_err, text = _tool_text(m)
    if is_err:
        return _message_kind(text), f"tool error: {text[:300]}"
    try:
        ev = json.loads(text)
    except ValueError:
        return _message_kind(text), f"tool answer is not JSON: {text[:200]}"
    if not isinstance(ev, dict):
        return "fail", "tool answer is not a JSON object"
    ev = _unwrap(ev)
    if "overall" not in ev and "details" not in ev:
        return _message_kind(text), f"tool answer without scores: {text[:200]}"
    return "answered", "ok"


def _unwrap(ev: dict) -> dict:
    """The Quickstart reads result.details: accept the evaluation either at the top or under `result`."""
    if "overall" not in ev and isinstance(ev.get("result"), dict):
        return ev["result"]
    return ev


def outer(rec_or_resp) -> dict:
    """The tool's JSON text as received (live: top-level app, audio, ccode, eof, request, result)."""
    if isinstance(rec_or_resp, C.Response):
        resp = rec_or_resp
    else:
        resp = C.Response(int(rec_or_resp.get("status") or 200), rec_or_resp.get("headers") or {},
                          rec_or_resp["body"].encode("utf-8"))
    m = rpc_reply(resp, None) or {}
    _, text = _tool_text(m)
    return json.loads(text)


def evaluation(rec_or_resp) -> dict:
    return _unwrap(outer(rec_or_resp))


def version(resp: C.Response, s: dict) -> str:
    srv = (s or {}).get("server") or {}
    v = f"{srv.get('name') or 'mcp'}-{srv.get('version') or 'unknown'}"
    try:
        o = outer(resp)
        ev = _unwrap(o)
    except (ValueError, TypeError):
        o, ev = {}, {}
    extra = [str(d[k]) for d in (o, ev) for k in ("version", "engine_version", "res")
             if isinstance(d.get(k), (str, int, float))]
    return "_".join([v] + extra)


# ── checking and reading an answer ─────────────────────────────────────────────

def _num01(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 100


def _fluency(ev):
    f = ev.get("fluency")
    if isinstance(f, dict):
        return f.get("overall")
    return f


def received_keys(ev: dict, top: dict | None) -> str:
    parts = []
    if isinstance(top, dict):
        parts.append(f"top level {sorted(top)}")
    if isinstance(ev, dict) and ev is not top:
        parts.append(f"scores object {sorted(ev)}")
    det = ev.get("details") if isinstance(ev, dict) else None
    if isinstance(det, list) and det and isinstance(det[0], dict):
        parts.append(f"details[0] {sorted(det[0])}")
    return "; ".join(parts)


def _is_han(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"


def _unit_text(d: dict, lang: str) -> str:
    """The text a details entry stands for. Real answers (2026-11-23): English `char` = the word in
    lower case; Mandarin `char` = the syllable in pinyin and `chn_char` = the character."""
    if lang == "zh" and isinstance(d.get("chn_char"), str):
        return d["chn_char"]
    return d["char"]


def _phone_label(p: dict):
    """Real Mandarin answers label a phone (initial or final) with `char`; the docs use `phoneme`."""
    return p.get("phoneme", p.get("char"))


def validate_eval(ev: dict, text: str, lang: str, top: dict | None = None) -> None:
    def bad(a, msg):
        raise C.ResponseFormatError(f"Chivox answer does not match assumption {a}: {msg}. Keys received: "
                                    f"{received_keys(ev, top)}. Check the adapter (bench/third_party/chivox_live.py) "
                                    "against the cached answer.")

    if not _num01(ev.get("overall")):
        bad("C3", f"`overall` = {ev.get('overall')!r}, expected a number in [0, 100]")
    for k in ("accuracy", "pron", "integrity"):
        if ev.get(k) is not None and not _num01(ev[k]):
            bad("C3", f"`{k}` = {ev[k]!r}")
    f = _fluency(ev)
    if f is not None and not _num01(f):
        bad("C3", f"fluency = {ev.get('fluency')!r}")
    pr = _prosodic(ev, lang)
    if pr is not None and not _num01(pr):
        bad("C3", f"prosody-like score ({'rhythm.overall' if lang == 'en' else 'tone'}) = {pr!r}")
    det = ev.get("details")
    if not isinstance(det, list) or (text.strip() and not det):
        bad("C4", "no `details` list")
    for i, d in enumerate(det):
        if not isinstance(d, dict) or not isinstance(d.get("char"), str) or not _num01(d.get("score")):
            bad("C4", f"details[{i}] = {str(d)[:120]}")
        if lang == "zh" and d.get("chn_char") is not None and not isinstance(d["chn_char"], str):
            bad("C4", f"details[{i}].chn_char = {d['chn_char']!r}")
        if d.get("tonescore") is not None and not _num01(d["tonescore"]):
            bad("C6'", f"details[{i}].tonescore = {d['tonescore']!r}, expected a number in [0, 100]")
        ph = d.get("phone")
        if ph is not None:
            if not isinstance(ph, list) or not all(isinstance(q, dict) and isinstance(_phone_label(q), str)
                                                   and _num01(q.get("score")) for q in ph):
                bad("C5", f"details[{i}].phone = {str(ph)[:120]}")
    if lang == "en":
        from bench.third_party import lingolix as L
        want = [L.key(t) for t in text.split() if L.key(t)]
        got = {L.key(d["char"]) for d in det}
        hit = sum(w in got for w in want)
        if want and hit < 0.5 * len(want):
            bad("C4", f"only {hit} of {len(want)} words of the sentence appear in `details`")
    else:
        han = [ch for ch in text if _is_han(ch)]
        got = "".join(_unit_text(d, "zh") for d in det)
        hit = sum(ch in got for ch in han)
        if han and hit < 0.5 * len(han):
            bad("C4", f"only {hit} of {len(han)} characters of the sentence appear in `details[].chn_char`")


def validate(resp: C.Response, item: C.Item) -> None:
    o = outer(resp)
    validate_eval(_unwrap(o), item.text, item.lang, o)


def _tone(x):
    m = re.search(r"[0-5]", str(x)) if x is not None else None
    return m.group() if m else None


def _prosodic(ev: dict, lang: str):
    """The prosody-like sentence score: Mandarin `tone` (sentence tone score); English `rhythm.overall`
    (not in the docs; present in real answers, an object {overall, sense, stress, tone})."""
    if lang == "zh":
        t = ev.get("tone")
        return t if isinstance(t, (int, float)) and not isinstance(t, bool) else None
    r = ev.get("rhythm")
    return r.get("overall") if isinstance(r, dict) else None


def zh_units(details: list, text: str) -> tuple[list, dict]:
    """(a, b, word score, tone score, sounds score) per details entry, as character spans of the text
    sent (Simplified, one character for one character of the OMPAL prompt, so the spans are the OMPAL
    unit spans). Each entry's chn_char is located left to right from the previous one, which skips
    whatever the service leaves out (punctuation); counts are returned for the report.
    Tone channel: the syllable's `tonescore` (real answers); the docs' tone_ref / tone_detected (equal ->
    100, else 0) only when no tonescore is given. Sounds channel: min score of the syllable's phones
    (real answers: initial and final)."""
    units, cur, cnt = [], 0, {"details": len(details), "han_in_text": sum(_is_han(c) for c in text), "placed": 0}
    for d in details:
        t = _unit_text(d, "zh").strip()
        j = text.find(t, cur) if t else -1
        if j < 0:
            continue
        if _num01(d.get("tonescore")):
            tone = float(d["tonescore"])
        else:
            tr, td = _tone(d.get("tone_ref")), _tone(d.get("tone_detected"))
            tone = None if tr is None or td is None else (100.0 if tr == td else 0.0)
        ph = [float(q["score"]) for q in d.get("phone") or []]
        units.append((j, j + len(t), float(d["score"]), tone, min(ph) if ph else None))
        cnt["placed"] += 1
        cur = j + len(t)
    return units, cnt


def view(rec: dict, item: C.Item) -> dict | None:
    o = outer(rec)
    ev = _unwrap(o)
    validate_eval(ev, item.text, item.lang, o)
    sent = {"total": ev.get("overall"), "accuracy": ev.get("accuracy"), "fluency": _fluency(ev),
            "prosodic": _prosodic(ev, item.lang), "completeness": ev.get("integrity"),
            "secondary": ev.get("pron") if item.lang == "en" else ev.get("overall")}
    words = [{"text": _unit_text(d, item.lang), "score": float(d["score"]),
              "phones": [(_phone_label(q), float(q["score"])) for q in d.get("phone") or []]} for d in ev["details"]]
    out = {"sentence": sent, "words": words}
    if item.lang == "zh":
        out["zh_units"], out["zh_alignment"] = zh_units(ev["details"], item.text)
    return out


# ── the local fake ─────────────────────────────────────────────────────────────

def fake_eval(text: str, lang: str, audio_sha: str) -> dict:
    """The `result` object in the layout of the real answers of 2026-11-23 (field names and nesting;
    random scores). English: one details entry per word (`char` = the word in lower case), no phones,
    `rhythm` object. Mandarin: one entry per character (`char` = pinyin, `chn_char` = character,
    `phone` = [initial, final] as {char, score}, `tone`, `tonescore`), sentence `tone`."""
    rng = random.Random(hashlib.sha256(f"{audio_sha}|{text}|{lang}".encode()).hexdigest())
    zh = lang == "zh"
    units = [c for c in text if _is_han(c)] if zh else re.findall(r"[\w']+", text)
    det, pos = [], 0
    for u in units:
        sc = int(rng.uniform(20, 60) if rng.random() < 0.12 else rng.uniform(70, 100))
        b = text.find(u, pos)
        pos = b + len(u)
        d = {"beginindex": b, "endindex": b + len(u) - 1, "start": 0, "end": 0, "dur": 0, "fluency": 100,
             "score": sc}
        if zh:
            d.update(char="x" + "ai"[:1 + rng.randint(0, 1)], chn_char=u, overall=sc, pron=sc, phn=sc,
                     tone=rng.randint(1, 4), tonescore=int(min(100, max(0, sc + rng.uniform(-20, 10)))), group=0,
                     confidence=[rng.randint(0, 100) for _ in range(5)],
                     phone=[{"char": "x", "score": int(min(100, max(0, sc + rng.uniform(-15, 10))))},
                            {"char": "ai", "score": int(min(100, max(0, sc + rng.uniform(-15, 10))))}])
        else:
            d.update(char=u.lower(), accent=0, indict=1, pause_ref=0, pause_score=100, senseref=0, sensescore=100,
                     stressref=0, stressscore=100, toneref=0, tonescore=100)
        det.append(d)
    m = int(sum(d["score"] for d in det) / len(det)) if det else 0
    res = {"overall": m, "accuracy": max(0, m - 2), "pron": min(100, m + 1), "integrity": 100,
           "fluency": {"overall": rng.randint(60, 100), "pause": rng.randint(0, 3), "speed": 65},
           "details": det, "forceout": 0, "info": {"clip": 0, "snr": 20.0, "tipId": 0, "trunc": 0, "volume": 2000},
           "pretime": 0, "rank": 100, "res": "FAKE.snt", "systime": 0, "textmode": 0, "usehookw": 0, "useref": 1,
           "version": "FAKE", "wavetime": 0}
    if zh:
        res.update(tone=rng.randint(50, 100), phn=m, chars={"chn_input": text})
    else:
        res["rhythm"] = {"overall": rng.randint(50, 100), "sense": 100, "stress": 100, "tone": 100}
    return res


# The two input schemas as tools/list returned them in the live smoke test (2026-11-23), descriptions
# shortened; 17 tools exist, the fake lists these two.
LIVE_SCHEMAS = {
    "en_sentence_eval": {"type": "object", "required": ["ref_text"], "properties": {
        "ref_text": {"type": "string"}, "audio_base64": {"type": "string"}, "audio_url": {"type": "string"},
        "accent": {"type": "number", "description": "1 British, 2 American, 3 either (default 3)"},
        "rank": {"type": "number", "description": "4 or 100 (default 100)"},
        "attachAudioUrl": {"type": "number", "description": "default 0"},
        "voiced": {"type": "number", "description": "0 lenient, 1 strict (default 1)"}}},
    "cn_sentence_eval": {"type": "object", "required": ["ref_text"], "properties": {
        "ref_text": {"type": "string"}, "audio_base64": {"type": "string"}, "audio_url": {"type": "string"},
        "rank": {"type": "number"}, "attachAudioUrl": {"type": "number"}}},
}
TIMEOUT_TOOL_ERROR = "评测失败: 请求错误: map[id:53000 msg:core is timeout]"


def tool_error(rid, text: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}], "isError": True}}


def tool_answer(rid, ev: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(ev, ensure_ascii=False)}]}}


class FakeTransport:
    """Local stand-in for the MCP server, following what the live smoke test showed: inline audio is
    read as MP3 (a WAV gets the "core is timeout" tool error), the answer is a JSON text with top-level
    app / audio / ccode / eof / request / result, and arguments outside the schema are refused.
    `script` maps a tools/call number (1-based) to a Response; `schemas` replaces tools/list."""

    def __init__(self, script: dict | None = None, sse: bool = False, session_id: str = "fake-session-1",
                 schemas: dict | None = None):
        self.calls: list[dict] = []
        self.script = dict(script or {})
        self.sse, self.session_id = sse, session_id
        self.schemas = LIVE_SCHEMAS if schemas is None else schemas
        self.n_tool = 0

    def post(self, url, *, headers, json_body=None, data=None, files=None, timeout=60.0) -> C.Response:
        self.calls.append({"url": url, "headers": dict(headers), "body": json_body})
        method = (json_body or {}).get("method")
        rid = (json_body or {}).get("id")
        if method == "initialize":
            m = {"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
                "serverInfo": {"name": "FAKE-chivox", "version": "dry-run"}}}
            return self._reply(m, {"Mcp-Session-Id": self.session_id})
        if method == "notifications/initialized":
            return C.Response(202, {}, b"")
        if method == "tools/list":
            return self._reply({"jsonrpc": "2.0", "id": rid, "result": {"tools": [
                {"name": k, "description": k, "inputSchema": v} for k, v in self.schemas.items()]}}, {})
        self.n_tool += 1
        if self.n_tool in self.script:
            return self.script[self.n_tool]
        name = json_body["params"]["name"]
        args = json_body["params"]["arguments"]
        extra = set(args) - set((self.schemas.get(name) or {}).get("properties") or {})
        if extra:
            return self._reply({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602,
                                                                       "message": f"Invalid params: {sorted(extra)}"}}, {})
        audio = base64.b64decode(args["audio_base64"])
        if audio[:4] == b"RIFF":
            return self._reply(tool_error(rid, TIMEOUT_TOOL_ERROR), {})
        lang = "zh" if name.startswith("cn_") else "en"
        res = fake_eval(args["ref_text"], lang, hashlib.sha256(audio).hexdigest())
        ev = {"app": {"applicationId": "FAKE"}, "audio": {"audioType": "mp3", "channel": 1, "sampleBytes": 2,
                                                          "sampleRate": 16000},
              "ccode": {"code": "fake", "len": 0, "time": "0"}, "eof": 1, "time": {}, "tokenId": "fake",
              "uuid": "fake",
              "request": {"coreType": "cn.sent.raw" if lang == "zh" else "en.sent.score", "refText": args["ref_text"],
                          "tokenId": "fake", **{k: v for k, v in args.items() if k not in ("ref_text", "audio_base64")}},
              "result": {**res, "delaytime": 0}}
        return self._reply(tool_answer(rid, ev), {})

    def _reply(self, m: dict, extra: dict) -> C.Response:
        if self.sse:
            return C.Response(200, {"Content-Type": "text/event-stream", **extra},
                              f"event: message\ndata: {json.dumps(m, ensure_ascii=False)}\n\n".encode())
        return C.Response(200, {"Content-Type": "application/json", **extra}, json.dumps(m, ensure_ascii=False).encode())


def main(argv=None) -> int:
    return C.main(sys.modules[__name__], argv)


if __name__ == "__main__":
    sys.exit(main())
