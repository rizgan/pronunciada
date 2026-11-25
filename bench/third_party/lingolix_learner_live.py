"""Lingolix on the Spanish learner set (Heroico / USMA test selection, OpenSLR 39), against Pronunciada.

    export LINGOLIX_API_KEY=...                                                # environment only
    python -m bench.third_party.lingolix_learner_live --plan                   # requests, audio minutes
    python -m bench.third_party.lingolix_learner_live --yes --minutes-cap 23   # send (cap: audio minutes)
    python -m bench.third_party.lingolix_learner_live --report-only            # results + per-item JSONL
    python -m bench.third_party.lingolix_learner_live --dry-run                # whole pipeline on a local fake

Approved by the owner (2026-11-23): the 600 Heroico / USMA clips of the learner suite's test selection
(the same items and exact prompt texts as the `heroico` set of live_core.load_items), about 22 audio
minutes, billed as overage. Nothing else can be sent (live_core.check_datasets).

API facts are those of the working harness bench/third_party/lingolix_live.py, reused, not re-derived:
endpoint POST https://api.lingolix.com/api/pronunciation/v3/check, multipart fields `sentence`,
`language_code` ("es") and file `speechdata` (16 kHz mono PCM16 WAV decoded locally, the format that
harness sends), `Authorization: Bearer $LINGOLIX_API_KEY`; limits 60 s / 10 MiB; status classification
(`classify_status`: 401 / 402 / 403 and the quota 429 stop the run, other 429 and 5xx are retried) and
response checks and reduction (`parse_response`: words with `accuracy` in [0, 1] and an error type).

Billing unit: audio time. The ledger counts minutes (seconds / 60 of the exact bytes sent) for EVERY
attempt; a live send needs --yes and --minutes-cap, the cap on this runner's own ledger
($PRONUNCIADA_CACHE/third_party/lingolix_learner_live/ledger.jsonl). The lingolix_live ledger is
read for information only, never written.

Sentence score (fixed before any answer): as in the Lingolix comparison on speechocean762
(lingolix.utterance_score / collect): the utterance score the service serves (`overall`), otherwise the
mean of its word scores. The report and the per-item file say which applied to each item.

Report (--report-only): the two pre-registered Heroico metrics of bench/learners/evaluate.py (USMA
native vs learner, prompt-matched; USMA learners vs USMA + HEROICO natives, prompt-matched) for
Lingolix, Pronunciada and cached Azure on the items Lingolix answered; 10,000 speaker resamples,
paired. Writes results/lingolix_learner_live.md, results/tables/lingolix_learner_live.json and the
per-item file <cache>/per_item.jsonl (see `after_report`).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from bench.third_party import live_core as C
from bench.third_party import lingolix_live as LL

NAME = "Lingolix"
STEM = "lingolix_learner_live"
UNIT = "minute"
KEY_ENV = LL.KEY_ENV
CAP_FLAG = "--minutes-cap"
WRITEUP = "lingolix_learner_live.md"
ALLOWED = ("heroico",)
PILOT = {"heroico": 600}
FULL_SIZE = {"heroico": 600}
HOST = LL.API_HOSTS[0]
URL = LL.DEFAULT_BASE + LL.ENDPOINT_PATH
OPTIONS = dict(LL.OPTIONS)               # endpoint, multipart fields, file field, content type
MAPPING = {"sentence_total": "overall if served, else the mean of words[].accuracy (lingolix.utterance_score)",
           "word": "words[].accuracy (lingolix_live.parse_response)", "scale": "0-1 (reported x 100)",
           "source": "bench/third_party/lingolix_live.py (working harness)"}
REPORT_NOTE = ("Sentence score: the `overall` Lingolix serves, otherwise the mean of its word scores (the convention "
               "of the Lingolix comparison on speechocean762); the per-item file records which applied.")


def api_lang(item: C.Item) -> str:
    return item.lang                      # "es": lingolix_live sends the benchmark language code as is


def cost(seconds: float) -> float:
    return seconds / 60.0


def over_limit(item: C.Item, audio: bytes) -> str | None:
    if item.seconds > LL.MAX_AUDIO_S or len(audio) > LL.MAX_AUDIO_BYTES:
        return "over 60 s or 10 MiB"
    return None


def endpoint_desc() -> str:
    return f"POST {URL}"


def existing_ledger_minutes() -> float | None:
    p = LL.LIVE / LL.LEDGER
    try:
        return LL.Ledger(p).total_s() / 60 if p.exists() else None
    except Exception:
        return None


def price_note(minutes: float) -> str:
    old = existing_ledger_minutes()
    return (f"Audio {minutes:.2f} min, all overage: about EUR {math.ceil(minutes - 1e-9) * LL.STARTER_OVERAGE_EUR_PER_MIN:.2f} "
            f"at EUR {LL.STARTER_OVERAGE_EUR_PER_MIN}/min (whole minutes charged; {LL.STARTER_PRICE_SOURCE})." +
            ("" if old is None else f" For information: the lingolix_live ledger holds {old:.2f} min (not touched)."))


def check_key(key: str) -> str | None:
    return None if len(key) >= 8 else f"{KEY_ENV} looks too short to be a key"


def policy() -> C.Policy:
    return C.Policy(rps=1.0)


def real_transport():
    return C.RequestsTransport((HOST,))


class _FakeAdapter:
    """lingolix_live's FakeTransport (documented response format) behind live_core's transport interface."""

    def __init__(self, inner=None):
        self.inner = inner or LL.FakeTransport()

    @property
    def calls(self):
        return self.inner.calls

    def post(self, url, *, headers, json_body=None, data=None, files=None, timeout=60.0) -> C.Response:
        r = self.inner.post(url, headers=headers, data=data, file=files[OPTIONS["file_field"]], timeout=timeout)
        return C.Response(r.status, dict(r.headers), r.body)


def fake_transport():
    return _FakeAdapter()


def new_session() -> dict:
    return {}


def ensure_session(s, **kw) -> None:
    return None


def session_info(s) -> dict:
    return {}


def send(s, item: C.Item, audio: bytes, h: str, *, transport, key: str, policy: C.Policy) -> C.Response:
    return transport.post(URL, headers={"Authorization": f"Bearer {key}"},
                          data={"sentence": item.text, "language_code": api_lang(item)},
                          files={OPTIONS["file_field"]: (f"{C.slug(item.id)}.wav", audio, OPTIONS["content_type"])},
                          timeout=policy.timeout_s)


def ledger_extra(resp: C.Response) -> dict:
    return {}


def classify(resp: C.Response, s=None) -> tuple[str, str]:
    kind = LL.classify_status(resp)
    snippet = resp.body[:300].decode("utf-8", "replace")
    if kind == "ok":
        return ("answered", "ok") if isinstance(resp.json(), dict) else ("fatal", "a 2xx answer that is not JSON (A4)")
    if kind == "fatal":
        return "fatal", LL._FATAL_HELP.get(resp.status, f"HTTP {resp.status}") + f": {snippet}"
    return kind, f"HTTP {resp.status}: {snippet}"          # retry | fail


def version(resp: C.Response, s=None) -> str:
    return str((resp.json() or {}).get("model_version") or "unknown")


def validate(resp: C.Response, item: C.Item) -> None:
    LL.parse_response(resp.json(), item.text, item.lang)


def sentence_score(raw: dict, words: list[dict]) -> tuple[float | None, str]:
    """lingolix.utterance_score's rule for Lingolix: `overall` if served, else the mean word score."""
    if isinstance(raw.get("overall"), (int, float)) and not isinstance(raw.get("overall"), bool):
        return float(raw["overall"]), "overall"
    sc = [w["score"] for w in words if isinstance(w.get("score"), (int, float))]
    return (sum(sc) / len(sc), "word mean") if sc else (None, "none")


def view(rec: dict, item: C.Item) -> dict | None:
    raw = json.loads(rec["body"])
    words = LL.parse_response(raw, item.text, item.lang)
    s, src = sentence_score(raw, words)
    return {"sentence": {"total": None if s is None else round(100.0 * s, 4), "source": src},
            "words": [{"text": w["text"], "score": round(100.0 * w["score"], 4), "phones": [], "error_type": w["error_type"]}
                      for w in words],
            "model_version": raw.get("model_version"), "received_utc": rec.get("received_utc"), "speaker": item.speaker}


def per_item_path(root: Path) -> Path:
    return Path(root) / "per_item.jsonl"


def after_report(A: dict, root: Path) -> None:
    """One JSON line per answered item: {"id", "speaker", "sentence_score" (0-100), "sentence_score_source"
    ("overall" | "word mean"), "words": [{"text", "score" (0-100), "error_type"}], "model_version",
    "received_utc"}; ordered by id. No audio, no text beyond the manifest's words."""
    d = A.get("heroico") or {}
    lines = []
    for i in sorted(d.get("views", {})):
        v = d["views"][i]
        lines.append(json.dumps({"id": i, "speaker": v["speaker"], "sentence_score": v["sentence"]["total"],
                                 "sentence_score_source": v["sentence"]["source"],
                                 "words": [{"text": w["text"], "score": w["score"], "error_type": w["error_type"]}
                                           for w in v["words"]],
                                 "model_version": v["model_version"], "received_utc": v["received_utc"]},
                                ensure_ascii=False))
    C.write_atomic(per_item_path(root), "\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {C.public_path(per_item_path(root))} ({len(lines)} items)", file=sys.stderr)


def extra_markdown(R: dict) -> list[str]:
    return []


def main(argv=None) -> int:
    return C.main(sys.modules[__name__], argv)


if __name__ == "__main__":
    sys.exit(main())
