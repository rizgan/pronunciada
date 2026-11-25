"""Shared machinery for the commercial-API benchmark runners (chivox_live, toneperfect_live).

Everything that keeps a paid API from being called by accident, or twice, lives here; the vendor
modules only say how to build one request, what it costs and how to read the answer.

  * Items. Only the public learner sets listed in ``DATASETS`` exist; a vendor module allows a
    subset of them and ``items()`` refuses anything else (no other loader exists).
  * Audio. Every recording is decoded locally to 16 kHz mono 16-bit PCM WAV (deterministic: the
    same file gives the same bytes); the Lingolix benchmark's own WAV files are sent byte for byte.
  * Cache. Every response body is written atomically the moment it arrives, under
    ``<root>/responses/<h[:2]>/<h>__<item>__<version>.json`` where ``h`` is the hash of the exact
    request (audio sha256, text, language, options). A request whose hash has a final answer
    (scored, or finally rejected) is never sent again, whatever the version string.
    Non-final error bodies go to ``<root>/errors/``.
  * Ledger. ``<root>/ledger.jsonl``: one ``attempt`` line per attempt, written and fsynced BEFORE the
    request goes out (UTC time, vendor, dataset, item id, request hash, cost, cost unit), one
    ``result`` line after (HTTP status or "network error"). Spent = the sum of attempt costs (every
    attempt counts, also failures, which the vendors say are free: the ledger is an upper bound).
  * Budget. A live send needs ``--yes`` and a cap; ``Budget.reserve`` books each attempt (retries
    included) against the cap under a lock before sending; the first refusal closes the run.
  * Credentials are read from one environment variable, never printed, logged or written: every
    file this module writes is scrubbed for the key, and kept headers exclude Authorization.
  * Dry run: a fake transport answers in the documented format; a socket guard refuses any
    connection to a non-local host; files go to ``<root>/DRYRUN_FAKE/``.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import email.utils
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
DRY_SUBDIR = "DRYRUN_FAKE"
LEDGER = "ledger.jsonl"

# Every set a runner may ever send. Licences: speechocean762 Apache-2.0 (OpenSLR 101); OMPAL
# CC BY 4.0; Heroico / USMA (OpenSLR 39) states Apache-2.0; Lingolix open benchmark v2026.10: MIT
# repository, test 1 audio = Common Voice clips (CC0), test 2 = speechocean762 clips.
DATASETS = {
    "speechocean762": "speechocean762 TEST, all 2,500 clips (125 speakers), Lingolix tier2x prompts",
    "ompal": "OMPAL TEST half (bench/eval_ompal.py), 745 clips incl. 3 native speakers",
    "heroico": "Heroico / USMA test selection of the learner suite (bench/learners/heroico.py), 600 clips",
    **{f"lingolix_test1_{x}": f"Lingolix open benchmark v2026.10 test 1, {x} (200 Common Voice clips)"
       for x in ("en", "fr", "es", "de", "ja", "ko")},
    "lingolix_test2": "Lingolix open benchmark v2026.10 test 2 (its 500 speechocean762 clips)",
}
# Named here so the refusal message is explicit; anything not in DATASETS is refused anyway.
NEVER = ("l2arctic", "epadb", "saa", "leap", "common_voice", "cv-real", "fleurs", "fleurs-swap")


class Fatal(RuntimeError):
    """Stop the run: authentication, billing, quota, or a response the adapter does not understand."""


class ResponseFormatError(Fatal):
    pass


class BudgetStop(Fatal):
    pass


class TransientError(RuntimeError):
    pass


class DatasetRefused(ValueError):
    pass


# ── items ──────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Item:
    dataset: str
    id: str
    speaker: str
    audio: Path
    text: str              # the exact text sent
    lang: str              # en, zh, es, fr, de, ja, ko
    seconds: float = 0.0   # filled from the bytes sent
    passthrough: bool = False
    meta: dict = dataclasses.field(default_factory=dict)


def check_datasets(names, allowed) -> list[str]:
    out = []
    for n in names:
        if n not in DATASETS or n not in allowed:
            why = "never sent by any runner (licence or purpose)" if any(n.startswith(x) for x in NEVER) else \
                "not an allowed set for this vendor"
            raise DatasetRefused(f"dataset {n!r} refused: {why}. Allowed: {', '.join(allowed)}")
        out.append(n)
    return out


def _t2s(text: str) -> str:
    from bench.eval_ompal import azure_text
    return azure_text(text, "zh-CN")


def load_items(dataset: str, allowed, *, zh_text: str = "simplified") -> list[Item]:
    """The items of one allowed set, in a fixed order (speaker, then id)."""
    check_datasets([dataset], allowed)
    if dataset == "speechocean762":
        from bench.third_party import lingolix as L
        its, spk = L.tier2x_items()
        paths = L.so762_paths([it["id"] for it in its])
        return [Item(dataset, it["id"], spk[it["id"]], paths[it["id"]], it["target"], "en") for it in its]
    if dataset == "lingolix_test2":
        from bench.third_party import lingolix as L
        man = L.tests()["tier2"]
        _, spk = L.tier2x_items()
        paths = L.so762_paths([x["id"] for x in man])
        out = [Item(dataset, x["id"], spk[x["id"]], paths[x["id"]], x["target"], "en") for x in man]
        return sorted(out, key=lambda i: (i.speaker, i.id))
    if dataset.startswith("lingolix_test1_"):
        from bench.third_party import lingolix as L
        lang = dataset.rsplit("_", 1)[1]
        out = []
        for it in L.tests()[f"tier1_{lang}"]:
            # Common Voice clips: no speaker id in the manifest (only accent / gender / age); each clip
            # is its own cluster, as in report.py's utterance resampling
            out.append(Item(dataset, it["id"], it["id"], L.tier1_wav(it), it["target"], lang,
                            passthrough=True))
        return out
    if dataset == "ompal":
        from bench.eval_ompal import records
        recs = sorted(records("test"), key=lambda r: (r["speaker"], r["id"]))
        conv = _t2s if zh_text == "simplified" else (lambda t: t)
        out = []
        for r in recs:
            t = conv(r["text"])
            assert len(t) == len(r["text"]), r["id"]          # t2s is char for char on this corpus
            out.append(Item(dataset, r["id"], r["speaker"], Path(r["audio"]), t, "zh",
                            meta={"native": r["labels"]["native"], "original_text": r["text"]}))
        return out
    if dataset == "heroico":
        from bench.learners.evaluate import suite
        recs = sorted((r for r in suite() if r["labels"]["corpus"] == "heroico"), key=lambda r: (r["speaker"], r["id"]))
        return [Item(dataset, r["id"], r["speaker"], Path(r["audio"]), r["text"], "es",
                     meta={"native": r["labels"]["native"], "subcorpus": r["labels"]["subcorpus"],
                           "prompt_id": r["labels"]["prompt_id"]}) for r in recs]
    raise DatasetRefused(dataset)


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def pilot_order(items: list[Item], seed: int) -> list[Item]:
    """A fixed, seeded, speaker-balanced order: speakers in a seeded order, each speaker's clips in a
    seeded order, then round robin over speakers. A pilot of n takes the first n, so a smaller pilot
    is always a subset of a larger one (and a 1-item smoke test is part of the pilot).

    Heroico's metric is prompt-matched (a native and a learner reading the same prompt), so there the
    order is prompt by prompt (prompts in a seeded order); within a prompt USMA natives and USMA
    learners alternate (speakers in a seeded order), then the HEROICO natives: a small pilot still
    has comparable pairs."""
    if items and items[0].dataset == "heroico":
        by_p: dict = {}
        for it in items:
            by_p.setdefault(it.meta["prompt_id"], []).append(it)
        out = []
        for pid in sorted(by_p, key=lambda x: _h(seed, "prompt", x)):
            grp = sorted(by_p[pid], key=lambda i: _h(seed, "spk", i.speaker, i.id))
            nat = [i for i in grp if i.meta["native"] and i.meta["subcorpus"] == "usma"]
            lrn = [i for i in grp if not i.meta["native"]]
            rest = [i for i in grp if i not in nat and i not in lrn]
            for k in range(max(len(nat), len(lrn))):
                out += [x[k] for x in (nat, lrn) if len(x) > k]
            out += rest
        return out
    by: dict = {}
    for it in items:
        by.setdefault(it.speaker, []).append(it)
    spk = sorted(by, key=lambda s: _h(seed, "spk", s))
    queues = [sorted(by[s], key=lambda i: _h(seed, "id", i.id)) for s in spk]
    out = []
    for k in range(max((len(q) for q in queues), default=0)):
        out += [q[k] for q in queues if len(q) > k]
    return out


# ── audio ──────────────────────────────────────────────────────────────────────

_AUDIO: dict = {}
_ALOCK = threading.Lock()


def audio_bytes(item: Item) -> bytes:
    """The exact bytes sent: 16 kHz mono PCM16 WAV decoded locally (or the file itself, passthrough)."""
    k = (str(item.audio), item.passthrough)
    with _ALOCK:
        if k in _AUDIO:
            return _AUDIO[k]
    if item.passthrough:
        b = Path(item.audio).read_bytes()
    else:
        import soundfile as sf
        x, sr = sf.read(str(item.audio), dtype="int16", always_2d=True)
        if sr != 16000:
            raise ValueError(f"{item.audio}: {sr} Hz; every allowed set is stored at 16 kHz")
        buf = io.BytesIO()
        sf.write(buf, x[:, 0], 16000, format="WAV", subtype="PCM_16")
        b = buf.getvalue()
    with _ALOCK:
        if len(_AUDIO) > 64:
            _AUDIO.clear()
        _AUDIO[k] = b
    return b


def wav_seconds(b: bytes) -> float:
    import soundfile as sf
    i = sf.info(io.BytesIO(b))
    return i.frames / i.samplerate


def request_hash(audio: bytes, text: str, lang: str, options: dict) -> tuple[str, dict]:
    inputs = {"audio_sha256": hashlib.sha256(audio).hexdigest(), "text": text, "language": lang, "options": options}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), inputs


# ── files ──────────────────────────────────────────────────────────────────────

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scrub(s: str, secret: str | None) -> str:
    if secret and len(secret) >= 4 and secret in s:
        s = s.replace(secret, "[REDACTED]")
    return s


def write_atomic(p: Path, text: str, secret: str | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(scrub(text, secret), encoding="utf-8")
    os.replace(tmp, p)


def slug(s) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", str(s or "unknown")).strip("-")[:80] or "unknown"


def find_answer(root: Path, h: str) -> dict | None:
    """The cached final answer of request hash h (any item id, any version), or None."""
    d = Path(root) / "responses" / h[:2]
    for p in sorted(d.glob(f"{h}__*.json")):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
    return None


def save_answer(root: Path, h: str, item: Item, version: str, rec: dict, secret: str | None) -> Path:
    p = Path(root) / "responses" / h[:2] / f"{h}__{slug(item.id)}__{slug(version)}.json"
    write_atomic(p, json.dumps(rec, ensure_ascii=False, indent=1), secret)
    return p


def save_error(root: Path, item: Item, aid: str, rec: dict, secret: str | None) -> None:
    write_atomic(Path(root) / "errors" / slug(item.dataset) / f"{slug(item.id)}__{slug(aid)}.json",
                 json.dumps(rec, ensure_ascii=False, indent=1), secret)


_KEEP = re.compile(r"version|model|request-id|ratelimit|retry-after|^date$|credits|engine|idempot|session|test|^x-",
                   re.I)


def keep_headers(h: dict) -> dict:
    return {k: v for k, v in (h or {}).items()
            if _KEEP.search(k) and k.lower() not in ("authorization", "set-cookie", "cookie", "x-api-key")}


# ── ledger and budget ──────────────────────────────────────────────────────────

class Ledger:
    """Append-only JSON lines. A line that cannot be read stops everything (the total is unknown)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    raise Fatal(f"ledger {self.path} line {n} cannot be read; fix it by hand before sending "
                                "anything (the total spent is unknown)") from None
        return out

    def spent(self) -> float:
        return float(sum(float(e.get("cost") or 0) for e in self.entries() if e.get("kind") == "attempt"))

    def append(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())


class Budget:
    """The cap in the vendor's billing unit. reserve() books one attempt before it is sent."""

    def __init__(self, cap: float | None, used: float, ledger: Ledger | None, unit: str, vendor: str):
        self.cap, self.used, self.start = cap, float(used), float(used)
        self.ledger, self.unit, self.vendor = ledger, unit, vendor
        self.lock = threading.Lock()
        self.closed: str | None = None
        self.n = 0

    def reserve(self, item: Item, cost: float, h: str | None) -> str:
        with self.lock:
            if self.closed:
                raise BudgetStop(self.closed)
            if self.cap is None:
                self.closed = "no cap given: nothing is sent"
                raise BudgetStop(self.closed)
            if self.used + cost > self.cap + 1e-9:
                self.closed = (f"{self.unit} cap reached: the next attempt ({item.dataset} {item.id}, {cost:g} "
                               f"{self.unit}) would take the total from {self.used:g} to {self.used + cost:g}, "
                               f"over the cap of {self.cap:g}; run stopped before sending it")
                raise BudgetStop(self.closed)
            aid = f"{utc_now()}-{os.getpid()}-{self.n:06d}"
            if self.ledger is not None:
                try:
                    self.ledger.append({"kind": "attempt", "attempt": aid, "utc": utc_now(), "vendor": self.vendor,
                                        "dataset": item.dataset, "id": item.id, "req": h, "cost": cost,
                                        "unit": self.unit, "seconds": round(item.seconds, 3), "status": "pending"})
                except OSError as e:
                    self.closed = f"the ledger cannot be written ({type(e).__name__}); run stopped"
                    raise BudgetStop(self.closed) from None
            self.used += cost
            self.n += 1
            return aid

    def result(self, aid: str, item: Item, status, extra: dict | None = None) -> None:
        if self.ledger is not None:
            self.ledger.append({"kind": "result", "attempt": aid, "utc": utc_now(), "vendor": self.vendor,
                                "dataset": item.dataset, "id": item.id, "status": status, **(extra or {})})

    def close(self, why: str) -> None:
        with self.lock:
            self.closed = self.closed or why


# ── transport ──────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Response:
    status: int
    headers: dict
    body: bytes

    def json(self):
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None


class RequestsTransport:
    """The real one (requests). Errors carry the exception type only, never the headers."""

    def __init__(self, allowed_hosts: tuple[str, ...]):
        self.allowed_hosts = allowed_hosts
        self._s = None

    def post(self, url: str, *, headers: dict, json_body=None, data=None, files=None, timeout: float = 60.0) -> Response:
        import requests
        from urllib.parse import urlparse

        if urlparse(url).hostname not in self.allowed_hosts:
            raise Fatal(f"refusing to send to {urlparse(url).hostname}: not the vendor's documented host")
        if self._s is None:
            self._s = requests.Session()
        try:
            r = self._s.post(url, headers=headers, json=json_body, data=data, files=files, timeout=timeout,
                             allow_redirects=False)
        except requests.RequestException as e:
            raise TransientError(type(e).__name__) from None
        return Response(r.status_code, dict(r.headers), r.content)

    def __repr__(self):
        return "RequestsTransport()"


def no_network_guard():
    from bench.third_party.lingolix_live import no_network_guard as g
    return g()


def retry_after_s(headers: dict) -> float | None:
    v = next((v for k, v in (headers or {}).items() if k.lower() == "retry-after"), None)
    if v is None:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        try:
            return max(0.0, (email.utils.parsedate_to_datetime(v) - dt.datetime.now(dt.timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


@dataclasses.dataclass
class Policy:
    max_attempts: int = 4          # bounded: at most 3 retries per item, transient errors only
    base_s: float = 2.0
    cap_s: float = 60.0
    max_retry_after_s: float = 600.0
    timeout_s: float = 90.0
    rps: float = 0.5               # at most one request every 2 s


# ── plan and run ───────────────────────────────────────────────────────────────

def plan(vendor, items: list[Item], root: Path, *, used: float, cap: float | None, refresh_rejected: bool = False) -> dict:
    """What a run would send: per set, items / cached (answered, rejected) / over the vendor's limits /
    to send / cost; units taken in order while used + planned <= cap (the first that does not fit ends
    the list). Nothing is sent."""
    rows, todo, seen = {}, [], set()
    for it in items:
        a = audio_bytes(it)
        it.seconds = wav_seconds(a)
        h, _ = request_hash(a, it.text, vendor.api_lang(it), vendor.OPTIONS)
        r = rows.setdefault(it.dataset, {"items": 0, "answered": 0, "rejected": 0, "over_limit": 0, "to_send": 0,
                                         "cost": 0.0, "audio_s": 0.0, "held_back": 0, "same_request_as_another_set": 0})
        r["items"] += 1
        if h in seen:                      # e.g. Lingolix test 2 = 500 of the speechocean762 requests
            r["same_request_as_another_set"] += 1
            continue
        seen.add(h)
        c = find_answer(root, h)
        if c is not None and (c.get("final") == "answered" or not refresh_rejected):
            r["answered" if c.get("final") == "answered" else "rejected"] += 1
            continue
        why = vendor.over_limit(it, a)
        if why:
            r["over_limit"] += 1
            continue
        todo.append((it, h, vendor.cost(it.seconds)))
    chosen, add, stopped = [], 0.0, None
    for it, h, c in todo:
        if cap is not None and used + add + c > cap + 1e-9:
            stopped = f"{it.dataset} {it.id} ({c:g} {vendor.UNIT}) does not fit under the cap"
            break
        chosen.append((it, h, c))
        add += c
    for it, h, c in chosen:
        rows[it.dataset]["to_send"] += 1
        rows[it.dataset]["cost"] += c
        rows[it.dataset]["audio_s"] += it.seconds
    for it, _, _ in todo[len(chosen):]:
        rows[it.dataset]["held_back"] += 1
    return {"rows": rows, "todo": chosen, "n_send": len(chosen), "cost": add, "used": used, "cap": cap,
            "left": None if cap is None else cap - used - add, "stopped_by": stopped,
            "unit": vendor.UNIT, "cost_all_unsent": sum(c for _, _, c in todo)}


def print_plan(p: dict, vendor, *, dry: bool, out=sys.stderr) -> None:
    print(("DRY RUN, FAKE DATA: nothing leaves this machine; a local fake answers.\n" if dry else
           f"LIVE: these requests are billed by {vendor.NAME} in {vendor.UNIT}s.\n") +
          f"endpoint: {vendor.endpoint_desc()}", file=out)
    print(f"{'set':<22}{'items':>7}{'answered':>10}{'rejected':>10}{'>limit':>8}{'to send':>9}{vendor.UNIT + 's':>9}"
          f"{'audio min':>11}{'held back':>11}", file=out)
    for t, r in p["rows"].items():
        print(f"{t:<22}{r['items']:>7}{r['answered']:>10}{r['rejected']:>10}{r['over_limit']:>8}{r['to_send']:>9}"
              f"{r['cost']:>9g}{r['audio_s'] / 60:>11.2f}{r['held_back']:>11}" +
              (f"   ({r['same_request_as_another_set']} identical to items of a set above: sent once)"
               if r["same_request_as_another_set"] else ""), file=out)
    cap = "none" if p["cap"] is None else f"{p['cap']:g}"
    print(f"{vendor.UNIT}s: spent so far (ledger) {p['used']:g}; cap {cap}; this run books {p['cost']:g} if no "
          f"attempt is retried" + ("" if p["left"] is None else f"; left under the cap after it {p['left']:g}") +
          ". Every attempt, retries included, is booked before it is sent; the run stops before any attempt "
          "that would cross the cap.", file=out)
    if p["stopped_by"]:
        print(f"held back: {p['stopped_by']}", file=out)
    print(f"requests to send: {p['n_send']}; {vendor.UNIT}s for every item not yet answered: "
          f"{p['cost_all_unsent']:g}. {vendor.price_note(p['cost'])}", file=out)


def run(vendor, todo, *, transport, key: str, root: Path, budget: Budget, policy: Policy,
        sleep=time.sleep, clock=time.monotonic, progress: bool = True) -> dict:
    """Send each (item, hash, cost) one at a time; cache each answer as it arrives.

    Optional vendor hooks (a vendor without them behaves exactly as before):
      encode_audio(item, wav) -> the bytes actually sent (the request hash stays that of the WAV and
          the vendor's OPTIONS, which name the encoding); record_extra(item, sent) -> fields added to
          every cached record; attempt_limit(detail) -> a lower attempt limit for that kind of
          failure; after_item(session, item, outcome, detail) -> may raise Fatal to stop the run."""
    st = {"sent": 0, "answered": 0, "rejected": 0, "failed": 0, "retries": 0, "failures": [], "versions": {},
          "stopped": None}
    nxt = 0.0
    rng = random.Random()
    session = vendor.new_session()
    for n, (it, h, cost) in enumerate(todo):
        if find_answer(root, h) is not None:
            continue                                         # answered meanwhile (another run): never twice
        audio = audio_bytes(it)
        it.seconds = wav_seconds(audio)
        cost = vendor.cost(it.seconds)
        _, inputs = request_hash(audio, it.text, vendor.api_lang(it), vendor.OPTIONS)
        why, outcome, detail = None, None, None
        try:
            payload = vendor.encode_audio(it, audio) if hasattr(vendor, "encode_audio") else audio
            limit = policy.max_attempts
            for attempt in range(policy.max_attempts):
                if policy.rps > 0:
                    now = clock()
                    if now < nxt:
                        sleep(nxt - now)
                    nxt = max(now, nxt) + 1.0 / policy.rps
                vendor.ensure_session(session, transport=transport, key=key, budget=budget, policy=policy)
                aid = budget.reserve(it, cost, h)
                t0 = time.perf_counter()
                try:
                    resp = vendor.send(session, it, payload, h, transport=transport, key=key, policy=policy)
                except TransientError as e:
                    resp, why = None, str(e)
                lat = round((time.perf_counter() - t0) * 1000, 1)
                budget.result(aid, it, resp.status if resp is not None else "network error",
                              vendor.ledger_extra(resp) if resp is not None else None)
                if resp is None:
                    kind, ra, detail = "retry", None, why
                else:
                    kind, detail = vendor.classify(resp, session)
                    ra = retry_after_s(resp.headers)
                    body = resp.body.decode("utf-8", "replace")
                    rec = {"schema": 1, "vendor": vendor.NAME, "dataset": it.dataset, "item": it.id,
                           "speaker": it.speaker, "req_hash": h, "request": inputs, "attempt": aid,
                           "received_utc": utc_now(), "status": resp.status, "headers": keep_headers(resp.headers),
                           "latency_ms": lat, "classified": kind, "detail": detail, "body": body,
                           "session": vendor.session_info(session)}
                    if hasattr(vendor, "record_extra"):
                        rec.update(vendor.record_extra(it, payload))
                    if kind in ("answered", "rejected"):
                        version = vendor.version(resp, session) if kind == "answered" else "rejected"
                        rec.update(final=kind, version=version)
                        save_answer(root, h, it, version, rec, key)
                        st["sent"] += 1
                        st[kind] += 1
                        outcome = kind
                        st["versions"][version] = st["versions"].get(version, 0) + 1
                        if kind == "answered":
                            vendor.validate(resp, it)            # after caching: a fix of the adapter costs nothing
                        else:
                            st["failures"].append({"dataset": it.dataset, "id": it.id, "error": scrub(detail, key)})
                        break
                    save_error(root, it, aid, rec, key)
                    if kind == "fatal":
                        raise Fatal(scrub(f"{vendor.NAME}: HTTP {resp.status}: {detail}; run stopped", key))
                    if kind == "fail":
                        outcome = "failed"
                        st["sent"] += 1
                        st["failed"] += 1
                        st["failures"].append({"dataset": it.dataset, "id": it.id, "error": scrub(detail, key)})
                        break
                    if kind == "session":
                        session.clear()
                    why = f"HTTP {resp.status}: {detail}"
                if hasattr(vendor, "attempt_limit"):
                    limit = min(limit, vendor.attempt_limit(detail))
                if attempt >= limit - 1:
                    outcome = "failed"
                    st["sent"] += 1
                    st["failed"] += 1
                    st["failures"].append({"dataset": it.dataset, "id": it.id,
                                           "error": scrub(f"gave up after {attempt + 1} attempts ({why})", key)})
                    break
                if ra is not None:
                    if ra > policy.max_retry_after_s:
                        raise Fatal(f"the API asked to wait {ra:.0f} s (Retry-After); run stopped, rerun later")
                    delay = ra + rng.uniform(0, 0.5)
                else:
                    delay = min(policy.cap_s, policy.base_s * 2 ** attempt) * rng.uniform(0.5, 1.0)
                st["retries"] += 1
                sleep(delay)
            if hasattr(vendor, "after_item"):
                vendor.after_item(session, it, outcome, detail)
        except Fatal as e:
            msg = scrub(str(e), key)
            budget.close(msg)
            st["stopped"] = msg
            break
        except KeyboardInterrupt:
            st["stopped"] = "interrupted (Ctrl-C); rerun the same command to resume"
            break
        if progress and (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(todo)} done, {budget.used - budget.start:g} {budget.unit}s booked", file=sys.stderr)
    st["booked"] = budget.used - budget.start
    st["session"] = vendor.session_info(session)        # run log only (Chivox: the tools/list schemas)
    return st


# ── statistics: paired speaker bootstrap ───────────────────────────────────────

def speaker_resamples(clusters, n: int, seed: int):
    """Yields index arrays, each a resample of clusters with replacement (numpy default_rng(seed))."""
    import numpy as np

    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])


def q95(v) -> list:
    import numpy as np
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [math.nan, math.nan]


def p_two(d) -> float:
    import numpy as np
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return math.nan
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def paired(fns: dict, clusters, n: int, seed: int, vendor: str, ref: str = "pronunciada") -> dict:
    """fns: {system: f(idx) -> float}. Every system on the same n speaker resamples; value and 95 %
    interval per system; vendor - ref with interval and two-sided p."""
    import numpy as np

    full = np.arange(len(clusters))
    out = {k: {"value": float(f(full))} for k, f in fns.items()}
    vals = {k: [] for k in fns}
    for idx in speaker_resamples(clusters, n, seed):
        for k, f in fns.items():
            vals[k].append(f(idx))
    for k in fns:
        out[k]["ci"] = q95(vals[k])
    if vendor in fns and ref in fns:
        d = np.asarray(vals[vendor], float) - np.asarray(vals[ref], float)
        out["delta"] = {"value": out[vendor]["value"] - out[ref]["value"], "ci": q95(d), "p": p_two(d),
                        "of": f"{vendor} - {ref}"}
    out["n_resamples"], out["seed"] = n, seed
    return out


def pearson_fn(y, s):
    import numpy as np
    y, s = np.asarray(y, float), np.asarray(s, float)

    def f(idx):
        a, b = s[idx], y[idx]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 3 or a.std() == 0 or b.std() == 0:
            return math.nan
        return float(np.corrcoef(a, b)[0, 1])
    return f


def spearman_fn(y, s):
    from scipy.stats import rankdata
    import numpy as np
    y, s = np.asarray(y, float), np.asarray(s, float)

    def f(idx):
        a, b = s[idx], y[idx]
        if len(a) < 3 or a.std() == 0 or b.std() == 0:
            return math.nan
        return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
    return f


def auc_low_fn(pos, score):
    """AUC of detecting pos with a goodness score (low score = flagged); NaN scores dropped."""
    import numpy as np
    from bench.metrics import roc_auc
    pos, score = np.asarray(pos, bool), np.asarray(score, float)

    def f(idx):
        p, s = pos[idx], score[idx]
        ok = np.isfinite(s)
        p, s = p[ok], s[ok]
        if p.sum() == 0 or p.sum() == len(p):
            return math.nan
        return float(roc_auc(p.astype(int), -s))
    return f


# ── report helpers ─────────────────────────────────────────────────────────────

def fmt(v, d=3) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def cell(x: dict | None) -> str:
    if not x or x.get("value") is None or not math.isfinite(x["value"]):
        return "—"
    ci = x.get("ci") or [math.nan, math.nan]
    return f"{x['value']:.3f} [{fmt(ci[0])}, {fmt(ci[1])}]"


def dcell(d: dict | None) -> str:
    if not d or not math.isfinite(d.get("value", math.nan)):
        return "—"
    p = d["p"]
    ps = "< 0.001" if p < 0.001 else f"{p:.3f}"
    return f"{d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}], p {ps}"


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        return clean(x.item())
    return x


def public_path(p: Path) -> str:
    s = str(p)
    for base, name in ((str(CACHE), "$PRONUNCIADA_CACHE"), (str(ROOT), "<repo>"), (str(Path.home()), "~")):
        if s.startswith(base):
            return name + s[len(base):]
    return s


# ── the command line shared by the runners ─────────────────────────────────────

def live_root(vendor, dry: bool, base: Path | None = None) -> Path:
    b = Path(base) if base else CACHE / "third_party" / getattr(vendor, "STEM", f"{vendor.NAME.lower()}_live")
    return b / DRY_SUBDIR if dry else b


def select_items(vendor, datasets: list[str], *, pilot: bool, limit: int | None, seed: int, loader=None) -> list[Item]:
    loader = loader or (lambda d: load_items(d, vendor.ALLOWED, zh_text=getattr(vendor, "ZH_TEXT", "simplified")))
    out = []
    for d in datasets:
        its = loader(d)
        n = limit if limit is not None else (vendor.PILOT.get(d, 0) if pilot else None)
        if n is not None:
            its = pilot_order(its, seed)[:max(0, n)]
        out += its
    return out


def collect_answers(vendor, items: list[Item], root: Path) -> dict:
    """{dataset: {"views": {id: view}, "answered", "rejected", "unanswered", "items", "versions",
    "first_utc", "last_utc"}} from the cache (sends nothing)."""
    out: dict = {}
    for it in items:
        a = audio_bytes(it)
        h, _ = request_hash(a, it.text, vendor.api_lang(it), vendor.OPTIONS)
        d = out.setdefault(it.dataset, {"views": {}, "records": {}, "answered": 0, "rejected": 0, "unanswered": 0, "items": 0,
                                        "speakers": set(), "speakers_answered": set(), "versions": {},
                                        "first_utc": None, "last_utc": None, "rejections": []})
        d["items"] += 1
        d["speakers"].add(it.speaker)
        rec = find_answer(root, h)
        if rec is None:
            d["unanswered"] += 1
            continue
        v = vendor.view(rec, it) if rec.get("final") == "answered" else None
        if v is None:
            d["rejected"] += 1
            d["rejections"].append({"id": it.id, "status": rec.get("status"), "detail": rec.get("detail")})
            continue
        d["answered"] += 1
        d["speakers_answered"].add(it.speaker)
        d["views"][it.id] = v
        if hasattr(vendor, "record"):
            d["records"][it.id] = vendor.record(rec, it.id)
        ver = rec.get("version") or "unknown"
        d["versions"][ver] = d["versions"].get(ver, 0) + 1
        t = rec.get("received_utc")
        if t:
            d["first_utc"] = min(x for x in (d["first_utc"], t) if x)
            d["last_utc"] = max(x for x in (d["last_utc"], t) if x)
    for d in out.values():
        d["n_speakers"], d["n_speakers_answered"] = len(d.pop("speakers")), len(d.pop("speakers_answered"))
    return out


def ledger_summary(ledger: Ledger) -> dict:
    out: dict = {}
    for e in ledger.entries():
        d = out.setdefault(e.get("dataset") or "session set-up (0 cost)", {"attempts": 0, "booked": 0.0, "statuses": {}})
        if e.get("kind") == "attempt":
            d["attempts"] += 1
            d["booked"] += float(e.get("cost") or 0)
        elif e.get("kind") == "result":
            s = str(e.get("status"))
            d["statuses"][s] = d["statuses"].get(s, 0) + 1
    return out


def evaluate_sets(vendor, A: dict, n_boot: int) -> dict:
    from bench.third_party import live_eval as LE

    R = {}
    name = vendor.NAME.lower()
    for ds, d in A.items():
        if not d["views"]:
            continue
        if ds == "speechocean762":
            R[ds] = LE.eval_so762({"so762-" + k: v for k, v in d["views"].items()}, n_boot=n_boot, vendor=name)
        elif ds == "ompal":
            R[ds] = LE.eval_ompal(d["views"], n_boot=n_boot, hit_boot=min(n_boot, LE.HIT_BOOT), vendor=name)
        elif ds == "heroico":
            R[ds] = LE.eval_heroico(d["views"], n_boot=n_boot, vendor=name)
    if hasattr(vendor, "extra_evaluation"):
        R.update(vendor.extra_evaluation(A, n_boot))
    return R


SET_TITLES = {"speechocean762": "speechocean762 test", "ompal": "OMPAL test", "heroico": "Heroico / USMA"}


def markdown(vendor, A: dict, R: dict, L: dict, *, dry: bool, n_boot: int = 10_000) -> str:
    from bench.third_party import live_eval as LE

    name = vendor.NAME
    lines = [f"# {name} API measured by us, against Pronunciada", ""]
    if dry:
        lines += [f"**DRY RUN, FAKE DATA: the {name} column comes from a local fake and says nothing about {name}.**", ""]
    if getattr(vendor, "REPORT_NOTE", None):
        lines += [vendor.REPORT_NOTE, ""]
    for ds, d in A.items():
        vs = ", ".join(f"{k} ({v})" for k, v in d["versions"].items()) or "none"
        full = vendor.FULL_SIZE.get(ds, d["items"])
        lines.append(f"- **{ds}**: {d['answered']} answered, {d['rejected']} rejected, {d['unanswered']} not sent, of "
                     f"{d['items']} items selected ({d['n_speakers_answered']} of {d['n_speakers']} speakers answered); "
                     f"versions {vs}; received {d['first_utc'] or '—'} to {d['last_utc'] or '—'} (UTC).")
        if d["answered"] and d["answered"] < full:
            lines.append(f"  - **SAMPLE**: these rows rest on {d['answered']} of the {full} clips of the set (a seeded, "
                         "speaker-balanced pilot). Read the intervals, not the point values; they describe this "
                         "sample, not the whole set.")
    lines += ["", ("Ledger (every attempt counted, also failures; per set): " +
                   "; ".join(f"{k}: {v['attempts']} attempts, {v['booked']:g} {vendor.UNIT}s booked, HTTP statuses "
                             f"{v['statuses']}" for k, v in L.items())) if L else "Ledger: empty.", ""]
    lines += [f"| set | metric | n | {name} [95% CI] | Pronunciada [95% CI] | {name} − Pronunciada [95% CI], p | "
              "Azure [95% CI] |", "|---|---|---|---|---|---|---|"]
    for ds, block in R.items():
        if ds in SET_TITLES:
            lines += LE.md_rows(SET_TITLES[ds], block, name.lower())
    lines += ["", f"Intervals: {n_boot:,} resamples of speakers with replacement (hit rates: {min(n_boot, 2000):,}), "
              "every system on the "
              "same resamples; p two-sided, paired. Only the items the vendor answered are compared; the Azure "
              "column comes from cached responses on the same items. Field mapping and assumptions: "
              f"results/{vendor.WRITEUP}."]
    extra = vendor.extra_markdown(R) if hasattr(vendor, "extra_markdown") else []
    return "\n".join(lines + ([""] + extra if extra else [])) + "\n"


def report(vendor, items: list[Item], root: Path, *, dry: bool, n_boot: int, out_md: Path | None = None,
           out_json: Path | None = None) -> dict:
    A = collect_answers(vendor, items, root)
    R = evaluate_sets(vendor, A, n_boot)
    L = ledger_summary(Ledger(root / LEDGER))
    stem = getattr(vendor, "STEM", vendor.NAME.lower() + "_live")
    if dry:
        out_md = out_md or root / "report" / f"{stem}.md"
        out_json = out_json or root / "report" / f"{stem}.json"
    else:
        out_md = out_md or ROOT / "results" / f"{stem}.md"
        out_json = out_json or ROOT / "results" / "tables" / f"{stem}.json"
    A2 = {k: {kk: vv for kk, vv in v.items() if kk not in ("views", "records")} for k, v in A.items()}
    doc = {"vendor": vendor.NAME, "dry_run": dry, "generated_utc": utc_now(), "answers": A2, "ledger": L,
           "results": R, "mapping": vendor.MAPPING, "cache": public_path(root), "n_boot": n_boot}
    write_atomic(Path(out_json), json.dumps(clean(doc), ensure_ascii=False, indent=1))
    write_atomic(Path(out_md), markdown(vendor, A, R, L, dry=dry, n_boot=n_boot))
    print(f"wrote {public_path(Path(out_md))} and {public_path(Path(out_json))}", file=sys.stderr)
    if hasattr(vendor, "after_report"):
        vendor.after_report(A, root)
    return doc


def main(vendor, argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog=f"python -m bench.third_party.{getattr(vendor, 'STEM', vendor.NAME.lower() + '_live')}",
                                 description=f"{vendor.NAME} benchmark runner: dry run by default; see "
                                             f"results/{vendor.WRITEUP}")
    ap.add_argument("--datasets", nargs="+", default=list(vendor.ALLOWED), help=f"of {', '.join(vendor.ALLOWED)}")
    ap.add_argument("--plan", action="store_true", help="print what would be sent and what it costs; send nothing")
    ap.add_argument("--pilot", action="store_true", help="the default pilot sample: " +
                    ", ".join(f"{k} {v}" for k, v in vendor.PILOT.items()))
    ap.add_argument("--limit", type=int, help="N items per set (seeded, speaker-balanced; a prefix of the pilot order)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yes", action="store_true", help="really send (live); also needs the cap")
    ap.add_argument(vendor.CAP_FLAG, dest="cap", type=float, help=f"hard cap in {vendor.UNIT}s, ledger total included")
    ap.add_argument("--dry-run", action="store_true", help="the whole pipeline against a local fake (no network)")
    ap.add_argument("--report-only", action="store_true", help="evaluate the cached answers; send nothing")
    ap.add_argument("--retry-rejected", action="store_true", help="send again the requests the API finally rejected")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    try:
        datasets = check_datasets(a.datasets, vendor.ALLOWED)
    except DatasetRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    dry = a.dry_run
    root = live_root(vendor, dry)
    sampled = a.pilot or a.limit is not None
    items = select_items(vendor, datasets, pilot=a.pilot, limit=a.limit, seed=a.seed)
    if a.report_only:
        report(vendor, items, root, dry=dry, n_boot=a.n_boot)
        return 0
    ledger = Ledger(root / LEDGER)
    used = ledger.spent()
    cap = a.cap if (a.cap is not None or not dry) else math.inf
    p = plan(vendor, items, root, used=used, cap=cap, refresh_rejected=a.retry_rejected)
    print_plan(p, vendor, dry=dry)
    if not sampled:
        print("(full run: every item of the selected sets)", file=sys.stderr)
    if a.plan:
        return 0
    if dry:
        transport, key = vendor.fake_transport(), "FAKE-KEY-dry-run-0000"
    else:
        key = os.environ.get(vendor.KEY_ENV) or ""
        problems = []
        if not a.yes:
            problems.append("--yes is missing")
        if a.cap is None:
            problems.append(f"{vendor.CAP_FLAG} is missing")
        if not key:
            problems.append(f"{vendor.KEY_ENV} is not set")
        else:
            why = vendor.check_key(key)
            if why:
                problems.append(why)
        if problems:
            print("nothing sent: " + "; ".join(problems) + f". A live send needs --yes, {vendor.CAP_FLAG} N and "
                  f"{vendor.KEY_ENV}.", file=sys.stderr)
            return 2
        transport = vendor.real_transport()
    budget = Budget(cap, used, ledger, vendor.UNIT, vendor.NAME)
    policy = vendor.policy()

    def go():
        return run(vendor, p["todo"], transport=transport, key=key, root=root, budget=budget, policy=policy,
                   sleep=(lambda s: None) if dry else time.sleep)

    if dry:
        with no_network_guard():
            st = go()
    else:
        st = go()
    st["finished_utc"] = utc_now()
    write_atomic(root / "runs" / f"{utc_now().replace(':', '')}-{os.getpid()}.json",
                 json.dumps(clean({"vendor": vendor.NAME, "dry_run": dry, "datasets": datasets, "pilot": a.pilot,
                                   "limit": a.limit, "seed": a.seed, "cap": a.cap, "endpoint": vendor.endpoint_desc(),
                                   "options": vendor.OPTIONS, **st}), ensure_ascii=False, indent=1), key)
    print(scrub(f"sent {st['sent']}: {st['answered']} answered, {st['rejected']} rejected, {st['failed']} failed, "
                f"{st['retries']} retries; {st['booked']:g} {vendor.UNIT}s booked; versions {st['versions']}" +
                (f"; STOPPED: {st['stopped']}" if st["stopped"] else ""), key), file=sys.stderr)
    if dry and not a.no_report:
        report(vendor, items, root, dry=True, n_boot=min(a.n_boot, 200))
    return 1 if st["stopped"] and not st["stopped"].startswith("interrupted") else 0
