"""OpenPronounce 1.1.0 (github.com/OpenPronounce/OpenPronounce, AGPL-3.0) on Pronunciada's held-out tests.

    python -m bench.heavy --need 30 --floor 12 -- \
        python -m bench.rivals.openpronounce --mode default --datasets speechocean762 [--max-records N]
    python -m bench.rivals.openpronounce --mode fast --datasets fleurs-swap --langs en,fr

Modes (the three ways the project ships it, all on CPU):
  default  `openpronounce clip.wav "text"`: phone recognizer facebook/wav2vec2-lv-60-espeak-cv-ft plus the
           word recognizer of the language (facebook/wav2vec2-large-960h for English, a jonatasgrosman
           XLSR-53 checkpoint otherwise) and the acoustic comparison with a synthetic reference voice.
  fast     `--fast`: acoustic=False, transcription=False (phone recognizer only).
  student  OPENPRONOUNCE_PHONEME_MODEL=OpenPronounce/wav2vec2-base-phones-en (the distilled phone model,
           English only) with `--fast`, the setting its model card documents; the library then picks
           word_mdd_en_student.json and scorer_en_student.json itself.

Offline: the default reference voice of the library is gTTS (Google Translate, online). It is replaced by
the project's documented offline backend, Piper (OPENPRONOUNCE_TTS=piper, default voice per language). In
the worker the `gtts` module is made unimportable and every non-local socket connection raises, so no
scoring call can reach the network; models are read from the local Hugging Face cache (HF_HUB_OFFLINE=1).
The reference voice feeds only `formula_score` (the pre-1.0 formula); the trained `score`, `scores` and
the word reports never read it.

Two changes that do not touch any output: prosody=False (the pitch / energy curves for display; the
scorer computes its own prosody features from the audio either way), and the recognizers' outputs are
memoised per audio file so the several prompts of one fleurs-swap recording share one forward pass.

The worker (this file run with --worker in OpenPronounce's own venv, nothing from Pronunciada imported)
answers one JSON line per (audio, text). The driver appends one JSON line per record to
$PRONUNCIADA_CACHE/round19/rivals/raw/<mode>/<dataset>.jsonl and skips records already there (resumable).
bench.rivals.convert turns these into the per-record files of the external-system contract.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VERSION = "1.1.0"
REV = "bd51333cebb9e7eb791bce45b710e520ca97e690"
STUDENT = "OpenPronounce/wav2vec2-base-phones-en"
CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
# OPENPRONOUNCE_WORK / OPENPRONOUNCE_VENV_PY: another work folder (raw outputs, HF_HOME) and venv
WORK = Path(os.environ.get("OPENPRONOUNCE_WORK") or CACHE / "round19" / "rivals")
VENV_PY = Path(os.environ.get("OPENPRONOUNCE_VENV_PY") or WORK / "venv" / "bin" / "python")
RAW = WORK / "raw"
LANGS = ("en", "fr", "es", "de", "it", "pt", "nl")
MODES = ("default", "fast", "student")


# ============================================================================ worker (OpenPronounce's venv)

def _offline_guard() -> None:
    import socket

    sys.modules["gtts"] = None                       # import gtts -> ImportError
    real_connect = socket.socket.connect

    def connect(self, address):
        if self.family == getattr(socket, "AF_UNIX", None):
            return real_connect(self, address)
        raise OSError(f"network disabled in the OpenPronounce worker (attempted {address!r})")

    socket.socket.connect = connect
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(OSError("network disabled"))


def worker(mode: str, threads: int) -> None:
    _offline_guard()
    import hashlib

    import numpy as np
    import torch

    torch.set_num_threads(threads)
    import openpronounce
    from openpronounce import audio, phones, speech
    from openpronounce.device import get_device

    memo: dict = {}

    def key(wav, *extra):
        a = np.ascontiguousarray(wav)
        return (hashlib.sha1(a.tobytes()).hexdigest(), a.shape, extra)

    def memoised(fn, name):
        def f(wav, *args, **kw):
            k = (name, key(wav, args, tuple(sorted(kw.items()))))
            if k not in memo:
                memo[k] = fn(wav, *args, **kw)
            return memo[k]
        return f

    phones.recognize_phones = memoised(phones.recognize_phones, "phones")
    speech.transcribe = memoised(speech.transcribe, "asr")
    speech.extract_embeddings = memoised(speech.extract_embeddings, "emb")
    opts = {"prosody": False}
    if mode in ("fast", "student"):
        opts.update(acoustic=False, transcription=False)
    print(json.dumps({"ready": True, "device": str(get_device()), "version": openpronounce.__version__,
                      "phone_model": phones.PHONE_MODEL_NAME, "threads": torch.get_num_threads()}), flush=True)
    last_audio = None
    for line in sys.stdin:
        q = json.loads(line)
        try:
            if q["audio"] != last_audio:
                memo.clear()
                last_audio = q["audio"]
            t0 = time.perf_counter()
            wav = audio.load(q["audio"])
            r = speech.compare_audio_with_text(wav, q["text"], lang=q["lang"], **opts)
            d = r["differences"]
            words = []
            for w in d.get("words", []):
                words.append({k: w[k] for k in ("position", "word", "confidence", "mispronounced", "start", "end")}
                             | {k: w[k] for k in ("missing", "missing_confidence", "guessed") if k in w}
                             | {"phones": [{"expected": p["expected"], "heard": p["heard"], "confidence": p["confidence"]}
                                           for p in w["phones"]]})
            exp = d.get("expected_phones")
            out = {"score": r["score"], "scores": r["scores"], "formula_score": r["formula_score"],
                   "acoustic_distance": r["acoustic_distance"], "transcribe": r["transcribe"],
                   "reliable": r["reliable"], "reasons": r["reasons"], "alignment": d.get("alignment"),
                   "errors": [{"position": e["position"], "word": e["word"], "confidence": e.get("confidence"),
                               "missing": bool(e.get("missing"))} for e in d["errors"]],
                   "words": words, "expected_phones": exp, "extra": d.get("extra", []),
                   "phoneme_error_rate": d.get("phoneme_error_rate"), "word_error_rate": d.get("word_error_rate"),
                   "audio_issues": r["audio_quality"]["issues"], "sec": round(time.perf_counter() - t0, 3)}
        except Exception as e:  # noqa: BLE001
            import traceback

            out = {"error": repr(e)[:400], "trace": traceback.format_exc()[-1500:]}
        print(json.dumps(out, ensure_ascii=False), flush=True)


# ============================================================================ driver (Pronunciada's venv)

def worker_env(mode: str) -> dict:
    env = dict(os.environ, HF_HOME=str(WORK / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OPENPRONOUNCE_TTS="piper", OPENPRONOUNCE_CACHE_DIR=str(WORK / "tts_cache"),
               OPENPRONOUNCE_DEVICE="cpu", TOKENIZERS_PARALLELISM="false", PYTORCH_ENABLE_MPS_FALLBACK="0")
    for k in ("PYTHONPATH", "OPENPRONOUNCE_PHONEME_MODEL", "OPENPRONOUNCE_TTS_VOICE", "OPENPRONOUNCE_G2P",
              "OPENPRONOUNCE_MODELS_DIR"):
        env.pop(k, None)
    if mode == "student":
        env["OPENPRONOUNCE_PHONEME_MODEL"] = STUDENT
    return env


class Worker:
    def __init__(self, mode: str, threads: int):
        self.p = subprocess.Popen([str(VENV_PY), "-I", str(Path(__file__).resolve()), "--worker", "--mode", mode,
                                   "--threads", str(threads)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, env=worker_env(mode), cwd=str(WORK))
        self.info = json.loads(self.p.stdout.readline())

    def __call__(self, audio: str, text: str, lang: str) -> dict:
        self.p.stdin.write(json.dumps({"audio": audio, "text": text, "lang": lang}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("worker died")
        return json.loads(line)

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait(timeout=120)


def so762_text(text: str) -> str:
    """benchmarks/speechocean762.py normalize_text: the corpus is upper case, give a normally cased sentence."""
    text = text.strip().lower()
    return text[:1].upper() + text[1:]


def lingolix_tasks() -> dict[str, list[dict]]:
    from bench.third_party import lingolix as L

    out = {}
    for lang in ("en", "fr", "es", "de"):
        out[f"lingolix-t1-{lang}"] = [{"dataset": f"lingolix-t1-{lang}", "id": it["id"], "lang": lang,
                                       "audio": str(L.tier1_wav(it)), "text": it["target"], "dur": 0.0}
                                      for it in L.tests()[f"tier1_{lang}"]]
    return out


LEARNER_CORPORA = ("saa", "heroico", "leap-de", "leap-en", "leap-de-prepost", "leap-en-prepost")


def learner_tasks() -> dict[str, list[dict]]:
    """The learner-suite corpora (bench.learners, suite4_test.jsonl) that are not scoreboard datasets."""
    from bench.learners import read_jsonl, root

    out = {c: [] for c in LEARNER_CORPORA}
    recs = list(read_jsonl(root() / "suite4_test.jsonl"))
    for c in ("leap-de", "leap-en"):
        if (root(c) / "prepost.jsonl").exists():
            recs += [{**r, "labels": {**r["labels"], "corpus": f"{c}-prepost"}} for r in read_jsonl(root(c) / "prepost.jsonl")]
    for r in recs:
        c = r["labels"]["corpus"]
        if c in out:
            out[c].append({"dataset": c, "id": r["id"], "lang": r["lang"], "audio": r["audio"], "text": r["text"],
                           "dur": r["dur"]})
    return out


def all_tasks(datasets: list[str]) -> dict[str, list[dict]]:
    out = {}
    if any(d in LEARNER_CORPORA for d in datasets):
        out.update({k: v for k, v in learner_tasks().items() if k in datasets})
        datasets = [d for d in datasets if d not in LEARNER_CORPORA]
        if not datasets:
            return out
    if any(not d.startswith("lingolix") for d in datasets):
        from bench.eval_zipa_baseline import tasks

        out.update({k: v for k, v in tasks().items() if k in datasets})
    if any(d.startswith("lingolix") for d in datasets):
        out.update({k: v for k, v in lingolix_tasks().items() if k in datasets})
    return out


def raw_path(mode: str, dataset: str) -> Path:
    return RAW / mode / f"{dataset}.jsonl"


def read_raw(mode: str, dataset: str) -> dict[str, dict]:
    p = raw_path(mode, dataset)
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                          # a line cut by a kill: rescored
                out[r["id"]] = r
    return out


def run(mode: str, datasets: list[str], langs: tuple, threads: int, max_records: int | None, first: int | None) -> int:
    T = all_tasks(datasets)
    todo = []
    for ds, ts in T.items():
        ts = [t for t in ts if t["lang"] in langs]
        if first:
            ts = ts[:first]
        done = {k for k, r in read_raw(mode, ds).items() if "error" not in r["out"]}
        todo += [t for t in ts if t["id"] not in done]
    # one language per worker (the library caches every recognizer it loads); records of one audio together
    todo.sort(key=lambda t: (t["lang"] != "en", t["lang"], t["dataset"], t["audio"], t["id"]))
    if max_records:
        todo = todo[:max_records]
    print(f"openpronounce {VERSION} [{mode}]: {len(todo)} records to score", flush=True)
    w, w_lang, t0, n, nerr = None, None, time.time(), 0, 0
    files = {}
    try:
        for t in todo:
            if t["lang"] != w_lang:
                if w is not None:
                    w.close()
                w, w_lang = Worker(mode, threads), t["lang"]
                print(f"worker [{w_lang}] ready: {w.info}", flush=True)
            text = so762_text(t["text"]) if t["dataset"] == "speechocean762" else t["text"]
            try:
                o = w(t["audio"], text, t["lang"])
            except RuntimeError as e:
                o = {"error": f"driver: {e!r}"}
                w = None
                w_lang = None
            nerr += "error" in o
            n += 1
            p = raw_path(mode, t["dataset"])
            if p not in files:
                p.parent.mkdir(parents=True, exist_ok=True)
                files[p] = open(p, "a", encoding="utf-8")
            files[p].write(json.dumps({"id": t["id"], "dataset": t["dataset"], "lang": t["lang"], "audio": t["audio"],
                                       "text": text, "mode": mode, "version": VERSION, "rev": REV, "out": o},
                                      ensure_ascii=False) + "\n")
            files[p].flush()
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}  {(time.time() - t0) / 60:.1f} min  errors {nerr}", flush=True)
    finally:
        for f in files.values():
            f.close()
        if w is not None:
            w.close()
    print(f"done: {n} records in {(time.time() - t0) / 60:.1f} min, {nerr} errors", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode", choices=MODES, default="default")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--datasets", default="speechocean762")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--max-records", type=int, default=None, help="stop after this many records (chunked runs)")
    ap.add_argument("--first", type=int, default=None, help="only the first N records of each dataset (smoke test)")
    a = ap.parse_args(argv)
    if a.worker:
        worker(a.mode, a.threads)
        return 0
    return run(a.mode, a.datasets.split(","), tuple(a.langs.split(",")), a.threads, a.max_records, a.first)


if __name__ == "__main__":
    sys.exit(main())
