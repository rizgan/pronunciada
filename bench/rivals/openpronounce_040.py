"""OpenPronounce 0.4.0 on the real-learner benchmark: the version that was current when Pronunciada 1.0.0
was released (PyPI: Pronunciada 1.0.0 on 17 November 2026; OpenPronounce 0.4.0 on 9 November, 1.0.0 on
18 November).

    python -m bench.heavy --lane cpu --need 8 --floor 6 -- python -m bench.rivals.openpronounce_040 \
        --datasets speechocean762,l2arctic,epadb,heroico,saa,leap-en,leap-de,leap-en-prepost,leap-de-prepost

OpenPronounce 0.4.0 from PyPI, installed with its `tts-piper` extra into its own virtual environment
($PRONUNCIADA_CACHE/third_party/venv-openpronounce-0.4.0); this file runs itself there with --worker
(JSON lines on stdin/stdout, nothing from Pronunciada imported). Scoring is the library's own
`compare_audio_with_text(load_audio(path), text, lang=...)` with its defaults, as in the 18 November
2026 snapshot (bench/external/openpronounce.py of that snapshot), with the same two changes, neither of
which touches the score: the reference voice is Piper (OPENPRONOUNCE_TTS=piper, the library's
documented offline backend, its default voice per language) instead of gTTS, which is online; the
display-only pitch and energy curves are not computed. The worker cannot reach the network (gtts is
made unimportable, non-local sockets raise) and reads its models from the local Hugging Face cache of
the learner benchmark. Device: CPU.

speechocean762 prompts are given in sentence case, as OpenPronounce's own speechocean762 benchmark does
(benchmarks/speechocean762.py normalize_text). --import-so762 (not used for the published tables) copies
the outputs of the 18 November 2026 run of the same version (MPS device, upper-case prompts) into the raw
format of this file; a fresh CPU run differs from it by a few points of sentence score (the Piper
reference voice) and not at all in the flagged words. Raw outputs: $PRONUNCIADA_CACHE/learnerbench/openpronounce-0.4.0/raw/<dataset>.jsonl,
one line per record, resumable. bench.learnerbench.report turns them into scores.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VERSION = "0.4.0"
REV = "58061a6b392ec49c1cd602a57d6fc29b2bfa1e9d"
CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
VENV_PY = CACHE / "third_party" / "venv-openpronounce-0.4.0" / "bin" / "python"
WORK = CACHE / "learnerbench" / "openpronounce-0.4.0"
MODELS = CACHE / "learnerbench" / "openpronounce"          # HF cache shared with the 1.1.0 runs
RAW = WORK / "raw"


# ============================================================================ worker (OpenPronounce's venv)

def _offline_guard() -> None:
    import socket

    sys.modules["gtts"] = None
    real_connect = socket.socket.connect

    def connect(self, address):
        if self.family == getattr(socket, "AF_UNIX", None):
            return real_connect(self, address)
        raise OSError(f"network disabled in the OpenPronounce worker (attempted {address!r})")

    socket.socket.connect = connect
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(OSError("network disabled"))


def worker(threads: int) -> None:
    _offline_guard()
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    import openpronounce
    from openpronounce import load_audio, speech
    from openpronounce import phones as P

    assert openpronounce.__version__ == VERSION, openpronounce.__version__
    speech.extract_f0 = lambda a, sr=16000: np.zeros(1)
    speech.extract_energy = lambda a: np.zeros(1)
    print(json.dumps({"ready": True, "version": openpronounce.__version__, "threads": threads}), flush=True)
    for line in sys.stdin:
        q = json.loads(line)
        try:
            t0 = time.time()
            r = speech.compare_audio_with_text(load_audio(q["audio"]), q["text"], lang=q["lang"])
            d = r["differences"]
            out = {"score": r["score"], "acoustic_distance": r["acoustic_distance"],
                   "phoneme_error_rate": d["phoneme_error_rate"], "word_error_rate": d["word_error_rate"],
                   "transcribe": r["transcribe"],
                   "errors": [{"position": e["position"], "word": e["word"], "confidence": e.get("confidence", 1.0),
                               "phones": e.get("phones")} for e in d["errors"]],
                   "sec": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            out = {"error": repr(e)[:400]}
        print(json.dumps(out, ensure_ascii=False), flush=True)
    del P


# ============================================================================ driver (Pronunciada's venv)

def worker_env() -> dict:
    env = dict(os.environ, HF_HOME=str(MODELS / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OPENPRONOUNCE_TTS="piper", OPENPRONOUNCE_CACHE_DIR=str(WORK / "tts_cache"),
               OPENPRONOUNCE_DEVICE="cpu", TOKENIZERS_PARALLELISM="false")
    for k in ("PYTHONPATH", "OPENPRONOUNCE_PHONEME_MODEL", "OPENPRONOUNCE_TTS_VOICE"):
        env.pop(k, None)
    return env


class Worker:
    def __init__(self, threads: int):
        WORK.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen([str(VENV_PY), "-I", str(Path(__file__).resolve()), "--worker",
                                   "--threads", str(threads)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, env=worker_env(), cwd=str(WORK))
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


def raw_path(dataset: str) -> Path:
    return RAW / f"{dataset}.jsonl"


def read_raw(dataset: str) -> dict[str, dict]:
    p, out = raw_path(dataset), {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out[r["id"]] = r
    return out


def import_so762() -> int:
    """The 18 November 2026 run (same version, same scoring code) into this file's raw format."""
    src = CACHE / "external" / "openpronounce" / "speechocean762"
    from bench.rivals.openpronounce import all_tasks

    tasks = {t["id"]: t for t in all_tasks(["speechocean762"])["speechocean762"]}
    RAW.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(raw_path("speechocean762"), "w", encoding="utf-8") as f:
        for p in sorted(src.glob("*.json")):
            d = json.loads(p.read_text())
            t = tasks.get(d["id"])
            if t is None:
                continue
            x = d.get("extra", {})
            if d.get("error") or d.get("gop_s") is None:
                o = {"error": d.get("error", "no score")}
            else:
                o = {"score": float(d["gop_s"]), "errors": x.get("errors", []), "transcribe": x.get("transcribe"),
                     "acoustic_distance": x.get("acoustic_distance"), "phoneme_error_rate": x.get("phoneme_error_rate"),
                     "word_error_rate": x.get("word_error_rate"), "sec": x.get("sec"),
                     "imported": "external/openpronounce 2026-11-18 (MPS)"}
            f.write(json.dumps({"id": t["id"], "dataset": "speechocean762", "lang": "en", "audio": t["audio"],
                                "text": t["text"], "version": VERSION, "rev": x.get("revision", REV), "out": o},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"imported {n} speechocean762 records", flush=True)
    return 0


def run(datasets: list[str], threads: int, first: int | None) -> int:
    from bench.rivals.openpronounce import all_tasks, so762_text

    T = all_tasks(datasets)
    todo = []
    for ds, ts in T.items():
        ts = [t for t in ts if t["lang"] in ("en", "es", "de")]
        if first:
            ts = ts[:first]
        done = {k for k, r in read_raw(ds).items() if "error" not in r["out"]}
        todo += [t for t in ts if t["id"] not in done]
    todo.sort(key=lambda t: (t["lang"] != "en", t["lang"], t["dataset"], t["audio"], t["id"]))
    print(f"openpronounce {VERSION}: {len(todo)} records to score", flush=True)
    w, w_lang, t0, n, nerr, files = None, None, time.time(), 0, 0, {}
    try:
        for t in todo:
            if t["lang"] != w_lang:
                if w is not None:
                    w.close()
                w, w_lang = Worker(threads), t["lang"]
                print(f"worker [{w_lang}] ready: {w.info}", flush=True)
            text = so762_text(t["text"]) if t["dataset"] == "speechocean762" else t["text"]
            try:
                o = w(t["audio"], text, t["lang"])
            except RuntimeError as e:
                o, w, w_lang = {"error": f"driver: {e!r}"}, None, None
            nerr += "error" in o
            n += 1
            p = raw_path(t["dataset"])
            if p not in files:
                p.parent.mkdir(parents=True, exist_ok=True)
                files[p] = open(p, "a", encoding="utf-8")
            files[p].write(json.dumps({"id": t["id"], "dataset": t["dataset"], "lang": t["lang"], "audio": t["audio"],
                                       "text": text, "version": VERSION, "rev": REV, "out": o},
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
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--datasets", default="speechocean762,l2arctic,epadb,heroico,saa,leap-en,leap-de,leap-en-prepost,leap-de-prepost")
    ap.add_argument("--first", type=int, default=None, help="only the first N records of each dataset (smoke test)")
    ap.add_argument("--import-so762", action="store_true")
    a = ap.parse_args(argv)
    if a.worker:
        worker(a.threads)
        return 0
    if a.import_so762:
        return import_so762()
    return run(a.datasets.split(","), a.threads, a.first)


if __name__ == "__main__":
    sys.exit(main())
