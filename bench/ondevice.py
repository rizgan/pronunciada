"""On-device cost of a pronunciation scorer, measured the same way for every system (CPU only).

    python -m bench.heavy --need 30 --floor 12 -- python -m bench.ondevice --system pronunciada-int8 --threads 4
    python -m bench.ondevice --system all --threads 1,4          # every system, one fresh process each
    python -m bench.ondevice --report                            # table of what was measured

Protocol, identical for every system:
  * clips: the first N (default 200) recordings of the speechocean762 test set in manifest order, with
    their prompt; audio decoding is inside the timed call for every system;
  * each (system, threads) pair runs in a fresh process; the driver notes the wall clock just before
    starting it. "load" = launch -> models loaded (imports included; a system that loads lazily is made to
    load explicitly); "cold start" = launch -> first result (load + one scoring call on clip 1);
  * after that warm-up every clip is scored once, timed with perf_counter: p50 / p95 seconds per clip,
    real-time factor = total seconds / total audio seconds (and the median per-clip RTF);
  * threads: torch.set_num_threads(n) and OMP/MKL env for torch systems, Scorer(threads=n) for
    Pronunciada (ONNX Runtime intra-op threads); GPU / MPS never used;
  * peak memory = maximum resident set size of the process (getrusage), read after the last clip;
  * weights on disk = the model files the system reads (listed in the output), package data included;
  * the machine's 1-minute load average before and after the run is recorded (the machine is shared:
    a run made under load is slower, so compare runs made at similar load).
Systems:
  pronunciada-fp32 / pronunciada-int8   pronunciada.Scorer("en", precision=...), default settings.
  openpronounce-default                 OpenPronounce 1.1.0 compare_audio_with_text defaults (both
                                        recognizers, reference voice (Piper, offline), prosody curves), = the CLI
                                        without --fast. The reference cache starts empty, so each new
                                        sentence pays one synthesis, as on first use.
  openpronounce-fast                    the same with acoustic, transcription and prosody off (= --fast).
  openpronounce-student                 --fast with the distilled phone model OpenPronounce/wav2vec2-base-phones-en.
  openpronounce-0.4.0                   OpenPronounce 0.4.0 compare_audio_with_text defaults (phone and word
                                        recognizers, Piper reference voice), its own venv and settings
                                        (bench.rivals.openpronounce_040); the version current at Pronunciada 1.0.0.
Writes results/tables/ondevice/<system>__t<threads>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# PRONUNCIADA_ONDEVICE_OUT, OPENPRONOUNCE_WORK, OPENPRONOUNCE_VENV_PY: another output folder, work folder
# (HF_HOME = <work>/hf) and venv (as in bench.rivals.openpronounce)
OUT = Path(os.environ.get("PRONUNCIADA_ONDEVICE_OUT") or ROOT / "results" / "tables" / "ondevice")
CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
OP_WORK = Path(os.environ.get("OPENPRONOUNCE_WORK") or CACHE / "round19" / "rivals")
OP_PY = Path(os.environ.get("OPENPRONOUNCE_VENV_PY") or OP_WORK / "venv" / "bin" / "python")
SYSTEMS = ("pronunciada-fp32", "pronunciada-int8", "openpronounce-default", "openpronounce-fast",
           "openpronounce-student", "openpronounce-0.4.0")


# ============================================================================ child (fresh process)

def _maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def _offline_guard() -> None:
    """As in bench.rivals.openpronounce: no gtts, no network connection from the OpenPronounce process."""
    import socket

    sys.modules["gtts"] = None
    real = socket.socket.connect

    def connect(self, address):
        if self.family == getattr(socket, "AF_UNIX", None):
            return real(self, address)
        raise OSError(f"network disabled ({address!r})")

    socket.socket.connect = connect


def _emit(**kw) -> None:
    print("@@" + json.dumps(kw), flush=True)


def child(system: str, threads: int, clips_file: str) -> None:
    clips = json.loads(Path(clips_file).read_text())
    if system.startswith("pronunciada"):
        from pronunciada.scorer import Scorer

        sc = Scorer("en", precision=system.split("-")[1], threads=threads)

        def score(path, text):
            return sc.score(path, text).prob
    else:
        _offline_guard()
        import torch

        torch.set_num_threads(threads)
        from openpronounce import audio, phones, speech

        opts = {} if system in ("openpronounce-default", "openpronounce-0.4.0") else {
            "acoustic": False, "transcription": False, "prosody": False}
        phones._load_model()
        if system in ("openpronounce-default", "openpronounce-0.4.0"):
            speech._load_models(speech.MODEL_NAME)

        def score(path, text):
            return speech.compare_audio_with_text(audio.load(path), text, lang="en", **opts)["score"]
    _emit(event="loaded", t=time.time())
    score(clips[0]["audio"], clips[0]["text"])
    _emit(event="first", t=time.time())
    times = []
    for c in clips:
        t0 = time.perf_counter()
        score(c["audio"], c["text"])
        times.append(time.perf_counter() - t0)
    _emit(event="done", times=times, maxrss_mb=_maxrss_mb())


# ============================================================================ driver

def weight_files(system: str) -> list[Path]:
    if system.startswith("pronunciada"):
        import pronunciada

        from pronunciada import acoustic

        prec = system.split("-")[1]
        data = Path(pronunciada.__file__).parent / "data"
        # the file the scorer loads: the one backbone file of the precision (cache layout of 1.1), or
        # the plain download in an older layout
        union = acoustic.union_model_path("zipa-small-crctc-500k", prec, str(CACHE))
        plain = CACHE / "models" / "zipa-small-crctc-500k" / acoustic.PRECISION_FILES[prec]
        model = Path(os.environ["PRONUNCIADA_BACKBONE"]) if os.environ.get("PRONUNCIADA_BACKBONE") else \
            Path(union) if acoustic.union_valid(union) else plain
        return [model] + [p for p in data.rglob("*") if p.is_file()]
    hub = (CACHE / "learnerbench" / "openpronounce" if system == "openpronounce-0.4.0" else OP_WORK) / "hf" / "hub"

    def snap(repo, names):
        d = hub / f"models--{repo.replace('/', '--')}"
        rev = (d / "refs" / "main").read_text().strip()
        return [(d / "snapshots" / rev / n).resolve() for n in names]

    py = _op040_py() if system == "openpronounce-0.4.0" else OP_PY
    pkg = py.parent.parent / "lib" / "python3.12" / "site-packages" / "openpronounce"
    files = [p for sub in ("models", "data") for p in (pkg / sub).rglob("*") if p.is_file()]
    if system == "openpronounce-student":
        files += snap("OpenPronounce/wav2vec2-base-phones-en", ["model.safetensors", "hidden_projection.safetensors"])
    else:
        files += _hf_weights(hub, "facebook/wav2vec2-lv-60-espeak-cv-ft")
    if system in ("openpronounce-default", "openpronounce-0.4.0"):
        files += _hf_weights(hub, "facebook/wav2vec2-large-960h")
        v = hub / "models--rhasspy--piper-voices"
        rev = (v / "refs" / "main").read_text().strip()
        files += [(v / "snapshots" / rev / "en" / "en_US" / "lessac" / "medium" / f).resolve()
                  for f in ("en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json")]
    return files


def _op040_py() -> Path:
    from bench.rivals.openpronounce_040 import VENV_PY

    return VENV_PY


def _hf_weights(hub: Path, repo: str) -> list[Path]:
    """The weight file transformers loads from a snapshot (safetensors when present, else the .bin)."""
    d = hub / f"models--{repo.replace('/', '--')}"
    rev = (d / "refs" / "main").read_text().strip()
    s = d / "snapshots" / rev
    for n in ("model.safetensors", "pytorch_model.bin"):
        if (s / n).exists():
            return [(s / n).resolve()]
    raise FileNotFoundError(f"no weights in {s}")


def clips(n: int) -> list[dict]:
    from bench.data import load

    return [{"id": r["id"], "audio": r["audio"], "text": r["text"], "dur": r["dur"]}
            for r in load("speechocean762", split="test")[:n]]


def measure(system: str, threads: int, n: int) -> dict:
    import numpy as np

    C = clips(n)
    with tempfile.TemporaryDirectory() as tmp:
        cf = Path(tmp) / "clips.json"
        cf.write_text(json.dumps(C))
        thr = str(threads)
        env = dict(os.environ, OMP_NUM_THREADS=thr, MKL_NUM_THREADS=thr, VECLIB_MAXIMUM_THREADS=thr,
                   OPENBLAS_NUM_THREADS=thr, TOKENIZERS_PARALLELISM="false")
        if system.startswith("pronunciada"):
            py = sys.executable
            cwd = str(ROOT)
        elif system == "openpronounce-0.4.0":
            from bench.rivals import openpronounce_040 as O4

            env.update(O4.worker_env())
            env["OPENPRONOUNCE_CACHE_DIR"] = str(Path(tmp) / "tts")
            py = str(O4.VENV_PY)
            cwd = tmp
        else:
            from bench.rivals.openpronounce import worker_env

            env.update(worker_env("student" if system == "openpronounce-student" else "default"))
            env["OPENPRONOUNCE_CACHE_DIR"] = str(Path(tmp) / "tts")
            py = str(OP_PY)
            cwd = tmp
        cmd = [py, "-I" if not system.startswith("pronunciada") else "-u", str(Path(__file__).resolve()),
               "--child", system, "--threads", thr, "--clips", str(cf)]
        if system.startswith("pronunciada"):
            cmd = [py, "-m", "bench.ondevice", "--child", system, "--threads", thr, "--clips", str(cf)]
        load_before = os.getloadavg()[0]
        t_launch = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    ev = {}
    for line in p.stdout.splitlines():
        if line.startswith("@@"):
            e = json.loads(line[2:])
            ev[e["event"]] = e
    if p.returncode != 0 or "done" not in ev:
        raise RuntimeError(f"{system} failed ({p.returncode}): {p.stderr[-2000:]}")
    t = np.asarray(ev["done"]["times"])
    dur = np.asarray([c["dur"] for c in C])
    files = weight_files(system)
    machine = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {"system": system, "threads": threads, "n_clips": int(len(t)), "audio_seconds": float(dur.sum()),
            "mean_clip_seconds": float(dur.mean()), "load_s": ev["loaded"]["t"] - t_launch,
            "cold_start_s": ev["first"]["t"] - t_launch, "p50_s": float(np.median(t)),
            "p95_s": float(np.quantile(t, 0.95)), "mean_s": float(t.mean()), "rtf": float(t.sum() / dur.sum()),
            "rtf_median": float(np.median(t / dur)), "peak_rss_mb": ev["done"]["maxrss_mb"],
            "weights_mb": sum(f.stat().st_size for f in files) / 1e6,
            "weight_files": [{"file": f.name, "mb": round(f.stat().st_size / 1e6, 1)} for f in files
                             if f.stat().st_size > 5e6],
            "machine": machine, "load_avg_1min_before": load_before, "load_avg_1min_after": os.getloadavg()[0], "date": time.strftime("%Y-%m-%d %H:%M"), "per_clip_s": [round(x, 4) for x in t]}


def report() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    print("| system | threads | load s | cold start s | p50 s/clip | p95 s/clip | RTF | peak RSS MB | weights MB | machine load |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in sorted(rows, key=lambda r: (SYSTEMS.index(r["system"]), r["threads"])):
        print(f"| {r['system']} | {r['threads']} | {r['load_s']:.1f} | {r['cold_start_s']:.1f} | {r['p50_s']:.3f} | "
              f"{r['p95_s']:.3f} | {r['rtf']:.3f} | {r['peak_rss_mb']:.0f} | {r['weights_mb']:.0f} | "
              f"{r.get('load_avg_1min_before', float('nan')):.1f} |")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default=None, help=f"one of {', '.join(SYSTEMS)} or all")
    ap.add_argument("--threads", default="4", help="comma-separated")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--child", default=None)
    ap.add_argument("--clips", default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.child:
        child(a.child, int(a.threads), a.clips)
        return
    if a.report:
        report()
        return
    systems = SYSTEMS if a.system == "all" else a.system.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    for s in systems:
        for th in map(int, a.threads.split(",")):
            r = measure(s, th, a.n)
            (OUT / f"{s}__t{th}.json").write_text(json.dumps(r, indent=1) + "\n")
            print(f"{s} t{th}: load {r['load_s']:.1f}s cold {r['cold_start_s']:.1f}s p50 {r['p50_s']:.3f}s "
                  f"p95 {r['p95_s']:.3f}s RTF {r['rtf']:.3f} RSS {r['peak_rss_mb']:.0f} MB "
                  f"weights {r['weights_mb']:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
