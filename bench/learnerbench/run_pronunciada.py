"""Score every benchmark item with Pronunciada through its public API (Scorer.score(audio, text)).

    python -m bench.learnerbench.run_pronunciada --variant fp32|int8 [--backbone FILE --name NAME] [--workers 3]

fp32 = the default (shipped ZIPA-small fp32 file); int8 = the shipped int8 file (precision="int8");
--backbone FILE sets PRONUNCIADA_BACKBONE for another backbone file (for example an on-device model),
stored under the system name given with --name. Default settings otherwise. One Scorer per worker
process, 1 ONNX thread each. Each item keeps:
  sentence  SentenceResult.scores (total, accuracy, fluency, prosodic) and prob
  api       bench.calib.api_run.summarize (the per-sound regression and probability outputs, word
            outputs and sentence details the scoreboard reads), English and Spanish / German items
  zh        bench.eval_ompal.summarize (word units by character span) for Mandarin items
A clip the scorer refuses (ValueError) is kept as {"status": "error"} and scores the floor later.
Resumable: items already saved are skipped. Records the package fingerprint and variant.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from bench.learnerbench import items, out_dir, save

_SC = None


def _init(lang: str, precision: str) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _SC
    import pronunciada

    _SC = pronunciada.Scorer(lang, threads=1, precision=precision)


def _one(rec: dict):
    from bench.calib.api_run import summarize

    t = time.perf_counter()
    try:
        res = _SC.score(rec["audio"], rec["text"])
    except ValueError as e:
        return rec["id"], {"status": "error", "error": str(e)[:200], "seconds": time.perf_counter() - t}
    sc = dict(res.scores or {})
    o = {"status": "Success", "prob": float(res.prob),
         "sentence": {k: (None if sc.get(k) is None else float(sc[k])) for k in ("total", "accuracy", "fluency", "prosodic")},
         "sound_name": _SC.sound_name[-1]}
    if rec["lang"] == "zh":
        from bench.eval_ompal import summarize as zsum

        o["zh"] = zsum(res, rec["text"])
    else:
        o["api"] = summarize(_SC, res, {})
    o["seconds"] = time.perf_counter() - t
    return rec["id"], o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("fp32", "int8"), default="fp32")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="first N items per language (smoke test)")
    a = ap.parse_args()
    if a.backbone:
        if not a.name:
            raise SystemExit("--backbone needs --name")
        os.environ["PRONUNCIADA_BACKBONE"] = str(Path(a.backbone).resolve())
    name = a.name or f"pronunciada-{a.variant}"
    import pronunciada

    from bench.learners.evaluate import REPO, fingerprint

    assert Path(pronunciada.__file__).resolve().is_relative_to(REPO), pronunciada.__file__
    d = out_dir(name)
    todo = defaultdict(list)
    for r in items():
        if not (d / f"{r['id']}.pkl").exists():
            todo[r["lang"]].append(r)
    if a.limit:
        todo = {k: v[: a.limit] for k, v in todo.items()}
    n = sum(len(v) for v in todo.values())
    print(f"{name}: pronunciada {pronunciada.__version__} fingerprint {fingerprint()}: {n} items to score", flush=True)
    t0 = time.time()
    for lang, rs in sorted(todo.items()):
        rs.sort(key=lambda r: -r["dur"])
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(lang, a.variant)) as ex:
            for k, (rid, o) in enumerate(ex.map(_one, rs, chunksize=2)):
                save(name, rid, o)
                if (k + 1) % 500 == 0:
                    print(f"  {lang}: {k + 1}/{len(rs)} ({time.time() - t0:.0f} s)", flush=True)
        print(f"  {lang}: {len(rs)} done ({time.time() - t0:.0f} s)", flush=True)
    meta = {"system": name, "version": pronunciada.__version__, "fingerprint": fingerprint(), "variant": a.variant,
            "backbone": os.path.basename(a.backbone) if a.backbone else None, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "_meta.json").write_text(json.dumps(meta))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
