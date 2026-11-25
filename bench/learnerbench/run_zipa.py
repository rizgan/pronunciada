"""Reference row "ZIPA alone": the backbone Pronunciada is built on, scored with the existing recipe of
bench/eval_zipa_baseline.py (the "gop" scoring the README and the scoreboard report as ZIPA alone),
unchanged, on every learner-benchmark item.

    python -m bench.heavy --lane cpu --need 25 --floor 12 -- python -m bench.learnerbench.run_zipa [--workers 3]

zipa-small-crctc-500k fp32 (the file Pronunciada loads), raw log-posteriors, no adapter, no calibration,
canonical phones of Pronunciada's text front end; phone = mean log-posterior of the canonical token over
its CTC Viterbi-aligned frames (floor -60), word = mean of its phones, sentence = mean of all phones (the
recipe's own sentence score, used for every sentence-level test, native-vs-learner included).
Outputs: $PRONUNCIADA_CACHE/learnerbench/outputs/zipa-alone/<id>.json (the recipe's record).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import bench.eval_zipa_baseline as Z
from bench.learnerbench import items, out_dir

NAME = "zipa-alone"


def _path(dataset: str, rid: str):
    return out_dir(NAME) / f"{rid.replace('/', '_')}.json"


def _group(g):
    Z.cache_path = _path
    return Z._score_group(g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from bench.pipeline.common import g2p_version

    gv = {}
    by = defaultdict(list)
    for r in items():
        if _path("", r["id"]).exists():
            continue
        lang = r["lang"]
        gv.setdefault(lang, g2p_version(lang))
        by[r["audio"]].append({"dataset": r["labels"]["corpus"], "id": r["id"], "lang": lang, "audio": r["audio"],
                               "text": r["text"], "g2p": gv[lang], "dur": r["dur"]})
    groups = sorted(by.values(), key=lambda g: -g[0]["dur"])
    print(f"{NAME}: {sum(len(g) for g in groups)} items to score", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=Z._init) as ex:
        for k, _ in enumerate(ex.map(_group, groups, chunksize=4)):
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(groups)} ({time.time() - t0:.0f} s)", flush=True)
    (out_dir(NAME) / "_meta.json").write_text(json.dumps({"recipe": "bench/eval_zipa_baseline.py gop",
                                                           "version": Z.VERSION, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
