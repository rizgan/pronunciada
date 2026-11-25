"""Assemble the learner suite test file: every fixed test selection in one plain JSONL.

    python -m bench.learners.suite      -> $PRONUNCIADA_CACHE/data/learners/suite_test.jsonl

One JSON object per recording: the benchmark manifest fields (id, audio = absolute path to 16 kHz
mono audio, lang, text = exact prompt, split, speaker, dur, labels) plus ``labels.corpus``,
``labels.native`` (bool) and, where human scores exist, ``labels.human_score`` (the corpus' overall
sentence score: speechocean762 ``total`` 0-10, OMPAL teacher ``accuracy`` 1-5). Any runner that
reads {audio, text, lang} can be pointed at it.

Members (test parts only; their dev parts stay aside):
  heroico         es  bench/learners/heroico.py (new)
  saa             en  bench/learners/saa.py (new)
  speechocean762  en  existing test split (bench/data/speechocean762.py), referenced in place
  ompal           zh  existing test half (bench/ompal_build.py), referenced in place

``python -m bench.learners.suite --four`` (added 2026-11-23) writes ``suite4_test.jsonl``: every line of
the round-19 file above, unchanged, plus the LeaP test selections (bench/learners/leap.py):
  leap-de         de  LeaP German read stories (new)
  leap-en         en  LeaP English read story (new, secondary English native-vs-learner test)
"""
from __future__ import annotations

import json

from bench.data import load as load_bench
from bench.learners import hours, load, root, write_jsonl

NEW = ("heroico", "saa")
FOUR = ("leap-de", "leap-en")


def existing() -> list[dict]:
    out = []
    for r in load_bench("speechocean762", split="test"):
        lab = dict(r["labels"])
        lab.update(corpus="speechocean762", native=False, l1="mandarin",
                   human_score=float(r["labels"]["sentence"]["total"]))
        out.append({**r, "labels": lab})
    for r in load_bench("ompal", split="test"):
        lab = dict(r["labels"])
        lab.update(corpus="ompal", native=bool(r["labels"]["native"]),
                   l1="mandarin" if r["labels"]["native"] else "french",
                   human_score=float(r["labels"]["sentence"]["accuracy"]))
        out.append({**r, "labels": lab})
    return out


def build(four: bool = False) -> list[dict]:
    recs = []
    for c in NEW + (FOUR if four else ()):
        for r in load(c, "test"):
            recs.append({**r, "labels": {**r["labels"], "corpus": c}})
    recs += existing()
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids)), "duplicate ids"
    name = "suite4_test" if four else "suite_test"
    n = write_jsonl(root() / f"{name}.jsonl", recs)
    summ = {}
    for r in recs:
        k = f"{r['labels']['corpus']}/{r['lang']}"
        s = summ.setdefault(k, {"items": 0, "learner_items": 0, "native_items": 0, "speakers": set(), "recs": []})
        s["items"] += 1
        s["native_items" if r["labels"]["native"] else "learner_items"] += 1
        s["speakers"].add(r["speaker"])
        s["recs"].append(r)
    table = {k: {"items": v["items"], "learner_items": v["learner_items"], "native_items": v["native_items"],
                 "speakers": len(v["speakers"]), "hours": hours(v["recs"])} for k, v in summ.items()}
    (root() / f"{name}.summary.json").write_text(json.dumps(table, indent=1))
    print(f"wrote {n} records -> {root() / (name + '.jsonl')}")
    print(json.dumps(table, indent=1))
    return recs


if __name__ == "__main__":
    import sys

    build(four="--four" in sys.argv[1:])
