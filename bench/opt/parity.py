"""Numerical parity of a candidate configuration against a reference, clip by clip.

    python -m bench.opt.parity run --ref '{"name":"main","code":"c1792d4"}' \
        --cand '{"name":"wt","code":"worktree"}' [--langs all] [--n 20] [--threads 2] [--out report.json]
    python -m bench.opt.parity compare <dump dir A> <dump dir B>

Clips: the committed practice list (bench/opt/clips.json: 20 FLEURS train recordings per language,
all ten languages). Each configuration scores them in its own fresh process (bench/opt/_child.py,
one process at a time), dumping per clip: the raw backbone log-posteriors and the adapted ones
(the adapter's output), SentenceResult.to_dict() without timing, the result rescored at
strictness 0 and 1, Scorer.details(), coaching(), tips() and the Azure layout.

Reported, per language and overall: max and mean |difference| of the raw and adapted posteriors
(finite entries; positions of -inf must agree) and of every final score, grouped as sentence /
word / sound / syllable (tone, stress) / rescore / details / coaching / tips / Azure layout; the
number of values compared, of values that differ, and of clips where any score differs at all
(numbers or strings). "identical" means max |diff| = 0 on every number, every string equal, every
posterior equal. Reference dumps are cached by the configuration's identity (source tree content,
precision, override files) under $PRONUNCIADA_CACHE/opt/parity/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import LANGS, load_spec, materialize, opt_root, practice_clips, spec_hash
from .proc import Child

CATS = ("sentence", "word", "sound", "syllable", "rescore", "details", "coaching", "tips", "azure layout")
# the reported probabilities themselves (strictness of the result), on top of the categories above
PROB_KEYS = (("sentence P", re.compile(r"^result\.prob$")), ("word P", re.compile(r"^result\.words\[\d+\]\.prob$")),
             ("sound P", re.compile(r"^result\.words\[\d+\]\.phones\[\d+\]\.prob$")))


def _flat(x, pre="", out=None):
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            _flat(v, f"{pre}.{k}" if pre else k, out)
    elif isinstance(x, list):
        out[pre + ".len"] = len(x)
        for i, v in enumerate(x):
            _flat(v, f"{pre}[{i}]", out)
    else:
        out[pre] = x
    return out


def category(key: str) -> str:
    top = key.split(".")[0].split("[")[0]
    if top == "result":
        if key.startswith("result.words["):
            if ".phones[" in key:
                return "sound"
            if ".syllables[" in key:
                return "syllable"
            return "word"
        return "sentence"
    return {"s0": "rescore", "s1": "rescore", "details": "details", "coaching": "coaching", "tips": "tips",
            "azure": "azure layout"}.get(top, "sentence")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_records(A: list[dict], B: list[dict]) -> dict:
    """Score dumps (lists of per-clip records, same clip order) -> difference statistics."""
    st = {c: {"values": 0, "numeric": 0, "differ": 0, "max_abs_diff": 0.0, "sum_abs_diff": 0.0}
          for c in CATS + tuple(n for n, _ in PROB_KEYS)}
    clips_diff, examples, other = 0, [], 0
    ida, idb = [r["id"] for r in A], [r["id"] for r in B]
    if ida != idb:
        other += 1
        examples.append("clip lists differ")
    for ra, rb in zip(A, B):
        fa = _flat({k: v for k, v in ra.items() if k not in ("id", "n")})
        fb = _flat({k: v for k, v in rb.items() if k not in ("id", "n")})
        any_diff = False
        for k in sorted(set(fa) | set(fb)):
            va, vb = fa.get(k, "<missing>"), fb.get(k, "<missing>")
            s = st[category(k)]
            s["values"] += 1
            if _num(va) and _num(vb):
                s["numeric"] += 1
                d = abs(float(va) - float(vb))
                if not math.isfinite(d):
                    d = math.inf
                s["sum_abs_diff"] += d
                s["max_abs_diff"] = max(s["max_abs_diff"], d)
                for pn, rx in PROB_KEYS:
                    if rx.match(k):
                        q = st[pn]
                        q["values"] += 1
                        q["numeric"] += 1
                        q["sum_abs_diff"] += d
                        q["max_abs_diff"] = max(q["max_abs_diff"], d)
                        q["differ"] += d > 0
                if d > 0:
                    s["differ"] += 1
                    any_diff = True
                    if len(examples) < 12:
                        examples.append(f"{ra['id']} {k}: {va} vs {vb}")
            elif va != vb:
                s["differ"] += 1
                any_diff = True
                if len(examples) < 12:
                    examples.append(f"{ra['id']} {k}: {str(va)[:60]!r} vs {str(vb)[:60]!r}")
        clips_diff += any_diff
    for s in st.values():
        s["mean_abs_diff"] = s.pop("sum_abs_diff") / s["numeric"] if s["numeric"] else 0.0
    return {"n_clips": len(A), "n_clips_any_score_differs": clips_diff, "by_category": st,
            "max_abs_diff": max(st[c]["max_abs_diff"] for c in CATS),
            "n_values": sum(st[c]["values"] for c in CATS),
            "n_values_differ": sum(st[c]["differ"] for c in CATS) + other, "examples": examples}


def compare_posteriors(pa: list[Path], pb: list[Path]) -> dict:
    out = {}
    for key in ("raw", "adapted"):
        mx, tot, n, shape_bad, inf_bad, differ = 0.0, 0.0, 0, 0, 0, 0
        for a, b in zip(pa, pb):
            with np.load(a) as za, np.load(b) as zb:
                xa, xb = za[key], zb[key]
            if xa.shape != xb.shape:
                shape_bad += 1
                continue
            fa, fb = np.isfinite(xa), np.isfinite(xb)
            if not np.array_equal(fa, fb) or not np.array_equal(xa[~fa], xb[~fb]):
                inf_bad += 1
            m = fa & fb
            d = np.abs(xa[m].astype(np.float64) - xb[m].astype(np.float64))
            if d.size:
                mx = max(mx, float(d.max()))
                tot += float(d.sum())
                n += int(d.size)
                differ += int((d > 0).sum())
        out[key] = {"max_abs_diff": mx, "mean_abs_diff": tot / n if n else 0.0, "values": n, "values_differ": differ,
                    "clips_shape_mismatch": shape_bad, "clips_nonfinite_mismatch": inf_bad,
                    "clips": min(len(pa), len(pb))}
    return out


def _read(d: Path) -> tuple[list[dict], list[Path]]:
    p = d / "outputs.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    posts = [d / "post" / f"{r['n']:05d}.npz" for r in recs]
    return recs, posts


def compare_dirs(a: Path, b: Path) -> dict:
    ra, pa = _read(Path(a))
    rb, pb = _read(Path(b))
    s = compare_records(ra, rb)
    s["posteriors"] = compare_posteriors(pa, pb)
    post_same = all(v["max_abs_diff"] == 0 and v["clips_shape_mismatch"] == 0 and v["clips_nonfinite_mismatch"] == 0
                    for v in s["posteriors"].values())
    s["identical"] = s["n_values_differ"] == 0 and post_same and len(ra) == len(rb)
    s["max_abs_diff_posteriors"] = max(v["max_abs_diff"] for v in s["posteriors"].values())
    return s


def dump(spec: dict, lang: str, clips: list[dict], threads: int, dest: Path, log=print) -> Path:
    """Dump a configuration's outputs on `clips` (cached: a complete dump is reused)."""
    done = dest / "DONE"
    if done.exists():
        return dest
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    tree = materialize(spec["code"])
    k = Child(spec, tree, lang, threads)
    errors = []
    try:
        for c in clips:
            r = k.score(c, dump=str(dest))
            if "error" in r:
                errors.append({"id": c["id"], "error": r["error"]})
    finally:
        k.close()
    (dest / "meta.json").write_text(json.dumps({"spec": spec, "tree": tree.name, "lang": lang, "threads": threads,
                                                "clips": [c["id"] for c in clips], "errors": errors}, indent=1))
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log(f"dumped {spec['name']} {lang}: {len(clips) - len(errors)} clips -> {dest}")
    return dest


def run(ref: dict, cand: dict, langs, n: int | None, threads: int, work: Path | None = None, log=print) -> dict:
    work = work or opt_root() / "parity"
    rt, ct = materialize(ref["code"]), materialize(cand["code"])
    rh, ch = spec_hash(ref, rt), spec_hash(cand, ct)
    out = {"kind": "opt_parity", "date": time.strftime("%Y-%m-%d %H:%M"), "ref": ref, "cand": cand,
           "ref_id": rh, "cand_id": ch, "threads": threads, "clip_set": "practice (bench/opt/clips.json)",
           "by_lang": {}}
    for lang in langs:
        clips = practice_clips([lang], n)
        tag = f"{lang}_t{threads}_n{len(clips)}"
        da = dump(ref, lang, clips, threads, work / rh / tag, log)
        db = dump(cand, lang, clips, threads, work / ch / tag, log)
        out["by_lang"][lang] = compare_dirs(da, db)
        r = out["by_lang"][lang]
        log(f"{lang}: identical={r['identical']} scores max|diff|={r['max_abs_diff']:.3g} "
            f"posteriors max|diff|={r['max_abs_diff_posteriors']:.3g} clips differing {r['n_clips_any_score_differs']}"
            f"/{r['n_clips']}")
    B = out["by_lang"].values()
    out["identical"] = all(r["identical"] for r in B)
    out["n_clips"] = sum(r["n_clips"] for r in B)
    out["n_clips_any_score_differs"] = sum(r["n_clips_any_score_differs"] for r in B)
    out["max_abs_diff"] = max(r["max_abs_diff"] for r in B)
    out["max_abs_diff_posteriors"] = {k: max(r["posteriors"][k]["max_abs_diff"] for r in B) for k in ("raw", "adapted")}
    out["n_values"] = sum(r["n_values"] for r in B)
    by = defaultdict(lambda: {"values": 0, "differ": 0, "max_abs_diff": 0.0, "numeric": 0, "sum": 0.0})
    for r in B:
        for c, s in r["by_category"].items():
            by[c]["values"] += s["values"]
            by[c]["differ"] += s["differ"]
            by[c]["max_abs_diff"] = max(by[c]["max_abs_diff"], s["max_abs_diff"])
            by[c]["numeric"] += s["numeric"]
            by[c]["sum"] += s["mean_abs_diff"] * s["numeric"]
    for c in by.values():
        c["mean_abs_diff"] = c.pop("sum") / c["numeric"] if c["numeric"] else 0.0
    out["by_category"] = dict(by)
    out["verdict"] = (f"exactly identical: max |diff| = 0 on {out['n_values']:,} values and every posterior, "
                      f"{out['n_clips']} clips" if out["identical"] else
                      f"not identical: {out['n_clips_any_score_differs']} of {out['n_clips']} clips have a differing "
                      f"score; max |diff| {out['max_abs_diff']:.3g} (scores), "
                      f"{out['max_abs_diff_posteriors']['raw']:.3g} (raw posteriors)")
    log(out["verdict"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--ref", default='{"name":"head","code":"HEAD"}')
    r.add_argument("--cand", required=True)
    r.add_argument("--langs", default="all")
    r.add_argument("--n", type=int, default=None)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--work", default=None)
    r.add_argument("--out", default=None)
    c = sp.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args(argv)
    for v in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT", "LINGOLIX_API_KEY"):
        os.environ.pop(v, None)
    if a.cmd == "compare":
        res = compare_dirs(Path(a.a), Path(a.b))
        print(json.dumps(res, indent=1))
        sys.exit(0 if res["identical"] else 1)
    langs = list(LANGS) if a.langs == "all" else a.langs.split(",")
    res = run(load_spec(a.ref, "ref"), load_spec(a.cand, "cand"), langs, a.n, a.threads,
              Path(a.work) if a.work else None)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    sys.exit(0 if res["identical"] else 1)


if __name__ == "__main__":
    main()
