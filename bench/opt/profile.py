"""End-to-end profiling of the scoring path: per-stage latency, cold start, peak RSS, disk footprint.

    # latency, one or more configurations compared on the same clips (paired, interleaved):
    python -m bench.heavy --lane cpu --need 25 --floor 12 -- python -m bench.opt.profile latency \
        --configs '{"name":"int8"}','{"name":"fp32","precision":"fp32"}' --langs all --threads 1,2,4 \
        --set practice --repeats 3 --out results/tables/opt_profile_baseline.json
    python -m bench.opt.profile rss --configs ... --langs en,ja --threads 1,4 --n 20 --out ...
    python -m bench.opt.profile footprint --config '{"name":"int8"}' --out results/tables/opt_footprint.json
    python -m bench.opt.profile selfcheck --config '{"name":"int8"}' --langs en,zh --n 3

Configurations are bench/opt/common.py specs (code, precision, backbone, adapter_dir, ort_options).

Latency protocol. Per (language, threads), one fresh process per configuration (bench/opt/_child.py).
Each process is started only after the previous one has loaded and scored its first clip, so load
and cold start are not measured under contention: load = launch -> Scorer built (imports included),
cold start = launch -> first score returned (clip 1). Then `repeats` rounds over every clip; inside
a round the configurations score each clip in turn (A B, A B, ...) and the order flips every round
(B A, ...), so slow drifts of the machine hit every configuration alike. Each call clears the
Scorer's posterior cache and passes the audio file path, so decoding is inside every timed call.
Stages are the Scorer's own SentenceResult.timing_ms (load = audio decode + resample, fbank,
model = ZIPA backbone, adapter, align, features, delivery, calibrate, sentence_head, prosody,
length, word_step, explain_away, assemble); wall = perf_counter around score(). No instrumentation
was added to the package for this: `selfcheck` shows that the profiling path (cache cleared every
call) gives outputs identical to plain repeated calls.
Steady-state statistics are over every (clip, round) sample: p50, mean, p90. The paired comparison
of configuration B against the first configuration A: per clip the median over rounds of wall ms,
d = B - A, mean and median of d over clips, 95 % clip-bootstrap interval of the mean (2,000
resamples, seed 0).

Peak RSS is the process's own maximum resident set size (getrusage), one fresh process per
(configuration, language, threads): (a) after loading only, (b) after scoring the clips.

Footprint: every file of the package (from the configuration's source tree) and of the acoustic
model folder, with its size; which files a Scorer of each language opens (an audit hook on open()
plus every ONNX Runtime session path, while building the Scorer and scoring one clip with
details(), coaching(), tips and the Azure layout); and the smallest install per language =
Python code + the backbone file it loads (+ the extra-outputs copy built next to it on first use,
when the language reads encoder activations) + tokens + the data files it opens. Third-party
packages that a language imports beyond the core runtime are listed with their installed size.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import LANGS, REPO, load_spec, machine, materialize, model_dir, practice_clips, so762_clips
from .proc import Child, load_only

STAGE_ORDER = ("load", "g2p", "audio_check", "fbank", "model", "adapter", "align", "features", "delivery",
               "calibrate", "sentence_head", "explain_away", "prosody", "length", "word_step", "assemble",
               "total", "wall")


def _langs(s: str) -> list[str]:
    return list(LANGS) if s == "all" else s.split(",")


def _clips(set_name: str, lang: str, n: int | None) -> list[dict]:
    if set_name == "so762":
        if lang != "en":
            raise SystemExit("the so762 set is English only")
        return so762_clips(n)
    return practice_clips([lang], n)


def _configs(s: list[str]) -> list[dict]:
    out = [load_spec(x, f"c{i}") for i, x in enumerate(s)]
    if len({c["name"] for c in out}) != len(out):
        raise SystemExit("configuration names must differ")
    return out


def _split_configs(s: str) -> list[str]:
    """Comma-separated specs; commas inside JSON objects are kept."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "{"
        depth -= ch == "}"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + ([cur] if cur else [])


def summarize(values) -> dict:
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "p50": float(np.median(v)), "mean": float(v.mean()), "p90": float(np.quantile(v, 0.9))}


def paired_diff(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """a, b: {clip id: [wall ms per round]} -> B - A on per-clip medians, clip bootstrap of the mean."""
    ids = sorted(set(a) & set(b))
    d = np.array([np.median(b[i]) - np.median(a[i]) for i in ids], float)
    ma = np.array([np.median(a[i]) for i in ids], float)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"n_clips": int(len(ids)), "mean_diff_ms": float(d.mean()), "median_diff_ms": float(np.median(d)),
            "mean_diff_ci": [float(lo), float(hi)], "rel_mean": float(d.mean() / ma.mean()) if ma.mean() else None}


# ----------------------------------------------------------------------------- latency

def latency(configs: list[dict], langs, threads, set_name: str, n: int | None, repeats: int,
            log=print) -> dict:
    trees = {c["name"]: materialize(c["code"]) for c in configs}
    out = {"kind": "opt_profile", "date": time.strftime("%Y-%m-%d %H:%M"), "machine": machine(),
           "configs": configs, "trees": {k: str(v.name) for k, v in trees.items()}, "clip_set": set_name,
           "repeats": repeats, "results": [], "paired": []}
    for lang in langs:
        clips = _clips(set_name, lang, n)
        for th in threads:
            kids, rows = {}, {}
            try:
                for c in configs:
                    k = Child(c, trees[c["name"]], lang, th)
                    first = k.score(clips[0])
                    if "error" in first:
                        raise RuntimeError(f"first clip failed: {first}")
                    kids[c["name"]] = k
                    rows[c["name"]] = {"config": c["name"], "lang": lang, "threads": th,
                                       "load_s": k.info["t"] - k.launched, "cold_start_s": first["t"] - k.launched,
                                       "maxrss_loaded_mb": k.info["maxrss_mb"], "samples": [], "errors": 0}
                names = [c["name"] for c in configs]
                for r in range(repeats):
                    order = names if r % 2 == 0 else names[::-1]
                    for clip in clips:
                        for name in order:
                            res = kids[name].score(clip)
                            if "error" in res:
                                rows[name]["errors"] += 1
                                continue
                            rows[name]["samples"].append({"id": clip["id"], "round": r, "wall_ms": res["wall_ms"],
                                                          **{f"t_{k}": v for k, v in res["timing_ms"].items()}})
                for name, k in kids.items():
                    rows[name]["maxrss_after_mb"] = k.ask(op="rss")["maxrss_mb"]
                    rows[name]["n_scored_after_load"] = 1 + len(rows[name]["samples"]) + rows[name]["errors"]
            finally:
                for k in kids.values():
                    k.close()
            for name, row in rows.items():
                S = row.pop("samples")
                keys = sorted({k[2:] for s in S for k in s if k.startswith("t_")},
                              key=lambda k: STAGE_ORDER.index(k) if k in STAGE_ORDER else 99)
                row["stages"] = {k: summarize([s.get(f"t_{k}") for s in S]) for k in keys}
                row["stages"]["wall"] = summarize([s["wall_ms"] for s in S])
                per = defaultdict(list)
                for s in S:
                    per[s["id"]].append(s["wall_ms"])
                row["per_clip_wall_ms"] = {i: [round(x, 3) for x in v] for i, v in per.items()}
                row["audio_s"] = {c["id"]: c["dur"] for c in clips}
                out["results"].append(row)
                w = row["stages"]["wall"]
                log(f"{name} {lang} t{th}: load {row['load_s']:.2f}s cold {row['cold_start_s']:.2f}s "
                    f"wall p50 {w['p50']:.1f} ms mean {w['mean']:.1f} p90 {w['p90']:.1f} "
                    f"RSS {row['maxrss_after_mb']:.0f} MB")
            base = rows[configs[0]["name"]]["per_clip_wall_ms"]
            for c in configs[1:]:
                pd_ = paired_diff(base, rows[c["name"]]["per_clip_wall_ms"])
                out["paired"].append({"lang": lang, "threads": th, "a": configs[0]["name"], "b": c["name"], **pd_})
                log(f"  {c['name']} - {configs[0]['name']}: {pd_['mean_diff_ms']:+.2f} ms "
                    f"[{pd_['mean_diff_ci'][0]:+.2f}, {pd_['mean_diff_ci'][1]:+.2f}]")
    out["machine_after"] = {"load_avg_1min": os.getloadavg()[0]}
    return out


# ----------------------------------------------------------------------------- RSS

def rss(configs: list[dict], langs, threads, set_name: str, n: int, log=print) -> dict:
    trees = {c["name"]: materialize(c["code"]) for c in configs}
    out = {"kind": "opt_rss", "date": time.strftime("%Y-%m-%d %H:%M"), "machine": machine(), "configs": configs,
           "clip_set": set_name, "n_clips": n, "results": []}
    for c in configs:
        for lang in langs:
            clips = _clips(set_name, lang, n)
            for th in threads:
                a = load_only(c, trees[c["name"]], lang, th)
                k = Child(c, trees[c["name"]], lang, th)
                try:
                    errs = sum("error" in k.score(x) for x in clips)
                    after = k.ask(op="rss")["maxrss_mb"]
                finally:
                    k.close()
                row = {"config": c["name"], "lang": lang, "threads": th, "load_only_mb": a["maxrss_mb"],
                       "load_s": a["load_s"], "after_n_mb": after, "n_scored": len(clips), "errors": errs,
                       "embeddings": a["embeddings"]}
                out["results"].append(row)
                log(f"{c['name']} {lang} t{th}: load-only {a['maxrss_mb']:.0f} MB, after {len(clips)} clips {after:.0f} MB")
    return out


# ----------------------------------------------------------------------------- footprint

def classify(rel: str) -> str:
    """Category of a file of the package (path relative to the package root) or of the model folder."""
    if rel.startswith("models/"):
        name = rel.split("/")[-1]
        if name == "tokens.txt":
            return "backbone tokens"
        if ".emb" in name and ".ort" in name:
            return "backbone (runtime-built pre-optimized copy)"
        return "backbone (runtime-built extra-outputs copy)" if ".emb" in name else "backbone"
    p = rel.split("/")
    if rel.endswith((".py", ".pyi")) or p[-1] == "py.typed":
        return "code"
    if len(p) > 2 and p[1] == "data":
        sub = p[2]
        return {"adapters": "adapter", "lex": "lexicon", "cjk": "lexicon", "calib": "calibration",
                "coach": "coaching", "prosody": "prosody", "length": "length", "native": "native statistics"
                }.get(sub, "other data")
    return "other"


def footprint_table(files: dict[str, int], used: dict[str, set[str]], backbone_for: dict[str, list[str]] | None = None,
                    third_party: dict[str, dict[str, int]] | None = None) -> dict:
    """files: {relative path: bytes} (package files as "pronunciada/...", model files as "models/...");
    used: {lang: set of relative paths opened by a Scorer of that language};
    backbone_for: {lang: model-folder files it needs that the trace cannot see being built ahead
    of time (e.g. the extra-outputs copy)}; third_party: {lang: {package: bytes}}.
    Returns per-file rows, totals per category, and the smallest install per language."""
    backbone_for = backbone_for or {}
    third_party = third_party or {}
    used = {l: set(u) | set(backbone_for.get(l, [])) for l, u in used.items()}
    rows = []
    for rel, size in sorted(files.items()):
        langs = sorted(l for l, u in used.items() if rel in u)
        rows.append({"file": rel, "bytes": int(size), "category": classify(rel), "used_by": langs})
    cats = defaultdict(lambda: {"files": 0, "bytes": 0})
    for r in rows:
        c = cats[r["category"]]
        c["files"] += 1
        c["bytes"] += r["bytes"]
    code = sum(r["bytes"] for r in rows if r["category"] == "code")
    per = {}
    for lang in sorted(used):
        need = set(used[lang]) | set(backbone_for.get(lang, []))
        sel = [r for r in rows if r["file"] in need and r["category"] != "code"]
        by = defaultdict(int)
        for r in sel:
            by[r["category"]] += r["bytes"]
        tp = third_party.get(lang, {})
        per[lang] = {"code_bytes": code, "data_bytes": int(sum(r["bytes"] for r in sel)), "by_category": dict(by),
                     "total_bytes": int(code + sum(r["bytes"] for r in sel)),
                     "third_party_bytes": int(sum(tp.values())), "third_party": tp,
                     "files": sorted(r["file"] for r in sel)}
    unused = [r["file"] for r in rows if not r["used_by"] and r["category"] not in ("code",)]
    return {"files": rows, "by_category": dict(cats), "smallest_install": per,
            "package_bytes": int(sum(r["bytes"] for r in rows if r["file"].startswith("pronunciada/"))),
            "data_files_opened_by_no_language": unused}


CORE = {"numpy", "onnxruntime", "kaldi_native_fbank", "soundfile", "_soundfile", "_soundfile_data", "scipy",
        "pronunciada", "cffi", "_cffi_backend"}


def _dir_bytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def footprint(spec: dict, langs, log=print) -> dict:
    tree = materialize(spec["code"])
    pkg = tree / "pronunciada"
    files = {f"pronunciada/{p.relative_to(pkg)}": p.stat().st_size for p in pkg.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    md = model_dir()
    files.update({f"models/{p.name}": p.stat().st_size for p in md.iterdir()
                  if p.is_file() and p.suffix in (".onnx", ".txt") and not p.name.endswith(".part")})
    if spec.get("backbone"):
        b = Path(spec["backbone"])
        files[f"models/{b.name}"] = b.stat().st_size
    used, backbone_for, third = {}, {}, {}
    site = Path(np.__file__).resolve().parents[1]
    for lang in langs:
        clip = practice_clips([lang], 1)[0]
        k = Child(spec, tree, lang, 1, trace_files=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                r = k.score(clip, dump=tmp)
                if "error" in r:
                    raise RuntimeError(f"{lang}: {r}")
            opened = k.ask(op="files")["files"]
        finally:
            k.close()
        u = set()
        tp = defaultdict(int)
        for f in opened:
            p = Path(f).resolve()
            if pkg.resolve() in p.parents:
                u.add(f"pronunciada/{p.relative_to(pkg.resolve())}")
            elif md.resolve() in p.parents or (spec.get("backbone") and p == Path(spec["backbone"]).resolve()):
                u.add(f"models/{p.name}")
            elif site in p.parents:
                top = p.relative_to(site).parts[0]
                name = top.split("-")[0].split(".")[0]
                if name not in CORE and top != "__pycache__" and not top.endswith((".dist-info", ".pth")):
                    tp[top] = _dir_bytes(site / top)
        used[lang] = u
        third[lang] = dict(tp)
        if k.info.get("backbone"):
            backbone_for[lang] = [f"models/{Path(k.info['backbone']).name}"]
        log(f"footprint {lang}: {len(u)} files opened, third-party {sorted(tp)}")
    t = footprint_table(files, used, backbone_for, third)
    t.update(kind="opt_footprint", date=time.strftime("%Y-%m-%d %H:%M"), config=spec, tree=tree.name,
             note="models/* sizes are the files in the local model cache; the backbone is downloaded on first use, "
                  "not shipped in the wheel. A '.emb' file is the copy with extra encoder outputs that the first "
                  "Scorer builds next to the downloaded model.")
    return t


# ----------------------------------------------------------------------------- first-ever load

def first_load(configs: list[dict], lang: str, threads: int, rounds: int = 3, log=print) -> dict:
    """Load and cold start on a model cache that holds only the downloaded files (model.onnx,
    model.int8.onnx, tokens.txt, linked), so every runtime-built file is built by the first
    process; then `rounds` later processes on the same cache. Reports the files the first process
    created and their sizes."""
    import shutil

    from .common import MODEL, cache_root, default_backbone

    real = cache_root()
    out = {"kind": "opt_first_load", "date": time.strftime("%Y-%m-%d %H:%M"), "machine": machine(), "lang": lang,
           "threads": threads, "results": []}
    for c in configs:
        tree = materialize(c["code"])
        tmp = Path(tempfile.mkdtemp(prefix="optfirst-", dir=str(real / "opt")))
        try:
            md = tmp / "models" / MODEL
            md.mkdir(parents=True)
            for f, src in (("model.onnx", default_backbone("fp32")), ("model.int8.onnx", default_backbone("int8")),
                           ("tokens.txt", real / "models" / MODEL / "tokens.txt")):
                (md / f).symlink_to(src)
            (tmp / "data").symlink_to(real / "data")
            clip = practice_clips([lang], 1)[0]
            old = os.environ.get("PRONUNCIADA_CACHE")
            os.environ["PRONUNCIADA_CACHE"] = str(tmp)
            try:
                runs = []
                for r in range(1 + rounds):
                    before = {p.name for p in md.iterdir()}
                    k = Child(c, tree, lang, threads)
                    first = k.score(clip)
                    k.close()
                    made = sorted({p.name for p in md.iterdir()} - before)
                    runs.append({"round": r, "load_s": k.info["t"] - k.launched, "cold_start_s": first["t"] - k.launched,
                                 "files_created": {n: (md / n).stat().st_size for n in made}})
                    log(f"first-load {c['name']} round {r}: load {runs[-1]['load_s']:.2f}s cold "
                        f"{runs[-1]['cold_start_s']:.2f}s created {list(runs[-1]['files_created'])}")
            finally:
                if old is None:
                    os.environ.pop("PRONUNCIADA_CACHE", None)
                else:
                    os.environ["PRONUNCIADA_CACHE"] = old
            later = runs[1:]
            out["results"].append({"config": c["name"], "first": runs[0],
                                   "later_load_s_median": float(np.median([x["load_s"] for x in later])),
                                   "later_cold_start_s_median": float(np.median([x["cold_start_s"] for x in later])),
                                   "runs": runs,
                                   "model_dir_bytes_after": {p.name: p.stat().st_size for p in md.iterdir()
                                                             if not p.is_symlink()}})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


# ----------------------------------------------------------------------------- self-check

def selfcheck(spec: dict, langs, n: int, log=print) -> dict:
    """Outputs of the profiling path (posterior cache cleared before every call, each clip scored
    twice) against a fresh process scoring each clip once: must be identical."""
    from .parity import compare_dirs

    tree = materialize(spec["code"])
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for lang in langs:
            clips = practice_clips([lang], n)
            a, b = Path(tmp) / lang / "a", Path(tmp) / lang / "b"
            k = Child(spec, tree, lang, 2)
            try:
                for c in clips:
                    k.score(c)                     # first call: discarded, as in a latency run
                    k.score(c, dump=str(a))
            finally:
                k.close()
            k = Child(spec, tree, lang, 2)
            try:
                for c in clips:
                    k.score(c, dump=str(b))
            finally:
                k.close()
            out[lang] = compare_dirs(a, b)
            log(f"selfcheck {lang}: identical={out[lang]['identical']} max|diff|={out[lang]['max_abs_diff']}")
    return out


# ----------------------------------------------------------------------------- main

def write(obj: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {p}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("latency", "rss"):
        s = sp.add_parser(name)
        s.add_argument("--configs", default='{"name":"int8"}')
        s.add_argument("--langs", default="all")
        s.add_argument("--threads", default="1,2,4")
        s.add_argument("--set", default="practice", choices=("practice", "so762"))
        s.add_argument("--n", type=int, default=None)
        s.add_argument("--out", required=True)
        if name == "latency":
            s.add_argument("--repeats", type=int, default=3)
    s = sp.add_parser("footprint")
    s.add_argument("--config", default='{"name":"int8"}')
    s.add_argument("--langs", default="all")
    s.add_argument("--out", required=True)
    s = sp.add_parser("firstload")
    s.add_argument("--configs", default='{"name":"fp32","precision":"fp32"}')
    s.add_argument("--lang", default="en")
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--out", required=True)
    s = sp.add_parser("selfcheck")
    s.add_argument("--config", default='{"name":"int8"}')
    s.add_argument("--langs", default="en,zh")
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    for v in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT", "LINGOLIX_API_KEY"):
        os.environ.pop(v, None)
    if a.cmd == "latency":
        r = latency(_configs(_split_configs(a.configs)), _langs(a.langs), [int(x) for x in a.threads.split(",")],
                    a.set, a.n, a.repeats)
        write(r, a.out)
    elif a.cmd == "rss":
        r = rss(_configs(_split_configs(a.configs)), _langs(a.langs), [int(x) for x in a.threads.split(",")],
                a.set, a.n or 20)
        write(r, a.out)
    elif a.cmd == "firstload":
        write(first_load(_configs(_split_configs(a.configs)), a.lang, a.threads, a.rounds), a.out)
    elif a.cmd == "footprint":
        write(footprint(load_spec(a.config), _langs(a.langs)), a.out)
    else:
        r = selfcheck(load_spec(a.config), _langs(a.langs), a.n)
        if a.out:
            write(r, a.out)
        sys.exit(0 if all(x["identical"] for x in r.values()) else 1)


if __name__ == "__main__":
    main()
