"""The zero-regression quality gate for size / speed candidates.

The user's rule: "Hard abort and revert any optimization that drops speechocean762 PCC by more than
0.002, degrades any language materially, or flips a single Microsoft Azure win."

    # once per code version and precision (runs the whole held-out suite; ~see STATUS):
    python -m bench.opt.gate record-baseline [--precision int8] [--code HEAD]
    # a candidate: backbone file, adapter folder, ONNX Runtime options and / or code
    python -m bench.opt.gate check --candidate '{"name":"int8-q2","backbone":"/path/model.onnx"}'
    python -m bench.opt.gate check --candidate '{"name":"wt","code":"worktree"}' --screen   # may only DISCARD
    python -m bench.opt.gate compare --baseline <run dir> --candidate <run dir>             # decision only

record-baseline and check run the suite through the machine-wide lock (they re-run themselves
under `python -m bench.heavy --lane cpu --need 25 --floor 12`). Azure numbers come from cached
responses only; every credential variable is removed from every job.

Decision (candidate vs baseline of the same precision; bench/opt/rows.py gives every scoreboard
row, bench/opt/paired.py the paired tests):
  FAIL if
  1. a speechocean762 Pearson row (sounds, words, the four sentence scores) drops by more than 0.002;
  2. an Azure tally row (84) moves down. The scoreboard's five classes are ordered ahead
     (significant) > ahead (n.s.) > level > behind (n.s.) > behind (significant); any move down
     fails, including the loss of significance of a win (default --class-rule strict; --class-rule
     coarse = only ahead -> level / behind and level -> behind, the architect's wording);
  3. a language degrades materially:
     a. any row of that language is significantly worse than the baseline (two-sided p < 0.05 in
        the row's own paired test, baseline in the place of Azure, worse direction), or
     b. a detection-AUC row drops by more than 0.002, or
     c. a native false-alarm row rises by more than 0.1 point (native acceptance falls by more
        than 0.1 point);
  4. a row has no paired test and no fixed tolerance (rates without per-clip outputs) and is
     worse at all (conservative; the five coaching figures quoted in running text are listed only);
  5. a row of the baseline is missing from the candidate.
  PASS otherwise. Every row that moved is printed, also inside tolerance.
  Identity: when every per-clip output file of the candidate's run equals the baseline's (timing
  fields aside) and every table value is equal, the gate passes by identity and says so.
  Screen (--screen): speechocean762 test (all of eval_english's rows) plus a fixed sample of the
  swap benchmark per language (the first 30 source recordings by id, all variants). It can only
  DISCARD (rule 1 on speechocean762, or a sample metric significantly worse, p < 0.05); a candidate
  that survives it still needs the full check.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import (child_env, git, load_spec, machine, materialize, opt_root, spec_hash)

ORDER = ("ahead (significant)", "ahead (not significant)", "level", "behind (not significant)", "behind (significant)")
COARSE = {"ahead (significant)": 0, "ahead (not significant)": 0, "level": 1, "behind (not significant)": 2,
          "behind (significant)": 2}
PCC_TOL = 0.002
AUC_TOL = 0.002
FA_TOL = 0.001                      # 0.1 percentage point, as a fraction
ALPHA = 0.05
PROSE_AREAS = ("coaching (heard)", "coaching (features)")
SELF_CHECK_TOL = 1e-9


# ----------------------------------------------------------------------------- decision (pure)

def decide(base_rows: list[dict], cand_rows: list[dict], paired: dict | None = None, class_rule: str = "strict",
           paired_errors: dict | None = None) -> dict:
    """PASS / FAIL with reasons, and every row that moved."""
    paired = paired or {}
    B = {r["key"]: r for r in base_rows}
    C = {r["key"]: r for r in cand_rows}
    fails, moved, notes = [], [], []
    n_paired = n_selfcheck_bad = n_p = n_sig = n_sig_worse = 0
    for key, b in B.items():
        c = C.get(key)
        if c is None:
            fails.append({"rule": "missing row", "key": key, "budget": True, "text": f"{key}: row missing from the candidate"})
            continue
        vb, vc = b["value"], c["value"]
        if vb is None or vc is None or (isinstance(vb, float) and math.isnan(vb)) or (isinstance(vc, float) and math.isnan(vc)):
            if not (vb is None and vc is None) and not (_nan(vb) and _nan(vc)):
                fails.append({"rule": "missing value", "key": key, "budget": True, "text": f"{key}: value {vb} -> {vc}"})
            continue
        dv = float(vc) - float(vb)
        hib = b["hib"]
        worse = dv < 0 if hib else dv > 0
        loss = -dv if hib else dv                       # > 0 = worse by this much
        pr = paired.get(key)
        selfcheck_ok = pr is not None and abs(pr["value_base"] - vb) <= SELF_CHECK_TOL and \
            abs(pr["value_cand"] - vc) <= SELF_CHECK_TOL
        if pr is not None and not selfcheck_ok:
            n_selfcheck_bad += 1
            notes.append(f"{key}: paired recomputation does not match the tables "
                         f"({pr['value_base']:.6g}/{pr['value_cand']:.6g} vs {vb:.6g}/{vc:.6g}); value rules only")
        if selfcheck_ok:
            n_paired += 1
        row_fail = []
        if b["kind"] == "pcc" and loss > PCC_TOL:
            row_fail.append((f"speechocean762 Pearson drops by {loss:.4f} (> {PCC_TOL})", True))
        if b["kind"] == "detection_auc" and loss > AUC_TOL:
            row_fail.append((f"detection AUC drops by {loss:.4f} (> {AUC_TOL})", True))
        if b["kind"] == "native_fa" and loss > FA_TOL:
            row_fail.append((f"native false alarms rise by {100 * loss:.2f} points (> {100 * FA_TOL:.1f})", True))
        if b["kind"] == "native_accept" and loss > FA_TOL:
            row_fail.append((f"native acceptance falls by {100 * loss:.2f} points (> {100 * FA_TOL:.1f})", True))
        if b.get("tally"):
            cb, cc = b["cat"], c["cat"]
            down = ORDER.index(cc) > ORDER.index(cb) if class_rule == "strict" else COARSE[cc] > COARSE[cb]
            if down:
                row_fail.append((f"Azure tally class {cb} -> {cc}", True))
        if selfcheck_ok and worse and pr["p"] < ALPHA:
            row_fail.append((f"significantly worse than the baseline (paired, p = {pr['p']:.3g}, "
                             f"delta {pr['delta']:+.4f} [{pr['ci'][0]:+.4f}, {pr['ci'][1]:+.4f}])", False))
        if not selfcheck_ok and worse and b["kind"] in ("rate", "other"):
            if b["area"] in PROSE_AREAS:
                notes.append(f"{key}: worse by {loss:.4g} (figure quoted in running text, no per-clip test; not gating)")
            else:
                row_fail.append((f"worse by {loss:.4g} and no paired test is available for this row", False))
        if dv != 0 or (b.get("cat") != c.get("cat")):
            moved.append({"key": key, "lang": b["lang"], "kind": b["kind"], "base": vb, "cand": vc, "delta": dv,
                          "worse": worse, "cat_base": b.get("cat"), "cat_cand": c.get("cat"),
                          "p": pr["p"] if selfcheck_ok else None, "ci": pr["ci"] if selfcheck_ok else None,
                          "fails": [t for t, _ in row_fail], "budget_fails": [t for t, bud in row_fail if bud]})
        if selfcheck_ok and np.isfinite(pr["p"]):
            n_p += 1
            n_sig += pr["p"] < ALPHA
            n_sig_worse += worse and pr["p"] < ALPHA
        for t, bud in row_fail:
            fails.append({"rule": t.split(" (")[0], "key": key, "lang": b["lang"], "budget": bud, "text": f"{key}: {t}"})
    extra = sorted(set(C) - set(B))
    if extra:
        notes.append(f"{len(extra)} rows only in the candidate: {extra[:5]}")
    for area, err in (paired_errors or {}).items():
        notes.append(f"paired tests of {area} not computed ({err}); its rows use the value rules")
    langs_failed = sorted({f.get("lang") for f in fails if f.get("lang")})
    bfails = [f for f in fails if f.get("budget")]
    listed = [f for f in fails if not f.get("budget")]
    chance = {"rows_with_paired_p": n_p, "expected_p_below_0.05_by_chance": round(ALPHA * n_p, 1),
              "observed_p_below_0.05": int(n_sig), "expected_worse_and_p_below_0.05": round(ALPHA / 2 * n_p, 1),
              "observed_worse_and_p_below_0.05": int(n_sig_worse)}
    return {"verdict": "FAIL" if fails else "PASS", "verdict_budget": "FAIL" if bfails else "PASS",
            "budget_fails": bfails, "budget_listed": listed, "chance": chance,
            "n_rows": len(B), "n_rows_paired": n_paired,
            "n_selfcheck_mismatch": n_selfcheck_bad, "n_moved": len(moved), "fails": fails, "moved": moved,
            "notes": notes, "languages_failed": langs_failed, "class_rule": class_rule}


def _nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ----------------------------------------------------------------------------- identity of two runs

IGNORE_KEY = re.compile(r"(timing|latency|elapsed|_ms$|^seconds$|^date$|^t$|fingerprint|^dir$|ours_cache|load_avg)")


def deep_equal(a, b, path="", diffs=None, limit=20) -> list:
    diffs = [] if diffs is None else diffs
    if len(diffs) >= limit:
        return diffs
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if IGNORE_KEY.search(str(k)):
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: only in one")
                continue
            deep_equal(a[k], b[k], f"{path}.{k}", diffs, limit)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep_equal(x, y, f"{path}[{i}]", diffs, limit)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or x.dtype.kind != y.dtype.kind:
            diffs.append(f"{path}: array shape/type {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}")
        elif x.dtype.kind in "fc":
            if not np.array_equal(x, y, equal_nan=True):
                diffs.append(f"{path}: arrays differ (max |d| {np.nanmax(np.abs(x - y)):.3g})")
        elif not np.array_equal(x, y):
            diffs.append(f"{path}: arrays differ")
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        pass
    elif a != b:
        diffs.append(f"{path}: {str(a)[:50]!r} vs {str(b)[:50]!r}")
    return diffs


def _outputs(run: Path) -> dict[str, Path]:
    """Per-clip output files of a suite run, keyed by a run-independent name."""
    c = Path(run) / "cache"
    out = {}
    for p in sorted((c / "calib" / "api").glob("*.pkl")):
        out[f"api/{p.name}"] = p
    for p in sorted((c / "bench" / "coach").glob("dump-*.pkl")):
        if not p.is_symlink():
            out[f"coach/{p.name}"] = p
    for p in sorted((c / "audit" / "contrast_v1_test").glob("*/chunk_*.pkl")):
        out[f"audit/{p.parent.name}/{p.name}"] = p
    for p in sorted((c / "opt_dumps").glob("*.pkl")):
        out[f"dumps/{p.name}"] = p
    lat = c / "cv-real" / "ours" / "latest.json"
    if lat.exists():
        d = c / "cv-real" / "ours" / json.loads(lat.read_text())["fingerprint"]
        for p in sorted(d.glob("*.json")):
            if p.name != "_meta.json":
                out[f"cv/{p.name}"] = p
    for lat in sorted((c / "ompal" / "ours").glob("latest_*.json")):
        d = c / "ompal" / "ours" / json.loads(lat.read_text())["dir"]
        for p in sorted(d.glob("*.json")):
            if p.name != "_meta.json":
                out[f"ompal/{p.name}"] = p
    return out


def _load_any(p: Path):
    return pickle.loads(p.read_bytes()) if p.suffix == ".pkl" else json.loads(p.read_text())


def identity(base: Path, cand: Path, tables_base: Path, tables_cand: Path, log=print) -> dict:
    ob, oc = _outputs(base), _outputs(cand)
    res = {"files_base": len(ob), "files_cand": len(oc), "missing": sorted(set(ob) ^ set(oc))[:20], "differ": {}}
    for k in sorted(set(ob) & set(oc)):
        d = deep_equal(_load_any(ob[k]), _load_any(oc[k]), k, limit=3)
        if d:
            res["differ"][k] = d
    tb = {p.name: p for p in Path(tables_base).glob("*.json")}
    tc = {p.name: p for p in Path(tables_cand).glob("*.json")}
    from .suite import table_outputs

    res["tables_differ"] = {}
    for name in table_outputs():
        if name not in tb or name not in tc:
            res["tables_differ"][name] = ["missing"]
            continue
        d = deep_equal(json.loads(tb[name].read_text()), json.loads(tc[name].read_text()), name, limit=3)
        if d:
            res["tables_differ"][name] = d
    res["identical"] = not res["missing"] and not res["differ"] and not res["tables_differ"] and len(ob) > 0
    log(f"identity: {len(ob)} per-clip output files; {len(res['differ'])} differ; "
        f"{len(res['tables_differ'])} tables differ -> identical={res['identical']}")
    return res


# ----------------------------------------------------------------------------- runs

def _in_heavy(argv: list[str]) -> None:
    """Re-run this command under the machine-wide lock unless already inside it."""
    if os.environ.get("OPT_IN_HEAVY") == "1":
        return
    env = {k: v for k, v in os.environ.items() if k not in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION",
                                                           "AZURE_SPEECH_ENDPOINT", "LINGOLIX_API_KEY")}
    env["OPT_IN_HEAVY"] = "1"
    cmd = [sys.executable, "-m", "bench.heavy", "--lane", "cpu", "--need", "25", "--floor", "12", "--",
           sys.executable, "-m", "bench.opt.gate", *argv]
    sys.exit(subprocess.run(cmd, env=env).returncode)


def baseline_dir(sha: str, precision: str) -> Path:
    return opt_root() / "baseline" / sha[:12] / precision


def latest_baseline(precision: str) -> Path:
    c = sorted((opt_root() / "baseline").glob(f"*/{precision}/rows.json"), key=lambda p: p.stat().st_mtime)
    if not c:
        raise SystemExit(f"no recorded baseline for {precision}; run record-baseline first")
    return c[-1].parent


def finish_run(run: Path) -> list[dict]:
    from .rows import extract

    rows = extract(run / "src" / "results" / "tables")
    (run / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    so = {r["measure"]: r["value"] for r in rows if r["area"] == "speechocean762"}
    (run / "so762.json").write_text(json.dumps(so, indent=1))
    return rows


def record_baseline(code: str, precision: str, parallel: int, serial: bool, log=print) -> Path:
    from .suite import run_suite, step_times

    sha = git("rev-parse", "--verify", code + "^{commit}")
    spec = load_spec({"name": f"baseline-{precision}", "code": sha, "precision": precision})
    run = baseline_dir(sha, precision)
    t0 = time.time()
    run_suite(spec, run, parallel=parallel, serial=serial, log=log)
    rows = finish_run(run)
    meta = {"spec": spec, "rows": len(rows), "wall_s": round(time.time() - t0), "steps": step_times(run),
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine()}
    (run / "baseline.json").write_text(json.dumps(meta, indent=1))
    log(f"baseline recorded: {run} ({len(rows)} rows, {meta['wall_s']} s)")
    return run


def compare_runs(base: Path, cand: Path, class_rule: str = "strict", tables_base: Path | None = None,
                 tables_cand: Path | None = None, skip_identity: bool = False, log=print) -> dict:
    from . import paired as PR
    from .rows import extract

    tb = Path(tables_base or base / "src" / "results" / "tables")
    tc = Path(tables_cand or cand / "src" / "results" / "tables")
    t0 = time.time()
    out = {"kind": "opt_gate", "date": time.strftime("%Y-%m-%d %H:%M"), "baseline": str(base), "candidate": str(cand)}
    if not skip_identity:
        idn = identity(base, cand, tb, tc, log)
        out["identity"] = idn
    rb, rc = extract(tb), extract(tc)
    if not skip_identity and out["identity"]["identical"]:
        d = decide(rb, rc, {}, class_rule)
        d["verdict"] = "PASS" if not d["fails"] else "FAIL"
        d["by_identity"] = True
        out.update(d)
        out["summary"] = ("PASS by identity: every per-clip output and every table value equals the baseline's"
                          if not d["fails"] else "outputs identical but rows differ (should not happen): FAIL")
    else:
        paired, errors = PR.all_rows(base, cand, log=log)
        d = decide(rb, rc, paired, class_rule, errors)
        d["by_identity"] = False
        out.update(d)
        out["summary"] = f"strict {d['verdict']} / budget {d['verdict_budget']}: {len(d['fails'])} failing checks " \
                         f"({len(d['budget_fails'])} under the budget rule), {d['n_moved']} rows moved, " \
                         f"{d['n_rows_paired']} of {d['n_rows']} rows with a paired test"
    out["analysis_s"] = round(time.time() - t0)
    return out


def print_decision(r: dict, file=sys.stdout) -> None:
    w = lambda s="": print(s, file=file)  # noqa: E731
    w(f"== gate: {r['summary']}")
    if "verdict_budget" in r:
        w(f"== strict rule: {r['verdict']} ({len(r['fails'])} failing checks); budget rule: {r['verdict_budget']} "
          f"({len(r.get('budget_fails', []))} failing checks)")
        c = r.get("chance", {})
        if c:
            w(f"== chance: {c['rows_with_paired_p']} rows with a paired p; expected p < 0.05 by chance "
              f"{c['expected_p_below_0.05_by_chance']}, observed {c['observed_p_below_0.05']}; worse and p < 0.05: "
              f"expected {c['expected_worse_and_p_below_0.05']}, observed {c['observed_worse_and_p_below_0.05']}")
    if r.get("fails"):
        w(f"-- failing checks ({len(r['fails'])}; [budget] = fails under the budget rule too, [listed] = strict only):")
        for f in r["fails"]:
            w(f"   FAIL {'[budget]' if f.get('budget') else '[listed]'} {f['text']}")
    w(f"-- every row that moved ({r['n_moved']}), candidate - baseline:")
    for m in sorted(r["moved"], key=lambda m: (bool(not m["fails"]), m["key"])):
        p = f" p={m['p']:.3g}" if m.get("p") is not None else " (no paired test)"
        cat = f" [{m['cat_base']} -> {m['cat_cand']}]" if m.get("cat_base") and m["cat_base"] != m["cat_cand"] else ""
        flag = "FAIL" if m["fails"] else ("worse" if m["worse"] else "better")
        w(f"   {flag:6s} {m['key']}: {m['base']:.4f} -> {m['cand']:.4f} ({m['delta']:+.4f}){p}{cat}")
    for n in r.get("notes", []):
        w(f"-- note: {n}")


def check(spec: dict, baseline: Path | None, parallel: int, serial: bool, class_rule: str, log=print) -> dict:
    from .suite import run_suite, step_times

    base = baseline or latest_baseline(spec["precision"])
    bmeta = json.loads((base / "baseline.json").read_text())
    if bmeta["spec"]["precision"] != spec["precision"]:
        raise SystemExit(f"baseline precision {bmeta['spec']['precision']} != candidate {spec['precision']}")
    tree = materialize(spec["code"])
    run = opt_root() / "checks" / f"{spec['name']}-{spec_hash(spec, tree)}"
    t0 = time.time()
    run_suite(spec, run, parallel=parallel, serial=serial, log=log)
    finish_run(run)
    res = compare_runs(base, run, class_rule, log=log)
    res["suite_s"] = round(time.time() - t0) - res["analysis_s"]
    res["steps"] = step_times(run)
    res["spec"] = spec
    (run / "gate.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return res


# ----------------------------------------------------------------------------- screen

SCREEN_SWAP_SOURCES = 30

_SCREEN_SWAP = r"""
import pickle, sys
from concurrent.futures import ProcessPoolExecutor
from bench.calib.api_run import _init
from bench.eval_languages import _swap_one
recs = pickle.load(open(sys.argv[1], "rb"))
out = {}
by = {}
for r in recs:
    by.setdefault(r["lang"], []).append(r)
for lang, rs in sorted(by.items()):
    with ProcessPoolExecutor(3, initializer=_init, initargs=(lang, "fp32", 1)) as ex:
        out.update(dict(ex.map(_swap_one, rs, chunksize=6)))
pickle.dump(out, open(sys.argv[2], "wb"))
"""


def screen_records():
    from bench.data import load

    recs = load("fleurs-swap", split="test")
    out = []
    for lang in sorted({r["lang"] for r in recs}):
        srcs = sorted({r["labels"]["src_id"] for r in recs if r["lang"] == lang}, key=str)[:SCREEN_SWAP_SOURCES]
        keep = set(srcs)
        out += [r for r in recs if r["lang"] == lang and r["labels"]["src_id"] in keep]
    return out


def screen(spec: dict, baseline: Path | None, log=print) -> dict:
    from bench.eval_swap import paired

    from .rows import extract
    from .suite import prepare, step_env

    base = baseline or latest_baseline(spec["precision"])
    tree = materialize(spec["code"])
    run = opt_root() / "screens" / f"{spec['name']}-{spec_hash(spec, tree)}"
    prepare(spec, run)
    env = step_env(spec, run)
    t0 = time.time()
    log_f = open(run / "logs" / "screen.log", "a")
    if not (run / "logs" / "english.done").exists():
        subprocess.run([sys.executable, "-m", "bench.eval_english", "--steps", "run,so762", "--workers", "4"],
                       cwd=run / "src", env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)
        (run / "logs" / "english.done").write_text("screen\n")
    recs = screen_records()
    rp, op = run / "cache" / "screen_swap_recs.pkl", run / "cache" / "screen_swap_out.pkl"
    if not op.exists():
        rp.write_bytes(pickle.dumps(recs))
        subprocess.run([sys.executable, "-c", _SCREEN_SWAP, str(rp), str(op)], cwd=run / "src", env=env,
                       stdout=log_f, stderr=subprocess.STDOUT, check=True)
    oc = {k: v["on"] for k, v in pickle.loads(op.read_bytes()).items()}
    ob = {}
    for lang in sorted({r["lang"] for r in recs}):
        p = base / "cache" / "calib" / "api" / f"fleurs-swap-{lang}.pkl"
        ob.update({k: v["on"] for k, v in pickle.loads(p.read_bytes()).items() if k in oc})
    P = paired(recs, oc, ob)
    reasons, moved = [], []
    rb = {r["key"]: r for r in json.loads((base / "rows.json").read_text()) if r["area"] == "speechocean762"}
    rc = {r["key"]: r for r in extract(run / "src" / "results" / "tables") if r["area"] == "speechocean762"}
    for k, b in rb.items():
        c = rc.get(k)
        if c is None:
            reasons.append(f"{k}: missing")
            continue
        loss = b["value"] - c["value"]
        moved.append(f"{k}: {b['value']:.4f} -> {c['value']:.4f} ({-loss:+.4f})")
        if loss > PCC_TOL:
            reasons.append(f"{k}: Pearson drops by {loss:.4f} (> {PCC_TOL})")
    for lang, m in P.items():
        for k, x in m.items():
            if not isinstance(x, dict):
                continue
            worse = x["delta"] < 0 if x.get("higher_is_better", True) else x["delta"] > 0
            if x["delta"] != 0:
                moved.append(f"swap sample {lang} {k}: {x['b']:.4f} -> {x['a']:.4f} ({x['delta']:+.4f}, p={x['p_two_sided']:.3g})")
            if worse and x["p_two_sided"] < ALPHA:
                reasons.append(f"swap sample {lang} {k}: significantly worse (p = {x['p_two_sided']:.3g})")
    res = {"kind": "opt_screen", "spec": spec, "baseline": str(base), "run": str(run),
           "verdict": "DISCARD" if reasons else "SURVIVES (full check still required)", "reasons": reasons,
           "moved": moved, "screen_s": round(time.time() - t0), "n_swap_records": len(recs)}
    (run / "screen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return res


def tables_vs_rev(run: Path, rev: str, log=print) -> dict:
    """The run's tables against the tables committed at `rev`, key by key (timing fields aside;
    top-level keys the suite does not compute, e.g. eval_english's speed / int8 blocks, are listed
    separately). Also the scoreboard rows (bench/opt/rows.py) value by value."""
    from .rows import extract
    from .suite import table_outputs

    import tempfile

    out = {"rev": rev, "run": str(run), "tables": {}}
    with tempfile.TemporaryDirectory() as tmp:
        data = subprocess.run(["git", "archive", rev, "results/tables"], cwd=Path(__file__).resolve().parents[2],
                              check=True, capture_output=True).stdout
        import io
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(tmp, filter="data")
        T = Path(tmp) / "results" / "tables"
        R = run / "src" / "results" / "tables"
        for name in table_outputs():
            a, b = json.loads((T / name).read_text()), json.loads((R / name).read_text())
            only = sorted(set(a) - set(b)) if isinstance(a, dict) else []
            if isinstance(a, dict):
                a = {k: v for k, v in a.items() if k in b}
            d = deep_equal(a, b, name, limit=20)
            out["tables"][name] = {"differences": d, "keys_only_committed": only}
            log(f"{name}: {'equal' if not d else f'{len(d)} differences'}"
                + (f" (keys not computed by the suite: {only})" if only else ""))
        rc, rr = {r["key"]: r for r in extract(T)}, {r["key"]: r for r in extract(R)}
        out["rows_differ"] = [{"key": k, "committed": rc[k]["value"], "run": rr.get(k, {}).get("value"),
                               "cat": [rc[k]["cat"], rr.get(k, {}).get("cat")]}
                              for k in rc if rr.get(k) is None or rr[k]["value"] != rc[k]["value"]
                              or rr[k]["cat"] != rc[k]["cat"]]
        out["n_rows"] = len(rc)
    out["equal"] = not out["rows_differ"] and all(not v["differences"] for v in out["tables"].values())
    log(f"rows: {out['n_rows']}, differing {len(out['rows_differ'])}; all tables equal: {out['equal']}")
    return out


# ----------------------------------------------------------------------------- main

def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("record-baseline")
    b.add_argument("--code", default="HEAD")
    b.add_argument("--precision", default="int8")
    c = sp.add_parser("check")
    c.add_argument("--candidate", required=True)
    c.add_argument("--screen", action="store_true")
    for x in (b, c):
        x.add_argument("--parallel", type=int, default=3)
        x.add_argument("--serial", action="store_true")
        x.add_argument("--no-lock", action="store_true", help="do not re-run under bench.heavy (tests only)")
    c.add_argument("--baseline", default=None)
    c.add_argument("--class-rule", default="strict", choices=("strict", "coarse"))
    m = sp.add_parser("compare")
    m.add_argument("--baseline", required=True)
    m.add_argument("--candidate", required=True)
    m.add_argument("--tables-base", default=None)
    m.add_argument("--tables-cand", default=None)
    m.add_argument("--class-rule", default="strict", choices=("strict", "coarse"))
    m.add_argument("--skip-identity", action="store_true")
    for x in (c, m):
        x.add_argument("--out", default=None)
    tv = sp.add_parser("tables-vs-rev")
    tv.add_argument("--run", required=True)
    tv.add_argument("--rev", required=True)
    tv.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "tables-vs-rev":
        r = tables_vs_rev(Path(a.run), a.rev)
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1, ensure_ascii=False, default=float))
        sys.exit(0 if r["equal"] else 1)
    for v in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT", "LINGOLIX_API_KEY"):
        os.environ.pop(v, None)
    if a.cmd in ("record-baseline", "check") and not a.no_lock:
        _in_heavy(argv)
    if a.cmd == "record-baseline":
        record_baseline(a.code, a.precision, a.parallel, a.serial)
        return
    if a.cmd == "compare":
        r = compare_runs(Path(a.baseline), Path(a.candidate), a.class_rule,
                         Path(a.tables_base) if a.tables_base else None,
                         Path(a.tables_cand) if a.tables_cand else None, a.skip_identity)
    else:
        spec = load_spec(a.candidate, "candidate")
        base = Path(a.baseline) if a.baseline else None
        if a.screen:
            r = screen(spec, base)
            print(json.dumps(r, indent=1, ensure_ascii=False))
            if a.out:
                Path(a.out).write_text(json.dumps(r, indent=1, ensure_ascii=False))
            sys.exit(1 if r["verdict"] == "DISCARD" else 0)
        r = check(spec, base, a.parallel, a.serial, a.class_rule)
    print_decision(r)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1, ensure_ascii=False, default=float))
    sys.exit(0 if r["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
