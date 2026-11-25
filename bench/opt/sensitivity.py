"""Backbone-level measurements for small-model candidates, on PRACTICE audio only.

    # main venv: fbank features of the 200 parity clips and of a disjoint calibration set
    python -m bench.opt.sensitivity feats
    # any venv with onnxruntime: posterior error against fp32 and fragility, per backbone file
    python bench/opt/sensitivity.py posterr <name>=<file> [...] [--out f.json] [--n 200]
    # tools venv: per-group sensitivity (one group of MatMuls at 4 bits, the rest fp32)
    python bench/opt/sensitivity.py layers [--n 40] [--out f.json]

Features: Scorer path (pronunciada.audio.load_audio + AcousticModel._features). Parity set = the
committed practice clips (bench/opt/clips.json, 20 FLEURS-train recordings per language). Calibration
set = 12 other FLEURS-train recordings per language (numpy default_rng(7) over the ids not in the
parity set), used only to calibrate static int8 activation scales.

posterr: per frame the backbone's log-posteriors (graph output log_probs, before the adapter) of the
candidate against model.onnx (fp32) on the same features: max and mean |dlogp| over all entries,
mean and max per-frame KL(fp32 || candidate), share of frames whose top token differs; then the
fragility test of bench/opt/fragility.py (features + Gaussian noise of sd 1e-4 / 3e-4, seed 0, the
candidate against itself). Sessions: 2 intra-op threads.

layers: MatMuls with weights are grouped by encoder stack and module (e.g. /encoder/3/*/self_attn,
feed_forward1, conv_module1, ...). For each group, a model with only that group quantized to 4 bits
(MatMulNBits, symmetric, block 32) is scored on the first --n parity clips; reported: mean KL and
max |dlogp| against fp32, and the group's fp32 weight bytes.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
SENS = CACHE / "opt" / "sensitivity"
FP32 = CACHE / "models" / "zipa-small-crctc-500k" / "model.onnx"   # resolved in main(): the cache may
# hold only the one all-outputs file; bench/opt/common.py default_backbone restores the download


def feats() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from bench.data import load
    from bench.opt.common import LANGS, clip_list
    from pronunciada.acoustic import AcousticModel
    from pronunciada.audio import load_audio

    SENS.mkdir(parents=True, exist_ok=True)
    am = AcousticModel(precision="fp32", threads=1)
    ids = clip_list()["practice"]["ids"]
    par, cal, meta = {}, {}, {"parity": [], "calib": []}
    for li, lang in enumerate(LANGS):
        recs = sorted(load("fleurs", split="train", lang=lang), key=lambda r: r["id"])
        by = {r["id"]: r for r in recs}
        for i in ids[lang]:
            par[f"{len(par):04d}"] = am._features(load_audio(by[i]["audio"]))
            meta["parity"].append(i)
        rest = [r for r in recs if r["id"] not in set(ids[lang])]
        rng = np.random.default_rng(7 + li)
        for j in sorted(rng.choice(len(rest), 12, replace=False)):
            cal[f"{len(cal):04d}"] = am._features(load_audio(rest[j]["audio"]))
            meta["calib"].append(rest[j]["id"])
    np.savez(SENS / "parity_feats.npz", **par)
    np.savez(SENS / "calib_feats.npz", **cal)
    (SENS / "feats.json").write_text(json.dumps(meta, indent=1))
    print(f"parity {len(par)} clips, calib {len(cal)} clips -> {SENS}")


def _session(path, threads=2):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    so.enable_mem_pattern = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _run(sess, x):
    lp, n = sess.run(["log_probs", "log_probs_len"], {"x": x[None].astype(np.float32),
                                                       "x_lens": np.array([len(x)], np.int64)})
    return lp[0, :int(n[0])].astype(np.float64)


def _load_feats(n=None):
    z = np.load(SENS / "parity_feats.npz")
    keys = sorted(z.files)
    if n:
        # spread over languages: every (len / n)-th clip
        step = max(1, len(keys) // n)
        keys = keys[::step][:n]
    return [z[k] for k in keys]


_REF: dict = {}


def _ref(X):
    key = id(X)
    if key not in _REF:
        s = _session(FP32)
        _REF[key] = [_run(s, x) for x in X]
    return _REF[key]


def compare(sess, X, R) -> dict:
    mx = tot = cnt = 0.0
    kls, kmax, flips, frames = [], 0.0, 0, 0
    for x, r in zip(X, R):
        c = _run(sess, x)
        f = np.isfinite(r) & np.isfinite(c)
        d = np.abs(r - c)[f]
        mx = max(mx, float(d.max()))
        tot += float(d.sum())
        cnt += d.size
        pr = np.exp(r)
        kl = np.sum(pr * (np.where(f, r - c, 0.0)), axis=1)
        kls.append(kl)
        kmax = max(kmax, float(kl.max()))
        flips += int((r.argmax(1) != c.argmax(1)).sum())
        frames += len(r)
    k = np.concatenate(kls)
    return {"max_abs_dlogp": mx, "mean_abs_dlogp": tot / cnt, "mean_kl": float(k.mean()), "max_kl": kmax,
            "p999_kl": float(np.quantile(k, 0.999)), "top_token_changed_share": flips / frames, "frames": frames}


def fragility(sess, X, eps) -> dict:
    rng = np.random.default_rng(0)
    mx, tot, cnt, flips, frames = 0.0, 0.0, 0, 0, 0
    for x in X:
        a = _run(sess, x)
        b = _run(sess, (x + rng.normal(0, eps, x.shape)).astype(np.float32))
        f = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a - b)[f]
        mx = max(mx, float(d.max()))
        tot += float(d.sum())
        cnt += d.size
        flips += int((a.argmax(1) != b.argmax(1)).sum())
        frames += len(a)
    return {"eps": eps, "max_abs_dlogp": mx, "mean_abs_dlogp": tot / cnt, "top_token_changed_share": flips / frames}


def posterr(specs, n, out) -> None:
    X = _load_feats(n)
    R = _ref(X)
    res = {"kind": "opt_posterr", "date": time.strftime("%Y-%m-%d %H:%M"), "n_clips": len(X), "results": {}}
    p = Path(out) if out else None
    if p and p.exists():
        res = json.loads(p.read_text())
    for s in specs:
        name, path = s.split("=", 1)
        t0 = time.time()
        sess = _session(path)
        r = {"file": Path(path).name, "bytes": Path(path).stat().st_size, "load_s": round(time.time() - t0, 3),
             "vs_fp32": compare(sess, X, R)}
        Xf = X[: max(20, len(X) // 10)]
        r["fragility"] = [fragility(sess, Xf, e) for e in (1e-4, 3e-4)]
        res["results"][name] = r
        v, fr = r["vs_fp32"], r["fragility"]
        print(f"{name}: {r['bytes'] / 1e6:.1f} MB | vs fp32 max {v['max_abs_dlogp']:.3g} mean {v['mean_abs_dlogp']:.3g} "
              f"KL {v['mean_kl']:.2e} top-token {100 * v['top_token_changed_share']:.2f} % | fragility 1e-4 max "
              f"{fr[0]['max_abs_dlogp']:.3g} ({100 * fr[0]['top_token_changed_share']:.2f} %), 3e-4 max "
              f"{fr[1]['max_abs_dlogp']:.3g}", flush=True)
        if p:
            p.write_text(json.dumps(res, indent=1))


def groups(m) -> dict:
    init = {t.name for t in m.graph.initializer}
    G = defaultdict(list)
    for nd in m.graph.node:
        if nd.op_type == "MatMul" and nd.input[1] in init:
            parts = nd.name.strip("/").split("/")
            if parts[0] == "encoder" and len(parts) > 3:
                mod = re.sub(r"\d+$", "", parts[3]) if parts[2] in ("encoder",) else parts[2]
                # /encoder/<s>/encoder/<l>/<module>/... or /encoder/<s>/<...>
                mod = parts[4] if parts[2] == "encoder" and len(parts) > 4 else parts[2]
                key = f"enc{parts[1]}/{re.sub(r'[0-9]+$', '', mod)}"
            else:
                key = parts[0]
            G[key].append(nd.name)
    return dict(G)


def layers(n, out) -> None:
    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    X = _load_feats(n)
    R = _ref(X)
    base = onnx.load(str(FP32))
    G = groups(base)
    wsize = {t.name: int(np.prod(t.dims)) * 4 for t in base.graph.initializer}
    node_w = {nd.name: nd.input[1] for nd in base.graph.node if nd.op_type == "MatMul"}
    res = {"kind": "opt_sensitivity", "date": time.strftime("%Y-%m-%d %H:%M"), "n_clips": len(X), "groups": {}}
    tmp = SENS / "tmp_layer.onnx"
    for g, nodes in sorted(G.items()):
        t0 = time.time()
        m = onnx.load(str(FP32))
        others = [x for gg, nn in G.items() if gg != g for x in nn]      # nodes_to_include is not honoured
        q = MatMulNBitsQuantizer(m, bits=4, block_size=32, is_symmetric=True, nodes_to_exclude=others)
        q.process()
        onnx.save(q.model.model, str(tmp))
        r = compare(_session(tmp), X, R)
        r.update(nodes=nodes, fp32_bytes=sum(wsize[node_w[x]] for x in nodes), s=round(time.time() - t0, 1))
        res["groups"][g] = r
        print(f"{g:28s} {len(nodes):3d} MatMuls {r['fp32_bytes'] / 1e6:6.1f} MB  KL {r['mean_kl']:.2e}  "
              f"max {r['max_abs_dlogp']:.3g}  top {100 * r['top_token_changed_share']:.2f} %", flush=True)
        Path(out).write_text(json.dumps(res, indent=1))
    tmp.unlink(missing_ok=True)


def layouts() -> None:
    """Mixed 4 / 8-bit layouts chosen from layers.json alone (practice-clip sensitivity):
    A = every MatMul of encoder stack 3 at 4 bits (the stack with the lowest KL per MB);
    B / C = groups at 4 bits in increasing order of mean KL per fp32 weight byte until 150 / 200 MB of
    fp32 weights are covered; everything else at 8 bits. Written to sensitivity/<A|B|C>.json."""
    G = json.loads((SENS / "layers.json").read_text())["groups"]
    rank = sorted(G, key=lambda g: G[g]["mean_kl"] / G[g]["fp32_bytes"])
    all_nodes = [n for g in G.values() for n in g["nodes"]]

    def write(name, four_groups, rule):
        four = {n for g in four_groups for n in G[g]["nodes"]}
        lay = {"rule": rule, "four_groups": four_groups, "bits8": sorted(set(all_nodes) - four), "symmetric": True,
               "block4": 32, "block8": 32}
        (SENS / f"{name}.json").write_text(json.dumps(lay, indent=1))
        print(f"{name}: {len(four)} MatMuls 4-bit ({sum(G[g]['fp32_bytes'] for g in four_groups) / 1e6:.1f} MB fp32 "
              f"weights), sum of group KL {sum(G[g]['mean_kl'] for g in four_groups):.2e}; groups {four_groups}")

    write("A", [g for g in G if g.startswith("enc3/")], "4-bit: every MatMul of encoder stack 3; 8-bit: the rest")
    for name, limit in (("B", 150e6), ("C", 200e6)):
        acc, sel = 0, []
        for g in rank:
            if acc >= limit:
                break
            sel.append(g)
            acc += G[g]["fp32_bytes"]
        write(name, sel, f"4-bit: groups in increasing KL per MB until {limit / 1e6:.0f} MB of fp32 weights; 8-bit: rest")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("feats", "posterr", "layers", "layouts"))
    ap.add_argument("specs", nargs="*")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd in ("posterr", "layers"):
        global FP32
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from common import default_backbone

        FP32 = default_backbone("fp32")
    if a.cmd == "feats":
        feats()
    elif a.cmd == "posterr":
        posterr(a.specs, a.n, a.out)
    elif a.cmd == "layouts":
        layouts()
    else:
        layers(a.n or 40, a.out or str(SENS / "layers.json"))


if __name__ == "__main__":
    main()
