"""Tables of the "On-device model" section of docs/optimization_and_mobile_report.md.

    python -m bench.opt.ondevice_table [--fill]

Inputs ($PRONUNCIADA_CACHE/opt/...): results/posterr_*.json (bench/opt/sensitivity.py posterr),
results/<name>.parity.json (bench.opt.candidates parity), profile_small/opt_profile_small.json and
opt_rss_small.json (bench/opt/run_profile_small.sh), results/<name>.screen.json / .gate.json, and
results/tables/mobile_bundles.json of the repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import REPO, opt_root

R = opt_root() / "results"
ORDER = ["fp32", "int8-dyn", "nb8-sym-b32", "nb8-asym-b32", "nb8-sym-b128", "dq8-sym-ch", "dq8-asym-ch",
         "dq8-sym-tensor", "dq8-asym-tensor", "mix-A", "mix-B", "mix-C", "nb4-sym-b32", "nb4-sym-b128", "s8-minmax"]
SPEC = {"int8-dyn": "int8-shipped-vs-fp32", "dq8-sym-ch": "dq8-sym-ch-noqdq"}
DESC = {"fp32": "fp32 (default)", "int8-dyn": "shipped dynamic int8 (MatMulInteger, activations quantized per call)",
        "nb8-sym-b32": "8-bit weights, MatMulNBits, symmetric, block 32",
        "nb8-asym-b32": "8-bit weights, MatMulNBits, asymmetric, block 32",
        "nb8-sym-b128": "8-bit weights, MatMulNBits, symmetric, block 128",
        "dq8-sym-ch": "int8 per-channel + DequantizeLinear (posterior columns: default ORT options; "
                      "all other columns with session.disable_quant_qdq=1)",
        "dq8-asym-ch": "uint8 per-channel + DequantizeLinear (default ORT options)",
        "dq8-sym-tensor": "int8 per-tensor + DequantizeLinear (default ORT options)",
        "dq8-asym-tensor": "uint8 per-tensor + DequantizeLinear (default ORT options)",
        "mix-A": "4-bit encoder stack 3, 8-bit rest (MatMulNBits, block 32)",
        "mix-B": "4-bit for the least sensitive 150 MB of fp32 weights, 8-bit rest",
        "mix-C": "4-bit for the least sensitive 200 MB, 8-bit rest",
        "nb4-sym-b32": "4-bit weights, MatMulNBits, block 32", "nb4-sym-b128": "4-bit weights, block 128",
        "s8-minmax": "static int8 (QDQ, per-channel weights, fixed activation scales, MinMax calibration)"}


def _j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def posterr() -> dict:
    out = {}
    for f in sorted(R.glob("posterr_*.json")):
        out.update(json.loads(f.read_text())["results"])
    return out


def latency() -> dict:
    P = _j(opt_root() / "profile_small" / "opt_profile_small.json")
    if not P:
        return {}
    out = {}
    for r in P["results"]:
        o = out.setdefault(r["config"], {"wall": {}, "model": {}, "cold": [], "load": []})
        o["wall"].setdefault(r["threads"], []).append(r["stages"]["wall"]["p50"])
        o["model"].setdefault(r["threads"], []).append(r["stages"].get("model", {}).get("p50"))
        o["cold"].append(r["cold_start_s"])
        o["load"].append(r["load_s"])
    pd = {}
    for p in P["paired"]:
        pd.setdefault(p["b"], {}).setdefault(p["threads"], []).append(p["rel_mean"])
    for k, v in out.items():
        v["paired_rel_vs_fp32"] = {t: float(np.median(x)) for t, x in pd.get(k, {}).items()}
    return out


def rss() -> dict:
    P = _j(opt_root() / "profile_small" / "opt_rss_small.json")
    out = {}
    for r in (P or {}).get("results", []):
        out.setdefault(r["config"], []).append((r["load_only_mb"], r["after_n_mb"]))
    return {k: (float(np.median([a for a, _ in v])), float(np.median([b for _, b in v]))) for k, v in out.items()}


def candidate_md() -> str:
    E, LAT, RSS = posterr(), latency(), rss()
    L = ["Posterior columns: the backbone alone against fp32 on the 200 practice clips (`sensitivity.py posterr`, "
         "21,596 frames; KL = mean per-frame KL(fp32 || candidate)); fragility: Gaussian feature noise of sd 1e-4 on "
         "20 clips, the candidate against itself (fp32: 0.0041 nats). Score columns: whole Scorer, 200 practice "
         "clips, max |difference| at sentence / word / sound level of the 0-1 probabilities (`candidates parity`). "
         "Latency: p50 ms per clip (about 10 s of audio), median over en / de / ja / zh, and the paired mean "
         "difference against the fp32 default; load: process start to Scorer built (every candidate loads its file "
         "through PRONUNCIADA_BACKBONE; `fp32-file` is fp32 loaded the same way). RSS: peak MB, load only / after "
         "10 clips, median over en / zh and 1 / 4 threads.", "",
         "| candidate | MB | max abs dlogp | mean abs dlogp | KL | top token changed | fragility 1e-4 max (top changed) | "
         "max diff sentence / word / sound P | p50 ms 1 / 2 / 4 thr (vs fp32) | load s | RSS MB load / after | screen | gate strict / budget |",
         "|---|--:|--:|--:|--:|--:|---|---|---|--:|---|---|---|"]
    for name in ORDER:
        e = E.get(name)
        if not e:
            continue
        v, fr = e["vs_fp32"], e["fragility"][0]
        sp = SPEC.get(name, name)
        par = _j(R / f"{sp}.parity.json")
        if par and name != "fp32":
            bc = par["by_category"]
            if "sentence P" in bc:
                pc = " / ".join(f"{bc[k]['max_abs_diff']:.3f} (mean {bc[k]['mean_abs_diff']:.4f})"
                                for k in ("sentence P", "word P", "sound P"))
            else:
                pc = "–"
        else:
            pc = "0 / 0 / 0" if name == "fp32" else "–"
        cfg = {"fp32": "fp32", "int8-dyn": "int8-shipped-vs-fp32", "dq8-sym-ch": "dq8-sym-ch-noqdq"}.get(name, name)
        la = LAT.get(cfg)
        if la:
            rel = la["paired_rel_vs_fp32"]
            lt = " / ".join(f"{np.median(la['wall'][t]):.0f}" + (f" ({100 * rel[t]:+.0f} %)" if t in rel else "")
                            for t in sorted(la["wall"]))
            ld = f"{np.median(la['load']):.2f}"
        else:
            lt = ld = "–"
        rs = RSS.get(cfg)
        rs = f"{rs[0]:.0f} / {rs[1]:.0f}" if rs else "–"
        sc = _j(R / f"{sp}.screen.json")
        sc = sc["verdict"].split(" (")[0] if sc else "–"
        g = _j(R / f"{sp}.gate.json")
        gs = f"{g['verdict']} / {g.get('verdict_budget', '–')}" if g else "–"
        L.append(f"| {name}: {DESC.get(name, '')} | {e['bytes'] / 1e6:.1f} | {v['max_abs_dlogp']:.3g} | {v['mean_abs_dlogp']:.3g} | "
                 f"{v['mean_kl']:.1e} | {100 * v['top_token_changed_share']:.2f} % | {fr['max_abs_dlogp']:.3g} "
                 f"({100 * fr['top_token_changed_share']:.2f} %) | {pc} | {lt} | {ld} | {rs} | {sc} | {gs} |")
    if "fp32-file" in LAT:
        L += ["", f"Load of fp32 through PRONUNCIADA_BACKBONE (`fp32-file`): {np.median(LAT['fp32-file']['load']):.2f} s; "
              f"fp32 with the pre-optimized graph (the default when this was measured; now `PRONUNCIADA_PREOPT=1`): {np.median(LAT['fp32']['load']):.2f} s."]
    return "\n".join(L)


ORT_LIB = [("macOS arm64 desktop, onnxruntime 1.32.0 wheel, libonnxruntime.dylib", 33.6, "measured here"),
           ("Android, pre-built onnxruntime-android AAR 1.18.0 (all ABIs, full operator set)", 24.4,
            "quoted: ONNX Runtime custom-build documentation example"),
           ("Android, custom reduced-operator AAR 1.18.0 (same example)", 7.5, "quoted, same source"),
           ("Android arm64, onnxruntime-mobile 1.11 libonnxruntime.so", 3.3, "quoted: ONNX Runtime 1.11 mobile docs"),
           ("iOS, onnxruntime-mobile-c 1.11 static library", 22.0, "quoted: ONNX Runtime 1.11 mobile docs (static, before linking)"),
           ("Web, ort-wasm-simd-threaded.wasm", 10.6, "quoted: hosted copies of onnxruntime-web, 10 to 12.3 MB")]


def size_md(cands=("fp32", "int8-dyn", "nb8-sym-b128", "dq8-sym-ch", "mix-A", "mix-B", "mix-C")) -> str:
    E = posterr()
    MB = _j(REPO / "results" / "tables" / "mobile_bundles.json")
    L = ["Total on device = backbone file + the language's scoring bundle (`results/tables/mobile_bundles.json`) + "
         "the ONNX Runtime library. The library sizes are not measured on a phone; see the table after this one.", "",
         "Backbone + bundle, MB (library not included):", "",
         "| lang | bundle MB | " + " | ".join(cands) + " |", "|---|--:|" + "--:|" * len(cands)]
    bb = {c: E[c]["bytes"] / 1e6 for c in cands if c in E}
    for row in MB["languages"]:
        b = row["bundle_bytes"] / 1e6
        L.append(f"| {row['lang']} | {b:.1f} | " + " | ".join(f"{bb[c] + b:.1f}" for c in cands if c in bb) + " |")
    L += ["", "ONNX Runtime library (a phone app ships one of these):", "", "| build | MB | source |", "|---|--:|---|"]
    L += [f"| {a} | {b:.1f} | {c} |" for a, b, c in ORT_LIB]
    L += ["", "MatMulNBits is an ONNX Runtime contrib operator (com.microsoft domain). The pre-built mobile and web "
          "packages include contrib operators; a reduced-operator build must list MatMulNBits (and, for the DequantizeLinear "
          "candidates, DequantizeLinear) in its operator configuration. Not verified on a phone."]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true")
    a = ap.parse_args(argv)
    c, s = candidate_md(), size_md()
    if a.fill:
        from .report import DOC, fill

        fill(DOC, "ondevice-candidates", c)
        fill(DOC, "ondevice-size", s)
    print(c + "\n\n" + s)


if __name__ == "__main__":
    main()
