"""Small-backbone candidates: weights stored in 4 to 8 bits, activations fp32 (and one static int8).

Runs in the tools virtualenv (onnx, onnxruntime 1.32.0 quantization tools; it does not import
pronunciada), from the repository root:
    ~/.cache/pronunciada/opt/venv_tools/bin/python bench/opt/build_small.py <name> [...]

Output: $PRONUNCIADA_CACHE/opt/candidates/<name>/model.onnx + build.json. Every candidate keeps the
intermediate tensors the Scorer reads as extra outputs (they are graph outputs while the graph is
rewritten, then removed, as in build_candidates.py). Source: model.onnx (fp32). Only the 290 MatMul
weight matrices (254 of the 260 MB) are changed; the rest stays fp32.

Names:
  nb<bits>-<sym|asym>-<b32|b128|ch>     onnxruntime MatMulNBitsQuantizer, QOperator format (MatMulNBits
                                        op; the kernel dequantizes blocks of weights and computes in fp32
                                        with the default accuracy level), block size 32 / 128 or per
                                        output channel (channel_wised_quantize)
  dq<bits>-<sym|asym>-<b128|ch>         the same quantizer in QDQ format: an int<bits> initializer, a
                                        DequantizeLinear and an fp32 MatMul (ONNX Runtime may fold the
                                        DequantizeLinear at load, so the weights then sit in memory as fp32)
  dq8-<sym|asym>-tensor                 one scale (and zero point) per weight matrix, written by this script:
                                        int8 / uint8 initializer -> DequantizeLinear -> MatMul
  mix-<layout file stem>                MatMulNBits, 8-bit for the nodes listed in
                                        $PRONUNCIADA_CACHE/opt/sensitivity/<stem>.json "bits8", 4-bit for the rest
                                        (both blockwise, block 32, symmetric unless the file says otherwise)
  s8-<minmax|percentile>                static QDQ int8 (onnxruntime quantize_static): per-channel int8 weights,
                                        uint8 activations with fixed scales calibrated on fbank features of
                                        FLEURS-train practice audio (calibration list written by
                                        bench/opt/sensitivity.py calib, disjoint from the parity clips)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_candidates import KEEP, MODELS, OUT, save, sha1, strip_outputs, with_outputs  # noqa: E402

CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")


def _nbits(m, bits, sym, block, fmt, include=None):
    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    kw = {}
    if block == "ch":
        kw["channel_wised_quantize"] = True
        block = 128
    exclude = None
    if include is not None:          # nodes_to_include is not honoured by onnxruntime 1.32: exclude the rest
        inc = set(include)
        exclude = [n.name for n in m.graph.node if n.op_type == "MatMul" and n.name not in inc]
    q = MatMulNBitsQuantizer(m, bits=bits, block_size=block, is_symmetric=sym,
                             quant_format=QuantFormat.QDQ if fmt == "dq" else QuantFormat.QOperator,
                             nodes_to_exclude=exclude, **kw)
    q.process()
    return q.model.model


def _per_tensor(m, sym: bool, per_channel: bool = False):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    init = {t.name: t for t in m.graph.initializer}
    new_nodes, drop = [], set()
    for n in m.graph.node:
        if n.op_type == "MatMul" and n.input[1] in init:
            w = numpy_helper.to_array(init[n.input[1]]).astype(np.float32)
            base = n.input[1]
            ax = 0 if per_channel else None          # W [K, N]: one scale per output column N
            red = (lambda f, a: f(a, axis=0)) if per_channel else (lambda f, a: f(a))
            if sym:
                s = np.maximum(red(np.max, np.abs(w)), 1e-12) / 127.0
                q = np.clip(np.round(w / s), -127, 127).astype(np.int8)
                zp = np.zeros_like(s, dtype=np.int8)
            else:
                lo, hi = np.minimum(red(np.min, w), 0), np.maximum(red(np.max, w), 0)
                s = np.maximum(hi - lo, 1e-12) / 255.0
                zp = np.clip(np.round(-lo / s), 0, 255).astype(np.uint8)
                q = np.clip(np.round(w / s) + zp.astype(np.float32), 0, 255).astype(np.uint8)
            s = np.asarray(s, np.float32)
            m.graph.initializer.extend([numpy_helper.from_array(q, base + "__q"),
                                        numpy_helper.from_array(s, base + "__s"),
                                        numpy_helper.from_array(np.asarray(zp), base + "__zp")])
            kw = {"axis": 1} if per_channel else {}
            new_nodes.append(helper.make_node("DequantizeLinear", [base + "__q", base + "__s", base + "__zp"],
                                              [base + "__dq"], name=base + "__DQ", **kw))
            n.input[1] = base + "__dq"
            drop.add(base)
    keep = [t for t in m.graph.initializer if t.name not in drop]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes + nodes)
    return m


class _Reader:
    """Calibration batches: fbank features of the calibration clips (written by sensitivity.py calib)."""

    def __init__(self, path: Path):
        import numpy as np

        z = np.load(path, allow_pickle=True)
        self.items = [z[k] for k in sorted(z.files)][::int(os.environ.get("OPT_CALIB_STEP", "3"))]
        self.i = 0

    def get_next(self):
        import numpy as np

        if self.i >= len(self.items):
            return None
        x = self.items[self.i].astype("float32")
        self.i += 1
        return {"x": x[None], "x_lens": np.array([len(x)], np.int64)}

    def rewind(self):
        self.i = 0


def build(name: str) -> dict:
    import onnx

    import onnxruntime as ort

    from common import default_backbone

    src = default_backbone("fp32")
    d = Path(os.environ["OPT_BUILD_DIR"]) / name if os.environ.get("OPT_BUILD_DIR") else OUT / name
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info = {"name": name, "source": src.name, "source_sha1": sha1(src), "source_bytes": src.stat().st_size,
            "onnx": onnx.__version__, "onnxruntime": ort.__version__, "date": time.strftime("%Y-%m-%d %H:%M")}
    p = d / "model.onnx"
    if name.startswith("nb"):
        kind, sym, block = name.split("-")
        fmt, bits = kind[:2], int(kind[2:])
        m = strip_outputs(_nbits(with_outputs(onnx.load(str(src))), bits, sym == "sym",
                                 block if block == "ch" else int(block[1:]), fmt))
        info["settings"] = (f"MatMulNBitsQuantizer(bits={bits}, block={block}, symmetric={sym == 'sym'}, "
                            f"format={'QDQ' if fmt == 'dq' else 'QOperator'})")
        save(m, d)
    elif name.startswith("dq8-"):
        sym, gran = name.split("-")[1] == "sym", name.split("-")[2]
        m = _per_tensor(onnx.load(str(src)), sym, per_channel=gran == "ch")
        info["settings"] = (f"per-{'channel' if gran == 'ch' else 'tensor'} {'symmetric int8' if sym else 'asymmetric uint8'}"
                            " initializer + DequantizeLinear + fp32 MatMul")
        save(m, d)
    elif name.startswith("mix-"):
        lay = json.loads((CACHE / "opt" / "sensitivity" / f"{name[4:]}.json").read_text())
        m = with_outputs(onnx.load(str(src)))
        b8 = set(lay["bits8"])
        all_mm = [n.name for n in m.graph.node if n.op_type == "MatMul" and
                  any(n.input[1] == t.name for t in m.graph.initializer)]
        four = [n for n in all_mm if n not in b8]
        sym = lay.get("symmetric", True)
        m = _nbits(m, 4, sym, lay.get("block4", 32), "nb", include=four)
        if b8:
            m = _nbits(m, 8, sym, lay.get("block8", 32), "nb", include=sorted(b8))
        m = strip_outputs(m)
        info["settings"] = (f"MatMulNBits: {len(b8)} MatMuls 8-bit, {len(four)} 4-bit (layout {name[4:]}: "
                            f"{lay.get('rule', '')})")
        info["bits8"] = sorted(b8)
        save(m, d)
    elif name.startswith("s8-"):
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

        meth = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}[name.split("-")[1]]
        tmp = d / "with_outputs.onnx"
        onnx.save(with_outputs(onnx.load(str(src))), str(tmp))
        cal = CACHE / "opt" / "sensitivity" / "calib_feats.npz"
        quantize_static(str(tmp), str(d / "q.onnx"), _Reader(cal), quant_format=QuantFormat.QDQ, per_channel=True,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, op_types_to_quantize=["MatMul"],
                        calibrate_method=meth, extra_options={"CalibTensorRangeSymmetric": False})
        m = strip_outputs(onnx.load(str(d / "q.onnx")))
        tmp.unlink()
        (d / "q.onnx").unlink()
        info["settings"] = (f"quantize_static(QDQ, MatMul only, per-channel QInt8 weights, QUInt8 activations, "
                            f"{name.split('-')[1]} calibration on {cal.name})")
        save(m, d)
    else:
        raise SystemExit(f"unknown candidate {name}")
    mm = onnx.load(str(p), load_external_data=False)
    produced = {o for n in mm.graph.node for o in n.output}
    info["missing_kept_tensors"] = [n for n in KEEP if n not in produced]
    info["bytes"] = p.stat().st_size
    info["sha1"] = sha1(p)
    from collections import Counter

    info["ops"] = dict(Counter(n.op_type for n in mm.graph.node if n.op_type in
                               ("MatMul", "MatMulNBits", "DequantizeLinear", "QuantizeLinear", "MatMulInteger")))
    info["build_s"] = round(time.time() - t0, 1)
    (d / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("name", "bytes", "missing_kept_tensors", "ops", "build_s")}), flush=True)
    return info


if __name__ == "__main__":
    for n in sys.argv[1:]:
        build(n)
